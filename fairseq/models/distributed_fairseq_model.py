# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import os
import signal
import threading

import torch
import torch.nn as nn

from fairseq import distributed_utils
from fairseq.legacy_distributed_data_parallel import LegacyDistributedDataParallel


logger = logging.getLogger(__name__)


_GOSSIP_DISABLED = False
try:
    import gossip
except ImportError:
    _GOSSIP_DISABLED = True


def DistributedFairseqModel(args, model, process_group):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
    """
    # determine which DDP class to extend
    assert isinstance(model, nn.Module)
    if args.tpu:
        ddp_class = TPUDistributedDataParallel
        init_kwargs = dict(
            module=model,
            process_group=process_group,
        )
    elif args.distributed_wrapper == "DDP" and args.ddp_backend == "c10d":
        ddp_class = nn.parallel.DistributedDataParallel
        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
        )
        # Maintain backward compatibility
        if "find_unused_parameters" in inspect.getargspec(ddp_class)[0]:
            init_kwargs["find_unused_parameters"] = args.find_unused_parameters
    elif args.distributed_wrapper == "DDP" and args.ddp_backend == "no_c10d":
        ddp_class = LegacyDistributedDataParallel
        init_kwargs = dict(
            module=model,
            buffer_size=2 ** 28,
            process_group=process_group,
        )
    elif args.distributed_wrapper == "SlowMo":
        if _GOSSIP_DISABLED:
            raise ImportError(
                "Cannot find gossip library. Please install from: "
                "github.com/facebookresearch/stochastic_gradient_push"
            )
        ddp_class = gossip.GossipDataParallel

        # The values of slowmo_momentum below were obtained by tuning on the
        # En-De 16 dataset by training the transformer_wmt_en_de_large model
        if args.slowmo_momentum is None:
            if args.distributed_world_size <= 16:
                args.slowmo_momentum = 0.0
            elif args.distributed_world_size <= 32:
                args.slowmo_momentum = 0.2
            elif args.distributed_world_size <= 64:
                args.slowmo_momentum = 0.5
            else:
                args.slowmo_momentum = 0.6

        init_kwargs = dict(
            module=model,
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            nprocs_per_node=args.nprocs_per_node,
            slowmo_momentum=args.slowmo_momentum,
            localsgd=(args.slowmo_algorithm == "LocalSGD"),
            localsgd_frequency=args.localsgd_frequency,
        )
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)

    heartbeat_timeout = getattr(args, "heartbeat_timeout", -1)

    class _DistributedFairseqModel(ddp_class):
        """
        Extend DistributedDataParallel to check for missing attributes in the
        wrapped module and to add a timeout to kill the job if no progress is
        made (--heartbeat-timeout).
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._heartbeat_timeout = heartbeat_timeout
            if self._heartbeat_timeout > 0:
                self._heartbeat = threading.Event()
                self._heartbeat_thread = threading.Thread(
                    target=self._check_heartbeat,
                    args=(os.getpid(),),
                    daemon=True,
                )
                self._heartbeat_thread.start()
            else:
                self._heartbeat = None

        def _check_heartbeat(self, parent_pid):
            self._heartbeat.wait()  # wait for the first forward pass
            while True:
                self._heartbeat.clear()
                success = self._heartbeat.wait(timeout=self._heartbeat_timeout)
                if not success:
                    logger.error((
                        "Killing job for not making progress in {} seconds. "
                        "Set --heartbeat-timeout=-1 to disable this timeout."
                    ).format(int(self._heartbeat_timeout)))
                    os.kill(parent_pid, signal.SIGKILL)
                    return

        def __getattr__(self, name):
            wrapped_module = super().__getattr__("module")
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

        def forward(self, *args, **kwargs):
            if self._heartbeat is not None:
                self._heartbeat.set()
            return super().forward(*args, **kwargs)

    return _DistributedFairseqModel(**init_kwargs)


class TPUDistributedDataParallel(nn.Module):

    def __init__(self, module, process_group):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = distributed_utils.get_world_size(self.process_group)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        gradients = []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if p.grad.requires_grad:
                raise RuntimeError(
                    "TPUDistributedDataParallel only works with gradients that don't "
                    "require grad"
                )
            gradients.append(p.grad)

        import torch_xla.core.xla_model as xm
        xm.all_reduce(
            'sum',
            gradients,
            scale=1. / self.world_size,
            groups=self.process_group[1],
        )
