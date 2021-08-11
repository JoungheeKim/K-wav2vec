# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec import (
    Wav2Vec2AsrConfig,
    TransformerEncoder,
    TransformerSentenceEncoderLayer,
)

from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer

@dataclass
class Wav2Vec2Ctc2Config(Wav2Vec2AsrConfig):
    additional_layers: int = field(
        default=2, metadata={"help": "number of additional layer in wav2vec 2.0"}
    )
    pass


@register_model("wav2vec_ctc2", dataclass=Wav2Vec2Ctc2Config)
class Wav2VecCtc2(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2Ctc2Config, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Ctc2Config, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder2(cfg, task.target_dictionary, task.additional_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

    def get_normalized_addprobs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["additional_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_addlogits(self, net_output):
        logits = net_output["additional_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits


    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

class Wav2VecEncoder2(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrConfig, tgt_dict=None, tgt_dict2=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model
        self.additional_layer = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=w2v_args.model.encoder_embed_dim,
                    ffn_embedding_dim=self.w2v_model.cfg.encoder_ffn_embed_dim,
                    num_attention_heads=self.w2v_model.cfg.encoder_attention_heads,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation_dropout=cfg.activation_dropout,
                    activation_fn=self.w2v_model.cfg.activation_fn,
                    layer_norm_first=self.w2v_model.cfg.layer_norm_first,
                )
                for _ in range(cfg.additional_layers)
            ]
        )

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

        if tgt_dict2 is not None:
            self.proj2 = Linear(d, len(tgt_dict2))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj2 = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj2 = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        additional_x = x
        for i, layer in enumerate(self.additional_layer):
            additional_x, z = layer(additional_x, self_attn_padding_mask=padding_mask, need_weights=False)

        if not tbc:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)
            additional_x = additional_x.transpose(0, 1)

        ## lower token out
        x = self.final_dropout(x)
        if self.proj:
            x = self.proj(x)

        ## upper token out
        additional_x = self.final_dropout(additional_x)
        if self.proj2:
            additional_x = self.proj2(additional_x)

        return {
            "encoder_out": x,  # T x B x C
            "additional_out": additional_x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
