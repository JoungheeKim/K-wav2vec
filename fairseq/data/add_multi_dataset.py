# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import BaseWrapperDataset, data_utils


class AddMultiDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        add_labels,
        pad,
        eos,
        add_pad,
        add_eos,
        batch_targets,
        process_label=None,
        add_process_label=None,
        add_to_input=False,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.add_labels = add_labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.add_pad = add_pad
        self.add_eos = add_eos
        self.process_label = process_label
        self.add_process_label = add_process_label
        self.add_to_input = add_to_input

    def get_label(self, index):
        return (
            self.labels[index]
            if self.process_label is None
            else self.process_label(self.labels[index])
        )

    def get_add_label(self, index):
        return (
            self.add_labels[index]
            if self.add_process_label is None
            else self.add_process_label(self.add_labels[index])
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)
        item["add_label"] = self.get_add_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        add_target = [s["add_label"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()

            collated["add_target_lengths"] = torch.LongTensor([len(t) for t in add_target])
            add_target = data_utils.collate_tokens(add_target, pad_idx=self.add_pad, left_pad=False)
            collated["add_ntokens"] = collated["add_target_lengths"].sum().item()
        else:
            collated["ntokens"] = sum([len(t) for t in target])
            collated["add_ntokens"] = sum([len(t) for t in add_target])

        collated["target"] = target
        collated["add_target"] = add_target

        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1).long()
            collated["net_input"]["prev_output_tokens"] = torch.cat(
                [eos, target], dim=-1
            ).long()
            collated["ntokens"] += target.size(0)

            add_eos = add_target.new_full((add_target.size(0), 1), self.add_eos)
            collated["add_target"] = torch.cat([add_target, add_eos], dim=-1).long()
            collated["net_input"]["prev_output_add_tokens"] = torch.cat(
                [add_eos, add_target], dim=-1
            ).long()
            collated["add_ntokens"] += add_target.size(0)

        return collated
