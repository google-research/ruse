# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implements task-collator to collate the samples in each batch."""
import torch
from typing import Dict

class TaskCollator:
  def __init__(self, tokenizer, data_args, tpu_num_cores=None):
    self.tokenizer = tokenizer
    self.pad_token_id = tokenizer.pad_token_id
    assert (
        self.pad_token_id is not None
    ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
    self.data_args = data_args
    self.tpu_num_cores = tpu_num_cores

  def __call__(self, batch) -> Dict[str, torch.Tensor]:
    # because of padding="longest" this does not work to be done in dataset part.
    batch = self._encode(batch)
    input_ids, attention_mask, labels = (
      batch["input_ids"],
      batch["attention_mask"],
      batch["labels"],
    )
    decoder_input_ids = self._shift_right_t5(labels)
    output_batch = {
      "input_ids": input_ids,
      "attention_mask": attention_mask,
      "decoder_input_ids": decoder_input_ids,
      "labels": labels,
    }
    output_batch["task"] = batch["task"]
    return output_batch

  def _shift_right_t5(self, input_ids):
    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = self.pad_token_id
    return shifted_input_ids

  def _encode(self, batch) -> Dict[str, torch.Tensor]:
    batch_encoding = self.tokenizer.prepare_seq2seq_batch(
      [x["src_texts"] for x in batch],
      tgt_texts=[x["tgt_texts"] for x in batch],
      max_length=self.data_args.max_source_length,
      max_target_length=self.data_args.max_target_length,
      padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
      return_tensors="pt"
    )
    tasks = [x["task"] for x in batch]
    # There should be only one task per batch.
    assert (len(set(tasks)) == 1)
    batch_encoding["task"] = tasks[0]
    return batch_encoding.data
