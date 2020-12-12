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
"""Implements a distributed sampler to sample different tasks in a way
to make sure that the same task is selected in each core."""
from typing import TypeVar

import numpy as np
import torch
from torch.utils.data import Sampler

T_co = TypeVar('T_co', covariant=True)


class MultiTaskBatchSampler(Sampler[T_co]):
  def __init__(self, dataset_sizes, batch_size: int, temperature, seed: int = 0) -> None:
    self.batch_size = batch_size
    self.dataset_sizes = dataset_sizes
    self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
    self.temperature = temperature
    self.seed = seed
    self.epoch = 0
    self.num_batches_per_epoch = (np.sum(dataset_sizes) + self.batch_size - 1) // self.batch_size

  def generate_tasks_distribution(self):
    total_size = sum(self.dataset_sizes)
    weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
    weights = weights / np.sum(weights)
    return torch.as_tensor(weights, dtype=torch.double)

  def __iter__(self):
    # TODO: we need to have shuffle here.
    # We have to ensure that:
    #    - each process gets the same task.
    #    - indices are per process.
    tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

    # For each batch, which task to use. torch.Generator ensures that this choice is consistent
    # accross processes.
    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch)
    batch_task_assignments = torch.multinomial(tasks_distribution,
                                               self.num_batches_per_epoch, replacement=True, generator=generator)
    for batch_task in batch_task_assignments:
      num_task_samples = self.dataset_sizes[batch_task]
      yield (self.dataset_offsets[batch_task] + torch.randint(low=0, high=num_task_samples,
                                                              size=(self.batch_size,), generator=generator)).tolist()

  def __len__(self):
    return self.num_batches_per_epoch

  def set_epoch(self, epoch):
    self.epoch = epoch
