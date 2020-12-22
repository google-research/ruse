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
from typing import TypeVar, Optional

import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)


class MultiTaskBatchSampler(Sampler[T_co]):
  """Defines a sampler to sample across multiple tasks."""
  def __init__(self, dataset_sizes, batch_size: int, temperature,
               num_replicas: Optional[int] = None, rank: Optional[int] = None,
               seed: int = 0, shuffle: bool=True) -> None:
    if num_replicas is None:
          if not dist.is_available():
              raise RuntimeError("Requires distributed package to be available")
          num_replicas = dist.get_world_size()
    if rank is None:
          if not dist.is_available():
              raise RuntimeError("Requires distributed package to be available")
          rank = dist.get_rank()
    if rank >= num_replicas or rank < 0:
          raise ValueError(
              "Invalid rank {}, rank should be in the interval"
              " [0, {}]".format(rank, num_replicas - 1))

    self.num_replicas = num_replicas
    self.rank = rank
    self.batch_size = batch_size
    self.dataset_sizes = dataset_sizes
    # By default we drop the last elements if dataset is not divisble by the number of ranks.
    self.rank_dataset_sizes = [dataset_size//self.num_replicas for dataset_size in self.dataset_sizes]
    self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
    self.total_sizes = [(dataset_size//self.num_replicas)*self.num_replicas for dataset_size in self.dataset_sizes]
    self.temperature = temperature
    self.seed = seed
    self.epoch = 0
    self.num_batches_per_epoch = (np.sum(dataset_sizes) + self.batch_size - 1) // self.batch_size // self.num_replicas
    self.shuffle = shuffle

  def generate_tasks_distribution(self):
    total_size = sum(self.dataset_sizes)
    weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
    weights = weights / np.sum(weights)
    return torch.as_tensor(weights, dtype=torch.double)

  def __iter__(self):
    # Defines torch generator, to make random choices consistent acorss cores and epochs
    # the seed needs to be set based on seed and epoch.
    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch)

    # Shuffles the data if needed.
    indices = []
    for dataset_size in self.dataset_sizes:
        if self.shuffle:
            indices.append(torch.randperm(dataset_size, generator=generator).tolist())
        else:
            indices.append(list(range(dataset_size)))

    # Shards the datasets across the all processes.
    self.rank_indices = []
    for i in range(len(self.dataset_sizes)):
        self.rank_indices.append(indices[i][self.rank:self.total_sizes[i]:self.num_replicas])

    # We have to ensure that:
    #    - each process gets the same task.
    #    - indices are per process.
    tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

    # For each batch, which task to use. torch.Generator ensures that this choice is consistent
    # accross processes.
    batch_task_assignments = torch.multinomial(tasks_distribution,
                                               self.num_batches_per_epoch, replacement=True, generator=generator)

    for batch_task in batch_task_assignments:
      num_task_samples = self.rank_dataset_sizes[batch_task]
      indices = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,), generator=generator).tolist()
      results = (self.dataset_offsets[batch_task] + torch.tensor(self.rank_indices[batch_task])[indices]).tolist()
      yield results

  def __len__(self):
    return self.num_batches_per_epoch

  def set_epoch(self, epoch):
    self.epoch = epoch
