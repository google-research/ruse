import datasets
from torch.utils.data.dataloader import DataLoader
import numpy as np
from seq2seq.data import TaskCollator, AutoTask
import torch
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from seq2seq.training_args import DataTrainingArguments
from typing import TypeVar, Optional
import torch.distributed as dist
import sys

T_co = TypeVar('T_co', covariant=True)


def shard_data(datasets, num_replicas, rank):
    """Returns the sharded data belonging to the given rank."""
    for i, dataset in enumerate(datasets):
        # shuffle needs to be per epoch as well.
        dataset = dataset.shuffle()
        sharded_dataset = dataset.shard(num_replicas, rank)
        datasets[i] = sharded_dataset
    return datasets

'''
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
      results= (self.dataset_offsets[batch_task] + torch.randint(low=0, high=num_task_samples,
                                                              size=(self.batch_size,), generator=generator)).tolist()
      print(results)
      yield results

  def __len__(self):
    return self.num_batches_per_epoch

  def set_epoch(self, epoch):
    self.epoch = epoch

'''
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
    #print("@@@@ dataset_sizes ", self.dataset_sizes)
    self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
    #print("@@@@ dataset offsets ", self.dataset_offsets)
    self.total_sizes = [(dataset_size//self.num_replicas)*self.num_replicas for dataset_size in self.dataset_sizes]
    #print("@@@ total_sizes ", self.total_sizes)
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
    #print("@@@ indices ", indices)

    # Shards the datasets across the all processes.
    self.rank_indices = []
    #self.rank_dataset_sizes = []
    for i in range(len(self.dataset_sizes)):
        #n = self.dataset_sizes[i]//self.num_replicas
        #print("all idnices ", indices[i])
        #print("selected ones ", indices[i][self.rank*n:self.rank*n+n])
        self.rank_indices.append(indices[i][self.rank:self.total_sizes[i]:self.num_replicas])
        #indices[i][self.rank*n:self.rank*n+n])
        #self.rank_dataset_sizes.append(len(self.rank_indices[i]))

    #print("@@@ rank_indices ", self.rank_indices)
    #sys.exit(0)
    #self.rank_dataset_offsets = torch.cumsum(torch.LongTensor([0] + self.rank_dataset_sizes), 0)

    # We have to ensure that:
    #    - each process gets the same task.
    #    - indices are per process.
    tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

    # For each batch, which task to use. torch.Generator ensures that this choice is consistent
    # accross processes.
    batch_task_assignments = torch.multinomial(tasks_distribution,
                                               self.num_batches_per_epoch, replacement=True, generator=generator)
    #print(batch_task_assignments)

    for batch_task in batch_task_assignments:
      num_task_samples = self.rank_dataset_sizes[batch_task]
      indices = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,), generator=generator).tolist()

      #print(indices[[0, 1]])
      #print("indices ", indices)
      #print("rank indices ", self.rank_indices[batch_task])
      results = (self.dataset_offsets[batch_task] + torch.tensor(self.rank_indices[batch_task])[indices]).tolist()
      #print("@@@ results ", results)
      yield results

  def __len__(self):
    return self.num_batches_per_epoch

  def set_epoch(self, epoch):
    self.epoch = epoch




if __name__ == "__main__":
    rank = 0
    num_replicas = 4
    dataset1 = AutoTask.get("squad").get_dataset(split="train", n_obs="16", add_prefix=False)
    dataset2 = AutoTask.get("cola").get_dataset(split="train", n_obs="32")
    train_datasets = [dataset1, dataset2]

    #train_datasets = shard_data(train_datasets, num_replicas=num_replicas, rank=rank)
    #dataset1 = datasets.load_dataset('glue', 'cola', split="train")
    #dataset2 = datasets.load_dataset('glue', 'rte', split="train")
    """
    print(dataset)
    seed = 42
    epoch = 1
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=g).tolist()  # type: ignore
    #shuffled_indices = shuffle(range(len(dataset)))
    shuffled_dataset = dataset.select(indices)
    print(len(shuffled_dataset))
    shuffled_dataset = dataset.select([1, 2, 3, 4])
    print(shuffled_dataset)
    print(len(shuffled_dataset))
    """
    #train_datasets = [dataset1, dataset2]

    dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
    train_dataset = datasets.concatenate_datasets(train_datasets)
    sampler = MultiTaskBatchSampler(dataset_sizes, batch_size=2, temperature=10, rank=0, num_replicas=1)

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    data_args = DataTrainingArguments(sampling=True,
                                      max_source_length=128, max_target_length=128, val_max_target_length=128,
                                      test_max_target_length=128, n_train=10, n_val=-1, n_test=-1, eval_beams=None,
                                      ignore_pad_token_for_loss=True)
    collator = TaskCollator(tokenizer, data_args, tpu_num_cores=0)
    dataloader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collator)
    for i, batch in enumerate(dataloader):
        pass
        #print(i, batch['task'], batch)
        #print("_"*100)

