import datasets 
from torch.utils.data.dataloader import DataLoader
import numpy as np
from seq2seq.tasks import AutoTask, TaskCollator
import torch
from torch.utils.data import Sampler, SequentialSampler
from typing import TypeVar, List
from transformers import AutoTokenizer
from seq2seq.training_args import  DataTrainingArguments

T_co = TypeVar('T_co', covariant=True)

class MultiTaskBatchSampler(Sampler[T_co]):
    def __init__(self, dataset_sizes, batch_size: int, temperature, seed: int = 0) -> None:
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0]+dataset_sizes), 0)
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.num_batches_per_epoch = (np.sum(dataset_sizes) + self.batch_size - 1)//self.batch_size

    def generate_tasks_distribution(self):
        total_size = sum(self.dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        # TODO: we need to have shuffle here?
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
            # TODO: this is always with replacement, we can also think to do it without
            #   replacement
            yield (self.dataset_offsets[batch_task] + torch.randint(low=0, high=num_task_samples,size=(self.batch_size,), generator=generator)).tolist()

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch


def shard_data(datasets, num_replicas, rank):
        """Returns the sharded data belonging to the given rank."""
        for i, dataset in enumerate(datasets):     
             sharded_dataset = dataset.shard(num_replicas, rank)
             datasets[i] = sharded_dataset 
        return datasets 


if __name__ == "__main__":
    rank = 0
    num_replicas = 4
    dataset1 = AutoTask.get("squad").get_dataset(split="train", n_obs="16", add_prefix=False)
    for d in dataset1:
        print(d)

    dataset2 = AutoTask.get("cola").get_dataset(split="train", n_obs="32")
    train_datasets = [dataset1, dataset2]
    train_datasets = shard_data(train_datasets, num_replicas=num_replicas, rank=rank)
    multitask_dataset = datasets.concatenate_datasets(train_datasets)
    dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
    batch_size = 2
    temperature = 10
    multitask_sampler = MultiTaskBatchSampler(dataset_sizes, batch_size, temperature)

    # Defining datacollator.
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    data_args= DataTrainingArguments(sampling=True, task=['mrpc'],
    max_source_length=128, max_target_length=128, val_max_target_length=128,
    test_max_target_length=128, n_train=10, n_val=-1, n_test=-1, eval_beams=None,
    ignore_pad_token_for_loss=True)
    collator = TaskCollator(tokenizer, data_args, tpu_num_cores=0, tasks=["cola", "squad"])
    dataloader = DataLoader(multitask_dataset, batch_sampler=multitask_sampler, collate_fn=collator)
    for i, batch in enumerate(dataloader):
        print(i, batch)