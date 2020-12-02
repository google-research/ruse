"""Multi-task iterative dataloaders in distributed mode."""


from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from itertools import cycle, islice

from transformers.file_utils import is_torch_tpu_available
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm



class TaskDataLoader:
    """Wrapper around dataloader to keep the task names."""
    def __init__(self, task_name, dataset, batch_size=8,
                 collate_fn=None, drop_last=False, num_workers=0, sampler=None):
        self.dataset = dataset
        self.task_name = task_name
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      collate_fn=collate_fn,
                                      drop_last=drop_last,
                                      num_workers=num_workers)
    def __len__(self):
        return self.dataset.num_rows

    def __iter__(self):
        for batch in self.data_loader:
            yield batch



class MultiTaskDataLoader:
    """Given a dictionary of task: dataset, returns a multi-task dataloader
    which uses temperature sampling to sample different datasets."""

    def __init__(self,  tasks_to_datasets, batch_size=8, collate_fn=None,
                 drop_last=False, num_workers=0, temperature=100.0):

        # Computes a mapping from task to dataloaders.
        self.task_to_dataloaders = {}
        for task, dataset in tasks_to_datasets.items():
            dataloader = TaskDataLoader(task, dataset, batch_size,
                collate_fn=collate_fn, drop_last=drop_last, num_workers=num_workers)
            self.task_to_dataloaders.update({task: dataloader})
        self.tasknames = list(self.task_to_dataloaders.keys())

        # Computes the temperature sampling weights.
        self.sampling_weights = self.temperature_sampling(self.dataloader_sizes.values(), temperature)
        self.dataiters = {k: cycle(v) for k, v in self.task_to_dataloaders.items()}

    def temperature_sampling(self, dataset_sizes, temp):
        total_size = sum(dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / temp) for size in dataset_sizes])
        return weights/np.sum(weights)

    @property
    def dataloader_sizes(self):
        if not hasattr(self, '_dataloader_sizes'):
            self._dataloader_sizes = {k: len(v) for k, v in self.task_to_dataloaders.items()}
        return self._dataloader_sizes

    def __len__(self):
        return sum(v for k, v in self.dataloader_sizes.items())

    def __iter__(self):
        outputs = {}
        for i in range(len(self)):
            taskname = np.random.choice(self.tasknames, p=self.sampling_weights)
            dataiter = self.dataiters[taskname]
            outputs["batch"] = next(dataiter)
            outputs["task"] = taskname
            yield outputs



class Trainer():
    """This is the trainer class which is responsible for distributing the data
    in case of multiple TPUs/GPUs."""
    def __init__(self, dataset_names_to_datasets):
        self.dataset_names_to_datasets = dataset_names_to_datasets
        self.batch_size = 8
        self.local_rank = -1 # this is not -1 in case of multi-gpu
        self.collate_fn = None
        self.drop_last = False
        self.num_workers = 0

    def get_sharded_data(self, num_replicas, rank):
        """Returns the sharded data belonging to the given rank."""
        sharded_dataset_names_to_datasets = {}
        for dataset_name, dataset in self.dataset_names_to_datasets:
            sharded_data = dataset.shard(num_replicas, rank)
            sharded_dataset_names_to_datasets.update({dataset_name: sharded_data})
        return sharded_dataset_names_to_datasets


    def get_train_dataset_shards(self):
        """In case of multiprocessing, returns the sharded data for the given rank."""
        if is_torch_tpu_available():
            if xm.xrt_world_size() > 1:
                return self.get_sharded_data(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.local_rank != -1:
                return self.get_sharded_data(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        else:
            return self.dataset_names_to_datasets


    def get_train_dataloader(self):
        """Returns the multi-task dataloader, each batch belongs
        to one task dataset."""
        dataset_names_to_datasets = self.get_train_dataset_shards()
        dataloader = MultiTaskDataLoader(dataset_names_to_datasets,
                                         batch_size=self.batch_size,
                                         collate_fn=self.collate_fn,
                                         drop_last=self.drop_last,
                                         num_workers=self.num_workers)
        return dataloader


if __name__ == "__main__":
    batch_size = 10
    num_shards = 2
    rank = 0
    dataset1 = load_dataset('glue', 'rte', split="train[:16]")
    dataset2 = load_dataset('glue', 'cola', split="train[:32]")
    trainer = Trainer({'dataset1': dataset1, 'dataset2': dataset2})
    dataloader = trainer.get_train_dataloader()
    for batch in islice(dataloader, 5):
        print(batch)



"""
mbsz = 128
wsz = 32
ordinal = 15
rank = ordinal
gbsz = mbsz * wsz
batch = torch.randn(4096, 512)
mb = torch.narrow(batch, 0, mbsz * rank, mbsz)

def narrow(loader):
    for item in loader:
        yield torch.narrow(item, 0, xm.get_ordinal() * mbsz, mbsz)

def rank_aware_loader(loader):
    for i, item in loader:
        if (i - xm.get_ordinal()):
            continue
        yield item
"""



