from torch.utils.data import DistributedSampler
import torch
from torch.utils.data.sampler import RandomSampler
import torch.distributed as dist
import math

from transformers.file_utils import is_torch_tpu_available



if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


def get_tpu_sampler(dataset: torch.utils.data.dataset.Dataset, data_groups=None):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)

    if data_groups is not None:
        return WeightedDistributedSampler(dataset, data_groups, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
    else:
        return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


def compute_data_groups(train_datasets):
    groups = torch.cat([torch.ones(len(x)) * i for i, x in enumerate(train_datasets)]).long()
    class_sample_count = torch.tensor([(groups == t).sum() for t in torch.unique(groups, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weights = torch.tensor([weight[t] for t in groups])
    return samples_weights.numpy()


"""
def compute_data_groups(train_datasets, tau=2):
    Z = sum(pow(len(dataset), tau) for dataset in train_datasets)
    sampling_weights = [(pow(len(dataset), tau)/Z) for dataset in train_datasets]
    groups = []
    for i, dataset in enumerate(train_datasets):
        groups.extend([sampling_weights[i] for _ in range(len(dataset))])
    return np.array(groups)
"""

from torch.utils.data import Sampler, Dataset
from typing import TypeVar, Optional, Iterator
T_co = TypeVar('T_co', covariant=True)


class WeightedDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, weights,  num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, replacement: bool = True) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.replacement = replacement
        self.weights = torch.as_tensor(weights, dtype=torch.double)

    """
    def calculate_weights(self, data_groups):
        class_sample_count = torch.tensor(
            [(data_groups == t).sum() for t in torch.unique(data_groups, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in data_groups])
        return samples_weight
    """


    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # reorder the data groups based on indices.
        weights = self.weights[indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # Gets data groups to compute the weighed indices from the selected ones.
        weights = weights[indices]
        assert len(weights) == self.num_samples
        rand_tensor = torch.multinomial(weights, self.num_samples, self.replacement).tolist()
        indices = [indices[i] for i in rand_tensor]

        return iter(indices)


    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


"""
def compute_data_groups(datasets, maximum=None, temperature=1.0, scale=1.0):
    #Mixing rate equal to the number of examples for the task.
    rates = np.array([len(dataset) for dataset in datasets])
    print(rates)
    rates = rates * scale
    if maximum:
        rates = np.minimum(rates, maximum)
    if temperature != 1.0:
        rates = rates ** (1.0 / temperature)
    rates = rates/np.sum(rates)

    dataset_rates = []
    for i, dataset in enumerate(datasets):
        temp = [rates[i]/len(dataset) for _ in range(len(dataset))]
        dataset_rates.extend(temp)

    return dataset_rates
"""
