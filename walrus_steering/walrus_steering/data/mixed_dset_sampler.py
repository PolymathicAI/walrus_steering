import inspect
import logging
from typing import Iterator, List

import torch
from torch.utils.data import Sampler

from .multidataset import MixedWellDataset

__all__ = [
    "MultisetSampler",
]


logger = logging.getLogger(__name__)


class MultisetSampler(Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: MixedWellDataset,
        base_sampler: type[Sampler[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        rank=0,
        distributed=True,
        recycle=True,
    ) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        self.base_sampler = base_sampler
        self.recycle = recycle
        if "drop_last" in inspect.signature(getattr(base_sampler, "__init__")).parameters:
            self.sub_samplers = [
                base_sampler(dataset, drop_last=drop_last)  # type: ignore
                for dataset in self.sub_dsets
            ]
        else:
            self.sub_samplers = [base_sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        generator = torch.Generator()
        # Ensure each worker on the same rank (GPU) sample the same dataset
        generator.manual_seed(100 * self.epoch + 10 * self.seed + self.rank)
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = int(
                torch.randint(
                    0, len(sampler_choices), size=(1,), generator=generator
                ).item()
            )
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            # Balance workload with different batch sizes
            sub_batch_size = self.dataset.effective_batch_sizes[dset_sampled]
            # Gather a batch of data from the same dataset
            # A drop last batch logic must be enforced
            # If a complete batch can be assembled, yield it
            # Otherwise move to the next dataset
            try:
                accumulated_batch: List[int] = []
                for _ in range(self.batch_size):
                    indices: List[int] = []
                    for _ in range(sub_batch_size):
                        indices.append(next(samplers[dset_sampled]) + offset)
                    # TODO Again, this is currently assuming we're not doing effective BS
                    accumulated_batch.append(indices[0])
            except StopIteration:
                # Selected sampler was exhausted
                if self.recycle:
                    count -= 1
                    samplers[index_sampled] = iter(self.sub_samplers[dset_sampled])
                    # sampler_choices[index_sampled] = self.base_sampler(self.sub_dsets[dset_sampled])
                else:
                    sampler_choices.pop(index_sampled)
                logger.info(
                    f"Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}"
                )
                continue
            else:
                yield from accumulated_batch
            if count >= self.max_samples:
                break

    def __len__(self) -> int:
        # Data loader len is len(sampler) / batch_size - so override len to max_samples * batch_size
        if self.recycle:
            return self.max_samples * self.batch_size
        return min(self.max_samples * self.batch_size, len(self.dataset))

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        self.epoch = epoch


class BatchedMultisetSampler(Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset.

    This version returns a full batch worth of indices at once and requires
    that the dataset can process list of indices"""

    def __init__(
        self,
        dataset: MixedWellDataset,
        base_sampler: type[Sampler[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        rank=0,
        distributed=True,
        recycle=True,
    ) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        self.base_sampler = base_sampler
        self.recycle = recycle
        if isinstance(base_sampler, torch.utils.data.DistributedSampler):
            self.sub_samplers = [
                base_sampler(dataset, drop_last=drop_last)  # type: ignore
                for dataset in self.sub_dsets
            ]
        else:
            self.sub_samplers = [base_sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        generator = torch.Generator()
        # Ensure each worker on the same rank (GPU) sample the same dataset
        generator.manual_seed(100 * self.epoch + 10 * self.seed + self.rank)
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = int(
                torch.randint(
                    0, len(sampler_choices), size=(1,), generator=generator
                ).item()
            )
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            # Balance workload with different batch sizes
            batch_factor = self.dataset.effective_batch_sizes[dset_sampled]
            dset_batch = max(1, int(self.batch_size * batch_factor))
            # Gather a batch of data from the same dataset
            # A drop last batch logic must be enforced
            # If a complete batch can be assembled, yield it
            # Otherwise move to the next dataset
            try:
                accumulated_batch: List[int] = []
                for _ in range(dset_batch):
                    accumulated_batch.append(next(samplers[dset_sampled]) + offset)
                accumulated_batch = [accumulated_batch]
            except StopIteration:
                # Selected sampler was exhausted
                if self.recycle:
                    count -= 1
                    samplers[index_sampled] = iter(self.sub_samplers[dset_sampled])
                    # sampler_choices[index_sampled] = self.base_sampler(self.sub_dsets[dset_sampled])
                else:
                    sampler_choices.pop(index_sampled)
                logger.info(
                    f"Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}"
                )
                continue
            else:
                yield from accumulated_batch
            if count >= self.max_samples:
                break

    def __len__(self) -> int:
        # Since each sample is a full batch, this is just self.max_samples
        if self.recycle:
            return self.max_samples  # * self.batch_size
        return min(self.max_samples, len(self.dataset))

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        self.epoch = epoch
