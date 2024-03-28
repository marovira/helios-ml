import math
import typing

import torch
import torch.utils.data as tud


class ResumableSampler(tud.Sampler):
    """
    Base class for samplers that are resumable.

    In this context, a resumable sampler is defined as a sampler that, when training stops
    and is later resumed, guarantees that the order in which the batches are returned is
    always the same and that they will continue from where the previous run left off.
    """

    def __init__(self, batch_size: int) -> None:
        """Create the sampler."""
        super().__init__()

        self._start_iter: int = 0
        self._epoch: int = 0
        self._batch_size = batch_size

    def _adjust_to_start_iter(self, indices: list[int]) -> list[int]:
        assert self._batch_size > 0

        start_index = self._start_iter * self._batch_size
        return indices[start_index:]

    @property
    def start_iter(self) -> int:
        """The starting iteration for the sampler."""
        return self._start_iter

    @start_iter.setter
    def start_iter(self, ite: int) -> None:
        self._start_iter = ite

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for seeding."""
        self._epoch = epoch


class ResumableRandomSampler(ResumableSampler):
    """
    Random sampler with resumable state.

    This allows training to stop and resume while guaranteeing that the order in which the
    batches will be returned stays consistent. It is effectively a replacement to the
    default RandomSampler from PyTorch.

    Args:
        data_source (Sized): the dataset to sample from.
        seed (int): the seed to use for setting up the random generator.
        batch_size (int): the number of samples per batch.
    """

    def __init__(
        self, data_source: typing.Sized, seed: int = 0, batch_size: int = 1
    ) -> None:
        """Create the sampler."""
        super().__init__(batch_size)

        self._data_source = data_source
        self._seed = seed
        self._num_samples = len(self._data_source)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._num_samples

    def __iter__(self) -> typing.Iterator[int]:
        """Retrieve the index of the next sample."""
        g = torch.Generator()
        g.manual_seed(self._epoch + self._seed)
        shuffling = torch.randperm(self._num_samples, generator=g)
        indices = shuffling.tolist()

        assert len(indices) == self._num_samples
        indices = self._adjust_to_start_iter(indices)
        return iter(indices)


class ResumableSequentialSampler(ResumableSampler):
    """
    Sequential sampler with resumable state.

    This allows training to stop and resume while guaranteeing that the order in which the
    batches will be returned stays consistent. It is effectively a replacement to the
    default SequentialSampler from PyTorch.

    Args:
        data_source (Sized): the dataset to sample from.
        batch_size (int): the number of samples per batch.
    """

    def __init__(self, data_source: typing.Sized, batch_size: int = 1):
        """Create the sampler."""
        super().__init__(batch_size)

        self._data_source = data_source
        self._num_samples = len(data_source)
        self._indices = list(range(self._num_samples))

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self._num_samples

    def __iter__(self) -> typing.Iterator[int]:
        """Retrieve the index of the next sample."""
        assert self._batch_size > 0
        indices = self._indices
        indices = self._adjust_to_start_iter(indices)
        return iter(indices)


class ResumableDistributedSampler(tud.DistributedSampler):
    """
    Distributed sampler with resumable state.

    This allows training to stop and resume while guaranteeing that the order in which the
    batches will be returned stays consistent. It is effectively a replacement to the
    default DistributedSampler from PyTorch.

    Args:
        dataset (torch.utils.data.Dataset): the dataset to sample from.
        num_replicas (int | None): number of processes for distributed training.
        rank (int | None): rank of the current process.
        shuffle (bool): if True, shuffle the indices.
        seed (int): random seed used to shuffle the sampler.
        drop_last (bool): If true, then drop the final sample to make it even across
        replicas.
        batch_size (int): the number of samples per batch.
    """

    def __init__(
        self,
        dataset: tud.Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1,
    ):
        """Create the sampler."""
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self._batch_size = batch_size
        self._start_iter = 0

    @property
    def start_iter(self) -> int:
        """The starting iteration for the sampler."""
        return self._start_iter

    @start_iter.setter
    def start_iter(self, ite: int) -> None:
        self._start_iter = ite

    def __iter__(self) -> typing.Iterator[int]:
        """Retrieve the index of the next sample."""
        # The code for this function was adapted from PyTorch's implementation of
        # DistributedSampler. The original license from PyTorch can be viewed here:
        # https://github.com/pytorch/pytorch/blob/main/LICENSE
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Trim off samples that we have already seen if we're restarting.
        assert self._batch_size > 0
        start_index = self._start_iter * self._batch_size
        indices = indices[start_index:]

        return iter(indices)


ResumableSamplerType = ResumableSampler | ResumableDistributedSampler
