import collections
import enum
import random
import typing

import numpy as np
import torch
import torch.utils.data as tud

from pyro import core

DATASET_REGISTRY = core.Registry("dataset")


def create_dataset(type_name: str, **kwargs):
    """
    Create a dataset of the given type.

    This uses DATASET_REGISTRY to look-up transform types, so ensure your datasets
    have been registered before using this function.

    Args:
        type_name (str): the type of the dataset to create.
        kwargs: any arguments you wish to pass into the dataset.

    Returns:
        nn.Module: the constructed transform.
    """
    return DATASET_REGISTRY.get(type_name)(**kwargs)


class DatasetSplit(enum.Enum):
    """The different dataset splits."""

    TRAIN = 0
    VALID = 1
    TEST = 2

    @staticmethod
    def from_str(label: str) -> "DatasetSplit":
        """
        Convert the given string to the corresponding enum value.

        Must be one of "train", "test", or "valid"

        Args:
            label (str): the label to convert.

        Returns:
            DatasetSplit: the corresponding value.
        """
        if label == "train":
            return DatasetSplit.TRAIN
        if label == "valid":
            return DatasetSplit.VALID
        if label == "test":
            return DatasetSplit.TEST

        raise ValueError(
            "invalid dataset split. Expected one of 'train', 'test', or "
            f"'valid', but received '{label}'"
        )


SampledDataLoader = collections.namedtuple("SampledDataLoader", ["dataloader", "sampler"])


def create_dataloader(
    dataset: tud.Dataset,
    random_seed: int = 0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    debug_mode: bool = False,
    is_distributed: bool = False,
) -> SampledDataLoader:
    """
    Create the dataloader for the given dataset.

    The sampler will only be returned if is_distributed is set to true.

    Args:
        dataset (Dataset): the dataset.
        random_seed (int): value to use as seed for the worker processes.
        batch_size (int): number of samplers per batch.
        shuffle (bool): if true, samples are randomly shuffled.
        num_workers (int): number of worker processes for loading data.
        pin_memory (bool): if true, use page-locked device memory.
        drop_last (bool): if true, remove the final batch.
        debug_mode (bool): if true, set number of workers to 0.
        is_distributed (bool): if true, create the distributed sampler.

    Returns:
        tuple[Dataloader, DistributedSampler | None]: the dataloader and sampler.
    """
    assert len(typing.cast(typing.Sized, dataset))

    sampler: tud.DistributedSampler | None = (
        tud.DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        if is_distributed
        else None
    )

    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(random_seed)

    return SampledDataLoader(
        tud.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=None if is_distributed else shuffle,
            num_workers=num_workers if not debug_mode else 0,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        sampler,
    )


class PyroDataModule:
    """
    The PyroDataModule standardizes the different splits, data preparation and transforms.

    The benefit is consistent data splits, data preparation, and transforms across runs.
    The API is similar to PyTorch Lightning's LightningDataModule, major difference is
    that this class does not have serialization capabilities.

    Ex:
        ```py
        from pyro import data

        class MyDataModule(data.PyroDataModule):
            def prepare_data(self):
                # Download or stream any data.

            def setup(self, split: DatasetSplit | None):
                # Create the datasets here. If None is given as split, instantiate all of
                # your datasets.
                self._train = ...
                self._val = ...

            def train_dataloader(self):
                # To create the dataloader, call self._create_dataloader with the
                # settings you require. It will automatically handle the creation of the
                # dataloader + sampler.
                return self._create_dataloader(self._train, ...)

            def val_dataloader(self):
                return self._create_dataloader(self._val, ...)

            def teardown(self):
                # Clear up any state here.
                ...

        ```
    """

    def __init__(self) -> None:
        """Create the DataModule."""
        self._is_distributed: bool = False

    @property
    def is_distributed(self) -> bool:
        """Flag controlling whether distributed training is being used or not."""
        return self._is_distributed

    @is_distributed.setter
    def is_distributed(self, val: bool) -> None:
        self._is_distributed = val

    def prepare_data(self) -> None:
        """
        Prepare data for training.

        This can include downloading datasets, or streaming them from external services.
        This function will be called on the primary process when using distributed
        training (will be called prior to initialization of the processes) so don't store
        any state here.
        """

    def setup(self, split: DatasetSplit | None = None) -> None:
        """
        Construct the dataset(s) associated with the given split.

        This will be called on each process if using distributed training. The special
        None value is used to signal the DataModule to create all of the datasets it
        holds.

        Args:
            split (DatasetSplit | None): the split to create.
        """

    def _create_dataloader(self, dataset: tud.Dataset, **kwargs) -> SampledDataLoader:
        """
        Construct the dataloader from the given dataset.

        You can use this function to automatically create the DataLoader (and Sampler)
        using the given arguments instead of having to create it yourself. Note that you
        DO NOT have to manage the distributed case, it will be automatically handled for
        you.

        Args:
            dataset (Dataset): the dataset.
            kwargs (dict): arguments for dataloader creation.

        Returns:
            SampledDataLoader: the dataloader + sampler pair.
        """
        return create_dataloader(dataset, is_distributed=self._is_distributed, **kwargs)

    def train_dataloader(self) -> SampledDataLoader | None:
        """
        Return the dataloader for the training split.

        Returns:
            SampledDataLoader | None: the dataloader.
        """
        return None

    def val_dataloader(self) -> SampledDataLoader | None:
        """
        Return the dataloader for the validation split.

        Returns:
            SampledDataLoader | None: the dataloader.
        """
        return None

    def test_dataloader(self) -> SampledDataLoader | None:
        """
        Return the dataloader for the testing split.

        Returns:
            SampledDataLoader | None: the dataloader.
        """
        return None

    def teardown(self) -> None:
        """Clean up any state after training is over."""
