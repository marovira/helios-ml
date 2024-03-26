import abc
import dataclasses
import enum
import random
import typing

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
) -> tuple[tud.DataLoader, tud.DistributedSampler | None]:
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
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(random_seed)

    return (
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


@dataclasses.dataclass
class DataLoaderParams:
    """
    Params used to create the dataloader object.

    Args:
        random_seed (int): value to use as seed for the worker processes.
        batch_size (int): number of samplers per batch.
        shuffle (bool): if true, samples are randomly shuffled.
        num_workers (int): number of worker processes for loading data.
        pin_memory (bool): if true, use page-locked device memory.
        drop_last (bool): if true, remove the final batch.
        debug_mode (bool): if true, set number of workers to 0.
        is_distributed (bool): if true, create the distributed sampler.
    """

    random_seed: int = 0
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    debug_mode: bool = False
    is_distributed: bool = False

    def to_dict(self) -> dict[str, typing.Any]:
        """Convert the params object to a dictionary using shallow copies."""
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }

    @classmethod
    def from_dict(cls, table: dict[str, typing.Any]):
        """Create a new params object from the given table."""
        params = cls()
        keys = [field.name for field in dataclasses.fields(params)]

        for key in keys:
            if key in table:
                setattr(params, key, table[key])

        return params


@dataclasses.dataclass
class Dataset:
    """
    The dataset and corresponding data loader params.

    Args:
        dataset (torch.utils.data.Dataset): the dataset.
        params (DataLoaderParams): the data loader params.
    """

    dataset: tud.Dataset
    params: DataLoaderParams

    def dict(self) -> dict[str, typing.Any]:
        """Convert to a dictionary."""
        res = self.params.to_dict()
        res["dataset"] = self.dataset
        return res


class PyroDataModule(abc.ABC):
    """
    Base class that groups together the creation of the main training datasets.

    The use of this class is to standardize the way datasets and their respective
    dataloaders are created, thereby allowing consistent settings across models.
    Ex:
        ```py
        from torchvision.datasets import CIFAR10
        from pyro import data

        class MyDataModule(data.PyroDataModule):
            def prepare_data(self) -> None:
                # Use this function to prepare the data for your datasets. This will be
                # called before the distributed processes are created (if using) so you
                # should not set any state here.
                CIFAR10(download=True) # download the dataset only.

            def setup(self) -> None:
                # Create the training dataset using a DataLoaderParams instance. Note that
                # you MUST assign it to self._train_dataset.
                self._train_dataset = self._create_dataset(CIFAR10(train=True),
                                                           DataLoaderParams(...))

                # It is also possible to create a dataset using a table of key-value pairs
                # that was loaded from a config file or manually created. Let's use one to
                # create the validation split:
                settings = {"batch_size": 1, ...}

                # We can now use it to assign to self._valid_dataset like this:
                self._valid_dataset = self._create_dataset(CIFAR10(train=False), settings)

                # Finally, if you need a testing split, you can create it like this:
                self._test_dataset = self._create_dataset(CIFAR10(train=False), settings)

            def teardown(self) -> None:
                # Use this function to clean up any state. It will be called after
                # training is done.
        ```
    """

    def __init__(self) -> None:
        """Create the data module."""
        self._is_distributed: bool = False
        self._train_dataset: Dataset | None = None
        self._valid_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None

    @property
    def is_distributed(self) -> bool:
        """Flag controlling whether distributed training is being used or not."""
        return self._is_distributed

    @is_distributed.setter
    def is_distributed(self, val: bool) -> None:
        self._is_distributed = val

    def prepare_data(self) -> None:  # noqa: B027
        """
        Prepare data for training.

        This can include downloading datasets, or streaming them from external services.
        This function will be called on the primary process when using distributed
        training (will be called prior to initialization of the processes) so don't store
        any state here.
        """

    @abc.abstractmethod
    def setup(self) -> None:
        """Construct all required datasets."""

    def train_dataloader(
        self,
    ) -> tuple[tud.DataLoader, tud.DistributedSampler | None] | None:
        """Create the train dataloader (if available)."""
        if self._train_dataset is None:
            return None
        return self._create_dataloader(self._train_dataset)

    def valid_dataloader(
        self,
    ) -> tuple[tud.DataLoader, tud.DistributedSampler | None] | None:
        """Create the valid dataloader (if available)."""
        if self._valid_dataset is None:
            return None
        return self._create_dataloader(self._valid_dataset)

    def test_dataloader(
        self,
    ) -> tuple[tud.DataLoader, tud.DistributedSampler | None] | None:
        """Create the test dataloader (if available)."""
        if self._test_dataset is None:
            return None
        return self._create_dataloader(self._test_dataset)

    def teardown(self) -> None:  # noqa: B027
        """Clean up any state after training is over."""

    def _create_dataset(
        self, dataset: tud.Dataset, params: DataLoaderParams | dict[str, typing.Any]
    ) -> Dataset:
        """
        Create a dataset object from a Torch dataset and a params.

        This is a convenience function to help you create the dataset objects. The params
        object can either be the fully filled in DataLoaderParams instance, or it can be a
        dict containing all of the necessary settings. If a dict is passed in, it will be
        used to populate the corresponding DataLoaderParams object.

        Args:
            dataset (torch.utils.data.Dataset): the dataset.
            params (DataLoaderParams | dict[str, Any]): either a params object or a dict.

        Returns:
            Dataset: the new dataset object.
        """
        return Dataset(
            dataset,
            params
            if isinstance(params, DataLoaderParams)
            else DataLoaderParams.from_dict(params),
        )

    def _create_dataloader(
        self, dataset: Dataset
    ) -> tuple[tud.DataLoader, tud.DistributedSampler | None]:
        dataset.params.is_distributed = self._is_distributed
        return create_dataloader(**dataset.dict())
