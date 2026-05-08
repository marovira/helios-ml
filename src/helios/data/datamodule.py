from __future__ import annotations

import abc
import copy
import dataclasses as dc
import enum
import types
import typing

import torch
import torch.utils.data as tud

from helios import core
from helios.core import rng

if typing.TYPE_CHECKING:
    from ..trainer import Trainer

from .samplers import (
    ResumableDistributedSampler,
    ResumableRandomSampler,
    ResumableSamplerType,
    ResumableSequentialSampler,
)


def _register_default_collate(registry: core.Registry) -> None:
    registry.register(tud.default_collate)


DATASET_REGISTRY = core.Registry("dataset")
"""
Global instance of the registry for datasets.

Example:
    .. code-block:: python

        import helios.data as hld

        # This automatically registers your dataset.
        @hld.DATASET_REGISTRY.register
        class MyDataset:
            ...

        # Alternatively you can manually register a dataset like this:
        hld.DATASET_REGISTRY.register(MyDataset)
"""

COLLATE_FN_REGISTRY = core.Registry("collate_fn")
"""
Global instance of the registry for collate functions.

Example:
    .. code-block:: python

        import helios.data as hld

        # This automatically registers your collate function.
        @hld.COLLATE_FN_REGISTRY
        def my_collate_fn():
            ...

        # Alternatively you can manually register a collate function like this:
        hld.COLLATE_FN_REGISTRY.register(my_collate_fn)
"""

_register_default_collate(COLLATE_FN_REGISTRY)


def create_dataset(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> tud.Dataset:
    """
    Create a dataset of the given type.

    This uses ``DATASET_REGISTRY`` to look-up dataset types, so ensure your datasets
    have been registered before using this function.

    Args:
        type_name: the type of the dataset to create.
        args: positional arguments to pass into the dataset.
        kwargs: keyword arguments to pass into the dataset.

    Returns:
        The constructed dataset.
    """
    return DATASET_REGISTRY.get(type_name)(*args, **kwargs)


def create_collate_fn(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> typing.Callable:
    """
    Create a collate function of the given type.

    This uses ``COLLATE_FN_REGISTRY`` to look-up collate functions, so ensure that your
    functions have been registered before using this function.
    In order to support regular functions as well as callable objects, this function
    behaves as follows:

    #. If ``type_name`` refers to a function-type, then ``args`` and ``kwargs`` are
        ignored.
    #. If ``type_name`` refers to an actual type (i.e. a class), then ``args`` and
        ``kwargs`` are forwarded upon instantiation.

    Args:
        type_name: the type of the function to create.
        args: positional arguments to pass into the collate class
        kwargs: keyword arguments to pass into the collate class

    Returns:
        The function
    """
    fn_type = COLLATE_FN_REGISTRY.get(type_name)
    if isinstance(fn_type, types.FunctionType):
        return fn_type
    assert isinstance(fn_type, type), f"Expected a type but found {type(fn_type)}"
    return fn_type(*args, **kwargs)


class DatasetSplit(enum.Enum):
    """The different dataset splits."""

    TRAIN = 0
    VALID = 1
    TEST = 2

    @staticmethod
    def from_str(label: str) -> DatasetSplit:
        """
        Convert the given string to the corresponding enum value.

        Must be one of "train", "test", or "valid"

        Args:
            label: the label to convert.

        Returns:
            The corresponding enum value.

        Raises:
            ValueError: if the given value is not one of "train", "test", or "valid".
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    rng.seed_rngs(worker_seed, skip_torch=True)


def create_dataloader(
    dataset: tud.Dataset,
    random_seed: int = rng.get_default_seed(),
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    is_distributed: bool = False,
    sampler: ResumableSamplerType | None = None,
    collate_fn: typing.Callable | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory_device: str = "",
    timeout: float = 0,
    multiprocessing_context: typing.Any | None = None,
) -> tuple[tud.DataLoader, ResumableSamplerType]:
    """
    Create the dataloader for the given dataset.

    If no sampler is provided, the choice of sampler will be determined based on the
    values of is_distributed and shuffle. Specifically, the following logic is used:

    * If is_distributed, then sampler is
      :py:class:`~helios.data.samplers.ResumableDistributedSampler`.
    * Otherwise, if shuffle then sampler is
      :py:class:`~helios.data.samplers.ResumableRandomSampler`, else
      :py:class:`~helios.data.samplers.ResumableSequentialSampler`.

    You may override this behaviour by providing your own sampler instance.

    .. warning::
        If you provide a custom sampler, then it **must** be derived from one of
        :py:class:`helios.data.samplers.ResumableSampler` or
        :py:class:`helios.data.samplers.ResumableDistributedSampler`.

    Args:
        dataset: the dataset to use.
        random_seed: value to use as seed for the worker processes. Defaults to the value
            returned by :py:func:`~helios.core.rng.get_default_seed`.
        batch_size: number of samplers per batch. Defaults to 1.
        shuffle: if true, samples are randomly shuffled. Defaults to false.
        num_workers: number of worker processes for loading data. Defaults to 0.
        pin_memory: if true, use page-locked device memory. Defaults to false.
        drop_last: if true, remove the final batch. Defaults to false.
        is_distributed: if true, create the distributed sampler. Defaults to false.
        sampler: (optional) sampler to use.
        collate_fn: (optional) function to merge batches.
        prefetch_factor: number of batches to prefetch per worker. Only valid when
            ``num_workers > 0``. Defaults to ``None``.
        persistent_workers: if true, keep worker processes alive between epochs. Only
            valid when ``num_workers > 0``. Defaults to false.
        pin_memory_device: target device for pinned memory when ``pin_memory`` is true.
            Defaults to ``""``.
        timeout: timeout in seconds for collecting a batch from workers. Defaults to 0.
        multiprocessing_context: method for spawning worker processes (e.g.
            ``"fork"``, ``"spawn"``, ``"forkserver"``). Defaults to ``None``.

    Returns:
        The dataloader and sampler.

    Raises:
        TypeError: if ``sampler`` is not ``None`` and not derived from one of
            :py:class:`~helios.data.samplers.ResumableDistributedSampler` or
            :py:class:`~helios.data.samplers.ResumableSampler`.
    """
    assert len(typing.cast(typing.Sized, dataset))

    if sampler is None:
        if is_distributed:
            sampler = ResumableDistributedSampler(
                dataset, shuffle=shuffle, drop_last=drop_last, seed=random_seed
            )
        else:
            if shuffle:
                sampler = ResumableRandomSampler(
                    dataset,  # type: ignore[arg-type]
                    seed=random_seed,
                    batch_size=batch_size,
                )
            else:
                sampler = ResumableSequentialSampler(
                    dataset,  # type: ignore[arg-type]
                    batch_size=batch_size,
                )

    elif not isinstance(sampler, typing.get_args(ResumableSamplerType)):
        raise TypeError(
            "error: expected sampler to derive from one of ResumableSampler or "
            f"ResumableDistributedSampler, but received {type(sampler)}"
        )

    return (
        tud.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler=sampler,
            worker_init_fn=_seed_worker,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
            timeout=timeout,
            multiprocessing_context=multiprocessing_context,
        ),
        sampler,
    )


@dc.dataclass
class DataLoaderParams:
    """
    Params used to create the dataloader object.

    Args:
        random_seed: value to use as seed for the worker processes.
        batch_size: number of samplers per batch.
        shuffle: if true, samples are randomly shuffled.
        num_workers: number of worker processes for loading data.
        pin_memory: if true, use page-locked device memory.
        drop_last: if true, remove the final batch.
        is_distributed: if true, create the distributed sampler.
        sampler: (optional) sampler to use.
        collate_fn: (optional) function to merge batches.
        prefetch_factor: number of batches to prefetch per worker. Only valid when
            ``num_workers > 0``.
        persistent_workers: if true, keep worker processes alive between epochs. Only
            valid when ``num_workers > 0``.
        pin_memory_device: target device for pinned memory when ``pin_memory`` is true.
        timeout: timeout in seconds for collecting a batch from workers.
        multiprocessing_context: method for spawning worker processes (e.g.
            ``"fork"``, ``"spawn"``, ``"forkserver"``).
    """

    random_seed: int = rng.get_default_seed()
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    is_distributed: bool | None = None
    sampler: ResumableSamplerType | None = None
    collate_fn: typing.Callable | None = None
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    pin_memory_device: str = ""
    timeout: float = 0
    multiprocessing_context: typing.Any | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        """Convert the params object to a dictionary using shallow copies."""
        return {field.name: getattr(self, field.name) for field in dc.fields(self)}

    @classmethod
    def from_dict(cls, table: dict[str, typing.Any]):
        """Create a new params object from the given table."""
        params = cls()
        keys = [field.name for field in dc.fields(params)]

        # Skip the sampler key, since that one needs to be created differently.
        for key in keys:
            if key in table and key != "sampler":
                setattr(params, key, table[key])

        return params


@dc.dataclass
class Dataset:
    """
    The dataset and corresponding data loader params.

    Args:
        dataset: the dataset.
        params: the data loader params.
    """

    dataset: tud.Dataset
    params: DataLoaderParams

    def dict(self) -> dict[str, typing.Any]:
        """Convert to a dictionary."""
        res = self.params.to_dict()
        res["dataset"] = self.dataset
        return res


class DataModule(abc.ABC):
    """
    Base class that groups together the creation of the main training datasets.

    The use of this class is to standardize the way datasets and their respective
    dataloaders are created, thereby allowing consistent settings across models.

    Example:
        .. code-block:: python

            from torchvision.datasets import CIFAR10
            from helios import data
            from helios.data import Dataset, DataLoaderParams

            class MyDataModule(data.DataModule):
                def prepare_data(self) -> None:
                    # Use this function to prepare the data for your datasets. This will
                    # be called before the distributed processes are created (if using)
                    # so you should not set any state here.
                    CIFAR10(download=True) # download the dataset only.

                def setup(self) -> None:
                    # Register the training phase(s) using _add_train_phase. The first
                    # call also sets self._train_dataset automatically.
                    self._add_train_phase(CIFAR10(train=True), DataLoaderParams(...))

                    # For multi-phase training, call _add_train_phase again:
                    self._add_train_phase(
                        CIFAR10(train=True, transform=...), DataLoaderParams(...)
                    )

                    # Validation and testing datasets use their own helpers.
                    # A dict of settings is also accepted in place of DataLoaderParams.
                    settings = {"batch_size": 1, ...}
                    self._add_valid_dataset(CIFAR10(train=False), settings)
                    self._add_test_dataset(CIFAR10(train=False), DataLoaderParams(...))

                def teardown(self) -> None:
                    # Use this function to clean up any state. It will be called after
                    # training is done.
    """

    def __init__(self) -> None:
        """Create the data module."""
        self._is_distributed: bool = False
        self._train_dataset: Dataset | None = None
        self._valid_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._trainer: Trainer | None = None
        self._train_phases: list[Dataset] = []
        self._current_train_phase: int = 0

    @property
    def is_distributed(self) -> bool:
        """Flag controlling whether distributed training is being used or not."""
        return self._is_distributed

    @is_distributed.setter
    def is_distributed(self, val: bool) -> None:
        self._is_distributed = val

    @property
    def trainer(self) -> Trainer:
        """Reference to the trainer."""
        return core.get_from_optional(self._trainer)

    @trainer.setter
    def trainer(self, t) -> None:
        self._trainer = t

    @property
    def train_dataset(self) -> tud.Dataset | None:
        """The training dataset (if available)."""
        if self._train_dataset is not None:
            return self._train_dataset.dataset
        return None

    @property
    def valid_dataset(self) -> tud.Dataset | None:
        """The validation dataset (if available)."""
        if self._valid_dataset is not None:
            return self._valid_dataset.dataset
        return None

    @property
    def test_dataset(self) -> tud.Dataset | None:
        """The testing dataset (if available)."""
        if self._test_dataset is not None:
            return self._test_dataset.dataset
        return None

    def prepare_data(self) -> None:  # noqa: B027
        """
        Prepare data for training.

        This can include downloading datasets, preparing caches, or streaming them from
        external services. This function will be called on the primary process when using
        distributed training (will be called prior to initialization of the processes) so
        don't store any state here.
        """

    @abc.abstractmethod
    def setup(self) -> None:
        """Construct all required datasets."""

    def train_dataloader(self) -> tuple[tud.DataLoader, ResumableSamplerType] | None:
        """Create the train dataloader (if available)."""
        if self._train_dataset is None:
            return None
        return self._create_dataloader(self._train_dataset)

    def valid_dataloader(self) -> tuple[tud.DataLoader, ResumableSamplerType] | None:
        """Create the valid dataloader (if available)."""
        if self._valid_dataset is None:
            return None
        return self._create_dataloader(self._valid_dataset)

    def test_dataloader(self) -> tuple[tud.DataLoader, ResumableSamplerType] | None:
        """Create the test dataloader (if available)."""
        if self._test_dataset is None:
            return None
        return self._create_dataloader(self._test_dataset)

    def teardown(self) -> None:  # noqa: B027
        """Clean up any state after training is over."""

    def get_train_steps_per_epoch(self) -> int:
        """
        Return the number of iterations per training epoch.

        The number is determined by constructing the training dataloader and returning its
        length. This function is generally useful for initialising schedulers that need
        the total number of steps per epoch, so you should call this within
        :py:meth:`~helios.model.model.Model.steup` or
        :py:meth:`~helios.plugins.plugin.Plugin.setup` and cache the result.

        This function is equivalent to:

            .. code-block:: python

                import helios.core as hlc
                # If called from within Model.setup()
                dataloader, _ = hlc.get_from_optional(
                        self.trainer.datamodule.train_dataloader())
                steps = len(dataloader)

        .. note::
            This function only applies to training datasets. If you need something similar
            for validation or testing, you can obtain the dataloader directly.

        Returns:
            The number of iterations per training epoch.

        Raises:
            RuntimeError: if the training dataloader hasn't been created.
        """
        result = self.train_dataloader()
        if result is None:
            raise RuntimeError(
                "get_train_steps_per_epoch() called but no training dataset "
                "has been configured."
            )
        dataloader, _ = result
        return len(dataloader)

    def advance_train_phase(self) -> None:
        """
        Advance the training dataset to the next phase.

        If the current phase is already the last one, this function does nothing.
        """
        if self._current_train_phase >= len(self._train_phases) - 1:
            return
        self._current_train_phase += 1
        self._train_dataset = self._train_phases[self._current_train_phase]

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Get the full state dictionary of the datamodule.

        The contents of the dictionary depend on whether training phases have been
        registered:
        * If they have, then this returns the current phase.
        * If they haven't, then this returns an empty dictionary.

        Returns:
            The state dictionary of the datamodule.
        """
        if not self._train_phases:
            return {}
        return {"phase": self._current_train_phase}

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Load the datamodule state from the given state dictionary.

        Args:
            state_dict: the state dictionary to load from.
        """
        if not self._train_phases or "phase" not in state_dict:
            return
        self._current_train_phase = state_dict["phase"]
        self._train_dataset = self._train_phases[self._current_train_phase]

    def _add_train_phase(
        self, dataset: tud.Dataset, params: DataLoaderParams | dict[str, typing.Any]
    ) -> None:
        """
        Add a training phase dataset.

        Training phases can be used to control the produced data at different stages
        during training. For example, suppose that we want to split training in two
        phases where the dataset is the same but the settings change. We can then add them
        like this:
            .. code-block:: python

                from torchvision.datasets import CIFAR10
                from torchvision.transforms import v2
                from helios import data
                from helios.data import Dataset, DataLoaderParams

                class MyDataModule(data.DataModule):
                    def setup(self) -> None:
                        # Phase 1: use RandomCrop
                        self._add_train_phase(CIFAR10(transform=v2.RadnomCrop(...)))
                        # Phase 2: use RandomResize
                        self._add_train_phase(CIFAR10(transform=v2.RandomResize(...)))
                        # ...

        This then introduces two phases which can be controlled by
        :py:meth:`~helios.model.model.Model.should_advance_dataset_phase`.
        The first call to this function will set the active dataset to the first phase
        automatically. If you do not require multi-phase training, you can simply insert
        one dataset and everything will work normally.

        Args:
            dataset: the dataset for this phase.
            params: either a :py:class:`DataLoaderParams` instance or a dict.
        """
        phase = Dataset(
            dataset,
            copy.deepcopy(params)
            if isinstance(params, DataLoaderParams)
            else DataLoaderParams.from_dict(params),
        )
        self._train_phases.append(phase)
        if len(self._train_phases) == 1:
            self._train_dataset = phase

    def _add_valid_dataset(
        self, dataset: tud.Dataset, params: DataLoaderParams | dict[str, typing.Any]
    ) -> None:
        """
        Set the validation dataset.

        Args:
            dataset: the validation dataset.
            params: either a :py:class:`DataLoaderParams` instance or a dict.
        """
        self._valid_dataset = Dataset(
            dataset,
            copy.deepcopy(params)
            if isinstance(params, DataLoaderParams)
            else DataLoaderParams.from_dict(params),
        )

    def _add_test_dataset(
        self, dataset: tud.Dataset, params: DataLoaderParams | dict[str, typing.Any]
    ) -> None:
        """
        Set the test dataset.

        Args:
            dataset: the test dataset.
            params: either a :py:class:`DataLoaderParams` instance or a dict.
        """
        self._test_dataset = Dataset(
            dataset,
            copy.deepcopy(params)
            if isinstance(params, DataLoaderParams)
            else DataLoaderParams.from_dict(params),
        )

    def _create_dataloader(
        self, dataset: Dataset
    ) -> tuple[tud.DataLoader, ResumableSamplerType]:
        # Only override the distributed flag if it hasn't been set by the user.
        if dataset.params.is_distributed is None:
            dataset.params.is_distributed = self._is_distributed
        return create_dataloader(**dataset.dict())
