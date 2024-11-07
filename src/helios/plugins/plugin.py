from __future__ import annotations

import abc
import dataclasses as dc
import typing

import torch

from helios import core

if typing.TYPE_CHECKING:
    from ..model import Model
    from ..trainer import Trainer, TrainingState

# ruff: noqa: B027


@dc.dataclass
class UniquePluginOverrides:
    """
    Set of flags that determine the unique overrides a plug-in can have.

    In order to avoid conflicts, two plug-ins should *not* be able to perform the same
    action twice. For example, it shouldn't be possible to have two distinct plug-ins
    perform processing on the training batch as that would cause undefined behaviour. This
    structure therefore holds all the possible overrides a plug-in might have that
    **must** remain unique.

    Args:
        training_batch: if true, the plug-in performs processing on the training batch.
        validation_batch: if true, the plug-in performs processing on the validation
            batch.
        testing_batch: if true, the plug-in performs processing on the testing batch.
        should_training_stop: if true, the plug-in can arbitrarily stop training.
    """

    training_batch: bool = False
    validation_batch: bool = False
    testing_batch: bool = False
    should_training_stop: bool = False


PLUGIN_REGISTRY = core.Registry("plug-in")
"""
Global instance of the registry for plug-ins.

By default, the registry contains the following plug-ins:

.. list-table:: Schedulers
    :header-rows: 1

    * - Plugin
      - Name
    * - :py:class:`helios.plugins.plugin.CUDAPlugin`
      - CUDAPlugin
    * - :py:class:`helios.plugins.optuna.OptunaPlugin`
      - OptunaPlugin

.. note::
    The :py:class:`~helios.plugins.optuna.OptunaPlugin` is only registered if the module
    is imported somewhere in the code. Otherwise it won't be registered.

Example:
    .. code-block:: python

        import helios.plug-ins as hlp

        # This automatically registers your plug-in
        @hlp.PLUGIN_REGISTRY
        class MyPlugin(hlp.Plugin):
            ...

        # Alternatively, you can manually register a plug-in like this:
        hlp.PLUGIN_REGISTRY.register(MyPlugin)
"""


def create_plugin(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> Plugin:
    """
    Create the plug-in for the given type.

    Args:
        type_name: the type of the plug-in to create.
        args: positional arguments to pass into the plug-in.
        kwargs: keyword arguments to pass into the plug-in.

    Returns:
        The plug-in.
    """
    return PLUGIN_REGISTRY.get(type_name)(*args, **kwargs)


class Plugin(abc.ABC):
    """
    Base class for plug-ins that extend the functionality of the Helios trainer.

    You can use this class to customize the behaviour of training to achieve a variety of
    objectives.
    The plug-ins have a similar API to the :py:class:`~helios.model.model.Model` class.
    The only major difference is that the plug-in functions are called *before* the
    corresponding model functions, providing the ability to override the model if
    necessary.

    Args:
        plug_id: the string with which the plug-in will be registered in the trainer
            plug-in table.
    """

    def __init__(self, plug_id: str):
        """Create the plug-in."""
        self._trainer: Trainer | None
        self._plug_id = plug_id

        self._is_distributed: bool = False
        self._map_loc: str | dict[str, str] = ""
        self._device: torch.device | None = None
        self._rank: int = 0
        self._overrides = UniquePluginOverrides()

    @property
    def unique_overrides(self) -> UniquePluginOverrides:
        """The set of unique overrides the plug-in uses."""
        return self._overrides

    @property
    def is_distributed(self) -> bool:
        """Flag controlling whether distributed training is being used or not."""
        return self._is_distributed

    @is_distributed.setter
    def is_distributed(self, val: bool) -> None:
        self._is_distributed = val

    @property
    def map_loc(self) -> str | dict[str, str]:
        """The location to map loaded weights from a checkpoint or pre-trained file."""
        return self._map_loc

    @map_loc.setter
    def map_loc(self, loc: str | dict[str, str]) -> None:
        self._map_loc = loc

    @property
    def device(self) -> torch.device:
        """The device on which the plug-in is running."""
        return core.get_from_optional(self._device)

    @device.setter
    def device(self, dev: torch.device) -> None:
        self._device = dev

    @property
    def rank(self) -> int:
        """The local rank (device id) that the plug-in is running on."""
        return self._rank

    @rank.setter
    def rank(self, r: int) -> None:
        self._rank = r

    @property
    def trainer(self) -> Trainer:
        """Reference to the trainer."""
        return core.get_from_optional(self._trainer)

    @trainer.setter
    def trainer(self, t: Trainer) -> None:
        self._trainer = t

    def configure_trainer(self, trainer: Trainer) -> None:
        """
        Configure the trainer before training or testing.

        This function can be used to set certain properties of the trainer. For example,
        it can be used to assign valid exceptions that the plug-in requires or to register
        the plug-in itself in the trainer.

        Args:
            trainer: the trainer instance.
        """

    def configure_model(self, model: Model) -> None:
        """
        Configure the model before training or testing.

        This function can be used to set certain properties of the model. For example, it
        can be used to override the save name of the model.
        """

    @abc.abstractmethod
    def setup(self) -> None:
        """Construct all required state for the plug-in."""

    def on_training_start(self) -> None:
        """Perform any necessary actions when training starts."""

    def process_training_batch(
        self, batch: typing.Any, state: TrainingState
    ) -> typing.Any:
        """
        Process the training batch.

        This function can be used to perform any processing on the training batch *prior*
        to the call to :py:meth:`~helios.model.model.Model.train_step`. For example, this
        can be used to filter out elements in a batch to reduce its size, or it can be
        used to move all elements in the batch to a set device.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """
        return batch

    def on_training_end(self) -> None:
        """Perform any necessary actions when training ends."""

    def on_validation_start(self, validation_cycle: int) -> None:
        """
        Perform any necessary actions when validation starts.

        Args:
            validation_cycle: the validation cycle number.
        """

    def process_validation_batch(self, batch: typing.Any, step: int) -> typing.Any:
        """
        Process the validation batch.

        This function can be used to perform any processing on the validation batch
        *prior* to the call to :py:meth:`~helios.model.model.Model.valid_step`. For
        example, this can be used to filter out elements in a batch to reduce its size,
        or it can be used to move all elements in the batch to a set device.

        Args:
            batch: the batch data returned from the dataset.
            step: the current validation batch.
        """
        return batch

    def on_validation_end(self, validation_cycle: int) -> None:
        """
        Perform any necessary actions when validation ends.

        Args:
            validation_cycle: the validation cycle number
        """

    def should_training_stop(self) -> bool:
        """
        Determine whether training should stop or continue.

        Returns:
            False if training should continue, true otherwise.
        """
        return False

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Load the plug-in state from the given state dictionary.

        Use this function to restore any state from a checkpoint.

        Args:
            state_dict: the state dictionary to load from.
        """

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Get the state dictionary of the plug-in.

        Use this function to save any state that you require for checkpoints.

        Returns:
            The state dictionary of the plug-in.
        """
        return {}

    def on_testing_start(self) -> None:
        """Perform any actions when testing starts."""

    def process_testing_batch(self, batch: typing.Any, step: int) -> typing.Any:
        """
        Process the testing batch.

        This function can be used to perform any processing on the testing batch
        *prior* to the call to :py:meth:`~helios.model.model.Model.test_step`. For
        example, this can be used to filter out elements in a batch to reduce its size,
        or it can be used to move all elements in the batch to a set device.

        Args:
            batch: the batch data returned from the dataset.
            step: the current testing batch number.
        """
        return batch

    def on_testing_end(self) -> None:
        """Perform any necessary actions when testing ends."""

    def _append_train_exceptions(
        self, exc: type[Exception] | list[type[Exception]], trainer: Trainer
    ) -> None:
        """
        Append exception type(s) to the list of valid train exceptions.

        Args:
            exc: valid exception type or a list of valid exception types.
            trainer: the trainer instance.
        """
        exc = core.convert_to_list(exc)  # type: ignore[arg-type]
        trainer.train_exceptions.extend(exc)  # type: ignore[arg-type]

    def _append_test_exceptions(
        self, exc: type[Exception] | list[type[Exception]], trainer: Trainer
    ) -> None:
        """
        Append exception type(s) to the list of valid test exceptions.

        Args:
            exc: valid exception type or a list of valid exception types.
            trainer: the trainer instance.
        """
        exc = core.convert_to_list(exc)  # type: ignore[arg-type]
        trainer.test_exceptions.extend(exc)  # type: ignore[arg-type]

    def _register_in_trainer(self, trainer: Trainer) -> None:
        """
        Register the plug-in instance in the trainer table.

        The plug-in will be registered with the value provided to ``plug_id``. If another
        plug-in is found with the same ID, an exception is raised.

        Args:
            trainer: the trainer instance.

        Raises:
            KeyError: if another plug-in with the same ID is found.
        """
        if self._plug_id in trainer.plugins:
            raise KeyError(
                f"error: a plug-in with id {self._plug_id} has already been registered"
            )

        trainer.plugins[self._plug_id] = self


@PLUGIN_REGISTRY.register
class CUDAPlugin(Plugin):
    """
    Plug-in to move elements of a training batch to a GPU.

    This plug-in can be used to move the elements of a training batch to the currently
    selected device automatically *prior* to the call to
    :py:meth:`~helios.model.model.Model.train_step`. The device is automatically assigned
    by the :py:class:`helios.trainer.Trainer` when training or testing starts.

    In order to cover the largest possible number of structures, the plug-in can handle
    the following containers:

    #. Single tensors
    #. Lists. Note that the elements of the list need not all be tensors. If any
       tensors are present, they are automatically moved to the device.
    #. Dictionaries. Similar to the list, not all the elements of the dictionary have
       to be tensors. Any tensors are detected automatically.

    .. warning::
        The plug-in is **not** designed to work with nested structures. In other words, if
        a list of dictionaries is passed in, the plug-in **will not** recognise any
        tensors contained inside the dictionary. Similarly, if a dictionary contains
        nested dictionaries (or any other container), the plug-in won't recognise them.

    .. warning::
        The use of this plug-in **requires** CUDA being enabled. If CUDA is not present,
        an exception is raised.

    .. note::
        If you require custom handling for your specific data types, you can override the
        behaviour of the plug-in by deriving from it. See the example below for details.

        Example:
            .. code-block:: python

                import helios.plug-ins as hlp

                class MyCUDAPlugin(hlp.CUDAPlugin):
                    def _move_collection_to_device(self, batch: <your-type>):
                        # Suppose our batch is a list:
                        for i in range(len(batch)):
                            batch[i] = batch[i].to(self.device)
    """

    plugin_id = "cuda"

    def __init__(self):
        """Create the plug-in."""
        super().__init__(self.plugin_id)
        core.cuda.requires_cuda_support()

        # Ensure that no-one else can manipulate the batch.
        self._overrides.training_batch = True
        self._overrides.validation_batch = True
        self._overrides.testing_batch = True

    def setup(self) -> None:
        """No-op setup function."""

    def configure_trainer(self, trainer: Trainer) -> None:
        """
        Register the plug-in instance into the trainer.

        The plug-in will be registered under the name ``cuda``.

        Args:
            trainer: the trainer instance.
        """
        self._register_in_trainer(trainer)

    def _move_collection_to_device(
        self, batch: torch.Tensor | list | dict | tuple
    ) -> torch.Tensor | list | dict | tuple:
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, list):
            for i in range(len(batch)):
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].to(self.device)
        elif isinstance(batch, tuple):
            as_list = list(batch)
            for i in range(len(as_list)):
                if isinstance(as_list[i], torch.Tensor):
                    as_list[i] = as_list[i].to(self.device)

            batch = tuple(as_list)
        elif isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
        else:
            raise RuntimeError(f"error: batch has unknown type {type(batch)}")

        return batch

    def process_training_batch(
        self, batch: typing.Any, state: TrainingState
    ) -> typing.Any:
        """
        Move the training batch to the GPU.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        return self._move_collection_to_device(batch)

    def process_validation_batch(self, batch: typing.Any, step: int) -> typing.Any:
        """
        Move the validation batch to the GPU.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        return self._move_collection_to_device(batch)

    def process_testing_batch(self, batch: typing.Any, step: int) -> typing.Any:
        """
        Move the testing batch to the GPU.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        return self._move_collection_to_device(batch)
