from __future__ import annotations

import abc
import dataclasses as dc
import typing

import torch

from helios import core

if typing.TYPE_CHECKING:
    from ..trainer import Trainer, TrainingState

# ruff: noqa: B027


@dc.dataclass
class UniquePluginOverrides:
    """
    Set of flags that determine the unique overrides a plugin can have.

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
    """

    training_batch: bool = False
    validation_batch: bool = False
    testing_batch: bool = False


PLUGIN_REGISTRY = core.Registry("plugin")
"""
Global instance of the registry for plugins.

By default, the registry contains the following plugins:

.. list-table:: Schedulers
    :header-rows: 1

    * - Plugin
      - Name
    * - helios.plugins.CUDAPlugin
      - CUDAPlugin

Example:
    .. code-block:: python

        import helios.plugins as hlp

        # This automatically registers your plugin
        @hlp.PLUGIN_REGISTRY
        class MyPlugin(hlp.Plugin):
            ...

        # Alternatively, you can manually register a plugin like this:
        hlp.PLUGIN_REGISTRY.register(MyPlugin)
"""


def create_plugin(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> Plugin:
    """
    Create the plugin for the given type.

    Args:
        type_name: the type of the plugin to create.
        args: positional arguments to pass into the plugin.
        kwargs: keyword arguments to pass into the plugin.

    Returns:
        The plugin.
    """
    return PLUGIN_REGISTRY.get(type_name)(*args, **kwargs)


class Plugin(abc.ABC):
    """
    Base class for plugins that extend the functionality of the Helios trainer.

    You can use this class to customize the behaviour of training to achieve a variety of
    objectives.
    The plugins have a similar API to the :py:class:`~helios.model.model.Model` class. The
    only major difference is that the plugin functions are called *before* the
    corresponding model functions, providing the ability to override the model if
    necessary.
    """

    def __init__(self):
        """Create the plugin."""
        self._trainer: Trainer | None

        self._is_distributed: bool = False
        self._map_loc: str | dict[str, str] = ""
        self._device: torch.device | None = None
        self._rank: int = 0
        self._overrides = UniquePluginOverrides()

    @property
    def unique_overrides(self) -> UniquePluginOverrides:
        """The set of unique overrides the plugin uses."""
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
        """The device on which the plugin is running."""
        return core.get_from_optional(self._device)

    @device.setter
    def device(self, dev: torch.device) -> None:
        self._device = dev

    @property
    def rank(self) -> int:
        """The local rank (device id) that the plugin is running on."""
        return self._rank

    @rank.setter
    def rank(self, r: int) -> None:
        self._rank = r

    @property
    def trainer(self) -> Trainer:
        """Reference to the trainer."""
        return core.get_from_optional(self._trainer)

    @trainer.setter
    def trainer(self, t) -> None:
        self._trainer = t

    @abc.abstractmethod
    def setup(self) -> None:
        """Construct all required state for the plugin."""

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
        """Perform any necessary actions when validation starts."""

    def process_validation_batch(
        self, batch: typing.Any, state: TrainingState
    ) -> typing.Any:
        """
        Process the validation batch.

        This function can be used to perform any processing on the validation batch
        *prior* to the call to :py:meth:`~helios.model.model.Model.valid_step`. For
        example, this can be used to filter out elements in a batch to reduce its size,
        or it can be used to move all elements in the batch to a set device.
        """
        return batch

    def on_validation_end(self) -> None:
        """Perform any necessary actions when validation ends."""

    def should_training_stop(self) -> bool:
        """
        Determine whether training should stop or continue.

        Returns:
            False if training should continue, true otherwise.
        """
        return False

    def on_testing_start(self) -> None:
        """Perform any actions when testing starts."""

    def process_testing_batch(
        self, batch: typing.Any, state: TrainingState
    ) -> typing.Any:
        """
        Process the testing batch.

        This function can be used to perform any processing on the testing batch
        *prior* to the call to :py:meth:`~helios.model.model.Model.test_step`. For
        example, this can be used to filter out elements in a batch to reduce its size,
        or it can be used to move all elements in the batch to a set device.
        """
        return batch

    def on_testing_end(self) -> None:
        """Perform any necessary actions when testing ends."""


@PLUGIN_REGISTRY.register
class CUDAPlugin(Plugin):
    """
    Plugin to move elements of a training batch to a GPU.

    This plugin can be used to move the elements of a training batch to the currently
    selected device automatically *prior* to the call to
    :py:meth:`~helios.model.model.Model.train_step`. The device is automatically assigned
    by the :py:class:`helios.trainer.Trainer` when training or testing starts.

    In order to cover the largest possible number of structures, the plugin can handle the
    following containers:
        #. Singe tensors
        #. Lists. Note that the elements of the list need not all be tensors. If any
          tensors are present, they are automatically moved to the device.
        #. Dictionaries. Similar to the list, not all the elements of the dictionary have
          to be tensors. Any tensors are detected automatically.

    .. warning::
        The plugin is **not** designed to work with nested structures. In other words, if
        a list of dictionaries is passed in, the plugin **will not** recognise any tensors
        contained inside the dictionary. Similarly, if a dictionary contains nested
        dictionaries (or any other container), the plugin won't recognise them.

    .. warning::
        The use of this plugin **requires** CUDA being enabled. If CUDA is not present, an
        exception is raised.

    .. note::
        If you require custom handling for your specific data types, you can override the
        behaviour of the plugin by deriving from it. See the example below for details.

        Example:
            .. code-block:: python

            import helios.plugins as hlp

            class MyCUDAPlugin(hlp.CUDAPlugin):
                def _move_collection_to_device(self, batch: <your-type>):
                    # Suppose our batch is a list:
                    for i in range(len(batch)):
                        batch[i] = batch[i].to(self.device)
    """

    def __init__(self):
        """Create the plugin."""
        super().__init__()
        core.cuda.requires_cuda_support()

        # Ensure that no-one else can manipulate the batch.
        self._overrides.training_batch = True
        self._overrides.validation_batch = True
        self._overrides.testing_batch = True

    def setup(self) -> None:
        """No-op setup function."""

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

    def process_validation_batch(
        self, batch: typing.Any, state: TrainingState
    ) -> typing.Any:
        """
        Move the validation batch to the GPU.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        return self._move_collection_to_device(batch)

    def process_testing_batch(
        self, batch: typing.Any, state: TrainingState
    ) -> typing.Any:
        """
        Move the testing batch to the GPU.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        return self._move_collection_to_device(batch)
