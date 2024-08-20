from __future__ import annotations

import abc
import typing

import torch

from helios import core

if typing.TYPE_CHECKING:
    from ..trainer import Trainer, TrainingState

# ruff: noqa: B027

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

    def prepare(self) -> None:
        """
        Prepare the plugin for training.

        This can include downloading auxiliary data, setting up additional state, etc.
        This function will be called on the primary process when using distributed
        training (will be called prior to initialization of the processes) so don't store
        any state here.
        """

    @abc.abstractmethod
    def setup(self) -> None:
        """Construct all required state for the plugin."""

    def on_training_start(self) -> None:
        """Perform any necessary actions when training starts."""

    def on_training_epoch_start(self, current_epoch: int) -> None:
        """
        Perform any necessary actions when a training epoch is started.

        Args:
            current_epoch: the epoch number that has just started.
        """

    def on_training_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform any actions when a training batch is started.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

    def on_training_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform any actions when a training batch ends.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

    def on_training_epoch_end(self, current_epoch: int) -> None:
        """
        Perform any necessary actions when a training epoch ends.

        Args:
            current_epoch: the epoch number that has just started.
        """

    def on_training_end(self) -> None:
        """Perform any necessary actions when training ends."""

    def on_validation_start(self, validation_cycle: int) -> None:
        """Perform any necessary actions when validation starts."""

    def on_validation_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform any necessary actions when a validation batch is started.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

    def on_validation_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform any necessary actions when a validation batch ends.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

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

    def on_testing_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform any necessary actions when a testing batch starts.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

    def on_testing_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform any necessary actions when a testing batch ends.

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

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

    def _move_collection_to_device(self, batch: torch.Tensor | list | dict):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, list):
            for i in range(len(batch)):
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].to(self.device)
        elif isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
        else:
            raise RuntimeError(f"error: batch has unknown type {type(batch)}")

    def on_training_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Move the batch to the GPU at the start of the training batch.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        self._move_collection_to_device(batch)

    def on_validation_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Move the batch to the GPU at the start of the validation batch.

        Args:
            batch: the batch returned by the training dataset.
            state: the current training state.
        """
        self._move_collection_to_device(batch)
