from __future__ import annotations

import abc
import typing

import torch

from helios import core
from helios.core import distributed as dist

if typing.TYPE_CHECKING:
    from ..trainer import Trainer, TrainingState


# Tell Ruff to ignore the empty-method-without-abstract-decorator check, since a lot of
# functions in the Model base class will be empty by default.
# ruff: noqa: B027


class Model(abc.ABC):
    """Base class for training models."""

    def __init__(self, save_name: str):
        """Create the model."""
        self._save_name: str = save_name
        self._trainer: Trainer | None

        self._is_distributed: bool = False
        self._map_loc: str | dict[str, str] = ""
        self._device: torch.device | None = None
        self._rank: int = 0

        self._loss_items: dict[str, torch.Tensor] = {}
        self._running_losses: dict[str, float] = {}
        self._val_scores: dict[str, float] = {}
        self._test_scores: dict[str, float] = {}

    @property
    def save_name(self) -> str:
        """The name of the model used for saving checkpoints and final networks."""
        return self._save_name

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
        """The device on which the tensors of the model are mapped to."""
        return core.get_from_optional(self._device)

    @device.setter
    def device(self, dev: torch.device) -> None:
        self._device = dev

    @property
    def rank(self) -> int:
        """The local rank (device id) that the model is running on."""
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
    def setup(self, fast_init: bool = False) -> None:
        """
        Initialize all the state necessary for training.

        Use this function to load all the networks, schedulers, optimizers, losses, etc.
        that you require for training. This will be called before training starts and
        after the distributed processes have been launched (if applicable).

        The fast_init flag is used to indicate that the model is going to be used to strip
        training data from any saved checkpoints or for testing. As such, you should
        ONLY load the network(s) and nothing else.

        Args:
            fast_init (bool): if True, only networks are loaded.
        """

    def load_for_testing(self) -> None:
        """
        Load the network(s) used for testing.

        When testing, the trainer will try to load the last checkpoint saved during
        training. If it is unable to find one, it will call this function to ensure the
        model loads the state of the network(s) for testing. If you wish to load your
        network using separate logic, use this function.
        """

    def load_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool = False
    ) -> None:
        """
        Load the model state from the given state dictionary.

        Use this function to restore any training state from a checkpoint. Note that any
        weights will have been automatically mapped to the correct device.

        The fast_init flag is used to indicate that the model is going to be used to strip
        training data from any saved checkpoints or for testing. As such, you should
        ONLY load the network(s) and nothing else.

        Args:
            state_dict (dict[str, Any]): the state dictionary to load from.
            fast_init (bool): if True, only networks need to be loaded.
        """

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Get the state dictionary of the model.

        Use this function to save any state that you require for checkpoints.

        Returns:
            dict[str, Any]: the state dictionary of the model.
        """
        return {}

    def trained_state_dict(
        self, *args: typing.Any, **kwargs: typing.Any
    ) -> dict[str, typing.Any]:
        """
        Get the state dictionary for the trained model.

        Use this function to save the state required for the final trained model. The
        returned dictionary should contain only the necessary information to re-create the
        network(s) along with any additional data you require.

        Args:
            args: positional arguments.
            kwargs: keyword arguments.

        Returns:
            dict[str, Any]: the state dictionary without any training data.
        """
        return {}

    def append_metadata_to_chkpt_name(self, chkpt_name: str) -> str:
        """
        Append additional data to the checkpoint filename.

        Use this function to append the value of the loss function(s), validation
        metric(s), or any extra metadata you wish to add to the name of the checkpoint.
        Note that the epoch and iteration numbers are added automatically. The extension
        will also be added automatically.

        Args:
            chkpt_name (str): the name of the checkpoint filename (without extension).

        Returns:
            str: the name with any additional metadata.
        """
        return chkpt_name

    def train(self) -> None:
        """Switch the model to training mode."""

    def on_training_start(self) -> None:
        """
        Perform any necessary actions when training starts.

        You may use this function to log the network architecture, hyper-params, etc.
        """

    def on_training_batch_start(self, state: TrainingState) -> None:
        """
        Perform any actions when a training batch is started.

        This function is called before train_step is called. By default, it will clear out
        the loss table, but you may also use it to do any additional tasks prior to the
        training step itself.

        Args:
            state (TrainingState): the current training state.
        """
        self._loss_items.clear()

    @abc.abstractmethod
    def train_step(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform a single training step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform the forward and backward passes for your
        network(s). If you use schedulers, they should be updated here as well. Note that
        you do not have to clear the losses or gather them. This will be handled
        automatically for you. Also keep in mind that the contents of the batch have NOT
        been moved to the current device. It is your responsibility to do so.

        Args:
            batch (Any): the batch data returned from the dataset.
            state (TrainingState): the current training state.
        """

    def on_training_batch_end(
        self, state: TrainingState, should_log: bool = False
    ) -> None:
        """
        Perform any actions when a training batch ends.

        This function is called after train_step is called. By default, it will gather all
        the losses stored in self._loss_items (if using distributed training) and will
        update the running losses using those values. You may also use this function to
        log your losses or perform any additional tasks after the training step.

        Args:
            state (TrainingState): the current training state.
            should_log (bool): if true, then logging should be performed.
        """
        if self._is_distributed:
            for _, loss in self._loss_items.items():
                dist.all_reduce_tensors(loss)

        for key, loss in self._loss_items.items():
            if loss is not None:
                if key not in self._running_losses:
                    self._running_losses[key] = loss.item()
                else:
                    self._running_losses[key] += loss.item()

    def on_training_end(self) -> None:
        """
        Perform any necessary actions when training ends.

        You may use this function to update any weight averaging networks, or any other
        tasks that should only happen at the end of training.
        """

    def eval(self) -> None:
        """Switch the model to evaluation mode."""

    def on_validation_start(self, validation_cycle: int) -> None:
        """
        Perform any necessary actions when validation starts.

        By default, this will clear out the table of validation values, but you may use it
        for any other tasks that should happen when validation begins.

        Args:
            validation_cycle (int): the validation cycle number.
        """
        self._val_scores.clear()

    def on_validation_batch_start(self, step: int) -> None:
        """
        Perform any actions when a validation batch is started.

        This function is called before valid_step is called. No steps are performed by
        default.

        Args:
            step (int): the current validation batch.
        """

    def valid_step(self, batch: typing.Any, step: int) -> None:
        """
        Perform a single validation step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform any steps necessary to compute the validation
        metric(s) for your network(s).

        Args:
            batch (Any): the batch data returned from the dataset.
            step (int): the current validation batch.
        """

    def on_validation_batch_end(self, step: int) -> None:
        """
        Perform any actions when a validation batch ends.

        This function is called after valid_step is called. No steps are performed by
        default.

        Args:
            step (int): the current validation batch.
        """

    def on_validation_end(self, validation_cycle: int) -> None:
        """
        Perform any necessary actions when validation ends.

        By default, this function will clear out the running loss table, but you may use
        this function to compute any final validation metrics as well as log them.

        Args:
            validation_cycle (int): the validation cycle number.
        """
        self._running_losses.clear()

    def have_metrics_improved(self) -> bool:
        """
        Determine whether the current validation results are an improvement or not.

        This is used when early stopping is enabled in the trainer to determine whether
        the stop cycle count should increase or not.

        Returns:
            bool: false if no improvements were seen in the last validation cycle.
        """
        return True

    def on_testing_start(self) -> None:
        """
        Perform any necessary actions when testing starts.

        By default, this will clear out the table of testing values, but you may use it
        for any other tasks that should happen when testing begins.
        """
        self._test_scores.clear()

    def on_testing_batch_start(self, step: int) -> None:
        """
        Perform any actions when a testing batch is started.

        This function is called before test_step is called. No steps are performed by
        default.

        Args:
            step (int): the current testing batch.
        """

    def test_step(self, batch: typing.Any, step: int) -> None:
        """
        Perform a single testing step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform any steps necessary to compute the testing
        metric(s) for your network(s).

        Args:
            batch (Any): the batch data returned from the dataset.
            step (int): the current validation batch.
        """

    def on_testing_batch_end(self, step: int) -> None:
        """
        Perform any actions when a testing batch ends.

        This function is called after test_step is called. No steps are performed by
        default.

        Args:
            step (int): the current testing batch.
        """

    def on_testing_end(self) -> None:
        """
        Perform any necessary actions when testing ends.

        You may use this function to compute any final testing metrics as well as log
        them.
        """
