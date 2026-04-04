from __future__ import annotations

import abc
import contextlib
import dataclasses as dc
import typing

import torch

from helios import core
from helios.core import distributed as dist

if typing.TYPE_CHECKING:
    from ..trainer import Trainer, TrainingState


@dc.dataclass
class AMPContext:
    """
    The AMP scaler state.

    You can use this class to gain access to the AMP state of the model. For example, you
    can do this:

    .. code-block:: python

        Class MyModel(helios.model.Model):
            def train_step(self, batch: typing.Any, state: TrainingState) -> None:
                with self.autocast():
                    loss = ...

                # Now scale the loss
                if self.amp_context is not None:
                    scaler = self.amp_context.scaler
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)

    Args:
        scaler: the :py:class:`torch.amp.GradScaler` (if enabled).
        dtype: the :py:class:`torch.dtype` of the scaler.
    """

    scaler: torch.amp.GradScaler
    dtype: torch.dtype = torch.float16


# Tell Ruff to ignore the empty-method-without-abstract-decorator check, since a lot of
# functions in the Model base class will be empty by default.
# ruff: noqa: B027


class Model(abc.ABC):
    """
    Base class that groups together the functionality needed to train networks.

    The use of this class is to standardize the way networks are created and trained. This
    allows the training code to be shared across multiple networks, reducing code
    duplication.
    The functions provided by the :py:class:`~helios.model.model.Model` class can be
    overridden to satisfy the individual needs of each of the network(s) that need to be
    trained.

    Example:
        Suppose the body of the training loop looks something like this:

        .. code-block:: python

            dataloader = ... # The dataloader for our dataset.
            net = ... # The network we wish to train.
            optimzer = ... # The optimizer
            criterion = ... # The loss function
            for batch in dataloder:
                inputs, labels = batch

                optimizer.zero_grad()
                outs = net(inputs)
                loss = criterion(outs, labels)
                loss.backward()
                optimizer.step()

        Then the code would be placed into a Model as follows:

        .. code-block:: python

            import helios.model as hlm
            import helios.trainer as hlt
            class MyModel(hlm.Model):
                def setup(self, fast_init: bool = False) -> None:
                    self._net = ...
                    self._optimizer = ...
                    self._criterion = ...

                def train_step(self, batch, state: hlt.TrainingState) -> None:
                    inputs, labels = batch

                    optimizer.zero_grad()
                    outs = net(inputs)
                    loss = criterion(outs, labels)
                    loss.backward()
                    optimizer.step()

    The example shown here is the most basic version of the training code and can be
    expanded in various ways by overriding the different available functions.

    Args:
        save_name: the name that will be used to identify the model when checkpoints are
            saved.
    """

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
        self._val_scores: dict[str, typing.Any] = {}
        self._test_scores: dict[str, typing.Any] = {}

        self._amp_context: AMPContext | None = None

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

    @property
    def metrics(self) -> dict[str, typing.Any]:
        """The metric table for the model."""
        return {}

    @property
    def amp_context(self) -> AMPContext | None:
        """The AMP state of the model."""
        return self._amp_context

    @amp_context.setter
    def amp_context(self, state: AMPContext) -> None:
        self._amp_context = state

    @abc.abstractmethod
    def setup(self, fast_init: bool = False) -> None:
        """
        Initialize all the state necessary for training.

        Use this function to load all the networks, schedulers, optimizers, losses, etc.
        that you require for training. This will be called before training starts and
        after the distributed processes have been launched (if applicable).

        The ``fast_init`` flag is used to indicate that the model should **not** load any
        training state. This can be used for testing or for other purposes.

        Args:
            fast_init: if True, only networks are loaded.
        """

    def create_scaler(
        self, dtype: torch.dtype = torch.float16, **kwargs: typing.Any
    ) -> None:
        """
        Create the :py:class:`torch.amp.GradScaler` for AMP training.

        This should be called in your :py:meth:`setup` function to enable Automatic Mixed
        Precision. Note that if the current device is not CUDA, then this function is a
        no-op.

        .. note::
            There is no need to pass in the "device" argument to
            :py:class:`torch.amp.GradScaler`. This is already handled internally. If the
            keyword is passed in, it will be removed.

        Args:
            dtype: the dtype to use for :py:func:`torch.amp.autocast`. Defaults to
                ``torch.float16``.
            kwargs: additional keyword arguments for :py:class:`torch.amp.GradScaler`.
        """
        if self._device is None or self.device.type != "cuda":
            return
        if "device" in kwargs:
            kwargs.pop("device")

        self.amp_context = AMPContext(
            scaler=torch.amp.GradScaler(device=self.device.type, **kwargs), dtype=dtype
        )

    def autocast(self) -> contextlib.AbstractContextManager:
        """
        Return a context manager for Automatic Mixed Precision.

        Return :py:class:`torch.amp.autocast` configured with the AMP dtype if AMP is
        enabled, otherwise returns a null context.

        Returns:
            The context manager.
        """
        if self.amp_context is not None:
            return torch.amp.autocast(
                device_type=self.device.type, dtype=self.amp_context.dtype
            )
        return contextlib.nullcontext()

    def clip_gradients(
        self,
        parameters: torch.Tensor | typing.Iterable[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        max_norm: float,
        **kwargs: typing.Any,
    ) -> None:
        """
        Clip gradient norms, handling AMP unscaling automatically.

        When AMP is active, gradients must be unscaled before clipping or the norm
        is computed on scaled values, producing incorrect results.
        This function performs the following steps:
            1. Call :py:meth:`torch.amp.GradScaler.unscale_` if AMP is active.
            1. Call :py:func:`torch.nn.utils.clip_grad_norm_`.
        If AMP is not acitve, then :py:func:`torch.nn.utils.clip_grad_norm_` is called
        directly.

        Call this between the backward pass and the optimizer step:

        .. code-block:: python

            if self.amp_context.enabled:
                scaler = self.amp_context.scaler
                scaler.scale(loss).backward()
                self.clip_gradients(self._net.parameters(), self._optimizer, max_norm=1.0)
                scaler.step(self._optimizer)
                scaler.update()
            else:
                loss.backward()
                self.clip_gradients(self._net.parameters(), self._optimizer, max_norm=1.0)
                self._optimizer.step()

        Args:
            parameters: the parameters whose gradients will be clipped.
            optimizer: the optimizer associated with the parameters. Only used when AMP
                is enabled to perform gradient unscaling.
            max_norm: the maximum norm of the gradients.
            kwargs: additional keyword arguments forwarded to
                :py:func:`torch.nn.utils.clip_grad_norm_`.
        """
        if self.amp_context is not None:
            self.amp_context.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm, **kwargs)

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

        Restores the state from the dictionary in the following order:
            1. Internal state
            1. User-state
        Note that any weights will have been automatically mapped to the correct device.
        Also note that internal state is only loaded if ``fast_init`` is set to False.

        Args:
            state_dict: the state dictionary to load from.
            fast_init: if True, only networks need to be loaded.
        """
        if (
            not fast_init
            and self.amp_context is not None
            and "_helios_amp_scaler" in state_dict
        ):
            self.amp_context.scaler.load_state_dict(state_dict["_helios_amp_scaler"])
        user_dict = {k: v for k, v in state_dict.items() if "_helios_" not in k}
        self.load_user_state_dict(user_dict, fast_init)

    def load_user_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool
    ) -> None:
        """
        Load the user-defined model state from the given state dictionary.

        Use this function to restore any training state from a checkpoint. Note that any
        weights will have already been automatically mapped to the correct device.

        The ``fast_init`` flag is used to indicate that the model should **not** load any
        training state. This can be used for testing or for other purposes. As such, you
        should only load the state of your network(s) and nothing else.

        Args:
            state_dict: the state dictionary to load from.
            fast_init: if True, only networks need be loaded
        """

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Get the full state dictionary of the model.

        The full dictionary is assembled like this:
            1. Call :py:meth:`user_state_dict` to gather any user state.
            1. Ensure that none of the keys are reserved.
            1. Add the model internal state.

        Reserved keys are prefixed with ``_helios_`` to avoid conflicts with user-keys. If
        a reserved key is found in the user-returned dictionary, then an error will be
        raised.

        Returns:
            The state dictionary of the model.

        Raises:
            KeyError: if :py:meth:`user_state_dict` contains a reserved key.
        """
        state = self.user_state_dict()
        reserved_keys = ("_helios_amp_scaler",)
        conflicts = [k for k in reserved_keys if k in state]
        if len(conflicts) > 0:
            raise KeyError(
                "user_state_dict() contains the following reserved keys: "
                f"{conflicts}. Reserved keys are for internal use only"
            )
        if self.amp_context is not None:
            state["_helios_amp_scaler"] = self.amp_context.scaler.state_dict()
        return state

    def user_state_dict(self) -> dict[str, typing.Any]:
        """
        Get the user-defined state dictionary of the model.

        Use this  function to save any state that you require for checkpoints.

        .. warning::
            Do **not** use any keys that begin with ``_helios_`` as these are reserved for
            internal use.

        Returns:
            The user-defined state dictionary of the model.
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
            The state dictionary without any training data.
        """
        return {}

    def types_for_safe_load(self) -> list[typing.Callable | tuple[typing.Callable, str]]:
        """
        Get additional types to safely load checkpoints.

        These types are appended to the global list of types from Helios and are sent to
        ``torch.serialization.add_safe_globals`` so checkpoints can be safely loaded.

        Returns:
            The list of safe types to load.
        """
        return []

    def append_metadata_to_chkpt_name(self, chkpt_name: str) -> str:
        """
        Append additional data to the checkpoint filename.

        Use this function to append the value of the loss function(s), validation
        metric(s), or any extra metadata you wish to add to the name of the checkpoint.

        .. note::
            The epoch and iteration numbers, alongside the file extension, are added
            automatically.

        Args:
            chkpt_name: the name of the checkpoint filename (without extension).

        Returns:
            The name with any additional metadata.
        """
        return chkpt_name

    def append_to_banner(self, banner: str) -> str:
        """
        Append additional information to the main banner printed on start-up.

        Use this function to add any extra information you wish to the main banner shown
        at start-up. Note that if ``print_banner`` is set to false in the
        :py:class:`~helios.trainer.Trainer`, then this message is not shown.

        Returns:
            The banner string with any additional information.
        """
        return banner

    def train(self) -> None:
        """Switch the model to training mode."""

    def on_training_start(self) -> None:
        """
        Perform any necessary actions when training starts.

        You may use this function to log the network architecture, hyper-params, etc.
        """

    def on_training_epoch_start(self, current_epoch: int) -> None:
        """
        Perform any necessary actions when a training epoch is started.

        This function is called whenever a new training epoch begins, at the top of the
        training loop. You may use this function to set any necessary training state.

        Args:
            current_epoch: the epoch number that has just started.
        """

    def on_training_batch_start(self, state: TrainingState) -> None:
        """
        Perform any actions when a training batch is started.

        This function is called before :py:meth:`train_step` is called. By default, it
        will clear out the loss table, but you may also use it to do any additional tasks
        prior to the training step itself.

        Args:
            state: the current training state.
        """
        self._loss_items.clear()

    def train_step(self, batch: typing.Any, state: TrainingState) -> None:
        """
        Perform a single training step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform the forward and backward passes for your
        network(s). If you use schedulers, they should be updated here as well. Note that
        you do not have to clear the losses or gather them. This will be handled
        automatically for you.

        .. warning::
            The contents of the batch **are not** moved to any devices prior to this call.
            It is your responsibility to move them (if necessary).

        Args:
            batch: the batch data returned from the dataset.
            state: the current training state.
        """

    def on_training_batch_end(
        self, state: TrainingState, should_log: bool = False
    ) -> None:
        """
        Perform any actions when a training batch ends.

        This function is called after :py:meth:`~helios.model.model.Model.train_step` is
        called. By default, it will gather all the losses stored in ``self._loss_items``
        (if using distributed training) and will update the running losses using those
        values. You may also use this function to log your losses or perform any
        additional tasks after the training step.

        Args:
            state: the current training state.
            should_log: if true, then logging should be performed. Defaults to false.
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

    def on_training_epoch_end(self, current_epoch: int) -> None:
        """
        Perform any necessary actions when a training epoch ends.

        This function is called at the bottom of the epoch loop. You may use this
        function to perform any training operations you require.

        Args:
            current_epoch: the epoch number that has just ended.
        """

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
            validation_cycle: the validation cycle number.
        """
        self._val_scores.clear()

    def on_validation_batch_start(self, step: int) -> None:
        """
        Perform any actions when a validation batch is started.

        This function is called before :py:meth:`valid_step` is called. No steps are
        performed by default.

        Args:
            step: the current validation batch.
        """

    def valid_step(self, batch: typing.Any, step: int) -> None:
        """
        Perform a single validation step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform any steps necessary to compute the validation
        metric(s) for your network(s).

        Args:
            batch: the batch data returned from the dataset.
            step: the current validation batch.
        """

    def on_validation_batch_end(self, step: int) -> None:
        """
        Perform any actions when a validation batch ends.

        This function is called after :py:meth:`valid_step` is called. No steps are
        performed by default.

        Args:
            step: the current validation batch.
        """

    def on_validation_end(self, validation_cycle: int) -> None:
        """
        Perform any necessary actions when validation ends.

        By default, this function will clear out the running loss table, but you may use
        this function to compute any final validation metrics as well as log them.

        Args:
            validation_cycle: the validation cycle number.
        """
        self._running_losses.clear()

    def have_metrics_improved(self) -> bool:
        """
        Determine whether the current validation results are an improvement or not.

        This is used when early stopping is enabled in the trainer to determine whether
        the stop cycle count should increase or not. This is called immediately after the
        validation cycle finishes.

        Returns:
            False if no improvements were seen in the last validation cycle.
        """
        return True

    def should_training_stop(self) -> bool:
        """
        Determine whether training should stop or continue.

        This is used in the event that training should stop when:

        * A validation metric crosses a certain threshold,
        * A loss value becomes invalid,
        * Any other circumstance under which training should stop immediately.

        This function is called by the :py:class:`~helios.trainer.Trainer` at the
        following times:

        * After a full training step has finished. That is, after
          :py:meth:`on_training_batch_start`, :py:meth:`train_step`, and
          :py:meth:`on_training_batch_end` have been called.
        * After a validation cycle has finished. That is, after *all* of the validation
          callbacks have been called.
        * After each epoch has concluded (if training by epoch). Note that this happens
          *after* :py:meth:`on_training_epoch_end`.

        Returns:
            False if training should continue, true otherwise.
        """
        return False

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

        This function is called before :py:meth:`~helios.model.model.Model.test_step` is
        called. No steps are performed by default.

        Args:
            step: the current testing batch.
        """

    def test_step(self, batch: typing.Any, step: int) -> None:
        """
        Perform a single testing step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform any steps necessary to compute the testing
        metric(s) for your network(s).

        Args:
            batch: the batch data returned from the dataset.
            step: the current validation batch.
        """

    def on_testing_batch_end(self, step: int) -> None:
        """
        Perform any actions when a testing batch ends.

        This function is called after :py:meth:`~helios.model.model.Model.test_step` is
        called. No steps are performed by default.

        Args:
            step: the current testing batch.
        """

    def on_testing_end(self) -> None:
        """
        Perform any necessary actions when testing ends.

        You may use this function to compute any final testing metrics as well as log
        them.
        """
