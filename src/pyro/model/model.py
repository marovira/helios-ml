import abc
import dataclasses
import pathlib
import typing

import torch

from pyro import core
from pyro.core import distributed as dist

MODEL_REGISTRY = core.Registry("model")

# Tell Ruff to ignore the empty-method-without-abstract-decorator check, since a lot of
# functions in the Model base class will be empty by default.
# ruff: noqa: B027


def create_model(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> "Model":
    """
    Create the model for the given type.

    Args:
        type_name (str): the type of the model to create.
        args: positional arguments to pass into the model.
        kwargs: keyword arguments to pass into the model.

    Returns:
        Model: the model.
    """
    return MODEL_REGISTRY.get(type_name)(*args, **kwargs)


def find_pretrained_file(root: pathlib.Path, name: str) -> pathlib.Path:
    """
    Find the pre-trained file in the given root.

    The assumption is the following:
        Given a root /models/cifar and a name resnet-50, then the name of the
        pre-trained file will contain cifar_resnet-50_ as a prefix. If no file is
        found, an exception is raised.

    Args:
        root (pathlib.Path): the root where the file is stored.
        net_name (str): the save name of the file.

    Returns:
        pathlib.Path: the path to the file.
    """
    for path in root.glob("*.pth"):
        file_name = str(path.stem)
        base_name = f"{str(root.stem)}_{name}_"
        if base_name in file_name:
            return path

    raise RuntimeError(
        f"error: unable to find a pretrained network named {name} at {str(root)}"
    )


@dataclasses.dataclass
class Checkpoint:
    """
    Represents the state loaded from a previously saved checkpoint.

    Args:
        path (Optional[pathlib.Path]): the path of the loaded checkpoint. If it's None,
        then no checkpoint is held and state_dict is also None.
        state_dict (Optional[dict[str, Any]]): the state dictionary of the loaded
        checkpoint. May be None, in which case path is also None and no checkpoint is
        held.
        epoch (int): the epoch the checkpoint was saved on.
        ite (int): the iteration the checkpoint was saved on.
    """

    path: pathlib.Path | None = None
    state_dict: dict[str, typing.Any] | None = None
    epoch: int = 0
    ite: int = 0


class Model(abc.ABC):
    """Base class for training models."""

    def __init__(self, save_name: str, display_name: str):
        """Create the model."""
        self._save_name: str = save_name
        self._display_name: str = display_name

        self._is_distributed: bool = False
        self._map_loc: str | dict[str, str] = ""
        self._device: torch.device | None = None

        self._loss_items: dict[str, torch.Tensor] = {}
        self._val_scores: dict[str, float] = {}

    @property
    def save_name(self) -> str:
        """The name of the model used for saving checkpoints and final networks."""
        return self._save_name

    @property
    def display_name(self) -> str:
        """The name of the model used for display/logging purposes."""
        return self._display_name

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

    @abc.abstractmethod
    def setup(self, fast_init: bool = False) -> None:
        """
        Initialize all the state necessary for training.

        Use this function to load all the networks, schedulers, optimizers, losses, etc.
        that you require for training. This will be called before training starts and
        after the distributed processes have been launched (if applicable).

        The fast_init flag is used to indicate that the model is going to be used to strip
        training data from any saved checkpoints. As such, you should ONLY load the
        network(s) and nothing else.

        Args:
            fast_init (bool): if True, only networks are loaded.
        """

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Load the model state from the given state dictionary.

        Use this function to restore any training state from a checkpoint. Note that any
        weights will have been automatically mapped to the correct device.

        Args:
            state_dict (dict[str, Any]): the state dictionary to load from.
        """

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Get the state dictionary of the model.

        Use this function to save any state that you require for checkpoints.

        Returns:
            dict[str, Any]: the state dictionary of the model.
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

    def strip_training_data(
        self, state_dict: dict[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """
        Remove any training data from the state dictionary.

        Use this function to convert a checkpoint into a pre-trained network. Note that
        the auxiliary training state (log information, training settings, etc.) will be
        removed automatically before this function is called. The returned dictionary
        should contain only the necessary information to re-create the network(s) along
        with any additional data you require.

        Args:
            state_dict (dict[str, Any]): the state dictionary.

        Returns:
            dict[str, Any]: the state dictionary without any training data.
        """
        return state_dict

    def train(self) -> None:
        """Switch the model to training mode."""

    def on_training_start(self) -> None:
        """
        Perform any necessary actions when training starts.

        You may use this function to log the network architecture, hyper-params, etc.
        """

    def on_training_batch_start(self) -> None:
        """
        Perform any actions when a training batch is started.

        This function is called before train_step is called. By default, it will clear out
        the loss table, but you may also use it to do any additional tasks prior to the
        training step itself.
        """
        self._loss_items.clear()

    @abc.abstractmethod
    def train_step(self, batch: typing.Any) -> None:
        """
        Perform a single training step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform the forward and backward passes for your
        network(s). If you use schedulers, they should be updated here as well. Note that
        you do not have to clear the losses or gather them. This will be handled
        automatically for you.
        """

    def on_training_batch_end(self) -> None:
        """
        Perform any actions when a training batch ends.

        This function is called after train_step is called. By default, it will gather all
        the losses (if using distributed training), but you may also use it to log your
        losses or perform any additional tasks after the training step.
        """
        if self._is_distributed:
            for _, loss in self._loss_items.items():
                dist.all_reduce_tensors(loss)

    def on_training_end(self) -> None:
        """
        Perform any necessary actions when training ends.

        You may use this function to update any weight averaging networks, or any other
        tasks that should only happen at the end of training.
        """

    def eval(self) -> None:
        """Switch the model to evaluation mode."""

    def on_validation_start(self) -> None:
        """
        Perform any necessary actions when validation starts.

        By default, this will clear out the table of validation values, but you may use it
        for any other tasks that should happen when validation begins.
        """
        self._val_scores.clear()

    def on_validation_batch_start(self) -> None:
        """
        Perform any actions when a validation batch is started.

        This function is called before valid_step is called. No steps are performed by
        default.
        """

    @abc.abstractmethod
    def valid_step(self, batch: typing.Any) -> None:
        """
        Perform a single validation step.

        The input is the returned value from the datasets you supplied to the trainer. In
        this function, you should perform any steps necessary to compute the validation
        metric(s) for your network(s).
        """

    def on_validation_batch_end(self) -> None:
        """
        Perform any actions when a validation batch ends.

        This function is called after train_step is called. No steps are performed by
        default.
        """

    def on_validation_end(self) -> None:
        """
        Perform any necessary actions when validation ends.

        You may use this function to compute any final validation metrics as well as log
        them.
        """

    def have_metrics_improved(self) -> bool:
        """
        Determine whether the current validation results are an improvement or not.

        This is used when early stopping is enabled in the trainer to determine whether
        the stop cycle count should increase or not.

        Returns:
            bool: false if no improvements were seen in the last validation cycle.
        """
        return True