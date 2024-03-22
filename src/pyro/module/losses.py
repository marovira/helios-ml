import abc
import typing

from torch import nn

from pyro import core

LOSS_REGISTRY = core.Registry("loss")


def create_loss(type_name: str, **kwargs) -> nn.Module:
    """
    Create the loss function for the given type.

    Args:
        type_name (str): the type of the loss to create.
        kwargs: any arguments you wish to pass into the loss.
    Returns:
        nn.Module: the loss function.
    """
    return LOSS_REGISTRY.get(type_name)(**kwargs)


class WeightedLoss(nn.Module, metaclass=abc.ABCMeta):
    """
    Defines a base class for weighted losses.

    The value of the final loss is determined by the following formula:
        loss = w * L
    where w is the weight, and L is the loss function.
    To use this class, do the following:
        ```py
        class MyLoss(WeightedLoss):
            ...
            def _eval(self, ...):
                return my_loss_function(...)
        ```
    Args:
        loss_weight (float): the weight of the loss function.
    """

    def __init__(self, loss_weight: float = 1.0):
        """Create the weighted loss."""
        super().__init__()
        self._loss_weight = loss_weight

    @abc.abstractmethod
    def _eval(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """
        Evaluate the loss function.

        Args:
            *args: arguments to the loss function.
            **kwargs: keyword arguments.

        Returns:
            Any: the result of the loss function.
        """

    def forward(self, *args, **kwargs) -> typing.Any:
        """
        Forward wrapper function.

        The final loss value will be computed as described above.

        Args:
            *args: arguments to the loss function.
            **kwargs: keyword arguments.

        Returns:
            Any: the weighted value of the loss function.
        """
        return self._loss_weight * self._eval(*args, **kwargs)
