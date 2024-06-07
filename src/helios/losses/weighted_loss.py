import abc
import typing

from torch import nn


class WeightedLoss(nn.Module, metaclass=abc.ABCMeta):
    """
    Defines a base class for weighted losses.

    The value of the final loss is determined by the following formula:

        .. math:: L_w = w * L

    where :math:`w` is the weight and :math:`L` is the loss function.

    Example:
        .. code-block:: python

            class MyLoss(WeightedLoss):
                ...
                def _eval(self, ...):
                    return my_loss_function(...)

    Args:
        loss_weight: the weight of the loss function. Defaults to 1.
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
            The result of the loss function.
        """

    def forward(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        """
        Forward wrapper function.

        The final loss value will be computed as described above.

        Args:
            *args: arguments to the loss function.
            **kwargs: keyword arguments.

        Returns:
            The weighted value of the loss function.
        """
        return self._loss_weight * self._eval(*args, **kwargs)
