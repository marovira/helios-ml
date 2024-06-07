import typing

from torch import nn

from helios import core

LOSS_REGISTRY = core.Registry("loss")
"""
Global instance of the registry for loss functions.

Example:
    .. code-block:: python

        import helios.losses as hll

        # This automatically registers your loss function.
        @hll.LOSS_REGISTRY.register
        class MyLoss:
            ...

        # Alternatively you can manually register a loss function like this:
        hll.LOSS_REGISTRY.register(MyLoss)
"""


def create_loss(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> nn.Module:
    """
    Create the loss function for the given type.

    Args:
        type_name: the type of the loss to create.
        args: positional arguments to pass into the loss.
        kwargs: keyword arguments to pass into the loss.
    Returns:
        The loss function.
    """
    return LOSS_REGISTRY.get(type_name)(*args, **kwargs)
