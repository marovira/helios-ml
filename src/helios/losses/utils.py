import typing

from torch import nn

from helios import core

LOSS_REGISTRY = core.Registry("loss")


def create_loss(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> nn.Module:
    """
    Create the loss function for the given type.

    Args:
        type_name (str): the type of the loss to create.
        args: positional arguments to pass into the loss.
        kwargs: keyword arguments to pass into the loss.
    Returns:
        nn.Module: the loss function.
    """
    return LOSS_REGISTRY.get(type_name)(*args, **kwargs)
