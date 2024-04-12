import typing

from torch import nn, optim

from helios import core


def _register_default_optimizers(registry: core.Registry) -> None:
    """
    Register the default optimizers.

    Args:
        registry (Registry): the optimizer registry.
    """
    registry.register(optim.Adam)
    registry.register(optim.AdamW)
    registry.register(optim.SGD)


OPTIMIZER_REGISTRY = core.Registry("optimizer")
_register_default_optimizers(OPTIMIZER_REGISTRY)


def create_optimizer(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> nn.Module:
    """
    Create the optimizer for the given type.

    Args:
        type_name (str): the type of the optimizer to create.
        args: positional arguments to pass into the optimizer.
        kwargs: keyword arguments to pass into the optimizer.
    Returns:
        nn.Module: the optimizer.
    """
    return OPTIMIZER_REGISTRY.get(type_name)(*args, **kwargs)
