from torch import nn, optim

from pyro import core


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


def create_optimizer(type_name: str, **kwargs) -> nn.Module:
    """
    Create the optimizer for the given type.

    Args:
        type_name (str): the type of the optimizer to create.
        kwargs: any arguments you wish to pass into the optimizer.
    Returns:
        nn.Module: the optimizer.
    """
    return OPTIMIZER_REGISTRY.get(type_name)(**kwargs)
