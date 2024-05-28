import typing

from torch import nn
from torch.optim import lr_scheduler

from helios import core


def _register_default_schedulers(registry: core.Registry):
    """
    Register the default Torch schedulers to the registry.

    List was obtained from
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Args:
        registry (Registry): the scheduler registry.
    """
    registry.register(lr_scheduler.LambdaLR)
    registry.register(lr_scheduler.MultiplicativeLR)
    registry.register(lr_scheduler.StepLR)
    registry.register(lr_scheduler.MultiStepLR)
    registry.register(lr_scheduler.ConstantLR)
    registry.register(lr_scheduler.LinearLR)
    registry.register(lr_scheduler.ExponentialLR)
    registry.register(lr_scheduler.PolynomialLR)
    registry.register(lr_scheduler.CosineAnnealingLR)
    registry.register(lr_scheduler.SequentialLR)
    registry.register(lr_scheduler.ReduceLROnPlateau)
    registry.register(lr_scheduler.CyclicLR)
    registry.register(lr_scheduler.OneCycleLR)
    registry.register(lr_scheduler.CosineAnnealingWarmRestarts)


SCHEDULER_REGISTRY = core.Registry("scheduler")
_register_default_schedulers(SCHEDULER_REGISTRY)


def create_scheduler(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> nn.Module:
    """
    Create the scheduler for the given type.

    Args:
        type_name (str): the type of the scheduler to create.
        args: positional arguments to pass into the scheduler.
        kwargs: keyword arguments to pass into the scheduler.

    Returns:
        nn.Module: the scheduler.
    """
    return SCHEDULER_REGISTRY.get(type_name)(*args, **kwargs)
