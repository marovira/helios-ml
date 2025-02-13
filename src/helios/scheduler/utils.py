import typing

from torch.optim import lr_scheduler

from helios import core


def _register_default_schedulers(registry: core.Registry) -> None:
    """
    Register the default Torch schedulers to the registry.

    List was obtained from
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Args:
        registry: the scheduler registry.
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
"""
Global instance of the registry for schedulers.

By default, the registry contains the following schedulers:

.. list-table:: Schedulers
    :header-rows: 1

    * - Scheduler
      - Name
    * - ``torch.optim.lr_scheduler.LambdaLR``
      - LambdaLR
    * - ``torch.optim.lr_scheduler.MultiplicativeLR``
      - MultiplicativeLR
    * - ``torch.optim.lr_scheduler.StepLR``
      - StepLR
    * - ``torch.optim.lr_scheduler.MultiStepLR``
      - MultiStepLR
    * - ``torch.optim.lr_scheduler.ConstantLR``
      - ConstantLR
    * - ``torch.optim.lr_scheduler.LinearLR``
      - LinearLR
    * - ``torch.optim.lr_scheduler.ExponentialLR``
      - ExponentialLR
    * - ``torch.optim.lr_scheduler.PolynomialLR``
      - PolynomialLR
    * - ``torch.optim.lr_scheduler.CosineAnnealingLR``
      - CosineAnnealingLR
    * - ``torch.optim.lr_scheduler.SequentialLR``
      - SequentialLR
    * - ``torch.optim.lr_scheduler.ReduceLROnPlateau``
      - ReduceLROnPlateau
    * - ``torch.optim.lr_scheduler.CyclicLR``
      - CyclicLR
    * - ``torch.optim.lr_scheduler.OneCycleLR``
      - OneCycleLR
    * - ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``
      - CosineAnnealingWarmRestarts
    * - :py:class:`helios.scheduler.schedulers.CosineAnnealingRestartLR`
      - CosineAnnealingRestartLR
    * - :py:class:`helios.scheduler.schedulers.MultiStepRestartLR`
      - MultiStepRestartLR

Example:
    .. code-block:: python

        import helios.optim as hlo
        import helios.scheduler as hls

        # This automatically registers your optimizer.
        @hls.SCHEDULER_REGISTRY.register
        class MyScheduler:
            ...

        # Alternatively you can manually register a scheduler. like this:
        hls.SCHEDULER_REGISTRY.register(MyScheduler)
"""
_register_default_schedulers(SCHEDULER_REGISTRY)


def create_scheduler(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> lr_scheduler.LRScheduler:
    """
    Create the scheduler for the given type.

    Args:
        type_name: the type of the scheduler to create.
        args: positional arguments to pass into the scheduler.
        kwargs: keyword arguments to pass into the scheduler.

    Returns:
        The scheduler.
    """
    return SCHEDULER_REGISTRY.get(type_name)(*args, **kwargs)
