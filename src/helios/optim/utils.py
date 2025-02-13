import typing

from torch import optim

from helios import core


def _register_default_optimizers(registry: core.Registry) -> None:
    """
    Register the Torch default optimizers.

    List was obtained from
    https://pytorch.org/docs/stable/optim.html#algorithms

    Args:
        registry: the optimizer registry.
    """
    registry.register(optim.Adadelta)
    registry.register(optim.Adagrad)
    registry.register(optim.Adam)
    registry.register(optim.AdamW)
    registry.register(optim.SparseAdam)
    registry.register(optim.Adamax)
    registry.register(optim.ASGD)
    registry.register(optim.LBFGS)
    registry.register(optim.NAdam)
    registry.register(optim.RAdam)
    registry.register(optim.RMSprop)
    registry.register(optim.Rprop)
    registry.register(optim.SGD)


OPTIMIZER_REGISTRY = core.Registry("optimizer")
"""
Global instance of the registry for optimizers.

By default, the registry contains the following optimizers:

.. list-table:: Optimizers
    :header-rows: 1

    * - Optimizer
      - Name
    * - ``torch.optim.Adadelta``
      - Adadelta
    * - ``torch.optim.Adagrad``
      - Adagrad
    * - ``torch.optim.Adam``
      - Adam
    * - ``torch.optim.AdamW``
      - AdamW
    * - ``torch.optim.SparseAdam``
      - SparseAdam
    * - ``torch.optim.Adamax``
      - Adamax
    * - ``torch.optim.ASGD``
      - ASGD
    * - ``torch.optim.LBFGS``
      - LBFGS
    * - ``torch.optim.NAdam``
      - NAdam
    * - ``torch.optim.RAdam``
      - RAdam
    * - ``torch.optim.RMSprop``
      - RMSprop
    * - ``torch.optim.Rprop``
      - Rprop
    * - ``torch.optim.SGD``
      - SGD

Example:
    .. code-block:: python

        import helios.optim as hlo

        # This automatically registers your optimizer.
        @hlo.OPTIMIZER_REGISTRY.register
        class MyOptimizer:
            ...

        # Alternatively you can manually register a optimizer like this:
        hlo.OPTIMIZER_REGISTRY.register(MyOptimizer)
"""
_register_default_optimizers(OPTIMIZER_REGISTRY)


def create_optimizer(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> optim.Optimizer:
    """
    Create the optimizer for the given type.

    Args:
        type_name: the type of the optimizer to create.
        args: positional arguments to pass into the optimizer.
        kwargs: keyword arguments to pass into the optimizer.
    Returns:
        The optimizer.
    """
    return OPTIMIZER_REGISTRY.get(type_name)(*args, **kwargs)
