import typing

from torch import nn

from helios import core


def _register_default_losses(registry: core.Registry) -> None:
    """
    Register the Torch default loss functions.

    List was obtained from https://pytorch.org/docs/stable/nn.html#loss-functions

    Args:
        registry: the loss registry.
    """
    registry.register(nn.L1Loss)
    registry.register(nn.MSELoss)
    registry.register(nn.CrossEntropyLoss)
    registry.register(nn.CTCLoss)
    registry.register(nn.NLLLoss)
    registry.register(nn.PoissonNLLLoss)
    registry.register(nn.GaussianNLLLoss)
    registry.register(nn.KLDivLoss)
    registry.register(nn.BCELoss)
    registry.register(nn.BCEWithLogitsLoss)
    registry.register(nn.MarginRankingLoss)
    registry.register(nn.HingeEmbeddingLoss)
    registry.register(nn.MultiLabelMarginLoss)
    registry.register(nn.HuberLoss)
    registry.register(nn.SmoothL1Loss)
    registry.register(nn.SoftMarginLoss)
    registry.register(nn.MultiLabelSoftMarginLoss)
    registry.register(nn.CosineEmbeddingLoss)
    registry.register(nn.MultiMarginLoss)
    registry.register(nn.TripletMarginLoss)
    registry.register(nn.TripletMarginWithDistanceLoss)


LOSS_REGISTRY = core.Registry("loss")
"""
Global instance of the registry for loss functions.

By default, the registry contains the following losses:

.. list-table:: Optimizers
    :header-rows: 1

    * - Loss
      - Name
    * - ``torch.nn.L1Loss``
      - L1Loss
    * - ``torch.nn.MSELoss``
      - MSELoss
    * - ``torch.nn.CrossEntropyLoss``
      - CrossEntropyLoss
    * - ``torch.nn.CTCLoss``
      - CTCLoss
    * - ``torch.nn.NLLLoss``
      - NLLLoss
    * - ``torch.nn.PoissonNLLLoss``
      - PoissonNLLLoss
    * - ``torch.nn.GaussianNLLLoss``
      - GaussianNLLLoss
    * - ``torch.nn.KLDivLoss``
      - KLDivLoss
    * - ``torch.nn.BCELoss``
      - BCELoss
    * - ``torch.nn.BCEWithLogitsLoss``
      - BCEWithLogitsLoss
    * - ``torch.nn.MarginRankingLoss``
      - MarginRankingLoss
    * - ``torch.nn.HingeEmbeddingLoss``
      - HingeEmbeddingLoss
    * - ``torch.nn.MultiLabelMarginLoss``
      - MultiLabelMarginLoss
    * - ``torch.nn.HuberLoss``
      - HuberLoss
    * - ``torch.nn.SmoothL1Loss``
      - SmoothL1Loss
    * - ``torch.nn.SoftMarginLoss``
      - SoftMarginLoss
    * - ``torch.nn.MultiLabelMarginLoss``
      - MultiLabelMarginLoss
    * - ``torch.nn.CosineEmbeddingLoss``
      - CosineEmbeddingLoss
    * - ``torch.nn.MultiMarginLoss``
      - MultiMarginLoss
    * - ``torch.nn.TripletMarginLoss``
      - TripletMarginLoss
    * - ``torch.nn.TripletMarginWithDistanceLoss``
      - TripletMarginWithDistanceLoss

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
_register_default_losses(LOSS_REGISTRY)


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
