import typing

import torch

from pyro import core
from pyro.module import losses, metrics, optimizers, schedulers


@losses.LOSS_REGISTRY.register
class SampleLoss(losses.WeightedLoss):
    def __init__(self):
        super().__init__(loss_weight=0.5)

    def _eval(self, x: torch.Tensor) -> torch.Tensor:
        return x


def check_registry(registry: core.Registry, registered_names: list[str]) -> None:
    assert len(typing.cast(typing.Sized, registry.keys())) != 0

    for name in registered_names:
        assert name in registry


class TestMetrics:
    def test_registry(self) -> None:
        check_registry(
            metrics.METRICS_REGISTRY,
            [
                "CalculatePSNR",
                "CalculateSSIM",
                "CalculateMAP",
                "CalculateMAE",
                "CalculatePrecision",
                "CalculateRecall",
                "CalculateF1",
            ],
        )


class TestLosses:
    def test_registry(self) -> None:
        check_registry(losses.LOSS_REGISTRY, ["SampleLoss"])

    def test_weighted_loss(self) -> None:
        loss = losses.create_loss("SampleLoss")
        assert isinstance(loss, SampleLoss)

        x = torch.tensor(10)
        y = loss(x)
        assert y == (x * 0.5)


class TestOptimizers:
    def test_registry(self) -> None:
        check_registry(optimizers.OPTIMIZER_REGISTRY, ["Adam", "AdamW", "SGD"])


class TestSchedulers:
    def test_registry(self) -> None:
        check_registry(
            schedulers.SCHEDULER_REGISTRY,
            [
                "MultiStepLR",
                "CosineAnnealingLR",
                "CosineAnnealingRestartLR",
                "MultiStepRestartLR",
            ],
        )
