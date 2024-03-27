import dataclasses
import typing

import torch
from torch import nn

from pyro import core
from pyro.model import losses, metrics, optimizers, schedulers


@losses.LOSS_REGISTRY.register
class SampleLoss(losses.WeightedLoss):
    def __init__(self):
        super().__init__(loss_weight=0.5)

    def _eval(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SampleModuleNoArgs(nn.Module):
    def forward(self, x: int) -> int:
        return x


class SampleModuleArgs(nn.Module):
    def __init__(self, val: int):
        super().__init__()

        self._val = val

    def forward(self, x: int) -> int:
        return x * self._val


class SampleModuleKwargs(nn.Module):
    def __init__(self, val: int = 1):
        super().__init__()

        self._val = val

    def forward(self, x: int) -> int:
        return x * self._val


class SampleModuleArgsAndKwargs(nn.Module):
    def __init__(self, val: int, opt_val: int = 1):
        super().__init__()

        self._val = val
        self._opt_val = opt_val

    def forward(self, x: int) -> int:
        return x * self._opt_val + self._val


@dataclasses.dataclass
class SampleEntry:
    sample_type: type
    exp_ret: int
    args: list[typing.Any] = dataclasses.field(default_factory=list)
    kwargs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


def check_registry(registry: core.Registry, registered_names: list[str]) -> None:
    assert len(typing.cast(typing.Sized, registry.keys())) != 0

    for name in registered_names:
        assert name in registry


def check_create_function(registry: core.Registry, create_fun: typing.Callable) -> None:
    in_val = 4
    test_table = [
        SampleEntry(SampleModuleNoArgs, exp_ret=in_val),
        SampleEntry(SampleModuleArgs, exp_ret=in_val * 2, args=[2]),
        SampleEntry(SampleModuleKwargs, exp_ret=in_val * 2, kwargs={"val": 2}),
        SampleEntry(
            SampleModuleArgsAndKwargs,
            exp_ret=in_val * 2 + 2,
            args=[2],
            kwargs={"opt_val": 2},
        ),
    ]

    for entry in test_table:
        registry.register(entry.sample_type)

    for entry in test_table:
        ret = create_fun(entry.sample_type.__name__, *entry.args, **entry.kwargs)
        assert isinstance(ret, entry.sample_type)
        val = ret(in_val)  # type: ignore[operator]
        assert val == entry.exp_ret


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

    def test_create(self) -> None:
        check_create_function(metrics.METRICS_REGISTRY, metrics.create_metric)


class TestLosses:
    def test_registry(self) -> None:
        check_registry(losses.LOSS_REGISTRY, ["SampleLoss"])

    def test_create(self) -> None:
        check_create_function(losses.LOSS_REGISTRY, losses.create_loss)

    def test_weighted_loss(self) -> None:
        loss = losses.create_loss("SampleLoss")
        assert isinstance(loss, SampleLoss)

        x = torch.tensor(10)
        y = loss(x)
        assert y == (x * 0.5)


class TestOptimizers:
    def test_registry(self) -> None:
        check_registry(optimizers.OPTIMIZER_REGISTRY, ["Adam", "AdamW", "SGD"])

    def test_create(self) -> None:
        check_create_function(optimizers.OPTIMIZER_REGISTRY, optimizers.create_optimizer)


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

    def test_create(self) -> None:
        check_create_function(schedulers.SCHEDULER_REGISTRY, schedulers.create_scheduler)