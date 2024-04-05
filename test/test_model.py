import dataclasses
import typing

import torch
import torch.nn.functional as F
from torch import nn

import pyro.model as pym
import pyro.trainer as pyt
from pyro import core
from pyro.model import losses, metrics, networks, optimizers, schedulers


@networks.NETWORK_REGISTRY.register
class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class TestNewtorks:
    def test_registry(self) -> None:
        check_registry(networks.NETWORK_REGISTRY, ["ClassifierNet"])

    def test_create(self) -> None:
        check_create_function(networks.NETWORK_REGISTRY, networks.create_network)


@pym.MODEL_REGISTRY.register
class MockModel(pym.Model):
    def __init__(self) -> None:
        super().__init__("mock-model")

    @property
    def loss_items(self) -> dict[str, torch.Tensor]:
        return self._loss_items

    @property
    def val_scores(self) -> dict[str, float]:
        return self._val_scores

    @property
    def test_scores(self) -> dict[str, float]:
        return self._test_scores

    def setup(self, fast_init: bool = False) -> None:
        pass

    def train_step(self, batch: typing.Any, state) -> None:
        pass

    def populate_loss(self) -> None:
        self._loss_items["foo"] = torch.tensor(1)
        self._loss_items["bar"] = torch.tensor(2)

    def populate_val_scores(self) -> None:
        self._val_scores["foo"] = 1
        self._val_scores["bar"] = 2

    def populate_test_scores(self) -> None:
        self._test_scores["foo"] = 1
        self._test_scores["bar"] = 2


class TestModel:
    def test_defaults(self) -> None:
        model = MockModel()
        state = pyt.TrainingState()

        assert model.save_name == "mock-model"
        assert model.state_dict() == {}

        chkpt_name = "chkpt"
        assert model.append_metadata_to_chkpt_name(chkpt_name) == chkpt_name

        state_dict = {"a": 1, "b": 2}
        assert model.strip_training_data(state_dict) == state_dict

        model.populate_loss()
        assert len(model.loss_items) != 0
        model.on_training_batch_start(state)
        assert len(model.loss_items) == 0

        model.populate_val_scores()
        assert len(model.val_scores) != 0
        model.on_validation_start(0)
        assert len(model.loss_items) == 0

        assert model.have_metrics_improved()

        model.populate_test_scores()
        assert len(model.test_scores) != 0
        model.on_testing_start()
        assert len(model.test_scores) == 0

    def test_create(self) -> None:
        check_registry(pym.MODEL_REGISTRY, ["MockModel"])
        pym.create_model("MockModel")
