import pytest
import torch
from torch import nn, optim

from helios import scheduler


class TestSchedulers:
    def test_registry(self, check_registry) -> None:
        check_registry(
            scheduler.SCHEDULER_REGISTRY,
            [
                # Pytorch schedulers.
                "LambdaLR",
                "MultiplicativeLR",
                "StepLR",
                "MultiStepLR",
                "ConstantLR",
                "LinearLR",
                "ExponentialLR",
                "PolynomialLR",
                "CosineAnnealingLR",
                "SequentialLR",
                "ReduceLROnPlateau",
                "CyclicLR",
                "OneCycleLR",
                "CosineAnnealingWarmRestarts",
                # Custom schedulers.
                "MultiStepRestartLR",
                "CosineAnnealingRestartLR",
                "LinearWarmupScheduler",
            ],
        )

    def test_create(self, check_create_function) -> None:
        check_create_function(scheduler.SCHEDULER_REGISTRY, scheduler.create_scheduler)


class TestLinearWarmupScheduler:
    def _make_optimizer(self, lr: float = 1.0) -> optim.Optimizer:
        param = nn.Parameter(torch.tensor(1.0))
        return optim.SGD([param], lr=lr)

    def test_warmup_lr(self) -> None:
        warmup_steps = 4
        base_lr = 1.0
        optimizer = self._make_optimizer(base_lr)
        inner = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        sched = scheduler.LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            scheduler=inner,
            warmup_start_factor=0.0,
        )

        # After init (last_epoch=0): LR = base_lr * 0.0 = 0.0
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)

        # Steps through warmup: factor = last_epoch / warmup_steps
        for step in range(1, warmup_steps + 1):
            sched.step()
            expected = base_lr * step / warmup_steps
            assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)

    def test_warmup_start_factor(self) -> None:
        warmup_steps = 4
        base_lr = 1.0
        start_factor = 0.5
        optimizer = self._make_optimizer(base_lr)
        inner = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        sched = scheduler.LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            scheduler=inner,
            warmup_start_factor=start_factor,
        )

        # After init (last_epoch=0): LR = base_lr * warmup_start_factor
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * start_factor)

        sched.step()  # last_epoch=1: factor = 0.5 + 0.5 * 1/4 = 0.625
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.625)

        for _ in range(warmup_steps - 1):
            sched.step()
        # After warmup_steps total explicit steps: LR = base_lr
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

    def test_post_warmup_delegates(self) -> None:
        warmup_steps = 2
        base_lr = 1.0
        gamma = 0.1
        optimizer = self._make_optimizer(base_lr)
        inner = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        sched = scheduler.LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            scheduler=inner,
        )

        # Advance through warmup
        for _ in range(warmup_steps):
            sched.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

        # First post-warmup step: StepLR(step_size=1) halves each step
        sched.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma)

        sched.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma**2)
