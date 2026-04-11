import math

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


class TestMultiStepRestartLR:
    def _make_optimizer(self, lr: float = 1.0) -> optim.Optimizer:
        param = nn.Parameter(torch.tensor(1.0))
        return optim.SGD([param], lr=lr)

    def test_milestone_decay(self) -> None:
        base_lr = 1.0
        gamma = 0.1
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.MultiStepRestartLR(optimizer, milestones=[2, 4], gamma=gamma)

        # epoch 0 (init): default restart at 0 with weight 1 → base_lr
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

        sched.step()  # epoch 1: no milestone
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

        sched.step()  # epoch 2: milestone hit once → lr * gamma
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma)

        sched.step()  # epoch 3: no milestone
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma)

        sched.step()  # epoch 4: milestone hit once → lr * gamma again
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma**2)

    def test_no_change_between_milestones(self) -> None:
        base_lr = 1.0
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.MultiStepRestartLR(optimizer, milestones=[5], gamma=0.1)

        for _ in range(4):
            sched.step()

        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

    def test_restart_resets_lr(self) -> None:
        base_lr = 1.0
        gamma = 0.1
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.MultiStepRestartLR(
            optimizer,
            milestones=[2],
            gamma=gamma,
            restarts=[0, 3],
            restart_weights=[1, 0.5],
        )

        # epoch 0: restart with weight 1
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

        sched.step()  # epoch 1: no change
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

        sched.step()  # epoch 2: milestone → lr * gamma
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma)

        sched.step()  # epoch 3: restart with weight 0.5 → initial_lr * 0.5
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * 0.5)

    def test_repeated_milestone_applies_higher_gamma_power(self) -> None:
        base_lr = 1.0
        gamma = 0.1
        optimizer = self._make_optimizer(base_lr)
        # Counter({2: 2}) — milestone 2 appears twice, so gamma**2 is applied at epoch 2
        sched = scheduler.MultiStepRestartLR(optimizer, milestones=[2, 2], gamma=gamma)

        sched.step()  # epoch 1
        sched.step()  # epoch 2: gamma**2

        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * gamma**2)

    def test_mismatched_restarts_and_weights_raises(self) -> None:
        optimizer = self._make_optimizer()
        with pytest.raises(AssertionError):
            scheduler.MultiStepRestartLR(
                optimizer, milestones=[2], restarts=[0, 5], restart_weights=[1]
            )


class TestCosineAnnealingRestartLR:
    def _make_optimizer(self, lr: float = 1.0) -> optim.Optimizer:
        param = nn.Parameter(torch.tensor(1.0))
        return optim.SGD([param], lr=lr)

    def test_single_period_starts_at_base_lr(self) -> None:
        base_lr = 1.0
        optimizer = self._make_optimizer(base_lr)
        scheduler.CosineAnnealingRestartLR(
            optimizer, periods=[4], restart_weights=[1], eta_min=0
        )
        # epoch 0: cos(pi*0/4)=1 → LR = eta_min + weight*0.5*base_lr*2 = base_lr
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

    def test_single_period_ends_at_eta_min(self) -> None:
        base_lr = 1.0
        eta_min = 0.0
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.CosineAnnealingRestartLR(
            optimizer, periods=[4], restart_weights=[1], eta_min=eta_min
        )
        for _ in range(4):
            sched.step()
        # epoch 4: cos(pi)=-1 → LR = eta_min
        assert optimizer.param_groups[0]["lr"] == pytest.approx(eta_min)

    def test_eta_min_is_floor(self) -> None:
        base_lr = 1.0
        eta_min = 0.1
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.CosineAnnealingRestartLR(
            optimizer, periods=[4], restart_weights=[1], eta_min=eta_min
        )
        # epoch 0: start → LR = eta_min + (base_lr - eta_min) = base_lr
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)

        for _ in range(4):
            sched.step()
        # epoch 4: end → LR = eta_min
        assert optimizer.param_groups[0]["lr"] == pytest.approx(eta_min)

    def test_midpoint_lr(self) -> None:
        base_lr = 1.0
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.CosineAnnealingRestartLR(
            optimizer, periods=[4], restart_weights=[1], eta_min=0
        )
        for _ in range(2):
            sched.step()
        # epoch 2: cos(pi*2/4) = cos(pi/2) = 0 → LR = 0.5*base_lr*(1+0) = 0.5
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * base_lr)

    def test_restart_weight_scales_peak(self) -> None:
        base_lr = 1.0
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.CosineAnnealingRestartLR(
            optimizer,
            periods=[2, 2],
            restart_weights=[1, 0.5],
            eta_min=0,
        )
        sched.step()  # epoch 1: period 1, weight=1, relative=1
        # LR = 0 + 1*0.5*(1+cos(pi/2)) = 0.5
        lr_period1 = optimizer.param_groups[0]["lr"]
        assert lr_period1 == pytest.approx(0.5)

        sched.step()  # epoch 2: end of period 1
        sched.step()  # epoch 3: period 2, weight=0.5, relative=1
        # LR = 0 + 0.5*0.5*(1+cos(pi/2)) = 0.25 — exactly half the period-1 midpoint
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5 * lr_period1)

    def test_lr_formula_matches_manual_computation(self) -> None:
        base_lr = 2.0
        eta_min = 0.2
        optimizer = self._make_optimizer(base_lr)
        sched = scheduler.CosineAnnealingRestartLR(
            optimizer, periods=[6], restart_weights=[1], eta_min=eta_min
        )
        for _ in range(3):
            sched.step()
        # epoch 3: eta_min + weight*0.5*(base_lr-eta_min)*(1+cos(pi*3/6))
        expected = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * 3 / 6))
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)

    def test_mismatched_periods_and_weights_raises(self) -> None:
        optimizer = self._make_optimizer()
        with pytest.raises(AssertionError):
            scheduler.CosineAnnealingRestartLR(
                optimizer, periods=[4, 4], restart_weights=[1]
            )
