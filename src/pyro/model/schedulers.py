import collections as col
import math
import typing

from torch import nn, optim
from torch.optim import lr_scheduler

from pyro import core


def _register_default_schedulers(registry: core.Registry):
    """
    Register the default schedulers to the registry.

    Args:
        registry (Registry): the scheduler registry.
    """
    registry.register(lr_scheduler.MultiStepLR)
    registry.register(lr_scheduler.CosineAnnealingLR)


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


def _get_position_from_periods(iteration: int, cummulative_period: list[int]) -> int:
    """
    Get position from a period list.

    Specifically, it returns the index of the right-closest number in the period list.
    For example, suppose cummulative_period is [100, 200, 300, 400]. Then:
        * If iteration == 50, return 0
        * If iteration == 210, return 2
        * If iteration == 300, return 2.

    Args:
        iteration(int): current iteration.
        cummulative_period (List[int]): cummulative period list.

    Returns:
        int: The position of the right-closest number in the period list
    """
    for i, period in enumerate(cummulative_period):
        if iteration <= period:
            return i

    return 0


@SCHEDULER_REGISTRY.register
class CosineAnnealingRestartLR(lr_scheduler.LRScheduler):
    """
    A cosine annealing with restarts LR scheduler.

    For example, given
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min = 1e-7

    Then the scheduler will have 4 cycles of 10 iterations each. At the 10th, 20th, and
    30th, the scheduler will restart with the weights in restart_weights.

    Inputs:
        optimizer (optim.Optimizer): the optimizer.
        periods (list[int]): period for each cosine annealing cycle.
        restart_weights (list[int]): restarts weights at each restart iteration.
        eta_min (float): The minimum lr.
        last_epoch (int): Used in _LRScheduler.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        periods: list[int],
        restart_weights: list[int] | None = None,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        """Create the scheduler."""
        if restart_weights is None:
            restart_weights = [1]

        self._periods = periods
        self._restart_weights = restart_weights
        self._eta_min = eta_min
        assert len(self._periods) == len(
            self._restart_weights
        ), "periods and restart_weights should have the same length."
        self._cumulative_period = [
            sum(self._periods[0 : i + 1]) for i in range(len(self._periods))
        ]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return the current learning rate."""
        idx = _get_position_from_periods(self.last_epoch, self._cumulative_period)
        current_weight = self._restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self._cumulative_period[idx - 1]
        current_period = self._periods[idx]

        return [
            self._eta_min
            + current_weight
            * 0.5
            * (base_lr - self._eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]


@SCHEDULER_REGISTRY.register
class MultiStepRestartLR(lr_scheduler.LRScheduler):
    """
    Multi-step with restarts LR scheduler.

    Args:
        optimizer (optim.Optimizer): Torch optimizer.
        milestones (List[int]): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio.
        restarts (List[int]): Restart iterations.
        restart_weights (List[int]): Restart weights at each restart iteration.
        last_epoch (int): Used in _LRScheduler.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        milestones: list[int],
        gamma: float = 0.1,
        restarts: list[int] | None = None,
        restart_weights: list[int] | None = None,
        last_epoch: int = -1,
    ):
        """Create the scheduler."""
        if restarts is None:
            restarts = [0]
        if restart_weights is None:
            restart_weights = [1]

        self._milestones = col.Counter(milestones)
        self._gamma = gamma
        self._restarts = restarts
        self._restart_weights = restart_weights
        assert len(self._restarts) == len(
            self._restart_weights
        ), "restarts and their weights do not match."
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return the current learning rate."""
        if self.last_epoch in self._restarts:
            weight = self._restart_weights[self._restarts.index(self.last_epoch)]
            return [group["initial_lr"] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self._milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self._gamma ** self._milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]
