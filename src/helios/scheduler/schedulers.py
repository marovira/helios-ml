import collections as col
import math

from torch import optim
from torch.optim import lr_scheduler

from .utils import SCHEDULER_REGISTRY


def _get_position_from_periods(iteration: int, cummulative_period: list[int]) -> int:
    """
    Get position from a period list.

    Specifically, it returns the index of the right-closest number in the period list.
    For example, suppose ``cummulative_period`` is ``[100, 200, 300, 400]``. Then:
    * If ``iteration == 50``, return 0
    * If ``iteration == 210``, return 2
    * If ``iteration == 300``, return 2.

    Args:
        iteration: current iteration.
        cummulative_period: cummulative period list.

    Returns:
        The position of the right-closest number in the period list
    """
    for i, period in enumerate(cummulative_period):
        if iteration <= period:
            return i

    return 0


@SCHEDULER_REGISTRY.register
class CosineAnnealingRestartLR(lr_scheduler.LRScheduler):
    """
    A cosine annealing with restarts LR scheduler.

    Example:
        Given

        .. code-block:: text

            periods = [10, 10, 10, 10]
            restart_weights = [1, 0.5, 0.5, 0.5]
            eta_min = 1e-7

        Then the scheduler will have 4 cycles of 10 iterations each. At the 10th, 20th,
        and 30th, the scheduler will restart with the weights in ``restart_weights``.

    Args:
        optimizer: the optimizer.
        periods: period for each cosine annealing cycle.
        restart_weights: (optional) restarts weights at each restart iteration.
        eta_min: The minimum lr. Defaults to 0
        last_epoch: Used in _LRScheduler. Defaults to -1.
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
        optimizer: torch optimizer.
        milestones: iterations that will decrease learning rate.
        gamma: decrease ratio. Defaults to 0.1.
        restarts: (optional) restart iterations.
        restart_weights: (optional) restart weights at each restart iteration.
        last_epoch: used in _LRScheduler. Defaults to -1.
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
