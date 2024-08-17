from __future__ import annotations

import abc
import typing

import torch

from helios import core

if typing.TYPE_CHECKING:
    from ..trainer import Trainer, TrainingState

# TODO: Remove this once we get everything ironed out.
# ruff: noqa: D101, D102, D107
# This one needs to stay!
# ruff: noqa: B027


class Plugin(abc.ABC):
    def __init__(self):
        self._trainer: Trainer | None

        self._is_distributed: bool = False
        self._map_loc: str | dict[str, str] = ""
        self._device: torch.device | None = None
        self._rank: int = 0

    @property
    def is_distributed(self) -> bool:
        """Flag controlling whether distributed training is being used or not."""
        return self._is_distributed

    @is_distributed.setter
    def is_distributed(self, val: bool) -> None:
        self._is_distributed = val

    @property
    def map_loc(self) -> str | dict[str, str]:
        """The location to map loaded weights from a checkpoint or pre-trained file."""
        return self._map_loc

    @map_loc.setter
    def map_loc(self, loc: str | dict[str, str]) -> None:
        self._map_loc = loc

    @property
    def device(self) -> torch.device:
        """The device on which the plugin is running."""
        return core.get_from_optional(self._device)

    @device.setter
    def device(self, dev: torch.device) -> None:
        self._device = dev

    @property
    def rank(self) -> int:
        """The local rank (device id) that the plugin is running on."""
        return self._rank

    @rank.setter
    def rank(self, r: int) -> None:
        self._rank = r

    @property
    def trainer(self) -> Trainer:
        """Reference to the trainer."""
        return core.get_from_optional(self._trainer)

    @trainer.setter
    def trainer(self, t) -> None:
        self._trainer = t

    @abc.abstractmethod
    def setup(self) -> None:
        pass

    def on_training_start(self) -> None:
        pass

    def on_training_epoch_start(self, current_epoch: int) -> None:
        pass

    def on_training_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_training_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_training_epoch_end(self, current_epoch: int) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_validation_start(self) -> None:
        pass

    def on_validation_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_validation_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def should_training_stop(self) -> bool:
        return False

    def on_testing_start(self) -> None:
        pass

    def on_testing_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_testing_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_testing_end(self) -> None:
        pass

    def on_tuning_start(self) -> None:
        pass

    def on_tuning_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_tuning_batch_end(self, batch: typing.Any, state: TrainingState) -> None:
        pass

    def on_tuning_end(self) -> None:
        pass
