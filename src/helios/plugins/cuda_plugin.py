from __future__ import annotations

import typing

import torch

from helios import core

if typing.TYPE_CHECKING:
    from ..trainer import TrainingState

from .plugin import Plugin

# TODO: Remove this once we get everything ironed out.
# ruff: noqa: D101, D102, D107


class CUDAPlugin(Plugin):
    def __init__(self):
        super().__init__()
        core.cuda.requires_cuda_support()

    def _move_collection_to_device(self, batch: torch.Tensor | list | dict):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, list):
            for i in range(len(batch)):
                if isinstance(batch[i], torch.Tensor):
                    batch[i] = batch[i].to(self.device)
        elif isinstance(batch, dict):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
        else:
            raise RuntimeError(f"error: batch has unknown type {type(batch)}")

    def on_training_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        self._move_collection_to_device(batch)

    def on_validation_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        self._move_collection_to_device(batch)

    def on_tuning_batch_start(self, batch: typing.Any, state: TrainingState) -> None:
        self._move_collection_to_device(batch)
