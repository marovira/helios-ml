import pathlib

import torch

import helios
import helios.core as hlc
import helios.trainer as hlt
from helios.chkpt_migrator import migrate_checkpoints_to_current_version


class TestChkptMigrator:
    def test_from_pre_release(self, tmp_path: pathlib.Path) -> None:
        state = {
            "training_state": hlt.TrainingState().dict(),
            "model": {},
            "rng": {},
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert all(
            key in new_state for key in ("version", "training_state", "model", "rng")
        )
        assert new_state["version"] == helios.__version__

    def test_from_pre_1_1(self, tmp_path: pathlib.Path) -> None:
        state = {
            "training_state": hlt.TrainingState().dict(),
            "model": {},
            "rng": {},
            "version": helios.__version__,
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert isinstance(new_state["training_state"], hlt.TrainingState)

    def test_from_current_release(self, tmp_path: pathlib.Path) -> None:
        state = {
            "training_state": hlt.TrainingState(),
            "model": {},
            "rng": {},
            "version": helios.__version__,
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert state == new_state
