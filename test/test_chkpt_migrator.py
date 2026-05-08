import pathlib

import torch

import helios
import helios.core as hlc
import helios.trainer as hlt
from helios.chkpt_migrator import migrate_checkpoints_to_current_version
from helios.model.model import _InternalStateKeys
from helios.trainer import _CheckpointKeys


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
            key in new_state
            for key in (
                _CheckpointKeys.VERSION,
                _CheckpointKeys.TRAINING_STATE,
                _CheckpointKeys.MODEL,
                _CheckpointKeys.RNG,
                _CheckpointKeys.LOGGERS,
            )
        )
        assert new_state[_CheckpointKeys.VERSION] == helios.__version__
        assert new_state[_CheckpointKeys.LOGGERS] == {}

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
        assert isinstance(new_state[_CheckpointKeys.TRAINING_STATE], hlt.TrainingState)
        assert new_state[_CheckpointKeys.LOGGERS] == {}

    def test_from_current_release(self, tmp_path: pathlib.Path) -> None:
        state = {
            _CheckpointKeys.TRAINING_STATE: hlt.TrainingState(),
            _CheckpointKeys.MODEL: {_InternalStateKeys.USER: {}},
            _CheckpointKeys.RNG: {},
            _CheckpointKeys.VERSION: helios.__version__,
            _CheckpointKeys.LOGGERS: {},
            _CheckpointKeys.DATAMODULE: {},
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert state == new_state

    def test_model_state_dict_migration(self, tmp_path: pathlib.Path) -> None:
        state = {
            "training_state": hlt.TrainingState(),
            "model": {"net": {"weight": 1.0}, "optimizer": {"lr": 0.01}},
            "rng": {},
            "version": helios.__version__,
            "loggers": {},
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert new_state[_CheckpointKeys.MODEL] == {
            _InternalStateKeys.USER: {"net": {"weight": 1.0}, "optimizer": {"lr": 0.01}}
        }

    def test_model_state_dict_migration_with_amp_scaler(
        self, tmp_path: pathlib.Path
    ) -> None:
        state = {
            "training_state": hlt.TrainingState(),
            "model": {"net": {}, "_helios_amp_scaler": {"scale": 65536.0}},
            "rng": {},
            "version": helios.__version__,
            "loggers": {},
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert new_state[_CheckpointKeys.MODEL] == {
            _InternalStateKeys.USER: {"net": {}, "_helios_amp_scaler": {"scale": 65536.0}}
        }

    def test_log_path_migration(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / "run.log"
        state = {
            "training_state": hlt.TrainingState(),
            "model": {},
            "rng": {},
            "version": helios.__version__,
            "log_path": log_file,
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert "log_path" not in new_state
        assert new_state[_CheckpointKeys.LOGGERS] == {"root": {"log_file": log_file}}

    def test_run_path_migration(self, tmp_path: pathlib.Path) -> None:
        run_path = tmp_path / "tensorboard" / "run_0"
        state = {
            "training_state": hlt.TrainingState(),
            "model": {},
            "rng": {},
            "version": helios.__version__,
            "run_path": run_path,
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert "run_path" not in new_state
        assert new_state[_CheckpointKeys.LOGGERS] == {
            "tensorboard": {"run_path": run_path}
        }

    def test_log_and_run_path_migration(self, tmp_path: pathlib.Path) -> None:
        log_file = tmp_path / "run.log"
        run_path = tmp_path / "tensorboard" / "run_0"
        state = {
            "training_state": hlt.TrainingState(),
            "model": {},
            "rng": {},
            "version": helios.__version__,
            "log_path": log_file,
            "run_path": run_path,
        }

        torch.save(state, tmp_path / "state.pth")
        migrate_checkpoints_to_current_version(tmp_path)

        new_state = hlc.safe_torch_load(tmp_path / "state.pth")
        assert "log_path" not in new_state
        assert "run_path" not in new_state
        assert new_state[_CheckpointKeys.LOGGERS] == {
            "root": {"log_file": log_file},
            "tensorboard": {"run_path": run_path},
        }
