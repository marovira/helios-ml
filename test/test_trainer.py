import copy
import pathlib

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch.utils import data as tud

import helios.trainer as hlt
from helios import data
from helios import model as hlm
from helios.core import rng
from helios.data import functional as F

# Ignore the use of private members so we can test them correctly.
# ruff: noqa: SLF001

DATASET_SIZE = 10


class RandomDataset(tud.Dataset):
    def __getitem__(self, index):
        gen = rng.get_default_numpy_rng().generator
        return gen.integers(0, 100, 3)

    def __len__(self):
        return DATASET_SIZE


class RandomDatamodule(data.DataModule):
    def setup(self) -> None:
        params = data.DataLoaderParams(
            batch_size=1, num_workers=0, random_seed=rng.get_default_seed()
        )

        self._train_dataset = self._create_dataset(RandomDataset(), params)
        self._valid_dataset = self._create_dataset(RandomDataset(), params)
        self._test_dataset = self._create_dataset(RandomDataset(), params)


class CheckFunModel(hlm.Model):
    def __init__(self) -> None:
        super().__init__("test-model")
        self.called_train_funs: dict[str, bool] = {
            "setup": False,
            "train": False,
            "on_training_start": False,
            "on_training_batch_start": False,
            "train_step": False,
            "on_training_batch_end": False,
            "on_training_end": False,
            "eval": False,
            "on_validation_start": False,
            "on_validation_batch_start": False,
            "valid_step": False,
            "on_validation_batch_end": False,
            "on_validation_end": False,
            "have_metrics_improved": False,
        }

        self.called_test_funs: dict[str, bool] = {
            "load_for_testing": False,
            "eval": False,
            "on_testing_start": False,
            "on_testing_batch_start": False,
            "test_step": False,
            "on_testing_batch_end": False,
            "on_testing_end": False,
        }

    def setup(self, fast_init: bool = False) -> None:
        self.called_train_funs["setup"] = True

    def load_for_testing(self) -> None:
        self.called_test_funs["load_for_testing"] = True

    def train(self) -> None:
        self.called_train_funs["train"] = True

    def on_training_start(self) -> None:
        self.called_train_funs["on_training_start"] = True

    def on_training_batch_start(self, state) -> None:
        self.called_train_funs["on_training_batch_start"] = True

    def train_step(self, batch, state) -> None:
        self.called_train_funs["train_step"] = True

    def on_training_batch_end(self, state, should_log: bool = False) -> None:
        self.called_train_funs["on_training_batch_end"] = True

    def on_training_end(self) -> None:
        self.called_train_funs["on_training_end"] = True

    def eval(self) -> None:
        self.called_train_funs["eval"] = True
        self.called_test_funs["eval"] = True

    def on_validation_start(self, validation_cycle) -> None:
        self.called_train_funs["on_validation_start"] = True

    def on_validation_batch_start(self, step) -> None:
        self.called_train_funs["on_validation_batch_start"] = True

    def valid_step(self, batch, step) -> None:
        self.called_train_funs["valid_step"] = True

    def on_validation_batch_end(self, step) -> None:
        self.called_train_funs["on_validation_batch_end"] = True

    def on_validation_end(self, cycle) -> None:
        self.called_train_funs["on_validation_end"] = True

    def have_metrics_improved(self) -> bool:
        self.called_train_funs["have_metrics_improved"] = True
        return True

    def on_testing_start(self) -> None:
        self.called_test_funs["on_testing_start"] = True

    def on_testing_batch_start(self, step) -> None:
        self.called_test_funs["on_testing_batch_start"] = True

    def test_step(self, batch, step) -> None:
        self.called_test_funs["test_step"] = True

    def on_testing_batch_end(self, step) -> None:
        self.called_test_funs["on_testing_batch_end"] = True

    def on_testing_end(self) -> None:
        self.called_test_funs["on_testing_end"] = True


class RestartModel(hlm.Model):
    def __init__(self, val_count: int = -1) -> None:
        super().__init__("test-restart")
        self.batches: list[npt.NDArray] = []
        self.val_count = val_count

    def setup(self, fast_init: bool = False) -> None:
        pass

    def on_training_batch_start(self, state: hlt.TrainingState) -> None:
        if self.val_count == state.validation_cycles:
            raise RuntimeError("stop")

    def train_step(self, batch: torch.Tensor, state) -> None:
        as_np = F.tensor_to_numpy(batch)
        self.batches.append(as_np)


class TestTrainingUnit:
    def test_from_str(self) -> None:
        assert hlt.TrainingUnit.from_str("epoch") == hlt.TrainingUnit.EPOCH
        assert hlt.TrainingUnit.from_str("iteration") == hlt.TrainingUnit.ITERATION

        with pytest.raises(ValueError):
            hlt.TrainingUnit.from_str("foo")


class TestTrainer:
    def test_find_chkpt(self, tmp_path: pathlib.Path) -> None:
        test_cases = [
            (tmp_path / "test_epoch_1_iter_10.pth", True),
            (tmp_path / "test.txt", False),
            (tmp_path / "test.pth", False),
            (tmp_path / "test_epoch_1.pth", False),
            (tmp_path / "test_iter_1.pth", False),
        ]

        for case in test_cases:
            file, exp = case
            file.touch()
            ret = hlt.find_last_checkpoint(tmp_path)
            if exp:
                assert ret is not None
            else:
                assert ret is None
            file.unlink()

    def check_trainer_flags(self, **kwargs) -> None:
        with pytest.raises(ValueError):
            hlt.Trainer(**kwargs)

    def test_trainer_validate(self, tmp_path: pathlib.Path) -> None:
        hlt.Trainer()
        invalid_root = tmp_path / "test.txt"
        invalid_root.touch()

        self.check_trainer_flags(total_steps=3.0)
        self.check_trainer_flags(chkpt_frequency=0)
        self.check_trainer_flags(print_frequency=0)
        self.check_trainer_flags(valid_frequency=0)
        self.check_trainer_flags(enable_deterministic=True, enable_cudnn_benchmark=True)
        self.check_trainer_flags(total_steps=float("inf"), early_stop_cycles=0)
        self.check_trainer_flags(enable_tensorboard=True, run_path=None)
        self.check_trainer_flags(enable_tensorboard=True, run_path=invalid_root)
        self.check_trainer_flags(enable_file_logging=True, log_path=None)
        self.check_trainer_flags(enable_file_logging=True, log_path=invalid_root)

    def test_trainer_device_flags(self) -> None:
        t = hlt.Trainer(use_cpu=True)
        assert t._use_cpu
        assert t._device == torch.device("cpu")
        assert t._map_loc == {"cuda:0": "cpu"}
        assert t._gpu_ids == []
        assert not t._is_distributed

        if torch.cuda.is_available():
            t = hlt.Trainer()
            devices = list(range(torch.cuda.device_count()))
            assert not t._use_cpu
            assert t._gpu_ids == devices
            assert t._is_distributed if len(devices) > 1 else not t._is_distributed

            with pytest.raises(ValueError):
                devs = copy.deepcopy(devices)
                devs.append(10)
                hlt.Trainer(gpus=devs)

            with pytest.raises(ValueError):
                hlt.Trainer(gpus=[len(devices)])

    def check_training_loops(
        self,
        trainer: hlt.Trainer,
        num_chkpts: int = 0,
        chkpt_root: pathlib.Path | None = None,
        fit: bool = True,
    ) -> None:
        datamodule = RandomDatamodule()
        model = CheckFunModel()

        called_funs: dict[str, bool]
        if fit:
            trainer.fit(model, datamodule)
            called_funs = model.called_train_funs
        else:
            trainer.test(model, datamodule)
            called_funs = model.called_test_funs

        for _, seen in called_funs.items():
            assert seen

        if fit:
            assert chkpt_root is not None
            assert num_chkpts == len(list(chkpt_root.glob("*.pth")))

    def test_fit_iter(self, tmp_path: pathlib.Path) -> None:
        self.check_training_loops(
            hlt.Trainer(
                train_unit=hlt.TrainingUnit.ITERATION,
                total_steps=10,
                valid_frequency=5,
                chkpt_frequency=5,
                use_cpu=True,
                enable_progress_bar=False,
                chkpt_root=tmp_path / "chkpt",
            ),
            2,
            tmp_path / "chkpt/test-model",
        )

    def test_fit_epoch(self, tmp_path: pathlib.Path) -> None:
        self.check_training_loops(
            hlt.Trainer(
                train_unit=hlt.TrainingUnit.EPOCH,
                total_steps=2,
                valid_frequency=1,
                chkpt_frequency=1,
                use_cpu=True,
                chkpt_root=tmp_path / "chkpt",
            ),
            2,
            tmp_path / "chkpt/test-model",
        )

    def test_testing(self) -> None:
        self.check_training_loops(hlt.Trainer(use_cpu=True), fit=False)

    def get_restart_trainer(
        self, unit: hlt.TrainingUnit, chkpt_root: pathlib.Path
    ) -> hlt.Trainer:
        if unit == hlt.TrainingUnit.ITERATION:
            total_steps = DATASET_SIZE
            valid_frequency = DATASET_SIZE // 2
        else:
            total_steps = 2
            valid_frequency = 1

        return hlt.Trainer(
            train_unit=unit,
            total_steps=total_steps,
            valid_frequency=valid_frequency,
            chkpt_frequency=valid_frequency,
            use_cpu=True,
            chkpt_root=chkpt_root,
        )

    def get_restart_model_and_datamodule(
        self, val_count: int = -1
    ) -> tuple[RandomDatamodule, RestartModel]:
        return RandomDatamodule(), RestartModel(val_count)

    def clear_chkpts(self, chkpt_root: pathlib.Path) -> None:
        for chkpt in chkpt_root.glob("*.pth"):
            chkpt.unlink()

    def check_batches(
        self, exp_batches: list[npt.NDArray], ret_batches: list[npt.NDArray]
    ) -> None:
        assert len(exp_batches) == len(ret_batches)
        for exp, ret in zip(exp_batches, ret_batches, strict=True):
            assert np.all(exp == ret)

    def check_restart_trainer(
        self, unit: hlt.TrainingUnit, tmp_path: pathlib.Path
    ) -> None:
        datamodule, model = self.get_restart_model_and_datamodule()
        trainer = self.get_restart_trainer(unit, tmp_path)
        chkpt_root = tmp_path / model.save_name

        trainer.fit(model, datamodule)
        batches = model.batches

        self.clear_chkpts(chkpt_root)
        del datamodule, model, trainer

        datamodule, model = self.get_restart_model_and_datamodule(1)
        trainer = self.get_restart_trainer(unit, tmp_path)
        with pytest.raises(RuntimeError):
            trainer.fit(model, datamodule)

        # Clear out everything again and restart.
        ret_batches = model.batches
        del datamodule, trainer, model

        datamodule, model = self.get_restart_model_and_datamodule()
        trainer = self.get_restart_trainer(unit, tmp_path)
        model.batches = ret_batches
        trainer.fit(model, datamodule)

        self.check_batches(batches, model.batches)

    def test_restart_iter(self, tmp_path: pathlib.Path) -> None:
        self.check_restart_trainer(hlt.TrainingUnit.ITERATION, tmp_path)

    def test_restart_epoch(self, tmp_path: pathlib.Path) -> None:
        self.check_restart_trainer(hlt.TrainingUnit.EPOCH, tmp_path)
