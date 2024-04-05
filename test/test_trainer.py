import copy
import pathlib

import pytest
import torch
from torch.utils import data as tud

import pyro.trainer as pyt
from pyro import data
from pyro import model as pym
from pyro.core import rng

# Ignore the use of private members so we can test them correctly.
# ruff: noqa: SLF001

DATASET_SIZE = 10


class RandomDataset(tud.Dataset):
    def __getitem__(self, index):
        gen = rng.get_default_numpy_rng().generator
        return gen.integers(0, 100, 3)

    def __len__(self):
        return DATASET_SIZE


class RandomDatamodule(data.PyroDataModule):
    def setup(self) -> None:
        params = data.DataLoaderParams(
            batch_size=1, num_workers=0, random_seed=rng.get_default_seed()
        )

        self._train_dataset = self._create_dataset(RandomDataset(), params)
        self._valid_dataset = self._create_dataset(RandomDataset(), params)
        self._test_dataset = self._create_dataset(RandomDataset(), params)


class Model(pym.Model):
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
            "eval": False,
            "on_testing_start": False,
            "on_testing_batch_start": False,
            "test_step": False,
            "on_testing_batch_end": False,
            "on_testing_end": False,
        }

    def setup(self, fast_init: bool = False) -> None:
        self.called_train_funs["setup"] = True

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


class TestTrainingUnit:
    def test_from_str(self) -> None:
        assert pyt.TrainingUnit.from_str("epoch") == pyt.TrainingUnit.EPOCH
        assert pyt.TrainingUnit.from_str("iteration") == pyt.TrainingUnit.ITERATION

        with pytest.raises(ValueError):
            pyt.TrainingUnit.from_str("foo")


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
            ret = pyt.find_last_checkpoint(tmp_path)
            if exp:
                assert ret is not None
            else:
                assert ret is None
            file.unlink()

    def check_trainer_flags(self, **kwargs) -> None:
        with pytest.raises(ValueError):
            pyt.Trainer(**kwargs)

    def test_trainer_validate(self, tmp_path: pathlib.Path) -> None:
        pyt.Trainer()
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
        t = pyt.Trainer(use_cpu=True)
        assert t._use_cpu
        assert t._device == torch.device("cpu")
        assert t._map_loc == {"cuda:0": "cpu"}
        assert t._gpu_ids == []
        assert not t._is_distributed

        if torch.cuda.is_available():
            t = pyt.Trainer()
            devices = list(range(torch.cuda.device_count()))
            assert not t._use_cpu
            assert t._gpu_ids == devices
            assert t._is_distributed if len(devices) > 1 else not t._is_distributed

            with pytest.raises(ValueError):
                devs = copy.deepcopy(devices)
                devs.append(10)
                pyt.Trainer(gpus=devs)

            with pytest.raises(ValueError):
                pyt.Trainer(gpus=[len(devices)])

    def check_training_loops(
        self,
        trainer: pyt.Trainer,
        num_chkpts: int = 0,
        chkpt_root: pathlib.Path | None = None,
        fit: bool = True,
    ) -> None:
        datamodule = RandomDatamodule()
        model = Model()

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
            pyt.Trainer(
                train_unit=pyt.TrainingUnit.ITERATION,
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
            pyt.Trainer(
                train_unit=pyt.TrainingUnit.EPOCH,
                total_steps=2,
                valid_frequency=1,
                chkpt_frequency=1,
                use_cpu=True,
                enable_progress_bar=False,
                chkpt_root=tmp_path / "chkpt",
            ),
            2,
            tmp_path / "chkpt/test-model",
        )

    def test_testing(self) -> None:
        self.check_training_loops(pyt.Trainer(), fit=False)
