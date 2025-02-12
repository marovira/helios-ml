import copy
import pathlib
import pickle
import typing

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import nn
from torch.utils import data as tud

import helios.core as hlc
import helios.plugins as hlp
import helios.trainer as hlt
from helios import data
from helios import model as hlm
from helios.core import rng
from helios.data import functional as F

# Ignore the use of private members so we can test them correctly.
# ruff: noqa: SLF001

DATASET_SIZE = 10


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)

    def forward(self, x):
        return self.conv1(x)


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
            "on_training_epoch_start": False,
            "on_training_batch_start": False,
            "train_step": False,
            "on_training_batch_end": False,
            "on_training_epoch_end": False,
            "on_training_end": False,
            "eval": False,
            "on_validation_start": False,
            "on_validation_batch_start": False,
            "valid_step": False,
            "on_validation_batch_end": False,
            "on_validation_end": False,
            "have_metrics_improved": False,
            "should_training_stop": False,
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
        assert self.called_train_funs["setup"]
        self.called_train_funs["train"] = True

    def on_training_start(self) -> None:
        self.called_train_funs["on_training_start"] = True

    def on_training_epoch_start(self, current_epoch) -> None:
        assert self.called_train_funs["train"]
        assert self.called_train_funs["on_training_start"]
        self.called_train_funs["on_training_epoch_start"] = True

    def on_training_batch_start(self, state) -> None:
        assert self.called_train_funs["on_training_epoch_start"]
        self.called_train_funs["on_training_batch_start"] = True

    def train_step(self, batch, state) -> None:
        assert self.called_train_funs["on_training_batch_start"]
        self.called_train_funs["train_step"] = True

    def on_training_batch_end(self, state, should_log: bool = False) -> None:
        assert self.called_train_funs["train_step"]
        self.called_train_funs["on_training_batch_end"] = True

    def on_training_epoch_end(self, current_epoch) -> None:
        assert self.called_train_funs["on_training_epoch_start"]
        self.called_train_funs["on_training_epoch_end"] = True

    def on_training_end(self) -> None:
        assert self.called_train_funs["on_training_start"]
        self.called_train_funs["on_training_end"] = True

    def eval(self) -> None:
        self.called_train_funs["eval"] = True
        self.called_test_funs["eval"] = True

    def on_validation_start(self, validation_cycle) -> None:
        assert self.called_train_funs["eval"]
        self.called_train_funs["on_validation_start"] = True

    def on_validation_batch_start(self, step) -> None:
        assert self.called_train_funs["eval"]
        assert self.called_train_funs["on_validation_start"]
        self.called_train_funs["on_validation_batch_start"] = True

    def valid_step(self, batch, step) -> None:
        assert self.called_train_funs["on_validation_batch_start"]
        self.called_train_funs["valid_step"] = True

    def on_validation_batch_end(self, step) -> None:
        assert self.called_train_funs["valid_step"]
        self.called_train_funs["on_validation_batch_end"] = True

    def on_validation_end(self, cycle) -> None:
        assert self.called_train_funs["on_validation_start"]
        self.called_train_funs["on_validation_end"] = True

    def have_metrics_improved(self) -> bool:
        assert self.called_train_funs["on_validation_end"]
        self.called_train_funs["have_metrics_improved"] = True
        return True

    def should_training_stop(self) -> bool:
        self.called_train_funs["should_training_stop"] = True
        return False

    def on_testing_start(self) -> None:
        self.called_test_funs["on_testing_start"] = True

    def on_testing_batch_start(self, step) -> None:
        assert self.called_test_funs["eval"]
        assert self.called_test_funs["on_testing_start"]
        self.called_test_funs["on_testing_batch_start"] = True

    def test_step(self, batch, step) -> None:
        assert self.called_test_funs["on_testing_batch_start"]
        self.called_test_funs["test_step"] = True

    def on_testing_batch_end(self, step) -> None:
        assert self.called_test_funs["test_step"]
        self.called_test_funs["on_testing_batch_end"] = True

    def on_testing_end(self) -> None:
        assert self.called_test_funs["on_testing_batch_end"]
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


class CheckpointModel(hlm.Model):
    def __init__(self) -> None:
        super().__init__("test-checkpoint")

        self._state = {
            "a": 1,
            "b": 2.0,
            "c": "x",
        }

    def setup(self, fast_init: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, typing.Any]:
        return self._state

    def load_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool = False
    ) -> None:
        assert self._state == state_dict


class AccumulationModel(hlm.Model):
    def __init__(self, accumulation_steps: int = 1) -> None:
        super().__init__("test-accumulation")
        self.accumulation_steps = accumulation_steps
        self.global_steps: int = 0
        self.accumulated_steps: int = 0

    def setup(self, fast_init: bool = False) -> None:
        pass

    def train_step(self, batch: torch.Tensor, state: hlt.TrainingState) -> None:
        self.global_steps += 1
        if self.global_steps % self.accumulation_steps == 0:
            self.accumulated_steps += 1

        assert state.global_iteration == self.global_steps
        assert state.current_iteration == self.accumulated_steps


class ExceptionModel(hlm.Model):
    def __init__(self, exc_type: type[Exception]) -> None:
        super().__init__("test-exception")
        self._exc_type = exc_type

    def setup(self, fast_init: bool = False) -> None:
        pass

    def on_training_start(self) -> None:
        raise self._exc_type("error")

    def on_testing_start(self) -> None:
        raise self._exc_type("error")


class EmptyPlugin(hlp.Plugin):
    def __init__(self):
        super().__init__("empty")

    def setup(self):
        pass


class CheckpointPlugin(hlp.Plugin):
    def __init__(self):
        super().__init__("chkpt")
        self._state = {
            "a": 1,
            "b": 2.0,
            "c": "x",
        }

    def setup(self):
        pass

    def configure_trainer(self, trainer: hlt.Trainer) -> None:
        self._register_in_trainer(trainer)

    def state_dict(self) -> dict[str, typing.Any]:
        return self._state

    def load_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool = False
    ) -> None:
        assert self._state == state_dict


class OverrideFlagsPlugin(hlp.Plugin):
    def __init__(
        self,
        training_batch: bool = False,
        validation_batch: bool = False,
        testing_batch: bool = False,
        should_training_stop: bool = False,
    ):
        super().__init__("override")
        self._overrides.training_batch = training_batch
        self._overrides.validation_batch = validation_batch
        self._overrides.testing_batch = testing_batch
        self._overrides.should_training_stop = should_training_stop

    def setup(self):
        pass


class CheckFunPlugin(hlp.Plugin):
    name: str = "check"

    def __init__(self):
        super().__init__(self.name)
        self._overrides.training_batch = True
        self._overrides.validation_batch = True
        self._overrides.testing_batch = True
        self._overrides.should_training_stop = True

        self.called_train_funs: dict[str, bool] = {
            "setup": False,
            "on_training_start": False,
            "process_training_batch": False,
            "on_training_end": False,
            "on_validation_start": False,
            "process_validation_batch": False,
            "on_validation_end": False,
            "should_training_stop": False,
        }

        self.called_test_funs: dict[str, bool] = {
            "on_testing_start": False,
            "process_testing_batch": False,
            "on_testing_end": False,
        }

    def setup(self) -> None:
        self.called_train_funs["setup"] = True

    def on_training_start(self) -> None:
        self.called_train_funs["on_training_start"] = True

    def process_training_batch(self, batch, state) -> typing.Any:
        self.called_train_funs["process_training_batch"] = True
        return batch

    def on_training_end(self) -> None:
        self.called_train_funs["on_training_end"] = True

    def on_validation_start(self, validation_cycle) -> None:
        self.called_train_funs["on_validation_start"] = True

    def process_validation_batch(self, batch, step) -> typing.Any:
        self.called_train_funs["process_validation_batch"] = True
        return batch

    def on_validation_end(self, validation_cycle) -> None:
        self.called_train_funs["on_validation_end"] = True

    def should_training_stop(self) -> bool:
        self.called_train_funs["should_training_stop"] = True
        return False

    def on_testing_start(self) -> None:
        self.called_test_funs["on_testing_start"] = True

    def process_testing_batch(self, batch, step) -> typing.Any:
        self.called_test_funs["process_testing_batch"] = True
        return batch

    def on_testing_end(self) -> None:
        self.called_test_funs["on_testing_end"] = True


class CheckPluginModel(hlm.Model):
    def __init__(self):
        super().__init__("test-plugin")

    def _get_plugin(self) -> CheckFunPlugin:
        plugin = self.trainer.plugins[CheckFunPlugin.name]
        assert isinstance(plugin, CheckFunPlugin)
        return plugin

    def setup(self, fast_init: bool = False) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["setup"]

    def on_training_start(self) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["on_training_start"]

    def train_step(self, batch, state) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["process_training_batch"]

    def on_training_end(self) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["on_training_end"]

    def on_validation_start(self, cycle) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["on_validation_start"]

    def valid_step(self, batch, step) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["process_validation_batch"]

    def on_validation_end(self, cycle) -> None:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["on_validation_end"]

    def should_training_stop(self) -> bool:
        plugin = self._get_plugin()
        assert plugin.called_train_funs["should_training_stop"]
        return False

    def on_testing_start(self) -> None:
        plugin = self._get_plugin()
        assert plugin.called_test_funs["on_testing_start"]

    def test_step(self, batch, step) -> None:
        plugin = self._get_plugin()
        assert plugin.called_test_funs["process_testing_batch"]

    def on_testing_end(self) -> None:
        plugin = self._get_plugin()
        assert plugin.called_test_funs["on_testing_end"]


class TestTrainingUnit:
    def test_from_str(self) -> None:
        assert hlt.TrainingUnit.from_str("epoch") == hlt.TrainingUnit.EPOCH
        assert hlt.TrainingUnit.from_str("iteration") == hlt.TrainingUnit.ITERATION

        with pytest.raises(ValueError):
            hlt.TrainingUnit.from_str("foo")


class TestTrainer:
    def test_register_types(self, tmp_path: pathlib.Path) -> None:
        test_dict = {"state": hlt.TrainingState(), "path": tmp_path}
        out_path = tmp_path / "register.pth"
        torch.save(test_dict, out_path)

        torch.serialization.clear_safe_globals()
        with pytest.raises(pickle.UnpicklingError):
            hlc.safe_torch_load(out_path)

        hlt.register_trainer_types_for_safe_load()
        reg_types = torch.serialization.get_safe_globals()
        exp_types = hlt.get_trainer_safe_types_for_load()
        assert set(reg_types) == set(exp_types)

        ret_dict = hlc.safe_torch_load(out_path)
        assert test_dict == ret_dict

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
        self.check_trainer_flags(use_cpu=True, gpus=[1])

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

    def check_map_location(
        self,
        out_path: pathlib.Path,
        start_device: torch.device,
        train_args: dict[str, typing.Any],
    ) -> None:
        net = SimpleNet().to(start_device)
        torch.save(net.state_dict(), out_path)

        t = hlt.Trainer(**train_args)
        t._configure_env()
        net = SimpleNet().to(t._device)
        data = hlc.safe_torch_load(out_path, map_location=t._map_loc)
        net.load_state_dict(data)

    def test_map_location(self, tmp_path: pathlib.Path) -> None:
        self.check_map_location(
            tmp_path / "cpu_to_cpu.pth", torch.device("cpu"), {"use_cpu": True}
        )

        if torch.cuda.is_available():
            self.check_map_location(
                tmp_path / "cpu_to_cuda.pth", torch.device("cpu"), {"gpus": [0]}
            )
            self.check_map_location(
                tmp_path / "cuda_to_cpu.pth", torch.device("cuda:0"), {"use_cpu": True}
            )

            available_ids = list(range(torch.cuda.device_count()))
            if len(available_ids) > 1:
                self.check_map_location(
                    tmp_path / "cpu_to_cuda1.pth", torch.device("cpu"), {"gpus": [1]}
                )
                self.check_map_location(
                    tmp_path / "cuda1_to_cpu.pth",
                    torch.device("cuda:1"),
                    {"use_cpu": True},
                )
                self.check_map_location(
                    tmp_path / "cuda0_to_cuda1.pth", torch.device("cuda:0"), {"gpus": [1]}
                )
                self.check_map_location(
                    tmp_path / "cuda1_to_cuda0.pth", torch.device("cuda:1"), {"gpus": [0]}
                )

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
            assert trainer.fit(model, datamodule)
            called_funs = model.called_train_funs
        else:
            assert trainer.test(model, datamodule)
            called_funs = model.called_test_funs

        for _, seen in called_funs.items():
            assert seen

        if fit:
            assert chkpt_root is not None
            assert num_chkpts == len(list(chkpt_root.glob("*.pth")))

            for chkpt_path in chkpt_root.glob("*.pth"):
                state_dict = hlc.safe_torch_load(chkpt_path)
                assert trainer._validate_state_dict(state_dict)

                # Add an extra key
                state_dict["tmp"] = "foo"
                assert trainer._validate_state_dict(state_dict)
                state_dict.pop("tmp")

                # Remove one of the valid keys
                state_dict.pop("rng")
                assert not trainer._validate_state_dict(state_dict)

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

        assert trainer.fit(model, datamodule)
        batches = model.batches

        self.clear_chkpts(chkpt_root)
        del datamodule, model, trainer

        datamodule, model = self.get_restart_model_and_datamodule(1)
        trainer = self.get_restart_trainer(unit, tmp_path)
        assert not trainer.fit(model, datamodule)

        # Clear out everything again and restart.
        ret_batches = model.batches
        del datamodule, trainer, model

        datamodule, model = self.get_restart_model_and_datamodule()
        trainer = self.get_restart_trainer(unit, tmp_path)
        model.batches = ret_batches
        assert trainer.fit(model, datamodule)

        self.check_batches(batches, model.batches)

    def test_restart_iter(self, tmp_path: pathlib.Path) -> None:
        self.check_restart_trainer(hlt.TrainingUnit.ITERATION, tmp_path)

    def test_restart_epoch(self, tmp_path: pathlib.Path) -> None:
        self.check_restart_trainer(hlt.TrainingUnit.EPOCH, tmp_path)

    def check_accumulation(self, num_steps: int) -> None:
        datamodule = RandomDatamodule()
        model = AccumulationModel(num_steps)
        trainer = hlt.Trainer(
            train_unit=hlt.TrainingUnit.ITERATION,
            total_steps=20,
            use_cpu=True,
            accumulation_steps=num_steps,
        )

        assert trainer.fit(model, datamodule)

    def test_accumulation(self) -> None:
        self.check_accumulation(1)
        self.check_accumulation(2)
        self.check_accumulation(4)
        self.check_accumulation(5)
        self.check_accumulation(10)

    def check_exception(
        self,
        exc_type: type[Exception],
        trainer: hlt.Trainer,
        fit: bool,
        raised_as_runtime: bool = False,
    ) -> None:
        datamodule = RandomDatamodule()
        model = ExceptionModel(exc_type)

        if not raised_as_runtime:
            with pytest.raises(exc_type):
                if fit:
                    trainer.fit(model, datamodule)
                else:
                    trainer.test(model, datamodule)
        else:
            if fit:
                assert not trainer.fit(model, datamodule)
            else:
                assert not trainer.test(model, datamodule)

    def test_trainer_exceptions(self) -> None:
        exception_types = [ValueError, RuntimeError, KeyError]
        trainer = hlt.Trainer()
        trainer.train_exceptions = exception_types[:2]
        for exc_type in trainer.train_exceptions:
            self.check_exception(exc_type, trainer, fit=True)
        self.check_exception(
            exception_types[-1], trainer, fit=True, raised_as_runtime=True
        )

        trainer.train_exceptions = []
        trainer.test_exceptions = exception_types[:2]
        for exc_type in trainer.test_exceptions:
            self.check_exception(exc_type, trainer, fit=False)
        self.check_exception(
            exception_types[-1], trainer, fit=False, raised_as_runtime=True
        )

    def test_append_plugins(self) -> None:
        trainer = hlt.Trainer()

        trainer.plugins["empty"] = EmptyPlugin()
        trainer._validate_plugins()

        batch_flags = {
            "training_batch": False,
            "validation_batch": False,
            "testing_batch": False,
            "should_training_stop": False,
        }

        for flag_name in batch_flags:
            batch_flags[flag_name] = True
            trainer.plugins[f"override_base_{flag_name}"] = OverrideFlagsPlugin(
                **batch_flags
            )

            with pytest.raises(ValueError):
                trainer.plugins[f"override_dup_{flag_name}"] = OverrideFlagsPlugin(
                    **batch_flags
                )
                trainer._validate_plugins()

            batch_flags[flag_name] = False

    def check_plugin_functions(self, trainer: hlt.Trainer, fit: bool = True) -> None:
        datamodule = RandomDatamodule()
        model = CheckPluginModel()

        name = CheckFunPlugin.name
        trainer.plugins[name] = CheckFunPlugin()

        called_funs: dict[str, bool]
        if fit:
            assert trainer.fit(model, datamodule)
            called_funs = trainer.plugins[name].called_train_funs  # type: ignore[attr-defined]
        else:
            assert trainer.test(model, datamodule)
            called_funs = trainer.plugins[name].called_test_funs  # type: ignore[attr-defined]

        for _, seen in called_funs.items():
            assert seen

    def test_plugin_functions(self) -> None:
        self.check_plugin_functions(
            hlt.Trainer(
                train_unit=hlt.TrainingUnit.ITERATION,
                total_steps=10,
                valid_frequency=10,
                use_cpu=True,
            )
        )

        self.check_plugin_functions(
            hlt.Trainer(
                train_unit=hlt.TrainingUnit.EPOCH,
                total_steps=1,
                valid_frequency=1,
                use_cpu=True,
            )
        )

        self.check_plugin_functions(hlt.Trainer(use_cpu=True), fit=False)

    def test_checkpoints(self, tmp_path: pathlib.Path) -> None:
        datamodule = RandomDatamodule()
        model = CheckpointModel()
        plugin = CheckpointPlugin()

        trainer = hlt.Trainer(
            train_unit=hlt.TrainingUnit.ITERATION,
            total_steps=10,
            valid_frequency=10,
            chkpt_root=tmp_path,
            chkpt_frequency=10,
            use_cpu=True,
        )
        plugin.configure_trainer(trainer)
        assert trainer.fit(model, datamodule)

        chkpt_root = tmp_path / model._save_name
        chkpts = list(chkpt_root.glob("*.pth"))
        assert len(chkpts) == 1

        chkpt = hlc.safe_torch_load(chkpts[0])
        model.load_state_dict(chkpt["model"])
        plugin.load_state_dict(chkpt[plugin._plug_id])
