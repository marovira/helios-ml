import dataclasses as dc
import functools
import pathlib
import typing

import optuna
import pytest
import torch

import helios.model as hlm
import helios.plugins as hlp
import helios.plugins.optuna as hlpo
import helios.trainer as hlt
from helios.core import rng

# Ignore the use of private members so we can test them correctly.
# ruff: noqa: SLF001


class ExceptionPlugin(hlp.Plugin):
    def __init__(self, exc_type: type[Exception] | list[type[Exception]]):
        super().__init__(f"exception_{str(exc_type)}")
        self._exc_type = exc_type

    def setup(self) -> None:
        pass

    def configure_trainer(self, t: hlt.Trainer) -> None:
        self._append_train_exceptions(self._exc_type, t)
        self._append_test_exceptions(self._exc_type, t)


class PluginModel(hlm.Model):
    def __init__(self, save_name: str):
        super().__init__(save_name)

    def setup(self, fast_init: bool = False) -> None:
        pass


class TestPlugins:
    def test_registry(self, check_registry) -> None:
        check_registry(hlp.PLUGIN_REGISTRY, ["CUDAPlugin", "OptunaPlugin"])

    def test_create(self, check_create_function) -> None:
        check_create_function(hlp.PLUGIN_REGISTRY, hlp.create_plugin)

    def test_append_exceptions(self) -> None:
        t = hlt.Trainer()
        plugin = ExceptionPlugin(RuntimeError)
        exc_list: list[type[Exception]] = [RuntimeError]

        plugin.configure_trainer(t)
        assert t.train_exceptions == exc_list
        assert t.test_exceptions == exc_list

        exc_list.extend([ValueError, TypeError])
        plugin = ExceptionPlugin([ValueError, TypeError])
        plugin.configure_trainer(t)

        assert t.train_exceptions == exc_list
        assert t.test_exceptions == exc_list


class TestCUDAPlugin:
    def check_batch_device(self, x: typing.Any, device: torch.device) -> None:
        if isinstance(x, torch.Tensor):
            assert x.device == device
        elif isinstance(x, dict):
            assert all(val.device == device for _, val in x.items())
        else:
            assert all(elem.device == device for elem in x)

    def check_batch_processing(
        self, plugin: hlp.CUDAPlugin, x: typing.Any, device: torch.device
    ) -> None:
        ret = plugin.process_training_batch(x, hlt.TrainingState())
        self.check_batch_device(ret, device)

        ret = plugin.process_validation_batch(x, 0)
        self.check_batch_device(ret, device)

        ret = plugin.process_testing_batch(x, 0)
        self.check_batch_device(ret, device)

    def test_plugin_id(self) -> None:
        assert hasattr(hlp.CUDAPlugin, "plugin_id")
        assert hlp.CUDAPlugin.plugin_id == "cuda"

    def test_batch_processing(self) -> None:
        if not torch.cuda.is_available():
            return

        def create_tensor() -> torch.Tensor:
            return torch.randn((1, 3, 32, 32)).to("cpu")

        device = torch.device("cuda:0")

        plugin = hlp.CUDAPlugin()
        plugin.is_distributed = False
        plugin.map_loc = {"cuda:0": "cuda:0"}
        plugin.device = device

        assert plugin.unique_overrides.training_batch
        assert plugin.unique_overrides.validation_batch
        assert plugin.unique_overrides.testing_batch

        self.check_batch_processing(plugin, create_tensor(), device)
        self.check_batch_processing(plugin, [create_tensor(), create_tensor()], device)
        self.check_batch_processing(plugin, (create_tensor(), create_tensor()), device)
        self.check_batch_processing(
            plugin, {"a": create_tensor(), "b": create_tensor()}, device
        )

    def test_configure(self) -> None:
        if not torch.cuda.is_available():
            return

        trainer = hlt.Trainer()
        plugin = hlp.CUDAPlugin()
        plugin.configure_trainer(trainer)

        assert "cuda" in trainer.plugins
        assert trainer.plugins["cuda"] == plugin


# Ignore the warnings coming from optuna.
@pytest.mark.filterwarnings(
    ("ignore::optuna.exceptions.ExperimentalWarning"), ("ignore::FutureWarning")
)
class TestOptunaPlugin:
    def test_plugin_id(self) -> None:
        assert hasattr(hlpo.OptunaPlugin, "plugin_id")
        assert hlpo.OptunaPlugin.plugin_id == "optuna"

    def test_invalid_storage(self) -> None:
        def objective(trial: optuna.Trial) -> int:
            plugin = hlpo.OptunaPlugin(trial, "accuracy")
            plugin.is_distributed = True
            with pytest.raises(ValueError):
                plugin.setup()

            return 0

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

    def test_configure(self) -> None:
        def objective(trial: optuna.Trial) -> int:
            plugin = hlpo.OptunaPlugin(trial, "accuracy")
            trainer = hlt.Trainer()

            plugin.configure_trainer(trainer)
            assert len(trainer.plugins) == 1
            assert trainer.plugins["optuna"] == plugin
            assert len(trainer.train_exceptions) == 1
            assert trainer.train_exceptions[0] == optuna.TrialPruned

            model = PluginModel("plugin-model")
            plugin.configure_model(model)
            assert model.save_name == "plugin-model_trial-0"

            return 0

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

    def check_in_range(
        self, val: typing.Any, t: type, low: typing.Any, high: typing.Any
    ) -> None:
        assert isinstance(val, t)
        assert low <= val <= high

    def check_in_sequence(self, val: typing.Any, t: type, seq: typing.Container) -> None:
        assert isinstance(val, t)
        assert val in seq

    def test_suggest(self) -> None:
        def objective(trial: optuna.Trial) -> int:
            plugin = hlpo.OptunaPlugin(trial, "accuracy")

            with pytest.raises(KeyError):
                plugin.suggest("foo", "bar")

            seq = [1, 2, 3]
            self.check_in_sequence(
                plugin.suggest("categorical", "val1", choices=seq), int, seq
            )

            low = 0
            high = 10
            self.check_in_range(
                plugin.suggest("int", "val2", low=low, high=high), int, low, high
            )

            high = 1
            self.check_in_range(
                plugin.suggest("float", "val3", low=low, high=high), float, low, high
            )

            return 0

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

    def test_state_dict(self) -> None:
        def objective(trial: optuna.Trial) -> int:
            plugin = hlpo.OptunaPlugin(trial, "accuracy")
            x = plugin.suggest("float", "x", low=-10, high=10)

            state_dict = plugin.state_dict()
            assert len(state_dict) == 1
            assert "x" in state_dict
            assert state_dict["x"] == x

            return 0

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

    def test_resume_trial(self, tmp_path: pathlib.Path) -> None:
        num_trials = 10
        storage_path = tmp_path / "trial_test.db"
        successful_trials = [False for _ in range(num_trials)]
        offset = 0
        study_args = {
            "study_name": "trial_test",
            "storage": f"sqlite:///{storage_path}",
            "load_if_exists": True,
        }

        def objective(trial: optuna.Trial) -> float:
            nonlocal offset
            plugin = hlpo.OptunaPlugin(trial, "accuracy")
            plugin.setup()

            model = PluginModel("plugin-model")
            plugin.configure_model(model)

            trial_num = trial.number
            if hlpo._ORIG_NUMBER_KEY in trial.user_attrs:
                trial_num = trial.user_attrs[hlpo._ORIG_NUMBER_KEY]
                assert model.save_name == f"plugin-model_trial-{trial_num}"

                # Artificially offset the trial number so we don't raise the exception
                # again.
                trial_num += 1
                offset = 1

            if trial_num == num_trials / 2:
                raise RuntimeError("half-way stop")

            successful_trials[trial_num - offset] = True
            offset = 0
            if trial_num % 2 == 1:
                raise optuna.TrialPruned()

            return 0

        def optimize(study: optuna.Study) -> None:
            study.optimize(
                objective,
                n_trials=num_trials,
                callbacks=[
                    optuna.study.MaxTrialsCallback(
                        num_trials,
                        states=(
                            optuna.trial.TrialState.COMPLETE,
                            optuna.trial.TrialState.PRUNED,
                        ),
                    )
                ],
            )

        study = hlpo.resume_study(study_args, backup_study=False)
        assert len(study.trials) == 0
        with pytest.raises(RuntimeError):
            optimize(study)

        del study

        study = hlpo.resume_study(study_args, backup_study=False)
        assert len(study.trials) == (num_trials // 2) + 1
        optimize(study)

        for v in successful_trials:
            assert v

    def test_study_backup(self, tmp_path: pathlib.Path) -> None:
        storage_path = tmp_path / "test.db"
        storage_path.touch(exist_ok=True)

        for _ in range(10):
            hlpo._backup_study(storage_path)

        # Ensure there are 11 "trials"
        backups = list(tmp_path.glob("*.db"))
        assert len(backups) == 11

        idx = [i for i, b in enumerate(backups) if b.name == "test.db"][0]
        backups.pop(idx)

        value = [False] * 10
        for p in backups:
            idx = int(p.stem.split("-")[-1])
            value[idx] = True

        assert all(v for v in value)

    def test_sampler_checkpoints(self, tmp_path: pathlib.Path) -> None:
        @dc.dataclass
        class TestRun:
            samples: list[float]
            chkpt_root: pathlib.Path

        num_trials = 10
        run1 = TestRun([], tmp_path / "run1")
        run2 = TestRun([], tmp_path / "run2")

        run1.chkpt_root.mkdir(exist_ok=True)
        run2.chkpt_root.mkdir(exist_ok=True)

        def objective(
            trial: optuna.Trial,
            raise_error: bool,
            run: TestRun,
        ) -> float:
            hlpo.checkpoint_sampler(trial, run.chkpt_root)
            res = trial.suggest_float("accuracy", 0, 1)
            if raise_error and trial.number == num_trials // 2:
                raise RuntimeError("half-way stop")
            run.samples.append(res)
            return res

        def create_study(
            sampler: optuna.samplers.BaseSampler | None = None,
        ) -> optuna.Study:
            return optuna.create_study(
                study_name="chkpt_test",
                sampler=optuna.samplers.TPESampler(seed=rng.get_default_seed())
                if sampler is None
                else sampler,
            )

        study = create_study()
        study.optimize(
            functools.partial(
                objective,
                raise_error=False,
                run=run1,
            ),
            n_trials=num_trials,
        )

        chkpts = [chkpt.stem for chkpt in run1.chkpt_root.glob("*.pkl")]
        chkpts.sort()
        assert len(chkpts) == num_trials
        assert all(chkpt == f"sampler_trial-{i}" for i, chkpt in enumerate(chkpts))
        del study

        study = create_study()
        sampler = hlpo.restore_sampler(run2.chkpt_root)
        assert sampler is None

        with pytest.raises(RuntimeError):
            study.optimize(
                functools.partial(
                    objective,
                    raise_error=True,
                    run=run2,
                ),
                n_trials=num_trials,
            )

        torch.serialization.clear_safe_globals()
        sampler = hlpo.restore_sampler(run2.chkpt_root)
        study = create_study(sampler)
        study.optimize(
            functools.partial(objective, raise_error=False, run=run2),
            n_trials=num_trials // 2,
        )
        assert run2.samples == run1.samples
