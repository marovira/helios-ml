import dataclasses as dc
import functools
import pathlib
import typing

import optuna
import pytest

import helios.model as hlm
import helios.plugins.optuna as hlpo
import helios.trainer as hlt
from helios.core import rng
from helios.plugins.optuna import utils


class PluginModel(hlm.Model):
    def __init__(self, save_name: str):
        super().__init__(save_name)

    def setup(self, fast_init: bool = False) -> None:
        pass


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


# Allow accessing private members so we can test things properly.
# ruff: noqa: SLF001
class TestOptunaUtils:
    def test_get_backup_name(self, tmp_path: pathlib.Path) -> None:
        test_entries = 4
        name = utils._get_backup_name(tmp_path, "test-file", utils._BackupType.FILE)
        assert name == "test-file_bkp-0"

        for i in range(test_entries):
            (tmp_path / f"test-file_bkp-{i}.txt").touch()

        name = utils._get_backup_name(tmp_path, "test-file", utils._BackupType.FILE)
        assert name == f"test-file_bkp-{test_entries}"

        # Now test folders
        name = utils._get_backup_name(tmp_path, "test-dir", utils._BackupType.DIR)
        assert name == "test-dir_bkp-0"

        for i in range(test_entries):
            (tmp_path / f"test-dir_bkp-{i}").mkdir()

        name = utils._get_backup_name(tmp_path, "test-dir", utils._BackupType.DIR)
        assert name == f"test-dir_bkp-{test_entries}"

    def test_backup_study(self, tmp_path: pathlib.Path) -> None:
        base_study = tmp_path / "base_study.db"
        base_study.touch()
        assert len(list(tmp_path.glob("*.db"))) == 1

        utils._backup_study(base_study)
        assert len(list(tmp_path.glob("*.db"))) == 2

        bkp_study = tmp_path / "base_study_bkp-0.db"
        assert bkp_study.exists()

    def test_backup_samplers(self, tmp_path: pathlib.Path) -> None:
        test_entries = 4
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        for i in range(test_entries):
            (base_dir / f"sampler_{i}.pkl").touch()

        def key(p: pathlib.Path) -> int:
            return int(p.stem.split("_")[-1])

        base_files = list(base_dir.glob("*.pkl"))
        base_files.sort(key=key)

        utils._backup_sampler_chkpts(base_dir)
        bkp_dir = tmp_path / "base_bkp-0"
        assert bkp_dir.exists()

        bkp_files = list(bkp_dir.glob("*.pkl"))
        bkp_files.sort(key=key)
        assert len(base_files) == len(bkp_files)

        for f, b in zip(base_files, bkp_files, strict=True):
            assert f.name == b.name

    def test_check_study_args(self) -> None:
        args: utils.StudyArgs = {}

        with pytest.raises(RuntimeError):
            utils._check_study_args(**args)

        args["storage"] = None
        with pytest.raises(RuntimeError):
            utils._check_study_args(**args)

        args["storage"] = ""
        with pytest.raises(RuntimeError):
            utils._check_study_args(**args)

        args["load_if_exists"] = False
        with pytest.raises(RuntimeError):
            utils._check_study_args(**args)

        args["load_if_exists"] = True
        args["storage"] = 1  # type: ignore[typeddict-item]
        with pytest.raises(TypeError):
            utils._check_study_args(**args)

        args["storage"] = "some_path.db"
        with pytest.raises(RuntimeError):
            utils._check_study_args(**args)

    def test_get_storage_path(self, tmp_path: pathlib.Path) -> None:
        db = tmp_path / "test.db"
        db.touch()
        args: utils.StudyArgs = {"storage": f"sqlite:///{db}"}

        path = utils._get_storage_path(**args)
        assert path == db

    def test_checkpoint_sampler(self, tmp_path: pathlib.Path) -> None:
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

        sampler = hlpo.restore_sampler(run2.chkpt_root)
        assert sampler is not None
        study = create_study(sampler)
        study.optimize(
            functools.partial(objective, raise_error=False, run=run2),
            n_trials=num_trials // 2,
        )
        assert run2.samples == run1.samples

    def test_create_study(self, tmp_path: pathlib.Path) -> None:
        args: utils.StudyArgs = {}
        args["storage"] = f"sqlite:///{str(tmp_path / 'basic.db')}"
        args["load_if_exists"] = True
        args["study_name"] = "basic"

        sampler_path = tmp_path / "samplers"
        sampler_path.mkdir()

        study = utils.create_or_load_study(sampler_path=sampler_path, **args)

        def objective(trial: optuna.Trial) -> float:
            hlpo.checkpoint_sampler(trial, sampler_path)
            res = trial.suggest_float("accuracy", 0, 1)
            return res

        study.optimize(objective, n_trials=2)

        del study

        # Now check backups.
        utils.create_or_load_study(
            sampler_path=sampler_path, backup_samplers=True, backup_study=True, **args
        )

        dbs = [p.name for p in tmp_path.glob("*.db")]
        assert len(dbs) == 2
        assert "basic_bkp-0.db" in dbs

        sampler_bkp = tmp_path / "samplers_bkp-0"
        assert sampler_bkp.exists()

        base_samplers = [p.name for p in sampler_path.glob("*.pkl")]
        base_samplers.sort()
        bkp_samplers = [p.name for p in sampler_bkp.glob("*.pkl")]
        bkp_samplers.sort()
        assert base_samplers == bkp_samplers

    def test_load_study(self, tmp_path: pathlib.Path) -> None:
        def create_study(
            name: str,
            add_sampler: bool,
        ) -> tuple[optuna.Study, pathlib.Path]:
            args: utils.StudyArgs = {
                "storage": f"sqlite:///{str(tmp_path / f'{name}.db')}",
                "load_if_exists": True,
                "study_name": name,
            }
            if add_sampler:
                args["sampler"] = optuna.samplers.TPESampler(seed=rng.get_default_seed())

            sampler_path = tmp_path / f"{name}_samplers"
            sampler_path.mkdir(exist_ok=True)

            return utils.create_or_load_study(
                sampler_path=sampler_path, **args
            ), sampler_path

        def objective(
            trial: optuna.Trial,
            samples: list[float],
            sampler_path: pathlib.Path,
            raise_at: int | None = None,
        ) -> float:
            utils.checkpoint_sampler(trial, sampler_path)
            res = trial.suggest_float("accuracy", 0, 1)
            if raise_at is not None and trial.number == raise_at:
                raise RuntimeError("stop")
            samples.append(res)
            return res

        base_samples: list[float] = []
        study, base_samplers = create_study("base", True)
        study.optimize(
            functools.partial(
                objective,
                samples=base_samples,
                sampler_path=base_samplers,
            ),
            n_trials=4,
        )
        assert len(base_samples) == 4
        del study

        samples: list[float] = []
        study, samplers = create_study("resume", True)
        with pytest.raises(RuntimeError):
            study.optimize(
                functools.partial(
                    objective, samples=samples, raise_at=2, sampler_path=samplers
                ),
                n_trials=4,
            )
        assert len(samples) == 2
        del study

        study, samplers = create_study("resume", False)
        study.optimize(
            functools.partial(objective, samples=samples, sampler_path=samplers),
            n_trials=2,
        )
        assert len(samples) == 4
        assert base_samples == samples

    def test_create_study_starting_from_trial(self, tmp_path: pathlib.Path) -> None:
        @dc.dataclass
        class TestRun:
            sampler_path: pathlib.Path
            samples1: list[float]
            samples2: list[float]
            active_sample: int = 1

            def push_sample(self, s: float) -> None:
                if self.active_sample == 1:
                    self.samples1.append(s)
                else:
                    self.samples2.append(s)

        def objective(trial: optuna.Trial, run: TestRun) -> float:
            hlpo.checkpoint_sampler(trial, run.sampler_path)
            res = trial.suggest_float("accuracy", 0, 1)
            run.push_sample(res)
            hlpo.checkpoint_sampler(trial, run.sampler_path)
            return res

        def get_sampler_names(path: pathlib.Path) -> list[str]:
            s = [p.stem for p in path.glob("*.pkl")]
            s.sort()
            return s

        num_trials = 4
        test_run = TestRun(tmp_path / "resume", [], [])
        test_run.sampler_path.mkdir()

        args: utils.StudyArgs = {}
        args["storage"] = f"sqlite:///{str(tmp_path / 'resume.db')}"
        args["load_if_exists"] = True
        args["study_name"] = "resume"

        study = utils.create_or_load_study(sampler_path=test_run.sampler_path, **args)
        test_run.active_sample = 1
        study.optimize(functools.partial(objective, run=test_run), n_trials=num_trials)
        del study

        base_samplers = get_sampler_names(test_run.sampler_path)
        assert len(base_samplers) == num_trials

        study = utils.create_study_starting_from_trial(
            2, sampler_path=test_run.sampler_path, **args
        )
        bkp_path = tmp_path / "resume_bkp-0"
        assert bkp_path.exists()
        bkp_samplers = get_sampler_names(bkp_path)
        assert bkp_samplers == base_samplers

        pruned_samplers = get_sampler_names(test_run.sampler_path)
        assert len(pruned_samplers) == 2
        assert pruned_samplers == base_samplers[:2]

        test_run.active_sample = 2
        study.optimize(functools.partial(objective, run=test_run), n_trials=2)

        assert len(study.trials) == 4
        assert len(test_run.samples2) == 2
        assert test_run.samples1[2:] == test_run.samples2

        resume_samplers = get_sampler_names(test_run.sampler_path)
        assert resume_samplers == base_samplers
