import typing

import optuna
import pytest
import torch

import helios.model as hlm
import helios.plugins as hlp
import helios.trainer as hlt
from helios.plugins.optuna import OptunaPlugin


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
        assert hasattr(OptunaPlugin, "plugin_id")
        assert OptunaPlugin.plugin_id == "optuna"

    def test_invalid_storage(self) -> None:
        def objective(trial: optuna.Trial) -> int:
            plugin = OptunaPlugin(trial, "accuracy")
            plugin.is_distributed = True
            with pytest.raises(ValueError):
                plugin.setup()

            return 0

        study = optuna.create_study()
        study.optimize(objective, n_trials=1)

    def test_configure(self) -> None:
        def objective(trial: optuna.Trial) -> int:
            plugin = OptunaPlugin(trial, "accuracy")
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
            plugin = OptunaPlugin(trial, "accuracy")

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
