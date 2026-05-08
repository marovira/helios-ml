import torch
from torch.utils import data as tud

import helios.model as hlm
import helios.plugins as hlp
import helios.trainer as hlt
from helios import data
from helios.core import rng


class _TinyDataset(tud.Dataset):
    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.zeros(1)

    def __len__(self) -> int:
        return 4


class _SimpleDatamodule(data.DataModule):
    def setup(self) -> None:
        params = data.DataLoaderParams(
            batch_size=1, num_workers=0, random_seed=rng.get_default_seed()
        )
        self._add_train_phase(_TinyDataset(), params)


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


@hlp.PLUGIN_REGISTRY.register
class _DummyPlugin(hlp.Plugin):
    def __init__(self) -> None:
        super().__init__("_dummy")

    def setup(self) -> None:
        pass


class TestPlugins:
    def test_registry(self, check_registry) -> None:
        check_registry(hlp.PLUGIN_REGISTRY, ["_DummyPlugin"])

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

    def test_configure_called_automatically(self) -> None:
        class TrackingPlugin(hlp.Plugin):
            def __init__(self) -> None:
                super().__init__("tracking")
                self.configure_trainer_called: bool = False
                self.configure_model_called: bool = False

            def setup(self) -> None:
                pass

            def configure_trainer(self, trainer: hlt.Trainer) -> None:
                self.configure_trainer_called = True

            def configure_model(self, model: hlm.Model) -> None:
                self.configure_model_called = True

        plugin = TrackingPlugin()
        trainer = hlt.Trainer(
            train_unit=hlt.TrainingUnit.EPOCH,
            total_steps=1,
            use_cpu=True,
        )
        trainer.register_plugin(plugin)
        trainer.fit(PluginModel("test"), _SimpleDatamodule())

        assert plugin.configure_trainer_called
        assert plugin.configure_model_called
