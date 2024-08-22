import typing

import torch

from helios import plugins, trainer


class ExceptionPlugin(plugins.Plugin):
    def __init__(self, exc_type: type[Exception] | list[type[Exception]]):
        super().__init__()
        self._exc_type = exc_type

    def setup(self) -> None:
        pass

    def configure_trainer(self, t: trainer.Trainer) -> None:
        self._append_train_exceptions(self._exc_type, t)
        self._append_test_exceptions(self._exc_type, t)


class TestPlugins:
    def test_registry(self, check_registry) -> None:
        check_registry(plugins.PLUGIN_REGISTRY, ["CUDAPlugin"])

    def test_create(self, check_create_function) -> None:
        check_create_function(plugins.PLUGIN_REGISTRY, plugins.create_plugin)

    def test_append_exceptions(self) -> None:
        t = trainer.Trainer()
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

    def check_batch_device(self, x: typing.Any, device: torch.device) -> None:
        if isinstance(x, torch.Tensor):
            assert x.device == device
        elif isinstance(x, dict):
            assert all(val.device == device for _, val in x.items())
        else:
            assert all(elem.device == device for elem in x)

    def check_batch_processing(
        self, plugin: plugins.CUDAPlugin, x: typing.Any, device: torch.device
    ) -> None:
        ret = plugin.process_training_batch(x, trainer.TrainingState())
        self.check_batch_device(ret, device)

        ret = plugin.process_validation_batch(x, 0)
        self.check_batch_device(ret, device)

        ret = plugin.process_testing_batch(x, 0)
        self.check_batch_device(ret, device)

    def test_cuda_plugin(self) -> None:
        if not torch.cuda.is_available():
            return

        def create_tensor() -> torch.Tensor:
            return torch.randn((1, 3, 32, 32)).to("cpu")

        device = torch.device("cuda:0")

        plugin = plugins.CUDAPlugin()
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
