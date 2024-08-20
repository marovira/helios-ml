import torch

from helios import plugins, trainer


class TestPlugins:
    def test_registry(self, check_registry) -> None:
        check_registry(plugins.PLUGIN_REGISTRY, ["CUDAPlugin"])

    def test_create(self, check_create_function) -> None:
        check_create_function(plugins.PLUGIN_REGISTRY, plugins.create_plugin)

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
        state = trainer.TrainingState()

        assert plugin.unique_overrides.training_batch
        assert plugin.unique_overrides.validation_batch
        assert plugin.unique_overrides.testing_batch

        func_list = [
            plugin.process_training_batch,
            plugin.process_validation_batch,
            plugin.process_testing_batch,
        ]

        # Single tensor
        for func in func_list:
            x = create_tensor()
            x = func(x, state)
            assert x.device == device

        # List
        for func in func_list:
            x = [create_tensor(), create_tensor()]  # type:ignore[assignment]
            x = func(x, state)
            assert all(elem.device == device for elem in x)

        # Tuple
        for func in func_list:
            x = (create_tensor(), create_tensor())  # type:ignore[assignment]
            x = func(x, state)
            assert all(elem.device == device for elem in x)

        # Dictionary
        for func in func_list:
            x = {"a": create_tensor(), "b": create_tensor()}  # type: ignore[assignment]
            x = func(x, state)
            assert all(val.device == device for _, val in x.items())  # type: ignore [attr-defined]
