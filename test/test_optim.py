from helios import optim


class TestOptimizers:
    def test_registry(self, check_registry) -> None:
        check_registry(optim.OPTIMIZER_REGISTRY, ["Adam", "AdamW", "SGD"])

    def test_create(self, check_create_function) -> None:
        check_create_function(optim.OPTIMIZER_REGISTRY, optim.create_optimizer)
