from helios import scheduler


class TestSchedulers:
    def test_registry(self, check_registry) -> None:
        check_registry(
            scheduler.SCHEDULER_REGISTRY,
            [
                # Pytorch schedulers.
                "LambdaLR",
                "MultiplicativeLR",
                "StepLR",
                "MultiStepLR",
                "ConstantLR",
                "LinearLR",
                "ExponentialLR",
                "PolynomialLR",
                "CosineAnnealingLR",
                "SequentialLR",
                "ReduceLROnPlateau",
                "CyclicLR",
                "OneCycleLR",
                "CosineAnnealingWarmRestarts",
                # Custom schedulers.
                "MultiStepRestartLR",
                "CosineAnnealingRestartLR",
            ],
        )

    def test_create(self, check_create_function) -> None:
        check_create_function(scheduler.SCHEDULER_REGISTRY, scheduler.create_scheduler)
