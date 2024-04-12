from helios import scheduler


class TestSchedulers:
    def test_registry(self, check_registry) -> None:
        check_registry(
            scheduler.SCHEDULER_REGISTRY,
            [
                "MultiStepLR",
                "CosineAnnealingLR",
                "CosineAnnealingRestartLR",
                "MultiStepRestartLR",
            ],
        )

    def test_create(self, check_create_function) -> None:
        check_create_function(scheduler.SCHEDULER_REGISTRY, scheduler.create_scheduler)
