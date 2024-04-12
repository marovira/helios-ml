from helios import metrics


class TestMetrics:
    def test_registry(self, check_registry) -> None:
        check_registry(
            metrics.METRICS_REGISTRY,
            [
                "CalculatePSNR",
                "CalculateSSIM",
                "CalculateMAP",
                "CalculateMAE",
                "CalculatePrecision",
                "CalculateRecall",
                "CalculateF1",
            ],
        )

    def test_create(self, check_create_function) -> None:
        check_create_function(metrics.METRICS_REGISTRY, metrics.create_metric)
