import math

import torch

from helios import metrics
from helios.core import rng
from helios.metrics import functional


class TestMetrics:
    def test_registry(self, check_registry) -> None:
        check_registry(
            metrics.METRICS_REGISTRY,
            [
                "CalculatePSNR",
                "CalculateSSIM",
                "CalculateMAP",
                "CalculateMAE",
            ],
        )

    def test_create(self, check_create_function) -> None:
        check_create_function(metrics.METRICS_REGISTRY, metrics.create_metric)

    def check_almost_equal(self, val1: float, val2: float) -> None:
        assert math.isclose(val1, val2, abs_tol=0.00001)

    def check_psnr_ssim(self, metric: type, metric_fun_torch, metric_fun) -> None:
        rng.seed_rngs()
        img = torch.rand((1, 3, 32, 32)) * 255
        img2 = torch.rand((1, 3, 32, 32)) * 255
        module = metric(1, "CHW")

        self.check_almost_equal(module(img, img2), metric_fun_torch(img, img2, 1))

        # Convert to numpy
        img = torch.squeeze(img).numpy()  # type: ignore[assignment]
        img2 = torch.squeeze(img2).numpy()  # type: ignore[assignment]

        self.check_almost_equal(module(img, img2), metric_fun(img, img2, 1, "CHW"))

    def test_psnr(self) -> None:
        self.check_psnr_ssim(
            metrics.CalculatePSNR,
            functional.calculate_psnr_torch,
            functional.calculate_psnr,
        )

    def test_ssim(self) -> None:
        self.check_psnr_ssim(
            metrics.CalculateSSIM,
            functional.calculate_ssim_torch,
            functional.calculate_ssim,
        )

    def test_mAP(self) -> None:
        rng.seed_rngs()
        gen = rng.get_default_numpy_rng().generator
        targs = gen.random((1, 10))
        preds = gen.random((1, 10))

        mAP = metrics.CalculateMAP()
        self.check_almost_equal(mAP(targs, preds), functional.calculate_mAP(targs, preds))

    def check_mae(self, scale: float) -> None:
        rng.seed_rngs()

        pred = torch.randn((1, 3, 32, 32)) * scale
        gt = torch.randn((1, 3, 32, 32)) * scale
        mae = metrics.CalculateMAE(scale)

        self.check_almost_equal(
            mae(pred, gt), functional.calculate_mae_torch(pred, gt, scale)
        )

        pred = pred.numpy()  # type: ignore[assignment]
        gt = gt.numpy()  # type: ignore[assignment]
        self.check_almost_equal(mae(pred, gt), functional.calculate_mae(pred, gt, scale))  # type: ignore[arg-type]

    def test_MAE(self) -> None:
        self.check_mae(1.0)
        self.check_mae(255.0)
