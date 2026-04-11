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
                "CalculateAccuracy",
                "CalculatePrecision",
                "CalculateRecall",
                "CalculateF1",
                "CalculateRMSE",
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

    def test_accuracy(self) -> None:
        # 3 samples, 3 classes.
        # top-1 predictions: [0, 1, 2], targets: [0, 2, 2] → 2 correct → 2/3
        # top-2 predictions: [[0,1],[1,2],[2,0]], targets: [0, 2, 2] → all correct → 1.0
        predictions = torch.tensor([[10.0, 5.0, 0.0], [0.0, 10.0, 5.0], [5.0, 0.0, 10.0]])
        targets = torch.tensor([0, 2, 2])

        self.check_almost_equal(
            functional.calculate_accuracy(predictions, targets, top_k=1), 2 / 3
        )
        self.check_almost_equal(
            functional.calculate_accuracy(predictions, targets, top_k=2), 1.0
        )

        acc1 = metrics.CalculateAccuracy(top_k=1)
        acc2 = metrics.CalculateAccuracy(top_k=2)
        self.check_almost_equal(acc1(predictions, targets), 2 / 3)
        self.check_almost_equal(acc2(predictions, targets), 1.0)

    def test_precision(self) -> None:
        # 4 samples, 2 classes.
        # predictions (class indices): [0, 0, 1, 1], targets: [0, 1, 0, 1]
        # class 0: TP=1, FP=1 → precision=0.5
        # class 1: TP=1, FP=1 → precision=0.5
        # macro precision = 0.5
        predictions_1d = torch.tensor([0, 0, 1, 1])
        predictions_2d = torch.tensor(
            [[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]
        )
        targets = torch.tensor([0, 1, 0, 1])

        self.check_almost_equal(
            functional.calculate_precision(predictions_1d, targets), 0.5
        )
        self.check_almost_equal(
            functional.calculate_precision(predictions_2d, targets), 0.5
        )

        prec = metrics.CalculatePrecision()
        self.check_almost_equal(prec(predictions_1d, targets), 0.5)
        self.check_almost_equal(prec(predictions_2d, targets), 0.5)

    def test_recall(self) -> None:
        # Same setup as precision test.
        # class 0: TP=1, FN=1 → recall=0.5
        # class 1: TP=1, FN=1 → recall=0.5
        # macro recall = 0.5
        predictions_1d = torch.tensor([0, 0, 1, 1])
        predictions_2d = torch.tensor(
            [[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]
        )
        targets = torch.tensor([0, 1, 0, 1])

        self.check_almost_equal(functional.calculate_recall(predictions_1d, targets), 0.5)
        self.check_almost_equal(functional.calculate_recall(predictions_2d, targets), 0.5)

        rec = metrics.CalculateRecall()
        self.check_almost_equal(rec(predictions_1d, targets), 0.5)
        self.check_almost_equal(rec(predictions_2d, targets), 0.5)

    def test_f1(self) -> None:
        # Precision=0.5, recall=0.5 → F1=0.5.
        predictions_1d = torch.tensor([0, 0, 1, 1])
        predictions_2d = torch.tensor(
            [[10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]
        )
        targets = torch.tensor([0, 1, 0, 1])

        self.check_almost_equal(functional.calculate_f1(predictions_1d, targets), 0.5)
        self.check_almost_equal(functional.calculate_f1(predictions_2d, targets), 0.5)

        f1 = metrics.CalculateF1()
        self.check_almost_equal(f1(predictions_1d, targets), 0.5)
        self.check_almost_equal(f1(predictions_2d, targets), 0.5)

    def test_rmse(self) -> None:
        # Identical tensors → RMSE = 0.
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        self.check_almost_equal(functional.calculate_rmse(predictions, targets), 0.0)

        # predictions=[0, 0], targets=[1, 0] → MSE=0.5, RMSE=sqrt(0.5).
        predictions = torch.tensor([0.0, 0.0])
        targets = torch.tensor([1.0, 0.0])
        expected = math.sqrt(0.5)
        self.check_almost_equal(functional.calculate_rmse(predictions, targets), expected)

        rmse = metrics.CalculateRMSE()
        self.check_almost_equal(rmse(predictions, targets), expected)
