import typing

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from helios import core

from .functional import (
    calculate_mae,
    calculate_mae_torch,
    calculate_mAP,
    calculate_psnr,
    calculate_psnr_torch,
    calculate_ssim,
    calculate_ssim_torch,
)

METRICS_REGISTRY = core.Registry("metrics")


def create_metric(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> nn.Module:
    """
    Create the metric function for the given type.

    Args:
        type_name (str): the type of the loss to create.
        args: positional arguments to pass into the metric.
        kwargs: keyword arguments to pass into the metric.

    Returns:
        Callable: the metric function
    """
    return METRICS_REGISTRY.get(type_name)(*args, **kwargs)


@METRICS_REGISTRY.register
class CalculatePSNR(nn.Module):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Implementation follows: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Note that the input_order is only needed if you plan to evaluate Numpy images. It can
    be left as default otherwise.

    Args:
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not
        involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default:
    """

    def __init__(
        self, crop_border: int, input_order: str = "HWC", test_y_channel: bool = False
    ):
        """Construct the PSNR metric."""
        super().__init__()

        self._crop_border = crop_border
        self._input_order = input_order
        self._test_y_channel = test_y_channel

    def forward(
        self, img: npt.NDArray | torch.Tensor, img2: npt.NDArray | torch.Tensor
    ) -> float:
        """
        Calculate the PSNR metric.

        Args:
            img (np.ndarray | torch.Tensor): Images with range [0, 255].
            img2 (np.ndarray | torch.Tensor): Images with range [0, 255].

        Returns:
            float: PSNR value.
        """
        if isinstance(img, torch.Tensor) and isinstance(img2, torch.Tensor):
            return calculate_psnr_torch(
                img, img2, self._crop_border, self._test_y_channel
            )
        assert isinstance(img, np.ndarray) and isinstance(img2, np.ndarray)
        return calculate_psnr(
            img, img2, self._crop_border, self._input_order, self._test_y_channel
        )


@METRICS_REGISTRY.register
class CalculateSSIM(nn.Module):
    """
    Calculate SSIM (structural similarity).

    Implementation follows: 'Image quality assesment: From error visibility to structural
    similarity'. Results are identical to those of the official MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not
        involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        test_y_channel (bool): Test on Y channel of YCbCr.

    Returns:
        float: SSIM.
    """

    def __init__(
        self, crop_border: int, input_order: str = "HWC", test_y_channel: bool = False
    ):
        """Construct the SSIM metric."""
        super().__init__()

        self._crop_border = crop_border
        self._input_order = input_order
        self._test_y_channel = test_y_channel

    def forward(
        self, img: npt.NDArray | torch.Tensor, img2: npt.NDArray | torch.Tensor
    ) -> float:
        """
        Calculate the SSIM metric.

        Args:
            img (np.ndarray | torch.Tensor): Images with range [0, 255].
            img2 (np.ndarray | torch.Tensor): Images with range [0, 255].

        Returns:
            float: PSNR value.
        """
        if isinstance(img, torch.Tensor) and isinstance(img2, torch.Tensor):
            return calculate_ssim_torch(
                img, img2, self._crop_border, self._test_y_channel
            )
        assert isinstance(img, np.ndarray) and isinstance(img2, np.ndarray)
        return calculate_ssim(
            img, img2, self._crop_border, self._input_order, self._test_y_channel
        )


@METRICS_REGISTRY.register
class CalculateMAP(nn.Module):
    """
    Calculate the mAP (Mean Average Precision).

    Implementation follows:
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
    """

    def forward(self, targs: npt.NDArray, preds: npt.NDArray) -> float:
        """
        Calculate the mAP (Mean Average Precision).

        Args:
            targs (np.ndarray): target (inferred) labels in range [0, 1].
            preds (np.ndarray): predicate labels in range [0, 1].

        Returns:
            float: the mAP score
        """
        return calculate_mAP(targs, preds)


@METRICS_REGISTRY.register
class CalculateMAE(nn.Module):
    """
    Compute the MAE (Mean-Average Precision) score.

    Implementation follows: https://en.wikipedia.org/wiki/Mean_absolute_error
    The scale argument is used in the event that the input tensors are not in the range
    [0, 1] but instead have been scaled to be in the range [0, N] where N is the factor.
    For example, if the tensors are images in the range [0, 255], then the scaling factor
    should be set to 255. If the tensors are already in the range [0, 1], then the scale
    can be omitted.

    Args:
        scale (float): scaling factor that was used on the input tensors (if any)
    """

    def __init__(self, scale: float = 1):
        """Construct the MAE metric."""
        super().__init__()
        self._scale = scale

    def forward(
        self, pred: npt.NDArray | torch.Tensor, gt: npt.NDArray | torch.Tensor
    ) -> float:
        """
        Calculate the MAE metric.

        Args:
            pred (np.ndarray | torch.Tensor): predicate (inferred) data.
            gt (np.ndarray | torch.Tensor): ground-truth data.

        Returns:
            float: the MAE score
        """
        if isinstance(pred, torch.Tensor) and isinstance(gt, torch.Tensor):
            return calculate_mae_torch(pred, gt, self._scale)
        assert isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray)
        return calculate_mae(pred, gt, self._scale)
