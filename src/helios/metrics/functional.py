import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from helios.data import functional


def _ssim(img: npt.NDArray[np.float64], img2: npt.NDArray[np.float64]) -> float:
    """
    SSIM implementation function. See calculate_ssim for details.

    Args:
        img: Images of type ``np.float64`` with range  :math:`[0, 1]`.
        img2: Images of type ``np.float64`` with range :math:`[0, 1]`.

    Returns:
        SSIM value.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel: npt.NDArray = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    # valid mode for window size 11
    mu1: npt.NDArray = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2: npt.NDArray = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq  # type: ignore[operator]
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq  # type: ignore[operator]
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def _ssim_torch(img: torch.Tensor, img2: torch.Tensor) -> float:
    """
    SSIM Torch implementation function. See :py:func:`.calculate_ssim_torch` for details.

    Args:
        img: Images of type ``torch.float64`` with range  :math:`[0, 1]`.
        img2: Images of type ``torch.float64`` with range :math:`[0, 1]`.

    Returns:
        float: SSIM.
    """
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    kernel: npt.NDArray = cv2.getGaussianKernel(11, 1.5)
    window = (
        torch.from_numpy(np.outer(kernel, kernel.transpose()))
        .view(1, 1, 11, 11)
        .expand(img.size(1), 1, 11, 11)
        .to(img.dtype)
        .to(img.device)
    )

    # pylint: disable=not-callable
    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2
    )

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return float(ssim_map.mean([1, 2, 3]))


def _average_precision(
    output: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
) -> float:
    """
    Calculate the average precision of the given inputs.

    See :py:func:`.calculate_map` for details.

    Args:
        output: predicate labels in range :math:`[0, 1]`.
        target: target (inferred) labels in range :math:`[0, 1]`.

    Returns:
        The average precision.
    """
    epsilon = 1e-8

    indices = output.argsort()[::-1]
    total_count = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count = np.cumsum(ind)
    total = pos_count[-1]
    pos_count[np.logical_not(ind)] = 0
    pp = pos_count / total_count
    precision_at_i = np.sum(pp)
    precision_at_i = precision_at_i / (total + epsilon)

    return precision_at_i


def _mae(pred: npt.NDArray, gt: npt.NDArray, scale: float = 1.0) -> float:
    """
    MAE implementation. See :py:func:`.calculate_mae` for details.

    Args:
        pred: predicate tensor.
        gt: ground-truth tensor.
        scale: the scaling factor used in the tensor data (if any). Defaults to 1.

    Returns:
        The MAE score.
    """
    n = np.prod(gt.shape)
    sum_error = np.sum(np.abs(np.subtract(np.float32(pred), np.float32(gt))))
    mae_error = np.divide(sum_error, n * scale + 1e-4)
    return float(mae_error)


def _mae_torch(pred: torch.Tensor, gt: torch.Tensor, scale: float = 1.0) -> float:
    """
    MAE Torch implementation. See :py:func:`.calculate_mae_torch` for details.

    Args:
        pred: predicate tensor.
        gt: ground-truth tensor.
        scale: the scaling factor used in the tensor data (if any). Defaults to 1.

    Returns:
        The MAE score.
    """
    n = torch.prod(torch.tensor(gt.shape))
    sum_error = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
    mae_error = torch.divide(sum_error, n * scale + 1e-4)
    return float(mae_error)


def calculate_psnr(
    img: npt.NDArray,
    img2: npt.NDArray,
    crop_border: int,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Implementation follows: `<https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__

    Args:
        img: Images with range :math:`[0, 255]`.
        img2: Images with range :math:`[0, 255]`.
        crop_border: Cropped pixels in each edge of an image. These pixels are not
            involved in the calculation.
        input_order: Whether the input order is "HWC" or "CHW". Defaults to "HWC".
        test_y_channel: Test on Y channel of YCbCr. Defaults to false.

    Returns:
        PSNR value.
    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    img = functional.convert_to_hwc(img, input_order=input_order)
    img2 = functional.convert_to_hwc(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = functional.to_y_channel(img)
        img2 = functional.to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)


def calculate_psnr_torch(
    img: torch.Tensor,
    img2: torch.Tensor,
    crop_border: int,
    test_y_channel: bool = False,
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Implementation follows: `<https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`__

    Args:
        img: Images with range :math:`[0, 255]`.
        img2: Images with range :math:`[0, 255]`.
        crop_border: Cropped pixels in each edge of an image. These pixels are not
            involved in the calculation.
        test_y_channel: Test on Y channel of YCbCr. Defaults to false.

    Returns:
        PSNR value.
    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = functional.rgb2ycbcr_torch(img, only_y=True)
        img2 = functional.rgb2ycbcr_torch(img2, only_y=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return float(10.0 * torch.log10(1.0 / (mse + 1e-8)))


def calculate_ssim(
    img: npt.NDArray,
    img2: npt.NDArray,
    crop_border: int,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> float:
    """
    Calculate SSIM (structural similarity).

    Implementation follows: 'Image quality assesment: From error visibility to structural
    similarity'. Results are identical to those of the official MATLAB code in
    `<https://ece.uwaterloo.ca/~z70wang/research/ssim/>`__.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img: Images with range :math:`[0, 255]`.
        img2: Images with range :math:`[0, 255]`.
        crop_border: Cropped pixels in each edge of an image. These pixels are not
            involved in the calculation.
        input_order: Whether the input order is "HWC" or "CHW". Defaults to "HWC"
        test_y_channel: Test on Y channel of YCbCr. Defaults to false.

    Returns:
        SSIM.
    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    img = functional.convert_to_hwc(img, input_order=input_order)
    img2 = functional.convert_to_hwc(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = functional.to_y_channel(img)
        img2 = functional.to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_ssim_torch(
    img: torch.Tensor,
    img2: torch.Tensor,
    crop_border: int,
    test_y_channel: bool = False,
) -> float:
    """
    Calculate SSIM (structural similarity).

    Implementation follows: 'Image quality assesment: From error visibility to structural
    similarity'. Results are identical to those of the official MATLAB code in
    `<https://ece.uwaterloo.ca/~z70wang/research/ssim/>`__.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img: Images with range :math:`[0, 255]`.
        img2: Images with range :math:`[0, 255]`.
        crop_border: Cropped pixels in each edge of an image. These pixels are not
            involved in the calculation.
        test_y_channel: Test on Y channel of YCbCr. Defaults to false.

    Returns:
        SSIM.
    """
    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = functional.rgb2ycbcr_torch(img, only_y=True)
        img2 = functional.rgb2ycbcr_torch(img2, only_y=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_torch(img * 255.0, img2 * 255.0)
    return ssim


def calculate_mAP(targs: npt.NDArray, preds: npt.NDArray) -> float:
    """
    Calculate the mAP (Mean Average Precision).

    Implementation follows:
    `<https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision>`__

    Args:
        targs: target (inferred) labels in range :math:`[0, 1]`.
        preds: predicate labels in range :math:`[0, 1]`.

    Returns:
        The mAP score
    """
    if np.size(preds) == 0:
        return 0

    class_count = 0
    mAP_sum = 0.0
    for k in range(preds.shape[1]):
        scores = preds[:, k]
        targets = targs[:, k]
        if np.sum(targets) == 0:
            continue
        mAP_sum += _average_precision(scores, targets)
        class_count += 1

    return 100 * (mAP_sum / class_count)


def calculate_mae(pred: npt.NDArray, gt: npt.NDArray, scale: float = 1.0) -> float:
    """
    Compute the MAE (Mean-Average Precision) score.

    Implementation follows: `<https://en.wikipedia.org/wiki/Mean_absolute_error>`__.
    The scale argument is used in the event that the input arrays are not in the range
    :math:`[0, 1]` but instead have been scaled to be in the range :math:`[0, N]` where
    :math:`N` is the factor. For example, if the arrays are images in the range
    :math:`[0, 255]`, then the scaling factor should be set to 255. If the arrays are
    already in the range :math:`[0, 1]`, then the scale can be omitted.

    Args:
        pred: predicate (inferred) array
        gt: ground-truth array
        scale: scaling factor that was used on the input arrays. Defaults to 1.

    Returns:
        The MAE score.
    """
    return _mae(pred, gt, scale)


def calculate_mae_torch(
    pred: torch.Tensor, gt: torch.Tensor, scale: float = 1.0
) -> float:
    """
    Compute the MAE (Mean-Average Precision) score.

    Implementation follows: `<https://en.wikipedia.org/wiki/Mean_absolute_error>`__.
    The scale argument is used in the event that the input arrays are not in the range
    :math:`[0, 1]` but instead have been scaled to be in the range :math:`[0, N]` where
    :math:`N` is the factor. For example, if the arrays are images in the range
    :math:`[0, 255]`, then the scaling factor should be set to 255. If the arrays are
    already in the range :math:`[0, 1]`, then the scale can be omitted.

    Args:
        pred: predicate (inferred) tensor
        gt: ground-truth tensor
        scale: scaling factor that was used on the input tensors. Defaults to 1.

    Returns:
        The MAE score.
    """
    return _mae_torch(pred, gt, scale)
