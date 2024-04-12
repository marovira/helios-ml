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
        img (np.ndarray): Images with range [0, 1] (float64).
        img2 (np.ndarray): Images with range [0, 1] (float64).

    Returns:
        float: SSIM.
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
    SSIM Torch implementation function. See calculate_ssim_torch for details.

    Args:
        img (torch.Tensor): Images with range [0, 1] (float64).
        img2 (torch.Tensor): Images with range [0, 1] (float64).

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
    Calculate the average precision of the given inputs. See calculate_map for details.

    Args:
        output (np.ndarray): predicate labels in range [0, 1].
        target (np.ndarray): target (inferred) labels in range [0, 1].

    Returns:
        float: the average precision.
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


def _mae_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    MAE Torch implementation. See calculate_mae_torch for details.

    Args:
        pred (torch.Tensor): predicate (inferred) images in range [0, 255]
        gt (torch.Tensor): ground-truth images in range [0, 255]

    Returns:
        float: the MAE score.
    """
    h, w = gt.shape[0:2]
    sum_error = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
    mae_error = torch.divide(sum_error, float(h) * float(w) * 2550 + 1e-4)
    return float(mae_error)


def _f1_score_torch(
    pd: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the following scores: precision, recall, and F1 for torch.

    See calculate_precision/recall/f1_torch for details.

    Args:
        pd (torch.Tensor): predicate tensor.
        gt (torch.Tensor): ground-truth tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the precision, recall, and F1
        scores, respectively.
    """
    gt_num = torch.sum((gt > 128).float() * 1)

    pp = pd[gt > 128]
    nn = pd[gt <= 128]

    pp_hist = torch.histc(pp, bins=255, min=0, max=255)
    nn_hist = torch.histc(nn, bins=255, min=0, max=255)

    pp_hist_flip = torch.flipud(pp_hist)
    nn_hist_flip = torch.flipud(nn_hist)

    pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
    nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

    precision = (pp_hist_flip_cum) / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
    recall = (pp_hist_flip_cum) / (gt_num + 1e-4)
    f1 = (1 + 0.3) * precision * recall / (0.3 * precision + recall + 1e-4)

    return (
        torch.reshape(precision, (1, precision.shape[0])),
        torch.reshape(recall, (1, recall.shape[0])),
        torch.reshape(f1, (1, f1.shape[0])),
    )


def calculate_psnr(
    img: npt.NDArray,
    img2: npt.NDArray,
    crop_border: int,
    input_order: str = "HWC",
    test_y_channel: bool = False,
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    Implementation follows: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (np.ndarray): Images with range [0, 255].
        img2 (np.ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not
        involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default:

    Returns:
        float: PSNR value.
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

    Implementation follows: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (torch.Tensor): Images with range [0, 255].
        img2 (torch.Tensor): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not
        involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default:

    Returns:
        float: PSNR value.
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
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (np.ndarray): Images with range [0, 255].
        img2 (np.ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not
        involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
        test_y_channel (bool): Test on Y channel of YCbCr.

    Returns:
        float: SSIM.
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
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (torch.Tensor): Images with range [0, 255].
        img2 (torch.Tensor): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not
        involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr.

    Returns:
        float: the SSIM value.
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
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    Args:
        targs (np.ndarray): target (inferred) labels in range [0, 1].
        preds (np.ndarray): predicate labels in range [0, 1].

    Returns:
        float: the mAP score
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


def calculate_mae_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute the MAE (Mean-Average Precision) score.

    Implementation follows: https://en.wikipedia.org/wiki/Mean_absolute_error

    Args:
        pred (torch.Tensor): predicate (inferred) images in range [0, 255]
        gt (torch.Tensor): ground-truth images in range [0, 255]

    Returns:
        float: the MAE score.
    """
    return _mae_torch(pred, gt)


def calculate_precision_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute the Precision score.

    Implementation follows:  https://en.wikipedia.org/wiki/Precision_and_recall

    Args:
        pred (torch.Tensor): predicate (inferred) images in range [0, 255]
        gt (torch.Tensor): ground-truth images in range [0, 255]

    Returns:
        float: the precision score.
    """
    return _f1_score_torch(pred, gt)[0].cpu().data.numpy()


def calculate_recall_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute the Recall score.

    Implementation follows:  https://en.wikipedia.org/wiki/Precision_and_recall

    Args:
        pred (torch.Tensor): predicate (inferred) images in range [0, 255]
        gt (torch.Tensor): ground-truth images in range [0, 255]

    Returns:
        float: the recall score.
    """
    return _f1_score_torch(pred, gt)[1].cpu().data.numpy()


def calculate_f1_torch(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    Compute the F1 score.

    Implementation follows: https://en.wikipedia.org/wiki/F-score

    Args:
        pred (torch.Tensor): predicate (inferred) images in range [0, 255]
        gt (torch.Tensor): ground-truth images in range [0, 255]

    Returns:
        float: the F1 score.
    """
    return _f1_score_torch(pred, gt)[2].cpu().data.numpy()
