import pathlib

import cv2
import numpy as np
import numpy.typing as npt
import PIL
import torch


def load_image(
    path: pathlib.Path, out_fmt: str = "", as_numpy: bool = True
) -> npt.NDArray | PIL.Image.Image:
    """
    Load the given image.

    ``out_fmt`` is a format string that can be passed in to PIL.Image.convert. Please
    `here <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert>`__
    for the list of accepted strings.
    If no string is passed, the image will be converted to RGB format.
    By default, the output is a NumPY array. If you need a PIL image instead, set
    ``as_numpy`` to false.

    Args:
        path: the path to the image to load.
        out_fmt: the format to convert the loaded image to. Defaults to empty.
        as_numpy: if true, the loaded image will be returned as a NumPY array, otherwise
            it is returned as a PIL image. Defaults to true.

    Returns:
        The loaded image.
    """
    with path.open(mode="rb") as infile:
        img = PIL.Image.open(infile)
        out = img.convert(out_fmt) if out_fmt != "" else img.convert("RGB")
        if as_numpy:
            return np.array(out)
        return out


def tensor_to_numpy(tens: torch.Tensor, as_float: bool = False) -> npt.NDArray:
    """
    Convert the given tensor to a numpy array.

    Args:
        tens: the tensor to convert in the range :math:`[0, 1]`
        as_float: whether to leave the output as float or convert to int.

    Returns:
        The converted array.
    """
    as_np = tens.squeeze().float().clamp_(0, 1).cpu().detach().numpy()
    if as_np.ndim == 3:
        as_np = np.transpose(as_np, (1, 2, 0))

    if not as_float:
        as_np = np.uint8((as_np * 255.0).round())

    return as_np


def show_tensor(tens: torch.Tensor, title: str = "debug window") -> None:
    """
    Display the image held by the tensor. Useful for debugging purposes.

    Args:
        tens: the image tensor to display in range :math:`[0, 1]`.
        title: the title of the displayed window. Defaults to "debug window".
    """
    img = tensor_to_numpy(tens)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def show_tensors(tens: torch.Tensor) -> None:
    """
    Show batches of tensors.

    Args:
        tens: the batch of tensors to display in range :math:`[0, 1]`.
    """
    if len(tens.shape) == 3:
        show_tensor(tens)
        return

    for b in range(tens.shape[0]):
        show_tensor(tens[b], title=f"tensor[{b}]")


def convert_to_hwc(img: npt.NDArray, input_order: str = "HWC") -> npt.NDArray:
    """
    Change the order of the input image channels so the result is in (h, w, c) order.

    If the input image is a single-channel image, then the return is (h, w, 1).

    Args:
        img: input image.
        input_order: the order of the channels of the input image. Must be one of 'HWC'
            or 'CHW'.

    Returns:
        The shuffled image.
    """
    assert input_order in ("HWC", "CHW")

    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img: npt.NDArray) -> npt.NDArray:
    """
    Return the Y (luma) channel of a YCbCr image.

    Args:
        img: input image in YCbCr format. Must be in the range :math:`[0, 255]`.

    Returns:
        The luma channel of the input image.
    """
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, only_y=True)
        img = img[..., None]
    return img * 255.0


def bgr2ycbcr(img: npt.NDArray, only_y: bool = False) -> npt.NDArray:
    """
    Convert the given numpy image array from BGR to YCBCR.

    If only the Y channel is required, set ``only_y`` to true.

    Args:
        img: the BGR image to convert.
        only_y: if true, only the luma (Y) channel will be returned.

    Returns:
        The converted image.
    """
    intype = img.dtype
    img.astype(np.float32)
    if intype != np.uint8:
        img *= 255

    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(
            img,
            [
                [24.966, 112.0, -18.214],
                [128.553, -74.203, -93.786],
                [65.481, -37.797, 112.0],
            ],
        ) / 255.0 + [16, 128, 128]

    if intype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(intype)


def rgb2ycbcr_torch(img: torch.Tensor, only_y: bool = False) -> torch.Tensor:
    """
    Convert the given torch Tensor image array from RGB to YCBCR.

    If only the Y channel is required, set ``only_y`` to true.

    Args:
        img: the BGR image to convert.
        only_y: if true, only the luma (Y) channel will be returned.

    Returns:
        The converted image.
    """
    if only_y:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor(
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ]
        ).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.0
    return out_img
