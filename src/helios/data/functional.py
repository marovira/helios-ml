import pathlib

import cv2
import numpy as np
import numpy.typing as npt
import PIL
import torch


def load_image(
    path: pathlib.Path, flags: int = cv2.IMREAD_COLOR, as_rgb: bool = True
) -> npt.NDArray:
    """
    Load the given image using OpenCV.

    ``flags`` correspond to the ``cv2.IMREAD_`` flags from OpenCV. Please see the full
    list of options
    `here <https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html>`__. If no
    value is passed, the image will be loaded using ``cv2.IMREAD_COLOR``, which will load
    it as an 8-bit BGR image.

    ``as_rgb`` can be used to automatically convert the image from OpenCV's default BGR to
    RGB using the following logic:
    * If the image is grayscale, then it is returned as is.
    * If the image is a 3-channel BGR, it is converted to RGB.
    * If the image is a 4-channel BGRA, it is converted to RGBA.
    If all these checks fail, the image is returned as is.

    Args:
        path: the path to the image to load.
        flags: the OpenCV flags to use when loading.
        as_rgb: if true, the image will be converted from BGR/BGRA to RGB/RGBA, otherwise
            the image is returned as is.

    Returns:
        The loaded image.
    """
    img = cv2.imread(str(path), flags=flags)
    if not as_rgb:
        return img

    if len(img.shape) == 2:
        return img

    c = img.shape[2]
    if c == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if c == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    raise RuntimeError(
        f"error: expected a 3 or 4 channel image but received {c} channels"
    )


def load_image_pil(
    path: pathlib.Path, out_fmt: str = "", as_numpy: bool = True
) -> npt.NDArray | PIL.Image.Image:
    """
    Load the given image using PIL.

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


def tensor_to_numpy(
    x: torch.Tensor,
    squeeze: bool = True,
    clamp: tuple[float, float] | None = (0, 1),
    transpose: bool = True,
    dtype: torch.dtype | None = torch.float,
    as_uint8: bool = True,
) -> npt.NDArray:
    r"""
    Convert the given tensor to a numpy array.

    Args:
        x: the tensor to convert.
        squeeze: if true, squeeze to remove all dimensions of size 1. Defaults to true.
        clamp: tuple containing min/max values to clamp the tensor to. Can be ``None`` if
            no clamping is desired. Defaults to ``(0, 1)``.
        transpose: if true, transpose the tensor so the dimensions are :math:`H \times W
            \times C` or :math:`B \times H \times W \times C`. Defaults to true.
        dtype: the type to convert the tensor to. Can be set to ``None`` if no conversion
            is desired. Defaults to ``torch.float``.
        as_uint8: if true, convert the final array to be of type uint8 and in the range
            :math:`[0, 255]`. Defaults to true.

    Returns:
        The converted array.
    """
    if squeeze:
        x.squeeze_()
    if clamp is not None:
        x.clamp_(min=clamp[0], max=clamp[1])
    if dtype is not None:
        x = x.to(dtype)
    as_np = x.cpu().detach().numpy()

    if transpose:
        as_np = (
            np.transpose(as_np, (1, 2, 0))
            if as_np.ndim == 3
            else np.transpose(as_np, (0, 2, 3, 1))
            if as_np.ndim == 4
            else as_np
        )

    if as_uint8:
        as_np = np.uint8((as_np * 255.0).round())  # type: ignore[assignment]
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
