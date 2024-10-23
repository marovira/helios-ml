import typing

import numpy.typing as npt
import PIL
import torch
import torchvision.transforms.v2 as T
from torch import nn

from helios import core

TRANSFORM_REGISTRY = core.Registry("transform")
"""
Global instance of the registry for transforms.

Example:
    .. code-block:: python

        import helios.data.transforms as hldt

        # This automatically registers your dataset.
        @hldt.TRANSFORM_REGISTRY.register()
        class MyTransform:
            ...

        # Alternatively you can manually register a dataset like this:
        hldt.TRANSFORM_REGISTRY.register(MyTransform)
"""


def create_transform(
    type_name: str, *args: typing.Any, **kwargs: typing.Any
) -> nn.Module:
    """
    Create a transform of the given type.

    This uses TRANSFORM_REGISTRY to look-up transform types, so ensure your transforms
    have been registered before using this function.

    Args:
        type_name: the type of the transform to create.
        args: positional arguments to pass into the transform.
        kwargs: keyword arguments to pass into the transform.

    Returns:
        The constructed transform.
    """
    return TRANSFORM_REGISTRY.get(type_name)(*args, **kwargs)


@TRANSFORM_REGISTRY.register
class ToImageTensor(nn.Module):
    """
    Convert an image (or list of images) to tensor(s).

    An image is meant to be a tensor, ndarray, or PIL image. The shape expected to be
    either [H, W, C] or [C, H, W].

    Args:
        dtype: the output type of the tensors.
        scale: if true, scale the values to the valid range. Defaults to true.
    """

    def __init__(self, dtype: torch.dtype = torch.float32, scale: bool = True):
        """Create the transform."""
        super().__init__()
        self._transform = T.Compose(
            [T.ToImage(), T.ToDtype(dtype, scale=scale), T.ToPureTensor()]
        )

    def forward(
        self,
        img: npt.NDArray
        | list[npt.NDArray]
        | tuple[npt.NDArray, ...]
        | PIL.Image.Image
        | list[PIL.Image.Image]
        | tuple[PIL.Image.Image, ...],
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        """
        Convert the input image(s) into tensor(s).

        The return type will match the type of the input. So, if the input is a single
        image, then the output will be a single tensor. If the input is a list or a tuple
        of images, the output will be a list or tuple of tensors.

        Args:
            img: image(s) to convert.

        Returns:
            The converted images.
        """
        out_tens: list[torch.Tensor] = []
        for elem in core.convert_to_list(img):
            out_tens.append(self._transform(elem))

        if len(out_tens) == 1:
            return out_tens[0]
        if isinstance(img, tuple):
            return tuple(out_tens)
        return out_tens
