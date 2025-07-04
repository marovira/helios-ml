import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveAvgPool2d(nn.Module):
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size :math:`H \times W`, for any input size. The number of output
    features is equal to the number of input planes.

    .. note::
        This class is a re-implementation of ``torch.nn.AdaptiveAvgPool2d`` that can be
        exported to ONNX and serves as a drop-in replacement for torch's version.

    Args:
        output_size: the target output size of the image of the form :math:`H \times W`.
            Can be a tuple ``(H, W)`` or a single ``H`` for a square image :math:`H \times
            H`. ``H`` and ``W`` can be either a ``int``, or ``None`` which means the size
            will be the same as that of the input.

    """  # noqa: E501

    def __init__(self, output_size: int | tuple[int | None, int | None]):
        """Create the pool."""
        super().__init__()

        self._validate_output_size(output_size)

        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        self._output_size = output_size

    def _validate_output_size(
        self, output_size: int | tuple[int | None, int | None]
    ) -> None:
        if not isinstance(output_size, int | tuple):
            raise RuntimeError("error: output_size must be an int or a tuple")
        if isinstance(output_size, tuple):
            if len(output_size) != 2:
                raise RuntimeError(
                    f"error: output_size must have size 2 but received {len(output_size)}"
                )
            if any(not isinstance(o, int | None) for o in output_size):
                raise RuntimeError("error: expected a tuple of int or None")

    def _get_output_size(self, x: torch.Tensor) -> npt.NDArray:
        x_size = np.array(x.shape[-2:])
        if self._output_size is None:
            return x_size

        # At this point, the output size should just be a tuple
        assert isinstance(self._output_size, tuple)
        x_size[0] = (
            self._output_size[0] if self._output_size[0] is not None else x_size[0]
        )
        x_size[1] = (
            self._output_size[1] if self._output_size[1] is not None else x_size[1]
        )
        return x_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the adaptive average pool on the input.

        Args:
            x: the input tensor.

        Returns:
            The pooled tensor.
        """
        output_size = self._get_output_size(x)
        stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size  # type: ignore[operator]
        x = F.avg_pool2d(x, kernel_size=list(kernel_size), stride=list(stride_size))
        return x
