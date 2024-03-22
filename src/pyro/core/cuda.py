import functools
import typing

import torch


def is_available() -> bool:
    """
    Check whether PyTorch has CUDA support.

    Returns:
        bool: True if CUDA support is found, False otherwise.
    """
    return torch.cuda.is_available()


def requires_cuda_support() -> None:
    """Ensure that CUDA support is found. If it isn't, an exception is raised."""
    if not is_available():
        raise RuntimeError("error: expected CUDA support but none was found")


def cuda_only(func: typing.Callable) -> typing.Callable:
    """
    Decorate functions that should only run when CUDA is available.

    Args:
        func (Callable): the function to mark.

    Returns:
        Callable: the marked function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_available():
            return func(*args, **kwargs)
        return None

    return wrapper


@cuda_only
def get_device_count() -> int:
    """
    Get the number of GPUs available in the system.

    Returns:
        int: the number of GPUs in the system.
    """
    return torch.cuda.device_count()
