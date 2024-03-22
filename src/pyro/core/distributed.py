import functools
import os
import typing

import torch
import torch.distributed as dist

from pyro.core import cuda


def main_only(func: typing.Callable) -> typing.Callable:
    """
    Mark functions that should only run on the main process.

    For distributed training, the main process is associated with the process whose rank
    is 0.

    Args:
        func (Callable): the function to mark

    Returns:
        Callable: the marked function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not cuda.is_available():
            return func(*args, **kwargs)

        rank = get_rank()
        if rank == 0:
            return func(*args, **kwargs)
        return None

    return wrapper


@cuda.cuda_only
def dist_only(func: typing.Callable) -> typing.Callable:
    """
    Mark functions that should only run when torch.distributed is available.

    Args:
        func (Callable): the function to mark

    Returns:
        Callable: the marked function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.is_available():
            return func(*args, **kwargs)
        return None

    return wrapper


@dist_only
def init_dist(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize the distributed process group.

    Args:
        rank (int): the rank of the current GPU
        world_size (int): the number of GPUs in the system
        backend (str): the backend to use.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend, rank=rank, world_size=world_size)


@dist_only
def shutdown_dist() -> None:
    """Shutdown the distributed process group."""
    dist.destroy_process_group()


@dist_only
def get_dist_info() -> tuple[int, int]:
    """
    Get the rank and world size of the current distributed run.

    If distributed training is not used, then 0 and 1 are returned as the values of rank
    and world size, respectively.

    Returns:
        tuple[int, int]: the rank and world size.
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


@dist_only
def get_rank() -> int:
    """
    Get the rank of the device the process is running on.

    If distributed trainig is not used, 0 is returned.

    Returns:
        int: the rank of the current device.
    """
    return get_dist_info()[0]


@dist_only
def gather_into_tensor(tensor: torch.Tensor, size: tuple[int]) -> torch.Tensor:
    """
    Gathers the tensors across all processes and merges them into a single tensor.

    Args:
        tensor (torch.Tensor): the tensor to merge
        size (tuple[int]): the dimensions of the output tensor.

    Returns:
        torch.Tensor: the resulting tensor containing all gathered tensors.
    """
    if not dist.is_initialized():
        raise RuntimeError("error: default process group has not been initialized")

    device = torch.device(f"cuda:{get_rank()}")
    out = torch.zeros(size, device=device, dtype=tensor.dtype)
    dist.all_gather_into_tensor(out, tensor)
    return out


@dist_only
def all_reduce_tensors(
    tensor: torch.Tensor | list[torch.Tensor], **kwargs
) -> torch.Tensor:
    """
    Reduces tensors across all processes so all have the same value.

    Args:
        tensor (torch.Tensor | list[torch.Tensor]): the input tensor(s) to reduce.
        If the input is a list of tensors, they will be concatenated into a single tensor
        kwargs (dict): additional options for torch.distributed.all_reduce

    Returns:
        torch.Tensor: the reduced tensor
    """
    if not dist.is_initialized():
        raise RuntimeError("error: default process group has not been initialized")

    value_tensor = torch.tensor(tensor).cuda() if isinstance(tensor, list) else tensor
    dist.all_reduce(value_tensor, **kwargs)
    return value_tensor
