import dataclasses as dc
import os
import typing

import torch
import torch.distributed as dist


def is_using_torchrun() -> bool:
    """
    Check if the current process was launched from ``torchrun``.

    This will check to see if the environment variables that are set by ``torchrun``
    exist. The list of variables is taken directly from the documentation and can be seen
    `here`_.

    .. _here: https://pytorch.org/docs/stable/elastic/run.html#environment-variables

    Returns:
        True if the run was started from ``torchrun``, false otherwise.
    """
    return all(
        key in os.environ
        for key in [
            "LOCAL_RANK",
            "RANK",
            "GROUP_RANK",
            "ROLE_RANK",
            "LOCAL_WORLD_SIZE",
            "WORLD_SIZE",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_RUN_ID",
        ]
    )


def get_distributed_backend(device_type: str, offload_ops_to_cpu: bool = False) -> str:
    """
    Get the PyTorch distributed backend based on device type.

    Args:
        device_type: device type to get the backend for.
        offload_ops_to_cpu: (optional) if true, then operations will be offloaded to CPU.

    Example:
        .. code-block:: python

            import helios.core.distributed as hlcd

            # Backend type is nccl
            backend = hlcd.get_distributed_backend("cuda")

            # Backend type is gloo
            backend = hlcd.get_distributed_backend("cpu")

            # Backend type is "cuda:nccl,cpu:gloo"
            backend = hlcd.get_distributed_backend("cuda", offload_ops_to_cpu=True)

    Returns:
        The name of the backend to use.
    """
    default_device_backend_map = dist.Backend.default_device_backend_map
    backend = "nccl"
    if device_type in default_device_backend_map:
        backend = default_device_backend_map[device_type]
    if offload_ops_to_cpu:
        backend = f"{device_type}:{backend},cpu:gloo"
    return backend


def init_dist(
    backend: str = "nccl", rank: int | None = None, world_size: int | None = None
) -> None:
    """
    Initialize the distributed process group.

    The optional values for ``rank`` and ``world_size`` **must** be omitted if
    distributed training is handled through ``torchrun``. If distributed training is
    started manually, then both arguments **must** be provided.

    Args:
        backend: the backend to use. Defaults to "nccl".
        rank: the (optional) rank of the current GPU.
        world_size: the (optional) number of GPUs in the system.

    Raises:
        ValueError: if either of ``rank`` or ``world_size`` are ``None`` (but not both).
    """
    if rank is None and world_size is None:
        dist.init_process_group(backend)
        return

    if rank is None or world_size is None:
        raise ValueError("error: rank and world_size cannot be None")

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend, rank=rank, world_size=world_size)


def shutdown_dist() -> None:
    """Shutdown the distributed process group."""
    dist.destroy_process_group()


def is_dist_active() -> bool:
    """Check if torch.distributed is active."""
    return dist.is_available() and dist.is_initialized()


@dc.dataclass
class DistributedInfo:
    """
    Bundle information regarding distributed state.

    To understand what these mean, consider a run being performed over two nodes, each
    with 2 GPUs. Suppose we have a single process per GPU. Then the following values are
    assigned:

    * ``local_rank``: 0 or 1 for both nodes. 0 for the first GPU, 1 for the second.
    * ``rank``: 0, 1, 2, 3 for each of the GPUs over the two nodes.
    * ``local_world_size``: 2 for both nodes.
    * ``world_size``: 4 (2 nodes with 2 workers per node).

    Args:
        local_rank: the local rank.
        rank: the global rank.
        local_world_size: the local world size.
        world_size: the global world size.
    """

    local_rank: int = 0
    rank: int = 0
    local_world_size: int = 1
    world_size: int = 1


def get_dist_info() -> DistributedInfo:
    """
    Get the distributed state of the current run.

    If distributed training is not used, then both ranks are set to 0 and both world sizes
    are set to 1.

    Returns:
        The information of the current distributed run.
    """
    info = DistributedInfo()
    if dist.is_available() and dist.is_initialized():
        info.local_rank = int(os.environ["LOCAL_RANK"])
        info.rank = int(os.environ["RANK"])
        info.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        info.world_size = int(os.environ["WORLD_SIZE"])

    return info


def get_local_rank() -> int:
    """
    Get the local rank of the device the process is running on.

    If distributed training is not used, 0 is returned.

    Returns:
        The local rank of the current device.
    """
    return get_dist_info().local_rank


def get_global_rank() -> int:
    """
    Get the global rank of the device the process is running on.

    If distributed training is not used, 0 is returned.

    Returns:
        The global rank of the current device.
    """
    return get_dist_info().rank


def gather_into_tensor(tensor: torch.Tensor, size: tuple[int]) -> torch.Tensor:
    """
    Gathers the tensors across all processes and merges them into a single tensor.

    Args:
        tensor: the tensor to merge
        size: the dimensions of the output tensor.

    Returns:
        The resulting tensor containing all gathered tensors.

    Raises:
        RuntimeError: if distributed training hasn't been initialised.
    """
    if not dist.is_initialized():
        raise RuntimeError("error: default process group has not been initialized")

    device = torch.device(f"{tensor.device.type}:{get_local_rank()}")
    out = torch.zeros(size, device=device, dtype=tensor.dtype)
    dist.all_gather_into_tensor(out, tensor)
    return out


def all_reduce_tensors(
    tensor: torch.Tensor | list[torch.Tensor], **kwargs
) -> torch.Tensor:
    """
    Reduces tensors across all processes so all have the same value.

    Args:
        tensor: the input tensor(s) to reduce. If the input is a list of tensors, they
            will be concatenated into a single tensor.
        kwargs: additional options for torch.distributed.all_reduce

    Returns:
        The reduced tensor.

    Raises:
        RuntimeError: if distributed training has not been initialised.
    """
    if not dist.is_initialized():
        raise RuntimeError("error: default process group has not been initialized")

    device = tensor[0].device if isinstance(tensor, list) else tensor.device
    value_tensor = torch.tensor(tensor).to(device) if isinstance(tensor, list) else tensor
    dist.all_reduce(value_tensor, **kwargs)
    return value_tensor


def _dist_print_wrapper(
    *args: typing.Any, rank_check: typing.Callable[[], bool], **kwargs: typing.Any
) -> None:
    if not is_dist_active():
        print(*args, **kwargs)
        return
    if rank_check():
        print(*args, **kwargs)


def global_print(*args: typing.Any, global_rank: int = 0, **kwargs: typing.Any) -> None:
    """
    Print wrapper that only prints on the specified global rank.

    Args:
        *args: positional arguments to pass in to Python's print.
        global_rank: the global rank the print should happen on. Defaults to 0.
        **kwargs: keyword arguments to pass in to Python's print.
    """
    _dist_print_wrapper(
        *args, rank_check=lambda: get_global_rank() == global_rank, **kwargs
    )


def local_print(*args: typing.Any, local_rank: int = 0, **kwargs: typing.Any) -> None:
    """
    Print wrapper that only prints on the specified local rank.

    Args:
        *args: positional arguments to pass in to Python's print.
        local_rank: the local rank the print should happen on. Defaults to 0.
        **kwargs: keyword arguments to pass in to Python's print.
    """
    _dist_print_wrapper(
        *args, rank_check=lambda: get_local_rank() == local_rank, **kwargs
    )


def safe_barrier(**kwargs: typing.Any) -> None:
    """
    Safe wrapper for torch.distributed.barrier.

    The wrapper is "safe" in the sense that it is valid to call this function regardless
    of whether the code is currently using distributed training or not.

    Args:
        **kwargs: keyword arguments to torch.distributed.barrier.
    """
    if not is_dist_active():
        return
    dist.barrier(**kwargs)
