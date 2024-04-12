import dataclasses
import os

import torch
import torch.distributed as dist


def is_using_torchrun() -> bool:
    """
    Check if the current process was launched from torchrun.

    This will check to see if the environment variables that are set by torchrun exist.
    The list of variables is taken directly from the documentation and can be seen here:
    https://pytorch.org/docs/stable/elastic/run.html#environment-variables

    Returns:
        bool: True if the run was started from torchrun, false otherwise.
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


def init_dist(
    backend: str = "nccl", rank: int | None = None, world_size: int | None = None
) -> None:
    """
    Initialize the distributed process group.

    Args:
        backend (str): the backend to use.
        rank (int): the rank of the current GPU
        world_size (int): the number of GPUs in the system
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


@dataclasses.dataclass
class DistributedInfo:
    """
    Bundle information regarding distributed state.

    To understand what these mean, consider a run being performed over two nodes, each
    with 2 GPUs. Suppose we have a single process per GPU. Then the following values are
    assigned:
        * local_rank: 0 or 1 for both nodes. 0 for the first GPU, 1 for the second.
        * rank: 0, 1, 2, 3 for each of the GPUs over the two nodes.
        * local_world_size: 2 for both nodes.
        * world_size: 4 (2 nodes with 2 workers per node).

    Args:
        local_rank (int): the local rank.
        rank (int): the global rank.
        local_world_size (int): the local world size.
        world_size (int): the global world size.
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
        DistributedInfo: the information of the current distributed run.
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
        int: the local rank of the current device.
    """
    return get_dist_info().local_rank


def get_global_rank() -> int:
    """
    Get the global rank of the device the process is running on.

    If distributed training is not used, 0 is returned.

    Returns:
        int: the global rank of the current device.
    """
    return get_dist_info().rank


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

    device = torch.device(f"cuda:{get_local_rank()}")
    out = torch.zeros(size, device=device, dtype=tensor.dtype)
    dist.all_gather_into_tensor(out, tensor)
    return out


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
