import dataclasses
import random

import torch


@dataclasses.dataclass
class RNGState:
    """
    Contains the random state for each of the random number generators used for training.

    Args:
        torch_state (torch.Tensor): random state for torch.
        cuda_state (torch.Tensor): random state for torch.cuda.
        rand_state (Tuple): random state for Python's random.
    """

    torch_state: torch.Tensor
    rand_state: tuple
    cuda_state: torch.Tensor | None = None

    dict = dataclasses.asdict


def seed_rngs(seed: int) -> None:
    """
    Seed the default RNGs with the given seed.

    Default RNGs are: PyTorch (+ CUDA if available) and Random.

    Args:
        seed (int): value to seed the random generators with.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def get_rng_state() -> RNGState:
    """
    Get the state for the default RNGs.

    Default RNGs are: PyTorch (+ CUDA if available) and Random.

    Returns:
        RandomState: the state of all RNGs.
    """
    return RNGState(
        torch_state=torch.get_rng_state(),
        cuda_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        rand_state=random.getstate(),
    )


def restore_rng_state(state: RNGState) -> None:
    """
    Restore the default RNGs from the given state.

    See get_rng_state for the list of default RNGs.

    Args:
        state (RandomState): the state of the RNGs
    """
    torch.set_rng_state(state.torch_state)
    if torch.cuda.is_available() and state.cuda_state is not None:
        torch.cuda.set_rng_state(state.cuda_state)

    random.setstate(state.rand_state)
