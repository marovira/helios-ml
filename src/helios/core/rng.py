from __future__ import annotations

import random
import typing
from collections import abc

import torch
from numpy import random as npr

_DEFAULT_RNG_SEED = 6691


def get_default_seed() -> int:
    """Return the default seed."""
    return _DEFAULT_RNG_SEED


class DefaultNumpyRNG:
    """
    Default RNG from Numpy.

    This is intended to serve as a replacement for the legacy random API from Numpy. The
    class wraps the new Generator instance, which is set to use PCG64. Functionality
    similar to the modules for PyTorch is provided for easy serialization and restoring.

    Args:
        seed (int | list[int] | tuple[int] | None): the initial seed to use.
    """

    def __init__(self, seed: int | list[int] | tuple[int] | None = None):
        """Create the default RNG."""
        self._generator = npr.Generator(npr.PCG64(seed))

    @property
    def generator(self) -> npr.Generator:
        """Return the Numpy Generator instance."""
        return self._generator

    def state_dict(self) -> abc.Mapping[str, typing.Any]:
        """
        Create a dictionary containing the RNG state.

        Returns:
            Mapping[str, Any]: the state of the RNG.
        """
        return self._generator.bit_generator.state

    def load_state_dict(self, state_dict: abc.Mapping[str, typing.Any]) -> None:
        """
        Restore the RNG from the given state dictionary.

        Args:
            state_dict (Mapping[str, Any]): the state dictionary.
        """
        self._generator.bit_generator.state = state_dict


_DEFAULT_RNG: DefaultNumpyRNG | None = None


def _get_safe_default_rng() -> DefaultNumpyRNG:
    global _DEFAULT_RNG
    if _DEFAULT_RNG is None:
        raise RuntimeError(
            "error: default RNG has not been created. Did you forget to call "
            "create_default_rng?"
        )
    return _DEFAULT_RNG


def create_default_numpy_rng(seed: int | list[int] | tuple[int] | None = None):
    """
    Initialize the default RNG with the given seed.

    Args:
        seed (int | list[int] | tuple[int] | None): the seed to use (if any).
    """
    global _DEFAULT_RNG
    _DEFAULT_RNG = DefaultNumpyRNG(seed=seed)


def get_default_numpy_rng() -> DefaultNumpyRNG:
    """
    Return the default RNG.

    Return:
        np.random.Generator: the random generator.
    """
    return _get_safe_default_rng()


def seed_rngs(seed: int | None = None, skip_torch: bool = False) -> None:
    """
    Seed the default RNGs with the given seed.

    If no seed is given, then the default seed from Helios will be used. The RNGs that
    will be seeded are: PyTorch (+ CUDA if available), stdlib random, and the default
    Numpy generator.
    The skip_torch flag is intended to be used when seeding worker processes for
    dataloaders. In those cases, the RNGs for PyTorch have already been seeded, so we
    shouldn't be re-seeding them.

    Args:
        seed (int | None): optional value to seed the random generators with.
        skip_torch (bool): if True, torch RNGs won't be seeded.
    """
    seed = get_default_seed() if seed is None else seed

    if not skip_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    create_default_numpy_rng(seed)


def get_rng_state_dict() -> dict[str, typing.Any]:
    """
    Get the state dict for the default RNGs.

    Default RNGs are: PyTorch (+ CUDA if available) and Random.

    Returns:
        RandomState: the state of all RNGs.
    """
    state = {
        "torch": torch.get_rng_state(),
        "rand": random.getstate(),
        "numpy": get_default_numpy_rng().state_dict(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()

    return state


def load_rng_state_dict(state_dict: dict[str, typing.Any]) -> None:
    """
    Restore the default RNGs from the given state dict.

    See get_rng_state for the list of default RNGs.

    Args:
        state_dict (dict[str, typing.Any]): the state of the RNGs
    """
    torch.set_rng_state(state_dict["torch"])
    random.setstate(state_dict["rand"])
    get_default_numpy_rng().load_state_dict(state_dict["numpy"])
    if torch.cuda.is_available() and "cuda" in state_dict:
        torch.cuda.set_rng_state(state_dict["cuda"])
