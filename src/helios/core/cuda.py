import torch
from torch import cuda


def requires_cuda_support() -> None:
    """
    Ensure that CUDA support is found, or raise an exception otherwise.

    Raises:
        RuntimeError: if no CUDA support is found.

    """
    if not cuda.is_available():
        raise RuntimeError("error: expected CUDA support but none was found")


class DisableCuDNNBenchmarkContext:
    """
    Allow disabling CuDNN benchmark within a scope.

    The intention is to facilitate the disabling of CuDNN benchmark for specific purposes
    (such as validation or testing) but then restoring it to its previous state upon
    leaving the scope. Note that if CUDA is not available, the scope does nothing.

    Example:
        .. code-block:: python

            torch.backends.cudnn.benchmark = True # Enable CuDNN
            ...
            with DisableCuDNNBenchmarkContext():
                # Benchmark is disabled.
                print(torch.backends.cudnn.benchmark) # <- Prints False
                ...

            print(torch.backends.cudnn.benchmark) # <- Prints true
    """

    def __init__(self) -> None:
        """Create the CuDNN context."""
        if cuda.is_available():
            self._init_value = torch.backends.cudnn.benchmark

    def __enter__(self) -> None:
        """Disable CuDNN benchmark."""
        if cuda.is_available():
            torch.backends.cudnn.benchmark = False

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Restore CuDNN benchmark to its starting state."""
        if cuda.is_available():
            torch.backends.cudnn.benchmark = self._init_value
