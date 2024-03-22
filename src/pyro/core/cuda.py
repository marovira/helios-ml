from torch import cuda


def requires_cuda_support() -> None:
    """Ensure that CUDA support is found. If it isn't, an exception is raised."""
    if not cuda.is_available():
        raise RuntimeError("error: expected CUDA support but none was found")
