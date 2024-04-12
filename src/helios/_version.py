import contextlib
from importlib import metadata as meta

with contextlib.suppress(meta.PackageNotFoundError):
    __version__ = meta.version("helios-ml")
