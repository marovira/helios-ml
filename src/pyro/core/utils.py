import os
import pathlib
import time
import typing

T = typing.TypeVar("T")


def get_from_optional(opt_var: T | None, raise_on_empty: bool = False) -> T:
    """
    Ensure the given variable is not None and return it.

    This is useful when dealing with variables that can be None at declaration but are set
    elsewhere. In those instances, Mypy is unable to determine that the variable was set,
    so it will issue a warning. The workaround is to add asserts, but that can get tedious
    very quickly. This function can be used as an alternative.
    Ex:
        ```py
        var: int | None = None
        # ... Set var to a valid value some place else.

        assert var is not None
        v = var

        # Alternatively:
        v = core.get_from_optional(var)
        ```

    Args:
        opt_var (Optional[T]): the optional variable.
        raise_on_empty (bool): if True, an exception is raised when the optional is None.

    Returns:
        T: the variable without the optional.
    """
    if not raise_on_empty:
        assert opt_var is not None
    else:
        raise RuntimeError("error: optional cannot be empty")
    return opt_var


class ChdirContext:
    """
    Allow switching between the current working directory and another within a scope.

    The intention is to facilitate temporary switches of the current working directory
    (such as when attempting to resolve relative paths) by creating a context in which the
    working directory is automatically switched to a new one. Upon exiting of the context,
    the original working directory is restored.

    Ex:
        ```py
        os.chdir(".")   # <- Starting working directory
        with ChdirContext("/new/path") as prev_cwd:
            # prev_cwd is the starting working directory
            Path.cwd() # <- This is /new/path now
            ...
        Path.cwd() # <- Back to the starting working directory.
        ```

    Args:
        target_path (pathlib.Path): the path to switch to.
    """

    def __init__(self, target_path: pathlib.Path):
        """
        Create the context manager with the given path.

        Args:
            target_path (pathlib.Path): the path to switch to.
        """
        self.start_path = pathlib.Path.cwd()
        self.target_path = target_path

    def __enter__(self) -> pathlib.Path:
        """
        Perform the switch from the current working directory to the new one.

        Returns:
            pathlib.Path: the previous working directory.
        """
        os.chdir(self.target_path)
        return self.start_path

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        """Restores the previous working directory."""
        os.chdir(self.start_path)


class AverageTimer:
    """
    Compute elapsed times using moving average.

    The timer will determine the elapsed time between a series of points using a sliding
    window moving average.

    Args:
        sliding_window (int): the number of steps of which the moving average will be
        computed.
    """

    def __init__(self, sliding_window: int = 200):
        """
        Create the timer with the given sliding window.

        Args:
            sliding_window (int): the number of steps of which the moving average will be
            computed.
        """
        self._sliding_window = sliding_window

        self._time_sum: float = 0
        self._step_count: int = 0
        self._current_time: float = 0
        self.start()

    def start(self) -> None:
        """Start the timer."""
        self._current_time = time.time()

    def record(self) -> None:
        """Record a new step in the timer."""
        self._step_count += 1
        self._time_sum += time.time() - self._current_time

        if self._step_count > self._sliding_window:
            self._step_count = 0
            self._time_sum = 0

        self._current_time = time.time()

    def get_average_time(self) -> float:
        """Return the moving average over the current step count."""
        return self._time_sum / self._step_count
