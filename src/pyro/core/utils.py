import os
import pathlib
import time
import typing

T = typing.TypeVar("T")


def get_from_optional(opt_var: T | None, raise_on_empty: bool = False) -> T:
    """
    Given a variable whose type is Optional[T], assert that the variable is not None and
    return it without the Optional type.
    Note: the raise_on_empty is intended for testing only.

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
    Context manager that allows easy switching between the current working directory
    (usually the invocation directory of the script) and a new path. Upon entering, the
    current directory will be switched to the new path and the previous working directory
    returned. Upon exiting, the original working directory will be restored.

    Ex:
        os.chdir(".")   # <- Starting working directory
        with ChdirContext("/new/path") as prev_cwd:
            # prev_cwd is the starting working directory
            Path.cwd() # <- This is /new/path now
            ...
        Path.cwd() # <- Back to the starting working directory.

    Args:
        target_path (pathlib.Path): the path to switch to.
    """

    def __init__(self, target_path: pathlib.Path):
        self.start_path = pathlib.Path.cwd()
        self.target_path = target_path

    def __enter__(self) -> pathlib.Path:
        os.chdir(self.target_path)
        return self.start_path

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        os.chdir(self.start_path)


class AverageTimer:
    """
    Timer with moving average determined by a sliding window.
    Used to determine the average time between splits.

    Args:
        sliding_window (int): the number of steps of which the moving average will be
        computed.
    """

    def __init__(self, sliding_window: int = 200):
        self._sliding_window = sliding_window

        self._time_sum: float = 0
        self._step_count: int = 0
        self._current_time: float = 0
        self.start()

    def start(self) -> None:
        """
        Starts the timer.
        """
        self._current_time = time.time()

    def record(self) -> None:
        """
        Records a new step in the timer.
        """
        self._step_count += 1
        self._time_sum += time.time() - self._current_time

        if self._step_count > self._sliding_window:
            self._step_count = 0
            self._time_sum = 0

        self._current_time = time.time()

    def get_average_time(self) -> float:
        """
        Returns the moving average over the current step count.
        """
        return self._time_sum / self._step_count
