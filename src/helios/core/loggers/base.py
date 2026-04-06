import abc
import pathlib
import typing
from datetime import datetime


class Logger(abc.ABC):
    """
    Base class for all Helios' loggers.

    Each logger sub-class is in charge of the following:
        * Their own storage (if applicable) under ``log_root``.
        * Their own persistent settings so they can be resumed.
    """

    @abc.abstractmethod
    def setup(
        self, run_name: str, log_root: pathlib.Path | None, is_resume: bool
    ) -> None:
        """
        Finish configuring the logger.

        Called after the logger has been created. If the logger is being restored, then
        this is called after :py:meth:`load_state_dict` is called. Each derived class
        should configure its own folder under ``log_root`` and create it as needed.

        Args:
            run_name: the name of the current run.
            log_root: root directory for all logs.  ``None`` means no on-disk
                output is desired.
            is_resume: ``True`` when continuing a previous run.
        """

    @abc.abstractmethod
    def flush(self) -> None:
        """Flush any buffered data to disk."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the logger and release any held resources."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, typing.Any]:
        """
        Return a dictionary containing the logger state.

        Returns:
            A dictionary that can be passed to :py:meth:`load_state_dict` to
            restore the logger to the same state.
        """

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Restore the logger state from a previously saved dictionary.

        Must be called *before* :py:meth:`setup` when resuming so that
        ``setup`` can use the saved paths.

        Args:
            state_dict: the state dictionary returned by a prior call to
                :py:meth:`state_dict`.
        """


def get_default_log_name(run_name: str) -> str:
    """
    Generate the default name used for loggers.

    Args:
        run_name: the name of the run.

    Returns:
        The string with the default name.
    """
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return run_name + f"_{current_time}"
