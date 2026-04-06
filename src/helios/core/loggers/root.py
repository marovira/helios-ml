import logging
import pathlib
import typing

from ..distributed import get_global_rank
from .base import Logger, get_default_log_name


class RootLogger(Logger):
    """
    Logger used to log while training, optionally to a file.

    The log file will be placed under ``log_root``. When resuming, the original file is
    loaded and new output is appended to the file.

    Args:
        capture_warnings: if ``True``, output of ``warnings.warn`` is captured
            in the log.
    """

    def __init__(self, capture_warnings: bool) -> None:
        """Create the root logger with stream output as default."""
        logging.captureWarnings(capture_warnings)
        self._logger = logging.getLogger("helios")
        self._rank = get_global_rank()
        self._format_str = "[%(asctime)s] [%(levelname)s]: %(message)s"
        self._log_file: pathlib.Path | None = None
        self._saved_log_file: pathlib.Path | None = None

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(self._format_str))
        stream_handler.setLevel(logging.WARNING)
        self._logger.addHandler(stream_handler)
        self._logger.propagate = False

        if self._rank != 0:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.INFO)

    def setup(
        self, run_name: str, log_root: pathlib.Path | None, is_resume: bool
    ) -> None:
        """
        Finish configuring the root logger.

        In particular, this function will create the file logger provided that the input
        path is not ``None``. If the path points to a file that already exists, then the
        logger will automatically append to the file, otherwise a new file will be
        created.

        Args:
            run_name: the name of the current run.
            log_root: root directory for log files.  ``None`` disables file
                logging.
            is_resume: ``True`` when continuing a previous run.
        """
        if log_root is None:
            return

        if is_resume and self._saved_log_file is not None:
            log_file = self._saved_log_file
        else:
            base_name = get_default_log_name(run_name)
            log_file = log_root / f"{base_name}.log"

        mode = "a" if log_file.exists() else "w"
        file_handler = logging.FileHandler(str(log_file), mode=mode, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(self._format_str))
        file_handler.setLevel(logging.INFO)
        self._logger.addHandler(file_handler)
        self._log_file = log_file

    @property
    def logger(self) -> logging.Logger:
        """Return the logger instance."""
        return self._logger

    @property
    def log_file(self) -> pathlib.Path | None:
        """
        Return the path to the current log file.

        If the path for the log file was originally ``None``, this will return ``None``
        as well.
        """
        return self._log_file

    def info(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log using the ``INFO`` tag.

        Only available for the main process in distributed training.

        Args:
            msg: the message to log.
            kwargs: keyword arguments to :py:meth:`logging.Logger.info`.
        """
        if self._rank != 0:
            return
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log using the ``WARNING`` tag.

        Args:
            msg: the message to log.
            kwargs: keyword arguments to :py:meth:`logging.Logger.warning`.
        """
        self._logger.warning(msg)

    def error(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log using the ``ERROR`` tag.

        Args:
            msg: the message to log.
            kwargs: keyword arguments to :py:meth:`logging.Logger.error`.
        """
        self._logger.error(msg)

    def exception(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log an exception.

        Args:
            msg: the message to log.
            kwargs: keyword arguments to :py:meth:`logging.Logger.exception`.
        """
        self._logger.exception(msg, **kwargs)

    def flush(self) -> None:
        """Flush all handlers."""
        for handler in self._logger.handlers:
            handler.flush()

    def close(self) -> None:
        """Close all handlers."""
        for handler in self._logger.handlers:
            handler.close()

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Return a dictionary containing the logger state.

        The state will be saved under a key called ``"log_file"`` holding the current
        log-file path (if using). If file logging is disabled, ``None`` is stored instead.

        Returns:
            A dictionary with the logger state.
        """
        return {"log_file": self._log_file}

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Restore the logger state from a previously saved dictionary.

        Args:
            state_dict: the state dictionary returned by :py:meth:`state_dict`.
        """
        self._saved_log_file = state_dict.get("log_file")
