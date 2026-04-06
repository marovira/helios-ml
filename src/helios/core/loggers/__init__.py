"""
Logging sub-package for Helios.

Holds all of the logger classes.
"""

import enum
import pathlib
import typing

from .base import Logger, get_default_log_name
from .root import RootLogger
from .tensorboard import TensorboardWriter
from .wandb import WandbArgs, WandbWriter


class LoggerType(enum.Enum):
    """Defines the types of loggers."""

    ROOT = "root"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


_ACTIVE_LOGGERS: dict[LoggerType, Logger] = {}


def create_loggers(
    enable_tensorboard: bool = True,
    capture_warnings: bool = True,
    wandb_args: WandbArgs | None = None,
) -> None:
    """
    Construct the logger instances and add them to the active table.

    The :py:class:`RootLogger` is always created, while additional loggers are only crated
    when their corresponding flag is ``True``. If a logger has already been created then
    this function does nothing, making it safe to call multiple times.

    In distributed training, this function should be called *after* the processes have
    been created to ensure each process gets a copy of the loggers.

    Args:
        enable_tensorboard: enable the Tensorboard writer.  Defaults to
            ``True``.
        capture_warnings: if ``True``, :py:func:`warnings.warn` output is
            captured by the root logger.  Defaults to ``True``.
        wandb_writer: an already-constructed :py:class:`WandbWriter` instance to
            register, or ``None`` to skip W&B logging.  Defaults to ``None``.
    """
    if LoggerType.ROOT not in _ACTIVE_LOGGERS:
        _ACTIVE_LOGGERS[LoggerType.ROOT] = RootLogger(capture_warnings=capture_warnings)

    if enable_tensorboard and LoggerType.TENSORBOARD not in _ACTIVE_LOGGERS:
        _ACTIVE_LOGGERS[LoggerType.TENSORBOARD] = TensorboardWriter()

    if wandb_args is not None and LoggerType.WANDB not in _ACTIVE_LOGGERS:
        _ACTIVE_LOGGERS[LoggerType.WANDB] = WandbWriter(**wandb_args)


def setup_loggers(
    run_name: str,
    log_root: pathlib.Path | None = None,
) -> None:
    """
    Call :py:meth:`~Logger.setup` on every active logger for a fresh run.

    This function should be called when the loggers don't need to continue from a previous
    run. If you need that, call :py:func:`restore_loggers` instead.

    Args:
        run_name: the name of the current run.
        log_root: root directory under which each logger will create its own
            subfolder.  ``None`` disables on-disk output.
    """
    for logger in _ACTIVE_LOGGERS.values():
        logger.setup(run_name, log_root, is_resume=False)


def restore_loggers(
    run_name: str,
    log_root: pathlib.Path | None = None,
    loggers_state: dict[str, dict[str, typing.Any]] | None = None,
) -> None:
    """
    Restore active loggers from a previous run.

    For each active logger whose name appears in the ``loggers_state`` dictionary:
        1. Call :py:meth:`~Logger.load_state_dict` so that their previous state is loaded.
        1. Call :py:meth:`~Logger.setup` so the loggers re-use the original paths.
    If an active logger does not have an entry in the dictionary, then it is configured to
    start fresh.

    Args:
        run_name: the name of the current run.
        log_root: root directory under which each logger will look for its
            subfolder.  ``None`` disables on-disk output.
        loggers_state: mapping of ``{logger_name: state_dict}`` as returned
            by a prior call to :py:func:`get_logger_state_dicts`.  ``None``
            is treated the same as an empty mapping.
    """
    if loggers_state is None:
        loggers_state = {}

    for logger_type, logger in _ACTIVE_LOGGERS.items():
        key = logger_type.value
        if key in loggers_state:
            logger.load_state_dict(loggers_state[key])
            logger.setup(run_name, log_root, is_resume=True)
        else:
            logger.setup(run_name, log_root, is_resume=False)


def get_logger(name: LoggerType) -> Logger:
    """
    Return the active logger identified by *name*.

    Args:
        name: the :py:class:`LoggerType` value identifying the desired logger.

    Returns:
        The requested :py:class:`Logger` instance.

    Raises:
        KeyError: if the requested logger has not been created.
    """
    if name not in _ACTIVE_LOGGERS:
        raise KeyError(
            f"error: logger '{name.value}' has not been created. "
            "Did you forget to call create_default_loggers?"
        )
    return _ACTIVE_LOGGERS[name]


def get_logger_state_dicts() -> dict[str, dict[str, typing.Any]]:
    """
    Return the state dictionaries of all active loggers.

    Returns:
        The dictionary containing the state of all active loggers.
    """
    return {
        logger_type.value: logger.state_dict()
        for logger_type, logger in _ACTIVE_LOGGERS.items()
    }


def flush_loggers() -> None:
    """Flush all active loggers."""
    for logger in _ACTIVE_LOGGERS.values():
        logger.flush()


def close_loggers() -> None:
    """Close all active loggers and remove them from the active table."""
    for logger in _ACTIVE_LOGGERS.values():
        logger.close()
    _ACTIVE_LOGGERS.clear()


def is_root_logger_active() -> bool:
    """Return ``True`` if the root logger has been created."""
    return LoggerType.ROOT in _ACTIVE_LOGGERS


def get_root_logger() -> RootLogger:
    """
    Get the root logger instance.

    Returns:
        The root logger.

    Raises:
        KeyError: if the root logger has not been created.
    """
    return typing.cast(RootLogger, get_logger(LoggerType.ROOT))


def get_tensorboard_writer() -> TensorboardWriter | None:
    """
    Return the Tensorboard writter.

    If Tensorboard is disabled, this function will return ``None``

    Returns:
        The Tensorboard logger, or ``None`` if it doesn't exist.
    """
    if LoggerType.TENSORBOARD not in _ACTIVE_LOGGERS:
        return None
    return typing.cast(TensorboardWriter, _ACTIVE_LOGGERS[LoggerType.TENSORBOARD])


__all__ = [
    "Logger",
    "LoggerType",
    "RootLogger",
    "TensorboardWriter",
    "WandbArgs",
    "WandbWriter",
    "get_default_log_name",
    "create_loggers",
    "setup_loggers",
    "restore_loggers",
    "get_logger",
    "get_logger_state_dicts",
    "flush_loggers",
    "close_loggers",
    "is_root_logger_active",
    "get_root_logger",
    "get_tensorboard_writer",
]
