import logging
import pathlib
import typing
from datetime import datetime

import numpy.typing as npt
import torch
import torch.utils.tensorboard as tb

from .distributed import get_rank


class RootLogger:
    """
    Logger used to log while training to a file.

    The log file will be placed in the logs folder as determined by the given path.

    Args:
        log_file (pathlib.Path): path to the log file.
    """

    def __init__(self, log_file: pathlib.Path | None = None):
        """Create the file logger with the given log file."""
        self._logger = logging.getLogger("pyro")
        self._rank = get_rank()

        format_str = "[%(asctime)s] [%(levelname)s]: %(message)s"
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(format_str))
        self._logger.addHandler(stream_handler)
        self._logger.propagate = False

        if self._rank != 0:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.INFO)

        if log_file is not None:
            file_handler = logging.FileHandler(str(log_file), "w")
            file_handler.setFormatter(logging.Formatter(format_str))
            self._logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        """
        Log using the INFO tag.

        Only available for the main process in distributed training.

        Args:
            msg (str): the message to log
        """
        if self._rank != 0:
            return

        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """
        Log using the WARNING tag.

        Args:
            msg (str): the message to log
        """
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """
        Log using the ERROR tag.

        Args:
            msg (str): the message to log
        """
        self._logger.error(msg)

    def flush(self) -> None:
        """Flush the log."""
        for handler in self._logger.handlers:
            handler.flush()

    def close(self) -> None:
        """Close the log."""
        for handler in self._logger.handlers:
            handler.close()


class TensorboardWriter:
    """
    Wrapper for the Tensorboard writer.

    Args:
        run_path (Path): path to the directory for the run.
    """

    def __init__(
        self,
        run_path: pathlib.Path,
    ):
        """Create the Tensorboard writer."""
        rank = get_rank()
        self._writer: tb.SummaryWriter | None = None

        if rank == 0:
            self._writer = tb.SummaryWriter(log_dir=str(run_path), flush_secs=20)

    def add_scalar(
        self, tag: str, scalar_value: float | str, global_step: int | None = None
    ) -> None:
        """
        Add scalar data to the log.

        Args:
            tag (str): name of the scalar to plot.
            scalar_value (float | str): the scalar to plot.
            global_step (int | None): the step for the given scalar.
        """
        if self._writer is None:
            return

        self._writer.add_scalar(tag, scalar_value, global_step, new_style=True)

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: dict[str, float], global_step: int | None
    ) -> None:
        """
        Add multiple scalars to the log.

        Args:
            main_tag (str): the parent name for the tags
            tag_scalar_dict (dict[str, float]): key-value pair storing tag and
            corresponding values.
            global_step (int | None): global step value to record.
        """
        if self._writer is None:
            return

        self._writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_image(
        self,
        tag: str,
        img_tensor: torch.Tensor | npt.NDArray,
        global_step: int | None = None,
        dataformats: str = "CHW",
    ) -> None:
        """
        Add image data to log.

        Args:
            tag (str): data identifier.
            img_tensor (torch.Tensor | np.ndarray): image data.
            global_step (int): global step value to record
            dataformats (str): image data format specification in the form CHW, HWC, HW,
            WH, etc.
        """
        if self._writer is None:
            return

        self._writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)

    def add_images(
        self,
        tag: str,
        img_tensor: torch.Tensor | npt.NDArray,
        global_step: int | None = None,
        dataformats: str = "NCHW",
    ) -> None:
        """
        Add batched image data to log.

        Args:
            tag (str): data identifier.
            img_tensor (torch.Tensor | np.ndarray): image data.
            global_step (int): global step value to record.
            dataformats (str): image data format specification in the form NCHW, NHWC,
            CHW, HWC, HW, WH, etc.
        """
        if self._writer is None:
            return

        self._writer.add_images(tag, img_tensor, global_step, dataformats=dataformats)

    def add_text(self, tag: str, text_string: str, global_step: int | None = None):
        """
        Add text data to log.

        Args:
            tag (str): data identifier.
            text_string (str): string to save.
            global_step (int): global step value to record.
        """
        if self._writer is None:
            return

        self._writer.add_text(tag, text_string, global_step)

    def add_graph(
        self,
        model: torch.nn.Module,
        input_to_model: torch.Tensor | list[torch.Tensor] | None = None,
        verbose: bool = False,
        use_strict_trace: bool = True,
    ) -> None:
        """
        Add graph data to log.

        Args:
            model (torch.nn.Module): model to draw.
            input_to_model(Optional[torch.Tensor | list[torch.Tensor]]): a variable
            or tuple of variables to be fed.
            verbose (bool): whether to print graph structure in console.
            use_strict_trace (bool): whether to pass keyword argument strict to
            torch.jit.trace.
        """
        if self._writer is None:
            return

        self._writer.add_graph(model, input_to_model, verbose, use_strict_trace)

    def add_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor | npt.NDArray,
        predictions: torch.Tensor | npt.NDArray,
        global_step: int | None = None,
        num_thresholds: int = 127,
    ) -> None:
        """
        Add a precision recall curve to the log.

        Plotting a precision-recall curve lets you understand your model's performance
        under different threshold settings. With this function, you provide the ground
        truth labeling (T/F) and prediction confidence (usually the output of your model)
        for each target. The TensorBoard UI will let you choose the threshold
        interactively.

        Args:
            tag (str): data identifier.
            labels (torch.Tensor | np.ndarray): ground truth data. Binary label for
            each element.
            predictions (torch.Tensor | np.ndarray): the probability that an element
            be classified as true. Value should be in [0, 1].
            global_step (Optional[int]): global step value to record.
            num_thresholds (int): number of thresholds used to draw the curve.
        """
        if self._writer is None:
            return

        self._writer.add_pr_curve(tag, labels, predictions, global_step, num_thresholds)

    def add_hparams(
        self,
        hparam_dict: dict[str, typing.Any],
        metric_dict: dict[str, typing.Any],
        hparam_domain_discrete: dict[str, list[typing.Any]] | None = None,
        run_name: str | None = None,
        global_step: int | None = None,
    ) -> None:
        """
        Add a set of hyper-parameters to be compared in the log.

        Args:
            hparam_dict (dict[str, Any]): each key-value pair in the dictionary is the
            name of the hyper-parameter and it's corresponding value. The type of the
            value can be one of bool, string, float, int, or None.
            metric_dict (dict[str, Any]): each key-value pair in the dictionary is the
            name of the metric and it's corresponding value. Note that the key used here
            should be unique in the Tensorboard record.
            hparam_domain_discrete: (Optional[dict[str, list[Any]]]): a dictionary that
            contains names of hyper-parameters and all the discrete values they can hold.
            run_name (str): name of the run, to be included as part of the logdir.
            global_step (Optional[int]): global step value to record.
        """
        if self._writer is None:
            return

        self._writer.add_hparams(
            hparam_dict, metric_dict, hparam_domain_discrete, run_name, global_step
        )

    def flush(self) -> None:
        """Flush any cached values to Tensorboard."""
        if self._writer is None:
            return

        self._writer.flush()

    def close(self) -> None:
        """Close the Tensorboard writer."""
        if self._writer is None:
            return

        self._writer.close()


ACTIVE_LOGGERS: dict[str, RootLogger | TensorboardWriter] = {}


def get_default_log_name(run_name: str) -> str:
    """
    Generate the default name used for log files or Tensorboard runs.

    Args:
        run_name (str): the name of the run.
    Returns:
        str: the string with the default name.
    """
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")

    return run_name + f"_{current_time}"


def create_default_loggers(
    run_name: str,
    logs_root: pathlib.Path,
    enable_tensorboard: bool = True,
    enable_file_logging: bool = False,
) -> None:
    """
    Construct the base loggers with the given paths (if any).

    Logs can be disabled by passing None as the corresponding root.

    Args:
        run_name (str): the name of the current run.
        runs_root (pathlib.Path | None): the path to store the Tensorboard logs.
        logs_root (pathlib.Path | None): the path to store

    """
    base_name = get_default_log_name(run_name)
    runs_root = logs_root / "runs"

    if enable_tensorboard and "tensorboard" not in ACTIVE_LOGGERS:
        run_path = runs_root / base_name
        ACTIVE_LOGGERS["tensorboard"] = TensorboardWriter(run_path)

    if "root" not in ACTIVE_LOGGERS:
        log_file = logs_root / f"{base_name}.log" if enable_file_logging else None
        ACTIVE_LOGGERS["root"] = RootLogger(log_file)


def flush_default_loggers() -> None:
    """Flushes the default loggers."""
    for _, logger in ACTIVE_LOGGERS.items():
        logger.flush()


def close_default_loggers() -> None:
    """Close the default loggers and clears them."""
    for _, logger in ACTIVE_LOGGERS.items():
        logger.close()

    ACTIVE_LOGGERS.clear()


def get_root_logger() -> RootLogger:
    """
    Get the root logger instance.

    Return:
        FileLogger: the root logger.
    """
    if "root" not in ACTIVE_LOGGERS:
        raise KeyError(
            "error: root logger has not been created. Did you forget to call "
            "create_base_loggers?"
        )
    return typing.cast(RootLogger, ACTIVE_LOGGERS["root"])


def get_tensorboard_writer() -> TensorboardWriter:
    """
    Return the Tensorboard writter.

    Return:
        TensorboardWriter: the Tensorboard logger.
    """
    if "tensorboard" not in ACTIVE_LOGGERS:
        raise KeyError(
            "error: tensorboard logger has not been created. Did you forget to call "
            "create_base_loggers?"
        )
    return typing.cast(TensorboardWriter, ACTIVE_LOGGERS["tensorboard"])
