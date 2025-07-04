# NOTE: when we release helios 2.0, this module should be renamed to loggers.py

import copy
import logging
import pathlib
import typing
from datetime import datetime

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch
import torch.utils.tensorboard as tb

from .distributed import get_global_rank
from .utils import get_from_optional


class RootLogger:
    """
    Logger used to log while training to a file.

    The log file will be placed in the logs folder as determined by the given path.

    Args:
        log_file: path to the log file.
    """

    def __init__(self: "RootLogger"):
        """Create the root logger with stream output as default."""
        self._logger = logging.getLogger("helios")
        self._rank = get_global_rank()
        self._format_str = "[%(asctime)s] [%(levelname)s]: %(message)s"
        self._log_file: pathlib.Path | None = None

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(self._format_str))
        stream_handler.setLevel(logging.WARNING)
        self._logger.addHandler(stream_handler)
        self._logger.propagate = False

        if self._rank != 0:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.INFO)

    def setup(self, log_file: pathlib.Path | None = None) -> None:
        """
        Finish configuring the root logger.

        In particular, this function will create the file logger provided that the input
        path is not ``None``. If the path points to a file that already exists, then the
        logger will automatically append to the file, otherwise a new file will be
        created.

        Args:
            log_file: the (optional) path to the log file.
        """
        if log_file is None:
            return

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
            msg: the message to log
            kwargs: keyword arguments to logging.info
        """
        if self._rank != 0:
            return

        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log using the ``WARNING`` tag.

        Args:
            msg: the message to log
            kwargs: keyword arguments to logging.warning
        """
        self._logger.warning(msg)

    def error(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log using the ``ERROR`` tag.

        Args:
            msg (str): the message to log
            kwargs (Any): keyword arguments to logging.error
        """
        self._logger.error(msg)

    def exception(self, msg: str, **kwargs: typing.Any) -> None:
        """
        Log an exception.

        Args:
            msg: the message to log
            kwargs: keyword arguments to ``logging.exception``
        """
        self._logger.exception(msg, **kwargs)

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
    Wrapper for the ``Tensorboard`` writer.

    Args:
        run_path: path to the directory for the run.
    """

    def __init__(self: "TensorboardWriter"):
        """Create the ``Tensorboard`` writer."""
        self._rank = get_global_rank()
        self._writer: tb.SummaryWriter | None = None
        self._run_path: pathlib.Path | None = None

    def setup(self, run_path: pathlib.Path) -> None:
        """
        Finish configuring the ``TensorboardWriter``.

        In particular, this function will create the writer instance and assign it to the
        given path. If the path already exists, Tensorboard will automatically append to
        the previous run. In distributed training, the writer will only be created on rank
        0.

        Args:
            run_path:: the path to the run folder.
        """
        if self._rank == 0:
            self._writer = tb.SummaryWriter(log_dir=str(run_path), flush_secs=20)
            self._run_path = run_path

    @property
    def run_path(self) -> pathlib.Path:
        """Return the path to the current run folder."""
        return get_from_optional(self._run_path)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float | str | torch.Tensor,
        global_step: int | None = None,
    ) -> None:
        """
        Add scalar data to the log.

        Args:
            tag: name of the scalar to plot.
            scalar_value: the scalar to plot.
            global_step: the (optional) step for the given scalar.
        """
        if self._writer is None:
            return

        self._writer.add_scalar(tag, scalar_value, global_step, new_style=True)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float | torch.Tensor],
        global_step: int | None,
    ) -> None:
        """
        Add multiple scalars to the log.

        Args:
            main_tag: the parent name for the tags
            tag_scalar_dict: key-value pair storing tag and corresponding values.
            global_step: (optional) global step value to record.
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
            tag: data identifier.
            img_tensor: image data.
            global_step: (optional) global step value to record
            dataformats: image data format specification in the form CHW, HWC, HW, WH,
                etc.
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
            tag: data identifier.
            img_tensor: image data.
            global_step: (optional) global step value to record.
            dataformats: image data format specification in the form NCHW, NHWC, CHW,
                HWC, HW, WH, etc.
        """
        if self._writer is None:
            return

        self._writer.add_images(tag, img_tensor, global_step, dataformats=dataformats)

    def add_figure(
        self,
        tag: str,
        figure: plt.Figure | list[plt.Figure],
        global_step: int | None = None,
        close: bool = True,
    ) -> None:
        """
        Render matplotlib figure into an image and add it to a summary.

        Args:
            tag: data identifier.
            figure: figure or a list of figures.
            global_step: (optional) global step value to record.
            close: flag to automatically close the figure.
        """
        if self._writer is None:
            return

        self._writer.add_figure(tag, figure, global_step, close)

    def add_text(self, tag: str, text_string: str, global_step: int | None = None):
        """
        Add text data to log.

        Args:
            tag: data identifier.
            text_string: string to save.
            global_step: (optional) global step value to record.
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

        PyTorch currently supports CPU tracing only, so the model and its input(s) will be
        automatically moved to the CPU prior to logging. Note that the model will be
        copied prior to moving to the CPU to ensure the input model is not affected.

        Args:
            model: model to draw.
            input_to_mode: a variable or tuple of variables to be fed.
            verbose: whether to print graph structure in console.
            use_strict_trace: whether to pass keyword argument strict to torch.jit.trace.
        """
        if self._writer is None:
            return

        # Transfer graph + input to the CPU for tracing.
        model_cpu = copy.deepcopy(model).cpu()

        input_cpu: torch.Tensor | list[torch.Tensor] | None
        if isinstance(input_to_model, torch.Tensor):
            input_cpu = input_to_model.cpu()
        elif isinstance(input_to_model, list):
            input_cpu = []
            for x in input_to_model:
                input_cpu.append(x.cpu())
        else:
            input_cpu = None

        self._writer.add_graph(model_cpu, input_cpu, verbose, use_strict_trace)

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
        truth labelling (T/F) and prediction confidence (usually the output of your model)
        for each target. The TensorBoard UI will let you choose the threshold
        interactively.

        Args:
            tag: data identifier.
            labels: ground truth data. Binary label for each element.
            predictions: the probability that an element be classified as true. Value
                should be in [0, 1].
            global_step: (optional) global step value to record.
            num_thresholds: number of thresholds used to draw the curve. Defaults to 127.
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
            hparam_dict: each key-value pair in the dictionary is the name of the
                hyper-parameter and it's corresponding value. The type of the value can
                be one of ``bool``, ``string``, ``float``, ``int``, or ``None``.
            metric_dict: each key-value pair in the dictionary is the name of the metric
                and it's corresponding value. Note that the key used here should be
                unique in the Tensorboard record.
            hparam_domain_discrete:: a dictionary that contains names of hyper-parameters
                and all the discrete values they can hold.
            run_name: name of the run, to be included as part of the logdir.
            global_step: (optional) global step value to record.
        """
        if self._writer is None:
            return

        if run_name is None:
            run_name = "."

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


_ACTIVE_LOGGERS: dict[str, RootLogger | TensorboardWriter] = {}


def get_default_log_name(run_name: str) -> str:
    """
    Generate the default name used for log files or Tensorboard runs.

    Args:
        run_name: the name of the run.

    Returns:
        The string with the default name.
    """
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")

    return run_name + f"_{current_time}"


def create_default_loggers(enable_tensorboard: bool = True) -> None:
    """
    Construct the ``RootLogger`` and ``TensorboardWriter`` instances.

    In distributed training, this function should be called AFTER the processes have been
    created to ensure each process gets a copy of the loggers.

    Args:
        enable_tensorboard: (optional) enables the tensorboard writer. Defaults to true.
    """
    # Root logger is always created.
    if "root" not in _ACTIVE_LOGGERS:
        _ACTIVE_LOGGERS["root"] = RootLogger()

    if enable_tensorboard and "tensorboard" not in _ACTIVE_LOGGERS:
        _ACTIVE_LOGGERS["tensorboard"] = TensorboardWriter()


def restore_default_loggers(
    log_path: pathlib.Path | None = None, run_path: pathlib.Path | None = None
) -> None:
    """
    Restore the default loggers from a previous run.

    This function should be called whenever the loggers need to continue logging to the
    same file/folder as a previous run.

    Args:
        log_path: (optional) path to the log file.
        run_path: (optional) path to the Tensorboard run folder.

    Raises:
        RuntimeError: if the root logger hasn't been created.
    """
    if "root" not in _ACTIVE_LOGGERS:
        raise RuntimeError(
            "error: root logger hasn't been created. Did you forget to call "
            "create_default_loggers?"
        )

    if log_path is not None:
        _ACTIVE_LOGGERS["root"].setup(log_path)

    if "tensorboard" in _ACTIVE_LOGGERS and run_path is not None:
        _ACTIVE_LOGGERS["tensorboard"].setup(run_path)


def setup_default_loggers(
    run_name: str,
    log_root: pathlib.Path | None = None,
    runs_root: pathlib.Path | None = None,
) -> None:
    """
    Call the setup functions on the default loggers.

    This function should be called when the loggers don't need to continue from a previous
    run. If you need that, call restore_default_loggers instead.

    Args:
        run_name: the name of the current run.
        log_root: (optional) path to the logs folder.
        runs_root: (optional) path to the Tensorboard runs folder.
    """
    base_name = get_default_log_name(run_name)
    run_path = None if runs_root is None else runs_root / base_name
    log_path = None if log_root is None else log_root / f"{base_name}.log"
    restore_default_loggers(log_path, run_path)


def flush_default_loggers() -> None:
    """Flushes the default loggers."""
    for _, logger in _ACTIVE_LOGGERS.items():
        logger.flush()


def close_default_loggers() -> None:
    """Close the default loggers and clears them."""
    for _, logger in _ACTIVE_LOGGERS.items():
        logger.close()

    _ACTIVE_LOGGERS.clear()


def get_root_logger() -> RootLogger:
    """
    Get the root logger instance.

    Return:
        The root logger.
    """
    if "root" not in _ACTIVE_LOGGERS:
        raise KeyError(
            "error: root logger has not been created. Did you forget to call "
            "create_base_loggers?"
        )
    return typing.cast(RootLogger, _ACTIVE_LOGGERS["root"])


def is_root_logger_active() -> bool:
    """Check if the root logger has been created."""
    return "root" in _ACTIVE_LOGGERS


def get_tensorboard_writer() -> TensorboardWriter | None:
    """
    Return the Tensorboard writter.

    If Tensorboard is disabled, this function will return ``None``

    Return:
        The Tensorboard logger or ``None`` if it doesn't exist.
    """
    if "tensorboard" not in _ACTIVE_LOGGERS:
        return None
    return typing.cast(TensorboardWriter, _ACTIVE_LOGGERS["tensorboard"])
