import copy
import pathlib
import typing

import matplotlib.pyplot as plt
import numpy.typing as npt
import torch

try:
    import torch.utils.tensorboard as tb

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

from ..distributed import get_global_rank
from ..utils import get_from_optional
from .base import Logger, get_default_log_name


class TensorboardWriter(Logger):
    """
    Wrapper for the Tensorboard ``SummaryWriter``.

    Data for the logger will be placed under ``log_root/tensorboard``. When resuming, the
    original run directory is restored and new data is appended to it.

    Requires the ``tensorboard`` package.  Install it with::

        pip install tensorboard
    """

    def __init__(self) -> None:
        """
        Create the Tensorboard writer.

        Raises:
            ImportError: if ``tensorboard`` is not installed.
        """
        if not _TENSORBOARD_AVAILABLE:
            raise ImportError(
                "tensorboard is required to use the TensorboardWriter. "
                "Install it with: pip install tensorboard"
            )
        self._rank = get_global_rank()
        self._writer: typing.Any = None
        self._run_path: pathlib.Path | None = None
        self._saved_run_path: pathlib.Path | None = None

    def setup(
        self, run_name: str, log_root: pathlib.Path | None, is_resume: bool
    ) -> None:
        """
        Finish configuring the ``TensorboardWriter``.

        In particular, this function will create the writer instance and assign it to the
        given path. If the path already exists, Tensorboard will automatically append to
        the previous run. In distributed training, the writer will only be created on rank
        0.

        Args:
            run_name: the name of the current run.
            log_root: root directory for logs.  ``None`` disables Tensorboard.
            is_resume: ``True`` when continuing a previous run.
        """
        if log_root is None:
            return

        if is_resume and self._saved_run_path is not None:
            run_path = self._saved_run_path
        else:
            tb_root = log_root / "tensorboard"
            tb_root.mkdir(parents=True, exist_ok=True)
            base_name = get_default_log_name(run_name)
            run_path = tb_root / base_name

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
            main_tag: the parent name for the tags.
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
        Add image data to the log.

        Args:
            tag: data identifier.
            img_tensor: image data.
            global_step: (optional) global step value to record.
            dataformats: image data format specification in the form CHW, HWC,
                HW, WH, etc.
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
        Add batched image data to the log.

        Args:
            tag: data identifier.
            img_tensor: image data.
            global_step: (optional) global step value to record.
            dataformats: image data format specification in the form NCHW, NHWC,
                CHW, HWC, HW, WH, etc.
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
        Render a matplotlib figure into an image and add it to a summary.

        Args:
            tag: data identifier.
            figure: figure or a list of figures.
            global_step: (optional) global step value to record.
            close: flag to automatically close the figure.
        """
        if self._writer is None:
            return
        self._writer.add_figure(tag, figure, global_step, close)

    def add_text(
        self, tag: str, text_string: str, global_step: int | None = None
    ) -> None:
        """
        Add text data to the log.

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
        Add graph data to the log.

        PyTorch currently supports CPU tracing only, so the model and its
        input(s) will be automatically moved to the CPU prior to logging.  The
        model is copied first so the caller's model is not affected.

        Args:
            model: model to draw.
            input_to_model: a variable or tuple of variables to be fed.
            verbose: whether to print graph structure in console.
            use_strict_trace: whether to pass keyword argument strict to
                ``torch.jit.trace``.
        """
        if self._writer is None:
            return

        model_cpu = copy.deepcopy(model).cpu()

        input_cpu: torch.Tensor | list[torch.Tensor] | None
        if isinstance(input_to_model, torch.Tensor):
            input_cpu = input_to_model.cpu()
        elif isinstance(input_to_model, list):
            input_cpu = [x.cpu() for x in input_to_model]
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
        Add a precision-recall curve to the log.

        Args:
            tag: data identifier.
            labels: ground truth data.  Binary label for each element.
            predictions: the probability that an element be classified as true.
                Value should be in [0, 1].
            global_step: (optional) global step value to record.
            num_thresholds: number of thresholds used to draw the curve.
                Defaults to 127.
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
            hparam_dict: each key-value pair is the name of a hyper-parameter
                and its value.  Valid types are ``bool``, ``str``, ``float``,
                ``int``, or ``None``.
            metric_dict: each key-value pair is the name of a metric and its
                value.  Keys must be unique in the Tensorboard record.
            hparam_domain_discrete: a dictionary containing hyper-parameter
                names and all the discrete values they can hold.
            run_name: name of the run, to be included as part of the log dir.
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

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Return a dictionary containing the writer state.

        The state will be saved under a key called ``"run_path"`` holding the current run
        folder. If Tensorboard was disabled, then ``None`` is stored instead.

        Returns:
            A dictionary with the logger state.
        """
        return {"run_path": self._run_path}

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Restore the writer state from a previously saved dictionary.

        Args:
            state_dict: the state dictionary returned by :py:meth:`state_dict`.
        """
        self._saved_run_path = state_dict.get("run_path")
