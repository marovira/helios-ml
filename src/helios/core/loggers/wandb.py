import pathlib
import typing

try:
    import wandb as wandb_sdk

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from ..distributed import get_global_rank
from .base import Logger


class WandbArgs(typing.TypedDict, total=False):
    """
    Arguments for constructing a :py:class:`WandbWriter`.

    ``project`` is the only required key; all others are optional.

    Keys:
        project: W&B project name.
        name: display name for the run shown in the W&B UI. If not provided, defaults to
            ``run_name`` provided by :py:meth:`~WandbWriter.setup`.
        config: hyper-parameter dictionary to associate with the run.
        extra_args: additional keyword arguments forwarded verbatim to
            :func:`wandb.init`.
    """

    project: typing.Required[str]
    name: str
    config: dict[str, typing.Any]
    extra_args: dict[str, typing.Any]


class WandbWriter(Logger):
    """
    Wrapper for the Weights & Biases ``wandb.init`` run.

    Data for the logger will be placed under ``log_root/wandb``.  When
    resuming, the original run ID is restored and new data is appended to it.

    Requires the ``wandb`` package.  Install it with::

        pip install wandb

    Args:
        project: W&B project name.
        name: optional display name for the run.
        config: optional hyper-parameter configuration dictionary to associate
            with the run.
        extra_args: optional extra keyword arguments forwarded verbatim to
            :func:`wandb.init`.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, typing.Any] | None = None,
        extra_args: dict[str, typing.Any] | None = None,
    ) -> None:
        """
        Create the W&B writer.

        Args:
            project: W&B project name.
            name: optional display name for the run.
            config: optional hyper-parameter configuration dictionary.
            extra_args: optional extra keyword arguments forwarded to
                :func:`wandb.init`.

        Raises:
            ImportError: if ``wandb`` is not installed.
        """
        if not _WANDB_AVAILABLE:
            raise ImportError(
                "wandb is required to use the WandbWriter. "
                "Install it with: pip install wandb"
            )
        self._project = project
        self._name = name
        self._config = config
        self._extra_args = extra_args if extra_args is not None else {}
        self._rank = get_global_rank()
        self._run: typing.Any = None
        self._run_id: str | None = None
        self._saved_run_id: str | None = None

    def setup(
        self, run_name: str, log_root: pathlib.Path | None, is_resume: bool
    ) -> None:
        """
        Finish configuring the ``WandbWriter``.

        In particular, this function will call :func:`wandb.init`. If a run ID was
        previously saved, then it will be forwarded to W&B so the run continues in place.
        In distributed training, the writer will only be created on rank 0.

        Args:
            run_name: the name of the current run; used as the W&B run name
                when no explicit ``name`` was given in ``__init__``.
            log_root: root directory for logs.  W&B data will be written
                under ``log_root/wandb/``.  ``None`` lets W&B choose its own
                default directory.
            is_resume: ``True`` when continuing a previous run.
        """
        if self._rank != 0:
            return

        log_dir: str | None = None
        if log_root is not None:
            wandb_root = log_root / "wandb"
            wandb_root.mkdir(parents=True, exist_ok=True)
            log_dir = str(wandb_root)

        run_id = (
            self._saved_run_id if is_resume and self._saved_run_id is not None else None
        )
        display_name = self._name if self._name is not None else run_name

        self._run = wandb_sdk.init(
            project=self._project,
            name=display_name,
            config=self._config,
            dir=log_dir,
            id=run_id,
            resume="allow" if run_id is not None else None,
            **self._extra_args,
        )
        self._run_id = self._run.id

    def flush(self) -> None:
        """No-op, W&B syncs its own data automatically."""

    def close(self) -> None:
        """Finish and close the W&B run."""
        if self._run is None:
            return
        self._run.finish()
        self._run = None

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Return a dictionary containing the writer state.

        The state will be saved under a key called ``"run_id"`` holding the current run
        ID. If W&B was disabled, then ``None`` is stored.

        Returns:
            A dictionary with the logger state.
        """
        return {"run_id": self._run_id}

    def load_state_dict(self, state_dict: dict[str, typing.Any]) -> None:
        """
        Restore the writer state from a previously saved dictionary.

        Args:
            state_dict: the state dictionary returned by :py:meth:`state_dict`.
        """
        self._saved_run_id = state_dict.get("run_id")
