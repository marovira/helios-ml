from __future__ import annotations

import dataclasses as dc
import enum
import itertools
import os
import pathlib
import re
import time
import typing

import packaging.version as pv
import torch
import torch.multiprocessing as mp
import torch.utils.data as tud
import tqdm

import helios.model as hlm
import helios.plugins as hlp
from helios import core, data
from helios.core import distributed as dist
from helios.core import logging, rng
from helios.data.samplers import ResumableSamplerType

from ._version import __version__


class TrainingUnit(enum.Enum):
    """Defines the types of units for training steps."""

    ITERATION = 0
    EPOCH = 1

    @classmethod
    def from_str(cls, label: str) -> TrainingUnit:
        """
        Convert the given string to the corresponding enum value.

        Must be one of "iteration" or "epoch".

        Args:
            label: the label to convert.

        Returns:
            The corresponding value.

        Raises:
            ValueError: if the given value is not one of "iteration" or "epoch".
        """
        if label == "iteration":
            return cls.ITERATION
        if label == "epoch":
            return cls.EPOCH

        raise ValueError(
            "invalid training unit. Expected one of 'iteration' or 'epoch' but "
            f"received '{label}'"
        )


class _TrainerMode(enum.Enum):
    """
    Defines the types of actions that the trainer can do.

    These values are used internally to figure out which function should be invoked from
    the distributed handler.
    """

    TRAIN = 0
    TEST = 1


@dc.dataclass
class _DistributedErrorState:
    """
    Holds the state necessary to correctly handle errors in distributed training.

    This class is pushed into the internal error handling queue and used by the main
    process when an error occurs to ensure the exception is handled correctly.

    Args:
        log_path: the path used by the root logger (if using).
    """

    log_path: pathlib.Path | None = None


@dc.dataclass
class TrainingState:
    """
    The training state.

    Args:
        current_iteration: the current iteration number. Note that this refers to
            the iteration in which gradients are updated. This may or may not be equal to
            the :py:attr:`global_iteration` count.
        global_iteration: the total iteration count.
        global_epoch: the total epoch count.
        validation_cycles: the number of validation cycles.
        dataset_iter: the current batch index of the dataset. This is reset every epoch.
        early_stop_count: the current number of validation cycles for early stop.
        average_iter_time: average time per iteration.
        running_iter: iteration count in the current validation cycle. Useful for
            computing running averages of loss functions.
    """

    current_iteration: int = 0
    global_iteration: int = 0
    global_epoch: int = 0
    validation_cycles: int = 0
    dataset_iter: int = 0
    early_stop_count: int = 0
    average_iter_time: float = 0
    running_iter: int = 0

    dict = dc.asdict


def get_trainer_safe_types_for_load() -> (
    list[typing.Callable | tuple[typing.Callable, str]]
):
    """
    Return the list of safe types for loading needed by the trainer.

    Returns:
        The list of types that need to be registered for safe loading.
    """
    return [TrainingState, pathlib.PosixPath, pathlib.WindowsPath]


def register_trainer_types_for_safe_load() -> None:
    """Register trainer types for safe loading."""
    torch.serialization.add_safe_globals(get_trainer_safe_types_for_load())


def find_last_checkpoint(root: pathlib.Path | None) -> pathlib.Path | None:
    """
    Find the last saved checkpoint (if available).

    The function assumes that checkpoint names contain ``epoch_<epoch>`` and
    ``iter_<iter>`` in the name, in which case it will return the path to the checkpoint
    with the highest epoch and/or iteration count.

    Args:
        root: the path where the checkpoints are stored.

    Returns:
        The path to the last checkpoint (if any).
    """
    if root is None:
        return None

    epoch = 0
    ite = 0
    last_chkpt = None
    regexp = re.compile(r"epoch_\d+_iter_\d+")
    for path in root.glob("*.pth"):
        if not regexp.search(path.stem):
            continue

        elems = str(path.stem).split("_")

        idx = elems.index("epoch") + 1
        e = int(elems[idx])

        idx = elems.index("iter") + 1
        i = int(elems[idx])

        if e > epoch or i > ite:
            epoch = e
            ite = i
            last_chkpt = path

    return last_chkpt


def _spawn_handler(
    rank: int,
    world_size: int,
    trainer: Trainer,
    datamodule: data.DataModule,
    model: hlm.Model,
    mode: _TrainerMode,
    queue: mp.Queue,
    error_queue: mp.Queue,
) -> None:
    """
    Spawn handler for distributed training.

    Args:
        rank: the rank id of the current process within the work group.
        world_size: number of processes in the work group.
        trainer: the trainer to use.
        datamodule: the datamodule to use.
        model: the model to use.
        mode: determines which operation needs to be performed.
    """
    dist.init_dist(rank=rank, world_size=world_size)

    trainer.model = model
    trainer.datamodule = datamodule
    trainer.rank = rank
    trainer.local_rank = rank
    trainer.queue = queue
    trainer._distributed_error_queue = error_queue  # noqa: SLF001

    try:
        if mode == _TrainerMode.TRAIN:
            trainer._train()  # noqa: SLF001
        elif mode == _TrainerMode.TEST:
            trainer._test()  # noqa: SLF001
    except Exception as e:
        dist.shutdown_dist()
        raise e

    dist.safe_barrier()
    dist.shutdown_dist()


class Trainer:
    """
    Automates the training, validation, and testing code.

    The trainer handles all of the required steps to setup the correct environment
    (including handling distributed training), the training/validation/testing loops, and
    any clean up afterwards.

    Args:
        run_name: name of the current run. Defaults to empty.
        train_unit: the unit used for training. Defaults to
            :py:attr:`TrainingUnit.ITERATION`.
        total_steps: the total number of steps to train for. Defaults to 0.
        valid_frequency: (optional) frequency with which to perform validation.
        chkpt_frequency: (optional) frequency with which to save checkpoints.
        print_frequency: (optional) frequency with which to log.
        accumulation_steps: number of steps for gradient accumulation. Defaults to 1.
        enable_cudnn_benchmark: enable/disable CuDNN benchmark. Defaults to false.
        enable_deterministic: enable/disable PyTorch deterministic. Defaults to false.
        early_stop_cycles: (optional) number of cycles after which training will stop if
            no improvement is seen during validation.
        use_cpu: (optional) if true, CPU will be used.
        gpus: (optional) IDs of GPUs to use.
        random_seed: (optional) the seed to use for RNGs.
        enable_tensorboard: enable/disable Tensorboard logging. Defaults to false.
        enable_file_logging: enable/disable file logging. Defaults to false.
        enable_progress_bar: enable/disable the progress bar(s). Defaults to false.
        chkpt_root: (optional) root folder in which checkpoints will be placed.
        log_path: (optional) root folder in which logs will be saved.
        run_path: (optional) root folder in which Tensorboard runs will be saved.
        src_root: (optional) root folder where the code is located. This is used to
            automatically populate the registries using
            :py:func:`~helios.core.utils.update_all_registries`.
        import_prefix: prefix to use when importing modules. See
            :py:func:`~helios.core.utils.update_all_registries` for details.
        print_banner: if true, the Helios banner with system info will be printed.
            Defaults to true.
    """

    def __init__(
        self,
        run_name: str = "",
        train_unit: TrainingUnit | str = TrainingUnit.EPOCH,
        total_steps: int | float = 0,
        valid_frequency: int | None = None,
        chkpt_frequency: int | None = None,
        print_frequency: int | None = None,
        accumulation_steps: int = 1,
        enable_cudnn_benchmark: bool = False,
        enable_deterministic: bool = False,
        early_stop_cycles: int | None = None,
        use_cpu: bool | None = None,
        gpus: list[int] | None = None,
        random_seed: int | None = None,
        enable_tensorboard: bool = False,
        enable_file_logging: bool = False,
        enable_progress_bar: bool = False,
        chkpt_root: pathlib.Path | None = None,
        log_path: pathlib.Path | None = None,
        run_path: pathlib.Path | None = None,
        src_root: pathlib.Path | None = None,
        import_prefix: str = "",
        print_banner: bool = True,
    ):
        """Create the trainer."""
        self._model: hlm.Model | None = None
        self._datamodule: data.DataModule | None = None
        self._local_rank: int = 0
        self._rank: int = 0

        self._use_cpu: bool = False
        self._device: torch.device | None = None
        self._map_loc: str | dict[str, str] = ""
        self._gpu_ids: list[int] = [] if gpus is None else gpus
        self._active_gpu: int = 0
        self._is_distributed: bool = False
        self._is_torchrun: bool = dist.is_using_torchrun()

        if isinstance(train_unit, str):
            train_unit = TrainingUnit.from_str(train_unit)

        self._train_unit = train_unit
        self._total_steps = total_steps
        self._accumulation_steps = accumulation_steps
        self._valid_frequency = valid_frequency
        self._chkpt_frequency = chkpt_frequency
        self._print_frequency = print_frequency
        self._enable_cudnn_benchmark = enable_cudnn_benchmark
        self._enable_deterministic = enable_deterministic
        self._early_stop_cycles = early_stop_cycles
        self._enable_tensorboard = enable_tensorboard
        self._enable_file_logging = enable_file_logging
        self._random_seed = rng.get_default_seed() if random_seed is None else random_seed
        self._enable_progress_bar = enable_progress_bar

        self._chkpt_root = chkpt_root
        self._log_path = log_path
        self._run_path = run_path

        self._src_root = src_root
        self._import_prefix = import_prefix

        self._run_name = run_name
        self._print_banner = print_banner

        self._train_exceptions: list[type[Exception]] = []
        self._test_exceptions: list[type[Exception]] = []

        self._plugins: dict[str, hlp.Plugin] = {}

        self._queue: mp.Queue | None = None
        self._distributed_error_queue: mp.Queue | None = None

        self._validate_flags(use_cpu)
        self._setup_device_flags(use_cpu)

    @property
    def model(self) -> hlm.Model:
        """Return the model."""
        return core.get_from_optional(self._model)

    @model.setter
    def model(self, model: hlm.Model) -> None:
        self._model = model

    @property
    def datamodule(self) -> data.DataModule:
        """Return the datamodule."""
        return core.get_from_optional(self._datamodule)

    @datamodule.setter
    def datamodule(self, datamodule: data.DataModule) -> None:
        self._datamodule = datamodule

    @property
    def local_rank(self) -> int:
        """Return the local rank of the trainer."""
        return self._local_rank

    @local_rank.setter
    def local_rank(self, r) -> None:
        self._local_rank = r
        if not self._use_cpu:
            self._active_gpu = self._gpu_ids[r]

    @property
    def rank(self) -> int:
        """Return the global rank of the trainer."""
        return self._rank

    @rank.setter
    def rank(self, r) -> None:
        self._rank = r

    @property
    def gpu_ids(self) -> list[int]:
        """Return the list of GPU IDs to use for training."""
        return self._gpu_ids

    @property
    def train_exceptions(self) -> list[type[Exception]]:
        """Return the list of valid exceptions for training."""
        return self._train_exceptions

    @train_exceptions.setter
    def train_exceptions(self, exc: list[type[Exception]]) -> None:
        self._train_exceptions = exc

    @property
    def test_exceptions(self) -> list[type[Exception]]:
        """Return the list of valid exceptions for testing."""
        return self._test_exceptions

    @test_exceptions.setter
    def test_exceptions(self, exc: list[type[Exception]]) -> None:
        self._test_exceptions = exc

    @property
    def plugins(self) -> dict[str, hlp.Plugin]:
        """Return the list of plug-ins."""
        return self._plugins

    @plugins.setter
    def plugins(self, plugs: dict[str, hlp.Plugin]) -> None:
        self._plugins = plugs

    @property
    def queue(self) -> mp.Queue | None:
        """
        Return the multi-processing queue instance.

        .. note::
            If training isn't distributed or if `torchrun`, then `None` is returned
            instead.
        """
        return self._queue

    @queue.setter
    def queue(self, q: mp.Queue) -> None:
        self._queue = q

    def fit(self, model: hlm.Model, datamodule: data.DataModule) -> bool:
        """
        Run the full training routine.

        Args:
            model: the model to run on.
            datamodule: the datamodule to use.

        Returns:
            True if the training process completed successfully, false otherwise.
        """
        try:
            self._launch(model, datamodule, _TrainerMode.TRAIN)
        except Exception as e:
            if not self._handle_exception(e, _TrainerMode.TRAIN):
                raise e
            return False

        return True

    def test(self, model: hlm.Model, datamodule: data.DataModule) -> bool:
        """
        Run the full testing routine.

        Args:
            model: the model to run on.
            datamodule: the datamodule to use.

        Returns:
            True if the training process completed successfully, false otherwise.
        """
        try:
            self._launch(model, datamodule, _TrainerMode.TEST)
        except Exception as e:
            if not self._handle_exception(e, _TrainerMode.TEST):
                raise e
            return False

        return True

    def _handle_exception(self, e: Exception, mode: _TrainerMode) -> bool:
        """
        Exception handler.

        Args:
            e: the raised exception.

        Returns:
            False if the exception should be allowed to continue up the stack. If true,
            the exception has been handled and should not be re-raised.
        """
        exc_list = (
            self._train_exceptions
            if mode == _TrainerMode.TRAIN
            else self._test_exceptions
        )
        if any(isinstance(e, exc) for exc in exc_list):
            logging.close_default_loggers()
            return False

        if logging.is_root_logger_active():
            root_logger = logging.get_root_logger()
            root_logger.exception("error: uncaught exception")
            logging.close_default_loggers()
        return True

    def _configure_env_for_distributed_error_handling(self) -> None:
        assert self._distributed_error_queue is not None
        if self._distributed_error_queue.empty():
            return

        state: _DistributedErrorState = self._distributed_error_queue.get_nowait()
        if self._enable_file_logging and state.log_path is not None:
            logging.create_default_loggers(enable_tensorboard=False)
            logging.restore_default_loggers(log_path=state.log_path)

    def _push_distributed_error_state(self, state: _DistributedErrorState) -> None:
        if self._distributed_error_queue is None:
            return
        self._distributed_error_queue.put(state)

    def _launch(
        self, model: hlm.Model, datamodule: data.DataModule, mode: _TrainerMode
    ) -> None:
        """
        Launch the function corresponding to the given mode.

        If distributed training is used, this will spawn the processes and call the
        handler.

        Args:
            model: the model to use.
            datamodule: the datamodule to use.
            mode:: the operation to perform.
        """
        datamodule.prepare_data()

        if self._is_distributed and not self._is_torchrun:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
            queue: mp.Queue = mp.Queue()
            error_queue: mp.Queue = mp.Queue()
            world_size = len(self._gpu_ids)
            try:
                mp.spawn(
                    _spawn_handler,
                    args=(
                        world_size,
                        self,
                        datamodule,
                        model,
                        mode,
                        queue,
                        error_queue,
                    ),
                    nprocs=world_size,
                    join=True,
                )
            except Exception as e:
                self._distributed_error_queue = error_queue
                self._configure_env_for_distributed_error_handling()
                raise e

            self.queue = queue
            return

        if self._is_torchrun and self._is_distributed:
            dist.init_dist()

        self.model = model
        self.datamodule = datamodule
        self.rank = dist.get_global_rank()
        self.local_rank = dist.get_local_rank()
        if mode == _TrainerMode.TRAIN:
            self._train()
        elif mode == _TrainerMode.TEST:
            self._test()

        if self._is_torchrun and self._is_distributed:
            dist.shutdown_dist()

        logging.close_default_loggers()

    def _train(self) -> None:
        """
        Train the model.

        This will ensure everything gets correctly initialised as well as select the
        appropriate training loop for the given training unit.
        """
        self._configure_env()
        self._setup_plugins()
        self._setup_datamodule()
        self._setup_model()
        self._prepare_roots()

        chkpt_path = find_last_checkpoint(self._chkpt_root)
        training_state = self._load_checkpoint(chkpt_path)

        log_path = logging.get_root_logger().log_file
        self._push_distributed_error_state(_DistributedErrorState(log_path=log_path))

        self._print_header(chkpt_path)

        self._execute_plugins("on_training_start")
        self.model.on_training_start()
        if self._train_unit == TrainingUnit.ITERATION:
            self._train_on_iteration(training_state)
        else:
            self._train_on_epoch(training_state)

        self._execute_plugins("on_training_end")
        self.model.on_training_end()

        logging.flush_default_loggers()
        logging.close_default_loggers()

        # If we're distributed, ensure that all processes are caught up before we exit.
        dist.safe_barrier()

    def _test(self) -> None:
        """
        Test the model.

        This will ensure everything gets correctly initialised and run the testing loop on
        the dataset.
        It will automatically try to load the last saved checkpoint for testing provided
        there is one. If no checkpoints are available, it is assumed the model is loading
        the correct state internally.
        """
        self._configure_env()
        self._setup_plugins()
        self._setup_datamodule()
        self._setup_model(fast_init=True)
        self._prepare_roots(mkdir=False)

        chkpt_path: pathlib.Path | None = None
        loaded: bool = False
        if self._chkpt_root is not None:
            chkpt_path = find_last_checkpoint(core.get_from_optional(self._chkpt_root))
            loaded = (
                self._load_checkpoint(chkpt_path, skip_rng=True, model_fast_init=True)
                != TrainingState()
            )

        # We failed to load the last checkpoint, so tell the model to load its state.
        if self._chkpt_root is None or not loaded:
            self.model.load_for_testing()

        log_path = logging.get_root_logger().log_file
        self._push_distributed_error_state(_DistributedErrorState(log_path=log_path))

        self._print_header(chkpt_path, for_training=False)

        if self.datamodule.test_dataloader() is None:
            return

        dataloader: tud.DataLoader
        sampler: ResumableSamplerType
        dataloader, sampler = core.get_from_optional(self.datamodule.test_dataloader())

        self._execute_plugins("on_testing_start")
        self.model.on_testing_start()

        enable_progress_bar = self._enable_progress_bar
        pbar_disabled = (
            self._is_distributed and self.rank != 0
        ) or not enable_progress_bar
        pbar = tqdm.tqdm(
            total=len(dataloader),
            desc="Testing",
            unit="it",
            disable=pbar_disabled,
            leave=False,
        )

        with core.cuda.DisableCuDNNBenchmarkContext():
            self.model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    batch = self._plugins_process_batch("testing", batch, step=idx)
                    self.model.on_testing_batch_start(idx)
                    self.model.test_step(batch, idx)
                    self.model.on_testing_batch_end(idx)
                    pbar.update()

            self._execute_plugins("on_testing_end")
            self.model.on_testing_end()

        dist.safe_barrier()

    def _configure_env(self) -> None:
        """
        Configure the training environment.

        This will seed the RNGs as well as setup any CUDA state (if using). It will also
        set all of the registries provided the source root is not None. This is to prevent
        the registries from being empty if distributed training is launched through spawn
        (note that ``torchrun`` doesn't have this problem).
        """
        register_trainer_types_for_safe_load()
        rng.seed_rngs(self._random_seed)
        torch.use_deterministic_algorithms(self._enable_deterministic)

        if not self._use_cpu:
            self._device = torch.device(f"cuda:{self._active_gpu}")
            self._map_loc = {"cuda:0": f"cuda:{self._active_gpu}"}
            torch.backends.cudnn.benchmark = self._enable_cudnn_benchmark
            torch.cuda.set_device(self._device)

        logging.create_default_loggers(self._enable_tensorboard)

        if self._src_root is not None:
            core.update_all_registries(
                self._src_root, recurse=True, import_prefix=self._import_prefix
            )

    def _setup_datamodule(self) -> None:
        """Finish setting up the datamodule."""
        self.datamodule.is_distributed = self._is_distributed
        self.datamodule.trainer = self
        self.datamodule.setup()

    def _setup_model(self, fast_init: bool = False) -> None:
        """
        Finish setting up the model.

        Args:
            fast_init: whether the model should setup its full state or not.
        """
        self.model.map_loc = self._map_loc
        self.model.is_distributed = self._is_distributed
        self.model.device = core.get_from_optional(self._device)
        self.model.rank = self.local_rank
        self.model.trainer = self
        self.model.setup(fast_init)

    def _setup_plugins(self) -> None:
        """Finish setting up the plug-ins."""
        for plugin in self._plugins.values():
            plugin.is_distributed = self._is_distributed
            plugin.map_loc = self._map_loc
            plugin.device = core.get_from_optional(self._device)
            plugin.rank = self.local_rank
            plugin.trainer = self
            plugin.setup()

    def _prepare_roots(self, mkdir=True) -> None:
        """Prepare the training roots."""
        name = self.model.save_name
        self._chkpt_root = (
            self._chkpt_root / name if self._chkpt_root is not None else None
        )
        if mkdir:
            if self._chkpt_root is not None:
                self._chkpt_root.mkdir(parents=True, exist_ok=True)

            if self._log_path is not None:
                self._log_path.mkdir(parents=True, exist_ok=True)
            if self._run_path is not None:
                self._run_path.mkdir(parents=True, exist_ok=True)

    def _print_header(
        self, chkpt_path: pathlib.Path | None, for_training: bool = True
    ) -> None:
        """Print the Helios header with system info to the logs."""
        root_logger = logging.get_root_logger()
        model = core.get_from_optional(self._model)
        banner = model.append_to_banner(core.get_env_info_str())

        if self._print_banner:
            dist.global_print(banner)

        if for_training:
            if chkpt_path is not None:
                msg = f"Resuming training from checkpoint {str(chkpt_path)}"
                root_logger.info(msg)
                dist.global_print(f"{msg}\n")
            elif self._print_banner:
                root_logger.info(banner)
        else:
            root_logger.info(core.get_env_info_str())
            msg = (
                f"Testing using checkpoint {str(chkpt_path)}"
                if chkpt_path is not None
                else "Testing from loaded model"
            )
            root_logger.info(msg)
            dist.global_print(f"{msg}\n")

    def _validate_flags(self, use_cpu: bool | None):
        """Ensure that all the settings and flags are valid."""
        if isinstance(self._total_steps, float) and self._total_steps != float("inf"):
            raise ValueError(
                "error: expected 'total_steps' to be of type 'int' or 'infinity', but "
                f"received {self._total_steps}"
            )

        if use_cpu is not None and use_cpu and len(self._gpu_ids) > 0:
            raise ValueError("error: cannot request CPU and GPU training")

        if self._chkpt_frequency is not None and self._chkpt_frequency == 0:
            raise ValueError("error: checkpoint frequency must be greater than 0 or None")

        if self._print_frequency is not None and self._print_frequency == 0:
            raise ValueError("error: print frequency must be greater than 0 or None")

        if self._valid_frequency is not None and self._valid_frequency == 0:
            raise ValueError("error: valid frequency must be greater than 0 or None")

        if self._enable_deterministic and self._enable_cudnn_benchmark:
            raise ValueError(
                "error: CUDA benchmark and deterministic flags are mutually exclusive"
            )

        if self._total_steps == float("inf") and self._early_stop_cycles == 0:
            raise ValueError(
                f"error: given 'total_steps' with value {self._total_steps}, "
                "'early_stop_cycles' must be non-zero"
            )

        if self._enable_tensorboard:
            if self._run_path is None:
                raise ValueError(
                    "error: Tensorboard requested but no run directory was given"
                )

            if self._run_path.exists() and not self._run_path.is_dir():
                raise ValueError("error: run path must be a directory")

        if self._enable_file_logging:
            if self._log_path is None:
                raise ValueError(
                    "error: file logging requested but no log directory was given"
                )

            if self._log_path.exists() and not self._log_path.is_dir():
                raise ValueError("error: log path must be a directory")

        if self._src_root is not None and not self._src_root.is_dir():
            raise ValueError("error: source root must be a directory")

    def _setup_device_flags(self, use_cpu: bool | None):
        """
        Configure the device state.

        If the CPU is being used, this will automatically set the correct settings. If the
        GPU will be used, then it will only verify that the GPU IDs are correct. The
        remaining state will be set afterwards.
        The ``use_cpu`` flag is used to determine whether the CPU will be used for
        training. If it is ``None``, then the value is determined by whether CUDA is
        available.

        Args:
            use_cpu: whether to use the CPU or not.
        """
        if use_cpu is None:
            use_cpu = not torch.cuda.is_available()

        if use_cpu:
            self._use_cpu = True
            self._device = torch.device("cpu")
            self._map_loc = {"cuda:0": "cpu"}
            self._gpu_ids = []
            self._is_distributed = False
            return

        if not torch.cuda.is_available():
            raise RuntimeError(
                "error: CUDA usage is requested, but CUDA is not available"
            )

        # At this point we know that CUDA exists and that we're supposed to use it. For
        # now, just verify that the GPU IDs are valid, but don't set the device or the map
        # location. Those need to be set after we launch distributed training (if using)
        # to ensure they get set to the correct thing.

        valid_ids = list(range(torch.cuda.device_count()))
        if len(self._gpu_ids) == 0:
            self._gpu_ids = valid_ids

        if len(self._gpu_ids) > len(valid_ids):
            raise ValueError(
                f"error: expected a maximum of {len(valid_ids)} GPU IDs but "
                f"received {len(self._gpu_ids)}"
            )

        for gpu_id in self._gpu_ids:
            if gpu_id not in valid_ids:
                raise ValueError(f"error: {gpu_id} is not a valid GPU")

        self._is_distributed = (
            len(self._gpu_ids) > 1
            if not self._is_torchrun
            else int(os.environ["WORLD_SIZE"]) > 1
        )

    def _validate_state_dict(self, state_dict: dict[str, typing.Any]) -> bool:
        """
        Ensure the state table is valid.

        This is to handle the eventuality that someone tries to load a checkpoint that
        wasn't generated by the trainer. It checks that all the required keys appear, and
        it also does a version check. Mainly: we guarantee support for checkpoints
        generated in versions less than or equal to the current version.

        Args:
            state_dict: the state dictionary.

        Returns:
            True if the checkpoint is valid, false otherwise.
        """
        required_keys = ("version", "training_state", "model", "rng")
        if not all(key in state_dict for key in required_keys):
            return False

        # Now check the version to see if it's compatible with us.
        cur_ver = pv.Version(__version__)
        chkpt_ver = pv.Version(state_dict["version"])
        return chkpt_ver <= cur_ver

    def _save_checkpoint(self, state: TrainingState) -> None:
        """
        Save the current training state to a checkpoint.

        This will automatically save the training state, RNG state, as well as the model
        state.

        Args:
            state: the current training state.
        """
        chkpt_root = core.get_from_optional(self._chkpt_root)

        epoch = state.global_epoch
        ite = state.current_iteration
        filename = f"{self.model.save_name}_epoch_{epoch}_iter_{ite}"
        filename = self.model.append_metadata_to_chkpt_name(filename)
        filename += ".pth"

        state_dict: dict[str, typing.Any] = {}
        state_dict["version"] = __version__
        state_dict["training_state"] = state
        state_dict["model"] = self.model.state_dict()
        state_dict["rng"] = rng.get_rng_state_dict()

        if self._enable_file_logging:
            state_dict["log_path"] = logging.get_root_logger().log_file

        if self._enable_tensorboard:
            writer = core.get_from_optional(logging.get_tensorboard_writer())
            state_dict["run_path"] = writer.run_path

        # Add the plug-ins (if using)
        for plug_id, plugin in self._plugins.items():
            state_dict[plug_id] = plugin.state_dict()

        # Safety check.
        assert self._validate_state_dict(state_dict)

        torch.save(state_dict, chkpt_root / filename)

    def _load_checkpoint(
        self,
        chkpt_path: pathlib.Path | None,
        skip_rng: bool = False,
        model_fast_init: bool = False,
    ) -> TrainingState:
        """
        Load the given checkpoint.

        Args:
            chkpt_path: path to the checkpoint to load.
            skip_rng: if True, skip the loading of the RNG states.
            model_fast_init: whether the model should setup its full state or not.

        Returns:
            Returns the loaded training state and ``True`` if the checkpoint was
                loaded successfully. Otherwise it returns an empty training state and
                ``False``.
        """
        if chkpt_path is None:
            logging.setup_default_loggers(self._run_name, self._log_path, self._run_path)
            return TrainingState()

        state_dict = core.safe_torch_load(chkpt_path, map_location=self._map_loc)
        if not self._validate_state_dict(state_dict):
            raise RuntimeError(
                f"error: the checkpoint found at {str(chkpt_path)} is not a "
                "valid checkpoint generated by Helios"
            )
        logging.restore_default_loggers(
            state_dict.get("log_path", None), state_dict.get("run_path", None)
        )
        if not skip_rng:
            rng.load_rng_state_dict(state_dict["rng"])
        self.model.load_state_dict(state_dict["model"], fast_init=model_fast_init)

        for plug_id, plugin in self._plugins.items():
            if plug_id in state_dict:
                plugin.load_state_dict(state_dict[plug_id])

        return state_dict["training_state"]

    def _train_on_iteration(self, state: TrainingState) -> None:
        """
        Run the main loop for iteration-based training.

        Args:
            state: the training state.
        """
        total_steps = self._total_steps
        save_freq = self._chkpt_frequency
        val_freq = self._valid_frequency
        print_freq = self._print_frequency
        accumulation_steps = self._accumulation_steps
        enable_progress_bar = self._enable_progress_bar
        early_stop_cycles = self._early_stop_cycles

        current_iteration_changed: bool = True
        training_done: bool = False
        root_logger = logging.get_root_logger()
        iter_timer = core.AverageTimer()

        pbar_disabled = (
            self._is_distributed and self.rank != 0
        ) or not enable_progress_bar
        pbar = tqdm.tqdm(
            total=total_steps if total_steps != float("inf") else None,
            desc="Training iterations",
            unit="it",
            disable=pbar_disabled,
            initial=state.current_iteration,
        )

        dataloader: tud.DataLoader
        sampler: ResumableSamplerType
        dataloader, sampler = core.get_from_optional(self.datamodule.train_dataloader())

        sampler.start_iter = state.dataset_iter
        self.model.train()

        for epoch in itertools.count(start=state.global_epoch):
            if training_done:
                break

            state.global_epoch += 1
            root_logger.info(f"Starting epoch {epoch + 1}")
            sampler.set_epoch(epoch)
            epoch_start = time.time()
            self.model.on_training_epoch_start(state.global_epoch)

            iter_timer.start()
            for batch in dataloader:
                state.global_iteration += 1
                if state.global_iteration % accumulation_steps == 0:
                    state.current_iteration += 1
                    state.running_iter += 1
                    current_iteration_changed = True
                else:
                    current_iteration_changed = False

                batch = self._plugins_process_batch("training", batch, state=state)
                self.model.on_training_batch_start(state)
                self.model.train_step(batch, state)
                iter_timer.record()
                state.average_iter_time = iter_timer.get_average_time()
                self.model.on_training_batch_end(
                    state,
                    should_log=(
                        False
                        if print_freq is None
                        else state.current_iteration % print_freq == 0
                        and current_iteration_changed
                    ),
                )
                # Depending on how fast the iteration loop is, it is possible that the
                # progress bar isn't refreshed every tick, so make sure it gets re-drawn.
                if state.global_iteration % accumulation_steps == 0 and not pbar.update():
                    pbar.refresh()
                state.dataset_iter += 1
                if (
                    self._plugins_should_training_stop()
                    or self.model.should_training_stop()
                ):
                    training_done = True
                    break

                if (
                    val_freq is not None
                    and state.current_iteration % val_freq == 0
                    and current_iteration_changed
                ):
                    self._validate(state.validation_cycles)
                    state.validation_cycles += 1
                    state.running_iter = 0
                    if not self.model.have_metrics_improved():
                        state.early_stop_count += 1
                    else:
                        state.early_stop_count = 0

                if (
                    save_freq is not None
                    and state.current_iteration % save_freq == 0
                    and self.rank == 0
                    and current_iteration_changed
                ):
                    self._save_checkpoint(state)

                if (
                    early_stop_cycles is not None
                    and state.early_stop_count >= early_stop_cycles
                ):
                    training_done = True
                    break

                if (
                    self._plugins_should_training_stop()
                    or self.model.should_training_stop()
                ):
                    training_done = True
                    break

                if state.current_iteration >= total_steps:
                    training_done = True
                    break

            state.dataset_iter = 0
            self.model.on_training_epoch_end(state.global_epoch)

            root_logger.info(
                f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s"
            )

    def _train_on_epoch(self, state: TrainingState) -> None:
        """
        Run the main loop for epoch-based training.

        Args:
            state: the training state.
        """
        total_steps = self._total_steps
        save_freq = self._chkpt_frequency
        val_freq = self._valid_frequency
        print_freq = self._print_frequency
        enable_progress_bar = self._enable_progress_bar
        early_stop_cycles = self._early_stop_cycles

        training_done: bool = False
        root_logger = logging.get_root_logger()
        iter_timer = core.AverageTimer()

        pbar_disabled = (
            self._is_distributed and self.rank != 0
        ) or not enable_progress_bar
        pbar = tqdm.tqdm(
            total=total_steps if total_steps != float("inf") else None,
            desc="Training epochs",
            unit="epoch",
            disable=pbar_disabled,
        )

        dataloader: tud.DataLoader
        sampler: ResumableSamplerType
        dataloader, sampler = core.get_from_optional(self.datamodule.train_dataloader())

        sampler.start_iter = state.dataset_iter
        self.model.train()
        iterator = (
            range(state.global_epoch, int(total_steps))
            if total_steps != float("inf")
            else itertools.count(start=state.global_epoch)
        )

        for epoch in iterator:
            if training_done:
                break

            state.global_epoch += 1

            if state.global_epoch > total_steps:
                training_done = True
                break

            root_logger.info(f"Starting epoch {epoch + 1}")
            sampler.set_epoch(epoch)
            epoch_start = time.time()
            self.model.on_training_epoch_start(state.global_epoch)
            iter_timer.start()
            with tqdm.tqdm(
                total=len(dataloader),
                desc=f"Epoch {epoch + 1}",
                unit="iter",
                disable=pbar_disabled,
                leave=False,
            ) as ite_pbar:
                for batch in dataloader:
                    state.global_iteration += 1
                    state.current_iteration += 1
                    state.running_iter += 1

                    batch = self._plugins_process_batch("training", batch, state=state)
                    self.model.on_training_batch_start(state)
                    self.model.train_step(batch, state)
                    iter_timer.record()
                    state.average_iter_time = iter_timer.get_average_time()
                    self.model.on_training_batch_end(
                        state,
                        should_log=(
                            False
                            if print_freq is None
                            else state.current_iteration % print_freq == 0
                        ),
                    )
                    state.dataset_iter += 1
                    if not ite_pbar.update():
                        ite_pbar.refresh()

                    if (
                        self._plugins_should_training_stop()
                        or self.model.should_training_stop()
                    ):
                        training_done = True
                        break

            state.dataset_iter = 0

            if val_freq is not None and state.global_epoch % val_freq == 0:
                self._validate(state.validation_cycles)
                state.running_iter = 0
                if not self.model.have_metrics_improved():
                    state.early_stop_count += 1
                else:
                    state.early_stop_count = 0
                state.validation_cycles += 1
            if (
                save_freq is not None
                and state.global_epoch % save_freq == 0
                and self.rank == 0
            ):
                self._save_checkpoint(state)

            self.model.on_training_epoch_end(state.global_epoch)
            root_logger.info(
                f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s"
            )
            pbar.update()
            if (
                early_stop_cycles is not None
                and state.early_stop_count >= early_stop_cycles
            ):
                training_done = True

            if self._plugins_should_training_stop() or self.model.should_training_stop():
                training_done = True

    def _validate(self, val_cycle: int) -> None:
        """
        Run the validation loop.

        Args:
            val_cycle: the current validation cycle number.
        """
        if self.datamodule.valid_dataloader() is None:
            return

        dataloader: tud.DataLoader
        sampler: ResumableSamplerType
        dataloader, sampler = core.get_from_optional(self.datamodule.valid_dataloader())

        enable_progress_bar = self._enable_progress_bar
        pbar_disabled = (
            self._is_distributed and self.rank != 0
        ) or not enable_progress_bar
        pbar = tqdm.tqdm(
            total=len(dataloader),
            desc="Validation",
            unit="it",
            disable=pbar_disabled,
            leave=False,
        )

        with core.cuda.DisableCuDNNBenchmarkContext():
            self.model.eval()
            self._execute_plugins("on_validation_start", validation_cycle=val_cycle)
            self.model.on_validation_start(val_cycle)
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    batch = self._plugins_process_batch("validation", batch, step=idx)
                    self.model.on_validation_batch_start(idx)
                    self.model.valid_step(batch, idx)
                    self.model.on_validation_batch_end(idx)

                    # Ensure the progress bar is updated in the event that the validation
                    # loop runs faster than the refresh rate of the progress bar.
                    if not pbar.update():
                        pbar.refresh()

            self.model.train()
            self._execute_plugins("on_validation_end", validation_cycle=val_cycle)
            self.model.on_validation_end(val_cycle)

        dist.safe_barrier()

    def _validate_plugins(self) -> None:
        seen_overrides: dict[str, str] = {}
        fields = dc.fields(hlp.UniquePluginOverrides)
        for _, plugin in self._plugins.items():
            for field in fields:
                name = field.name
                if getattr(plugin.unique_overrides, name):
                    if name not in seen_overrides:
                        seen_overrides[name] = str(type(plugin))
                    else:
                        raise ValueError(
                            f"error: override field {name} has already been overridden "
                            f"by {seen_overrides[name]}"
                        )

    def _execute_plugins(self, func_name: str, **kwargs: typing.Any) -> None:
        for plugin in self._plugins.values():
            func = getattr(plugin, func_name)
            func(**kwargs)

    def _plugins_process_batch(
        self, mode: str, batch: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        func_name = f"process_{mode}_batch"
        override_name = f"{mode}_batch"

        for plugin in self._plugins.values():
            if getattr(plugin.unique_overrides, override_name):
                return getattr(plugin, func_name)(batch=batch, **kwargs)

        return batch

    def _plugins_should_training_stop(self) -> bool:
        return any(plugin.should_training_stop() for plugin in self._plugins.values())
