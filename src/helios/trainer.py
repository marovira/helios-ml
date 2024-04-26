from __future__ import annotations

import dataclasses
import enum
import itertools
import os
import pathlib
import re
import time
import typing

import torch
import torch.distributed as td
import torch.multiprocessing as mp
import torch.utils.data as tud
import tqdm

import helios.model as hlm
from helios import core, data
from helios.core import distributed as dist
from helios.core import logging, rng
from helios.data.samplers import ResumableSamplerType


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
            label (str): the label to convert.

        Returns:
            TrainingUnit: the corresponding value.
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


@dataclasses.dataclass
class TrainingState:
    """
    The training state.

    Args:
        current_iteration (int): the current iteration number. Note that this refers to
        the iteration in which gradients are updated. This may or may not be equal to the
        global_iteration count (see below).
        global_iteration (int): the total iteration count.
        global_epoch (int): the total epoch count.
        validation_cycles (int): the number of validation cycles.
        dataset_iter (int): the current batch index of the dataset. This is reset every
        epoch.
        early_stop_count (int): the current number of validation cycles for early stop.
        average_iter_time (float): average time per iteration.
        running_iter (int): iteration count in the current validation cycle. Useful for
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

    dict = dataclasses.asdict


def find_last_checkpoint(root: pathlib.Path) -> pathlib.Path | None:
    """
    Find the last saved checkpoint (if available).

    The function assumes that checkpoint names contain 'epoch_<epoch>' and 'iter_<iter>'
    in the name, in which case it will return the path to the checkpoint with the highest
    epoch and/or iteration count.

    Args:
        root (pathlib.Path): the path where the checkpoints are stored.

    Returns:
        pathlib.Path | None: the path to the last checkpoint (if any).
    """
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


def spawn_handler(
    rank: int,
    world_size: int,
    trainer: Trainer,
    datamodule: data.DataModule,
    model: hlm.Model,
    mode: _TrainerMode,
) -> None:
    """
    Spawn handler for distributed training.

    Args:
        rank (int): the rank id of the current process within the work group.
        world_size (int): number of processes in the work group.
        trainer (Trainer): the trainer to use.
        datamodule (DataModule): the datamodule to use.
        model (Model): the model to use.
        mode (_TrainerMode): determines which operation needs to be performed.
    """
    dist.init_dist(rank=rank, world_size=world_size)

    trainer.model = model
    trainer.datamodule = datamodule
    trainer.rank = rank
    trainer.local_rank = rank
    if mode == _TrainerMode.TRAIN:
        trainer._train()  # noqa: SLF001
    elif mode == _TrainerMode.TEST:
        trainer._test()  # noqa: SLF001

    dist.shutdown_dist()


class Trainer:
    """
    Automates the training, validation, and testing code.

    The trainer handles all of the required steps to setup the correct environment
    (including handling distributed training), the training/validation/testing loops, and
    any clean up afterwards.

    Args:
        run_name (str): name of the current run.
        train_unit (TrainingUnit | str): the unit used for training.
        total_steps (int | float): the total number of steps to train for.
        valid_frequency (int): frequency with which to perform validation.
        chkpt_frequency (int): frequency with which to save checkpoints.
        print_frequency (int): frequency with which to log.
        accumulation_steps (int): number of steps for gradient accumulation.
        enable_cudnn_benchmark (bool): enable/disable CuDNN benchmark.
        enable_deterministic (bool): enable/disable PyTorch deterministic.
        early_stop_cycles (int): number of cycles after which training will stop if no
        improvement is seen during validation.
        use_cpu: (bool | None): if True, CPU will be used.
        gpus (list[int] | None): IDs of GPUs to use.
        random_seed (int | None): the seed to use for RNGs.
        enable_tensorboard (bool): enable/disable Tensorboard logging.
        enable_file_logging (bool): enable/disable file logging.
        enable_progress_bar (bool): enable/disable the progress bar(s).
        chkpt_root (pathlib.Path): root folder in which checkpoints will be placed.
        log_path (pathlib.Path): root folder in which logs will be saved.
        run_path (pathlib.Path): root folder in which Tensorboard runs will be saved.
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
    ):
        """Create the trainer."""
        self._model: hlm.Model | None = None
        self._datamodule: data.DataModule | None = None
        self._local_rank: int = 0
        self._rank: int = 0
        self._callbacks: dict[str, typing.Callable] = {}

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

        self._validate_flags()
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
    def callbacks(self) -> dict[str, typing.Callable]:
        """Return the table of callbacks."""
        return self._callbacks

    def register_callback(self, name: str, callback: typing.Callable) -> None:
        """
        Register a callback.

        Callbacks are used to inject additional code or behaviour into the training code.
        You may call these at any time through the trainer instance that is attached to
        the datamodule and the model.

        Args:
            name (str): the name of the callback.
            callback (typing.Callable): the callback to register.
        """
        if name in self._callbacks:
            raise KeyError(f"error: {name} has already been registered as a callback")

        self._callbacks[name] = callback

    def fit(self, model: hlm.Model, datamodule: data.DataModule) -> None:
        """
        Run the full training routine.

        Args:
            model (pym.Model): the model to run on.
            datamodule (data.DataModule): the datamodule to use.
        """
        try:
            self._launch(model, datamodule, _TrainerMode.TRAIN)
        except Exception as e:
            if logging.is_root_logger_active():
                root_logger = logging.get_root_logger()
                root_logger.exception("error: uncaught exception")
                logging.close_default_loggers()
            raise RuntimeError("error: uncaught exception") from e

    def test(self, model: hlm.Model, datamodule: data.DataModule) -> None:
        """
        Run the full testing routine.

        Args:
            model (pym.Model): the model to run on.
            datamodule (data.DataModule): the datamodule to use.
        """
        try:
            self._launch(model, datamodule, _TrainerMode.TEST)
        except Exception as e:
            if logging.is_root_logger_active():
                root_logger = logging.get_root_logger()
                root_logger.exception("error: uncaught exception")
                logging.close_default_loggers()
            raise RuntimeError("error: uncaught exception") from e

    def _launch(
        self, model: hlm.Model, datamodule: data.DataModule, mode: _TrainerMode
    ) -> None:
        """
        Launch the function corresponding to the given mode.

        If distributed training is used, this will spawn the processes and call the
        handler.

        Args:
            model (Model): the model to use.
            datamodule (DataModule): the datamodule to use.
            mode: (_TrainerMode): the operation to perform.
        """
        datamodule.prepare_data()

        if self._is_distributed and not self._is_torchrun:
            world_size = len(self._gpu_ids)
            mp.spawn(
                spawn_handler,
                args=(world_size, self, datamodule, model, mode),
                nprocs=world_size,
                join=True,
            )
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

    def _train(self) -> None:
        """
        Train the model.

        This will ensure everything gets correctly initialised as well as select the
        appropriate training loop for the given training unit.
        """
        self._configure_env()
        self._setup_datamodule()
        self._setup_model()
        self._prepare_roots()

        chkpt_path = find_last_checkpoint(core.get_from_optional(self._chkpt_root))
        training_state, resume_training = self._load_checkpoint(chkpt_path)

        self._print_header(chkpt_path)

        self.model.on_training_start()
        if self._train_unit == TrainingUnit.ITERATION:
            self._train_on_iteration(training_state, resume_training)
        else:
            self._train_on_epoch(training_state)

        self.model.on_training_end()

        logging.flush_default_loggers()
        logging.close_default_loggers()

        # If we're distributed, ensure that all processes are caught up before we exit.
        if self._is_distributed:
            td.barrier()

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
        self._setup_datamodule()
        self._setup_model(fast_init=True)
        self._prepare_roots(mkdir=False)

        chkpt_path: pathlib.Path | None = None
        loaded: bool = False
        if self._chkpt_root is not None:
            chkpt_path = find_last_checkpoint(core.get_from_optional(self._chkpt_root))
            _, loaded = self._load_checkpoint(
                chkpt_path, skip_rng=True, model_fast_init=True
            )

        # We failed to load the last checkpoint, so tell the model to load its state.
        if self._chkpt_root is None or not loaded:
            self.model.load_for_testing()

        self._print_header(chkpt_path, for_training=False)

        if self.datamodule.test_dataloader() is None:
            return

        dataloader: tud.DataLoader
        sampler: ResumableSamplerType
        dataloader, sampler = core.get_from_optional(self.datamodule.test_dataloader())

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
            self.model.on_testing_start()
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    self.model.on_testing_batch_start(idx)
                    self.model.test_step(batch, idx)
                    self.model.on_testing_batch_end(idx)
                    pbar.update()

            self.model.on_testing_end()

        if self._is_distributed:
            td.barrier()

    def _configure_env(self) -> None:
        """
        Configure the training environment.

        This will seed the RNGs as well as setup any CUDA state (if using). It will also
        set all of the registries provided the source root is not None. This is to prevent
        the registries from being empty if distributed training is launched through spawn
        (note that torchrun doesn't have this problem).
        """
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
            fast_init (bool): whether the model should setup its full state or not.
        """
        self.model.map_loc = self._map_loc
        self.model.is_distributed = self._is_distributed
        self.model.device = core.get_from_optional(self._device)
        self.model.rank = self.local_rank
        self.model.trainer = self
        self.model.setup(fast_init)

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

        self._print(core.get_env_info_str())

        if for_training:
            if chkpt_path is not None:
                msg = f"Resuming training from checkpoint {str(chkpt_path)}"
                root_logger.info(msg)
                self._print(f"{msg}\n")
            else:
                root_logger.info(core.get_env_info_str())
        else:
            root_logger.info(core.get_env_info_str())
            msg = (
                f"Testing using checkpoint {str(chkpt_path)}"
                if chkpt_path is not None
                else "Testing from loaded model"
            )
            root_logger.info(msg)
            self._print(f"{msg}\n")

    def _validate_flags(self):
        """Ensure that all the settings and flags are valid."""
        if isinstance(self._total_steps, float) and self._total_steps != float("inf"):
            raise ValueError(
                "error: expected 'total_steps' to be of type 'int' or 'infinity', but "
                f"received {self._total_steps}"
            )

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
        The use_cpu flag is used to determine whether the CPU will be used for training.
        If it is None, then the value is determined by whether CUDA is available.

        Args:
            use_cpu (bool | None): whether to use the CPU or not.
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

    def _save_checkpoint(self, state: TrainingState) -> None:
        """
        Save the current training state to a checkpoint.

        This will automatically save the training state, RNG state, as well as the model
        state.

        Args:
            state (TrainingState): the current training state.
        """
        chkpt_root = core.get_from_optional(self._chkpt_root)

        epoch = state.global_epoch + 1
        ite = state.current_iteration
        filename = f"{self.model.save_name}_epoch_{epoch}_iter_{ite}"
        filename = self.model.append_metadata_to_chkpt_name(filename)
        filename += ".pth"

        state_dict: dict[str, typing.Any] = {}
        state_dict["training_state"] = state.dict()
        state_dict["model"] = self.model.state_dict()
        state_dict["rng"] = rng.get_rng_state_dict()

        if self._enable_file_logging:
            state_dict["log_path"] = logging.get_root_logger().log_file

        if self._enable_tensorboard:
            writer = core.get_from_optional(logging.get_tensorboard_writer())
            state_dict["run_path"] = writer.run_path

        torch.save(state_dict, chkpt_root / filename)

    def _load_checkpoint(
        self,
        chkpt_path: pathlib.Path | None,
        skip_rng: bool = False,
        model_fast_init: bool = False,
    ) -> tuple[TrainingState, bool]:
        """
        Load the given checkpoint.

        Args:
            chkpt_path (pathlib.Path): path to the checkpoint to load.
            skip_rng (bool): if True, skip the loading of the RNG states.
            model_fast_init (bool): whether the model should setup its full state or not.

        Returns:
            tuple[TrainingState, bool]: returns the loaded training state and True if the
            checkpoint was loaded successfully. Otherwise it returns an empty training
            state and False.
        """
        if chkpt_path is None:
            logging.setup_default_loggers(self._run_name, self._log_path, self._run_path)
            return TrainingState(), False

        state_dict = torch.load(chkpt_path, map_location=self._map_loc)
        logging.restore_default_loggers(
            state_dict.get("log_path", None), state_dict.get("run_path", None)
        )
        if not skip_rng:
            rng.load_rng_state_dict(state_dict["rng"])
        self.model.load_state_dict(state_dict["model"], fast_init=model_fast_init)
        return TrainingState(**state_dict["training_state"]), True

    def _train_on_iteration(self, state: TrainingState, resume_training: bool) -> None:
        """
        Run the main loop for iteration-based training.

        Args:
            state (TrainingState): the training state.
            resume_training (bool): if True, then training is resumed from the state.
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
            root_logger.info(f"Starting epoch {epoch + 1}")
            sampler.set_epoch(epoch)
            epoch_start = time.time()

            iter_timer.start()
            for batch in dataloader:
                if state.global_iteration % accumulation_steps == 0:
                    state.current_iteration += 1
                    state.running_iter += 1
                    current_iteration_changed = True
                else:
                    current_iteration_changed = False

                state.global_iteration += 1
                if state.current_iteration > total_steps:
                    training_done = True
                    break

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

            state.dataset_iter = 0
            state.global_epoch += 1

            root_logger.info(
                f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s"
            )
            if (
                early_stop_cycles is not None
                and state.early_stop_count >= early_stop_cycles
            ):
                training_done = True

    def _train_on_epoch(self, state: TrainingState) -> None:
        """
        Run the main loop for epoch-based training.

        Args:
            state (TrainingState): the training state.
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

            if state.global_epoch > total_steps:
                training_done = True
                break

            root_logger.info(f"Starting epoch {epoch + 1}")
            sampler.set_epoch(epoch)
            epoch_start = time.time()
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

            state.dataset_iter = 0
            state.global_epoch += 1

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

            root_logger.info(
                f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s"
            )
            pbar.update()
            if (
                early_stop_cycles is not None
                and state.early_stop_count >= early_stop_cycles
            ):
                training_done = True

    def _validate(self, val_cycle: int) -> None:
        """
        Run the validation loop.

        Args:
            val_cycle (int): the current validation cycle number.
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
            self.model.on_validation_start(val_cycle)
            with torch.no_grad():
                for idx, batch in enumerate(dataloader):
                    self.model.on_validation_batch_start(idx)
                    self.model.valid_step(batch, idx)
                    self.model.on_validation_batch_end(idx)

                    # Ensure the progress bar is updated in the event that the validation
                    # loop runs faster than the refresh rate of the progress bar.
                    if not pbar.update():
                        pbar.refresh()

            self.model.train()
            self.model.on_validation_end(val_cycle)

        if self._is_distributed:
            td.barrier()

    def _print(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Wrap Python's print function for distributed training.

        Specifically, this function will ensure that only rank 0 prints messages to the
        screen. All other ranks will do nothing.

        Args:
            args: named arguments for print.
            kwargs: keyword arguments for print.
        """
        if self.rank != 0:
            return
        print(*args, **kwargs)
