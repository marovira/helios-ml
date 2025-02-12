# Optuna tutorial code.
#
# The code is adapted from PyTorch's "Training a Classifier" tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import os
import pathlib
import typing

import optuna
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
from optuna.trial import TrialState
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

import helios.core as hlc
import helios.data as hld
import helios.model as hlm
import helios.nn as hln
import helios.optim as hlo
import helios.trainer as hlt
from helios.core import logging
from helios.plugins.optuna import OptunaPlugin


class CIFARDataModule(hld.DataModule):
    def __init__(self, root: pathlib.Path, batch_size: int):
        self._root = root / "data"
        self._batch_size = batch_size

    def prepare_data(self):
        torchvision.datasets.CIFAR10(root=self._root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self._root, train=False, download=True)

    def setup(self) -> None:
        transforms = T.Compose(
            [
                hld.transforms.ToImageTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        params = hld.DataLoaderParams()
        params.batch_size = self._batch_size
        params.shuffle = True
        params.num_workers = 2
        params.drop_last = True
        self._train_dataset = self._create_dataset(
            torchvision.datasets.CIFAR10(
                root=self._root,
                train=True,
                download=False,
                transform=transforms,
            ),
            params,
        )

        params.drop_last = False
        params.shuffle = False
        params.is_distributed = False
        self._valid_dataset = self._create_dataset(
            torchvision.datasets.CIFAR10(
                root=self._root,
                train=False,
                download=False,
                transform=transforms,
            ),
            params,
        )


@hln.NETWORK_REGISTRY.register
class Net(nn.Module):
    def __init__(self, l1: int = 120, l2: int = 84):
        super().__init__()

        self._conv1 = nn.Conv2d(3, 6, 5)
        self._pool = nn.MaxPool2d(2, 2)
        self._conv2 = nn.Conv2d(6, 16, 5)
        self._fc1 = nn.Linear(16 * 5 * 5, l1)
        self._fc2 = nn.Linear(l1, l2)
        self._fc3 = nn.Linear(l2, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool(F.relu(self._conv1(x)))
        x = self._pool(F.relu(self._conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = self._fc3(x)
        return x


class ClassifierModel(hlm.Model):
    def __init__(self, batch_size: int):
        super().__init__("classifier")
        self._tune_params: dict[str, float | int] = {}
        self._tune_params["batch_size"] = batch_size

    @property
    def metrics(self) -> dict[str, typing.Any]:
        if not self._val_scores or self._val_scores["total"] == 0:
            return {}

        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct // total
        return {"accuracy": accuracy}

    def setup(self, fast_init: bool = False) -> None:
        plugin = self.trainer.plugins[OptunaPlugin.plugin_id]
        assert isinstance(plugin, OptunaPlugin)

        # Assign the tunable parameters so we can log them as hyper-parameters when
        # training ends.
        self._tune_params["l1"] = plugin.trial.suggest_categorical(
            "l1", [2**i for i in range(9)]
        )
        self._tune_params["l2"] = plugin.trial.suggest_categorical(
            "l2", [2**i for i in range(9)]
        )
        self._tune_params["lr"] = plugin.trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        self._net = Net(
            l1=self._tune_params["l1"],  # type: ignore[arg-type]
            l2=self._tune_params["l2"],  # type: ignore[arg-type]
        ).to(self.device)

        if self.is_distributed:
            self._net = DDP(
                self._net,
                device_ids=[self.rank],
                output_device=self.rank,
            )  # type: ignore[assignment]

        self._criterion = nn.CrossEntropyLoss().to(self.device)
        self._optimizer = hlo.create_optimizer(
            "SGD",
            self._net.parameters(),
            lr=self._tune_params["lr"],
            momentum=0.9,
        )

    def load_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool = False
    ) -> None:
        if not self.is_distributed:
            consume_prefix_in_state_dict_if_present(state_dict["net"], "module.")

        self._net.load_state_dict(state_dict["net"])
        self._optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self) -> dict[str, typing.Any]:
        return {"net": self._net.state_dict(), "optimizer": self._optimizer.state_dict()}

    def train(self) -> None:
        self._net.train()

    def train_step(self, batch: typing.Any, state: hlt.TrainingState) -> None:
        inputs: torch.Tensor
        labels: torch.Tensor

        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self._optimizer.zero_grad()

        outputs = self._net(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()
        self._optimizer.step()  # type: ignore[operator]

        self._loss_items["loss"] = loss

    def on_training_batch_end(
        self, state: hlt.TrainingState, should_log: bool = False
    ) -> None:
        super().on_training_batch_end(state, should_log)

        if should_log:
            root_logger = logging.get_root_logger()
            tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

            loss_val = self._loss_items["loss"]

            root_logger.info(
                f"[{state.global_epoch}, {state.global_iteration:5d}] "
                f"loss {loss_val:.3f}, "
                f"running loss: {loss_val / state.running_iter:.3f} "
                f"avg time: {state.average_iter_time:.2f}s"
            )
            tb_logger.add_scalar("train/loss", loss_val, state.global_iteration)
            tb_logger.add_scalar(
                "train/running loss",
                loss_val / state.running_iter,
                state.global_iteration,
            )

    def on_training_end(self) -> None:
        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct / total
        tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())
        tb_logger.add_hparams(
            self._tune_params,
            {"hparam/accuracy": accuracy, "hparam/loss": self._loss_items["loss"].item()},
        )

        if self.is_distributed and self.rank == 0:
            assert self.trainer.queue is not None
            self.trainer.queue.put(
                {"accuracy": accuracy, "loss": self._loss_items["loss"].item()}
            )

    def eval(self) -> None:
        self._net.eval()

    def on_validation_start(self, validation_cycle: int) -> None:
        super().on_validation_start(validation_cycle)

        self._val_scores["total"] = 0
        self._val_scores["correct"] = 0
        self._val_scores["loss"] = 0
        self._val_scores["steps"] = 0

    def valid_step(self, batch: typing.Any, step: int) -> None:
        images: torch.Tensor
        labels: torch.Tensor
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self._net(images)
        _, predicted = torch.max(outputs.data, 1)
        self._val_scores["total"] += labels.size(0)
        self._val_scores["correct"] += (predicted == labels).sum().item()

        loss = self._criterion(outputs, labels)
        self._val_scores["loss"] += loss.cpu().numpy()
        self._val_scores["steps"] += 1

    def on_validation_end(self, validation_cycle: int) -> None:
        root_logger = logging.get_root_logger()
        tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct // total

        root_logger.info(f"[Validation {validation_cycle}] accuracy: {accuracy}")
        tb_logger.add_scalar("validation/accuracy", accuracy, validation_cycle)
        tb_logger.add_scalar(
            "validation/loss",
            self._val_scores["loss"] / self._val_scores["steps"],
            validation_cycle,
        )

        # Report metrics to the trial
        plugin = self.trainer.plugins[OptunaPlugin.plugin_id]
        assert isinstance(plugin, OptunaPlugin)
        plugin.report_metrics(validation_cycle)

    def append_metadata_to_chkpt_name(self, chkpt_name: str) -> str:
        if len(self._val_scores) == 0:
            return chkpt_name

        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct / total
        chkpt_name += f"_accuracy{round(accuracy, 4)}"
        return chkpt_name


def objective(trial: optuna.Trial) -> float:
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8, 16])

    datamodule = CIFARDataModule(pathlib.Path.cwd(), batch_size)
    model = ClassifierModel(batch_size)
    plugin = OptunaPlugin(trial, "accuracy")

    trainer = hlt.Trainer(
        run_name="cifar10",
        train_unit=hlt.TrainingUnit.EPOCH,
        total_steps=2,
        valid_frequency=1,
        chkpt_frequency=1,
        print_frequency=10,
        enable_tensorboard=True,
        enable_file_logging=True,
        enable_progress_bar=True,
        enable_deterministic=True,
        print_banner=True,
        chkpt_root=pathlib.Path.cwd() / "chkpt",
        log_path=pathlib.Path.cwd() / "logs",
        run_path=pathlib.Path.cwd() / "runs",
    )

    plugin.configure_trainer(trainer)
    plugin.configure_model(model)
    trainer.fit(model, datamodule)
    plugin.check_pruned()

    if trainer.queue is None:
        return model.metrics["accuracy"]

    metrics = trainer.queue.get()
    return metrics["accuracy"]


if __name__ == "__main__":
    # Set the CUBLAS workspace setting to allow determinism to be used in CUDA >= 10.2.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    study = optuna.create_study(
        direction="maximize",
        study_name="classifier",
        storage="sqlite:///classifier.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial: ")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
