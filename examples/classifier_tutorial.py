# Getting started tutorial code.
#
# The code is adapted from PyTorch's "Training a Classifier" tutorial:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import os
import pathlib
import typing

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
from torch import nn

import helios.core as hlc
import helios.data as hld
import helios.model as hlm
import helios.optim as hlo
import helios.trainer as hlt
from helios.core import logging


class CIFARDataModule(hld.DataModule):
    def __init__(self, root: pathlib.Path) -> None:
        super().__init__()
        self._root = root / "data"

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(root=self._root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self._root, train=False, download=True)

    def setup(self) -> None:
        # Use the ToImageTensor transform from Helios to automate the conversion from
        # images to tensors.
        transforms = T.Compose(
            [
                hld.transforms.ToImageTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        params = hld.DataLoaderParams()
        params.batch_size = 4
        params.shuffle = True
        params.num_workers = 2
        params.drop_last = True
        self._train_dataset = self._create_dataset(
            torchvision.datasets.CIFAR10(
                root=self._root, train=True, download=False, transform=transforms
            ),
            params,
        )

        # The dataloader params are copied when the dataset is created, so we can safely
        # change the options for the validation dataset without interfering with the ones
        # for training.
        params.drop_last = False
        params.shuffle = False
        self._valid_dataset = self._create_dataset(
            torchvision.datasets.CIFAR10(
                root=self._root, train=False, download=False, transform=transforms
            ),
            params,
        )


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ClassifierModel(hlm.Model):
    def __init__(self) -> None:
        super().__init__("classifier")

    def setup(self, fast_init: bool = False) -> None:
        # Note that when we create the network and loss function, we immediately move them
        # to the current device, which has been set by the trainer.
        self._net = Net().to(self.device)
        self._criterion = nn.CrossEntropyLoss().to(self.device)

        # Note that SGD is shipped as part of the default optimizers from Helios, so we
        # can directly request it from create_optimizer instead of building it ourselves.
        self._optimizer = hlo.create_optimizer(
            "SGD", self._net.parameters(), lr=0.001, momentum=0.9
        )

    def load_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool = False
    ) -> None:
        # Note that we don't have to re-map the weights ourselves. They have already been
        # re-mapped for us by the trainer when it loaded the checkpoint.
        self._net.load_state_dict(state_dict["net"])
        self._criterion.load_state_dict(state_dict["criterion"])
        self._optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self) -> dict[str, typing.Any]:
        return {
            "net": self._net.state_dict(),
            "criterion": self._criterion.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def train(self) -> None:
        # If we had more networks, we would shift them to training mode here.
        self._net.train()

    def on_training_start(self) -> None:
        tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

        x = torch.randn((1, 3, 32, 32)).to(self.device)
        tb_logger.add_graph(self._net, x)

    def train_step(self, batch: typing.Any, state: hlt.TrainingState) -> None:
        # Due to the simplicity of the code, we do both the forward and backward passes in
        # the training step, but you could also split it between the train_step and
        # on_training_batch_end if your setup is more complex.
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self._optimizer.zero_grad()

        outputs = self._net(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()
        self._optimizer.step()  # type: ignore[operator]

        # Note that we save the value of the loss function to the _loss_items dictionary.
        # This allows the model to automatically gather the losses for us if we are
        # training in distributed mode.
        self._loss_items["loss"] = loss

    def on_training_batch_end(
        self,
        state: hlt.TrainingState,
        should_log: bool = False,
    ) -> None:
        # If we were training in distributed mode, calling the base Model's function will
        # automatically gather all loss values saved to self._loss_items.
        super().on_training_batch_end(state, should_log)

        # This flag is set to true whenever the number of iterations is a multiple of the
        # logging frequency we set when creating the trainer.
        if should_log:
            root_logger = logging.get_root_logger()
            tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

            loss_val = self._loss_items["loss"]

            root_logger.info(
                f"[{state.global_epoch + 1}, {state.global_iteration:5d}] "
                f"loss: {loss_val:.3f}, "
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
        # For our example, we're going to save the hyper-params to the tensorboard log so
        # we can compare with other runs.
        # Notice that self._val_scores is active because we validate every epoch. If your
        # validation frequency is different, you may need to alter this code.
        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct // total
        writer = hlc.get_from_optional(logging.get_tensorboard_writer())
        writer.add_hparams(
            {"lr": 0.001, "momentum": 0.9, "epochs": 2},
            {"hparam/accuracy": accuracy, "hparam/loss": self._loss_items["loss"].item()},
        )

    def eval(self) -> None:
        self._net.eval()

    def on_validation_start(self, validation_cycle: int) -> None:
        # The base function will automatically clear the validation scores for us.
        super().on_validation_start(validation_cycle)

        # If you need to add further data to the table, you can do so here. In our case,
        # we're going to add the total number of labels seen and how many of those were
        # correct so we can compute the accuracy metric.
        self._val_scores["total"] = 0
        self._val_scores["correct"] = 0

    def valid_step(self, batch: typing.Any, step: int) -> None:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self._net(images)

        _, predicted = torch.max(outputs.data, 1)
        self._val_scores["total"] += labels.size(0)
        self._val_scores["correct"] += (predicted == labels).sum().item()

    def on_validation_end(self, validation_cycle: int) -> None:
        root_logger = logging.get_root_logger()
        tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

        # Grab the validation scores and compute the accuracy metric so we can log it. If
        # we were in distributed mode, you would need to gather the values here.
        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct // total

        root_logger.info(f"[Validation {validation_cycle}] accuracy: {accuracy}")
        tb_logger.add_scalar("val", accuracy, validation_cycle)


if __name__ == "__main__":
    # Set the CUBLAS workspace setting to allow determinism to be used in CUDA >= 10.2.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    datamodule = CIFARDataModule(pathlib.Path.cwd())
    model = ClassifierModel()

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
        chkpt_root=pathlib.Path.cwd() / "chkpt",
        log_path=pathlib.Path.cwd() / "logs",
        run_path=pathlib.Path.cwd() / "runs",
    )

    trainer.fit(model, datamodule)
