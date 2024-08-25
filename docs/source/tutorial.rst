Training a Classifier with Helios
#################################

As a simple example, we're going to implement the
`Training a Classifier <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_
tutorial from PyTorch. For the sake of brevity, we will assume that the reader is familiar
with the process of training networks using PyTorch and focus exclusively on the steps
necessary to accomplish the same task using Helios.

.. note::
   The code for this tutorial is available
   `here <https://github.com/marovira/helios-ml/blob/master/examples/cifar10/cifar10.py>`__.

Project Structure
=================

The first thing we're going to do is create a folder where our virtual environment and the
code will live.

.. code-block:: bash

    mkdir classifier
    cd classifier

Next, let's create a virtual environment and install Helios. All necessary dependencies
will be installed automatically.

.. code-block:: bash

    python3 -m venv .venv # For Windows, replace with python
    . .venv/bin/activate
    # . .venv/Scripts/activate for Windows.
    pip install helios
    touch cifar.py

With that done, let's begin by defining how our data will be managed.


Managing Datasets
=================

We start by setting up our data. In the tutorial, the dataset is downloaded through the
`torchvision`, so we have to make sure we download it as well. In Helios, datasets are
managed through the :py:class:`~helios.data.datamodule.DataModule`. Note that this shares
some similarities with the corresponding class from PyTorch Lightning, so if you're
familiar with that it will be easier to follow along. First, let's add our imports.

.. code-block:: python

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

These will be all the imports we'll need for this tutorial. Next, let's create a new class
for our data:

.. code-block:: python

    class CIFARDataModule(hld.DataModule):
        def __init__(self, root: pathlib.Path) -> None:
            super().__init__()
            self._root = root / "data"

The datamodule will take in as an argument the root where the datasets will be downloaded
to. Next, let's add the code to download the data:

.. code-block:: python

        def prepare_data(self) -> None:
            torchvision.datasets.CIFAR10(root=self._root, train=True, download=True)
            torchvision.datasets.CIFAR10(root=self._root, train=False, download=True)

The :py:meth:`~helios.data.datamodule.DataModule.prepare_data` function will be called
automatically by the Trainer before training starts. If we were training with multiple
GPUs, this would be called *prior* to the creation of the distributed context. Now let's
make the datasets themselves:

.. code-block:: python

        def setup(self) -> None:
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

            params.drop_last = False
            params.shuffle = False
            self._valid_dataset = self._create_dataset(
                torchvision.datasets.CIFAR10(
                    root=self._root, train=False, download=False, transform=transforms
                ),
                params,
            )

There's a few things to note here:

#. Helios ships with a transform that automatically converts images (or arrays of images)
   from their NumPY representation to tensors called
   :py:class:`~helios.data.transforms.ToImageTensor`. The class is ultimately equivalent
   to the following:

   .. code-block:: python

      import torchvision.transforms.v2 as T

      to_image_tensor = T.Compose(
        [T.ToImage(), T.ToDType(dtype=torch.float32, scale=scale), T.ToPureTensor()]
      )

#. The :py:class:`~helios.data.datamodule.DataLoaderParams` object wraps all of the
   settings used to create the dataloader and sampler pair. This is where you can set
   options like batch sizes, number of workers, whether the dataset should be shuffled,
   etc.
#. The ``params`` object can be freely re-used without worrying about settings interfering
   with each other. As soon as ``_create_dataset`` is called, the ``params`` object is
   deep-copied to avoid conflicts.

Making the Model
================

Network
-------

With the datasets ready, we can now turn our attention to the network. The code will be
identical to the one from PyTorch so we won't explain any details.

.. code-block:: python

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

With the network ready, we can implement the other main class from Helios: the model. The
:py:class:`~helios.model.model.Model` class serves as the main holder for the training
code itself. The functionality is provided through different callback functions that are
used by the :py:class:`~helios.trainer.Trainer` at specific points in time. The first one
is the :py:meth:`~helios.model.Model.setup` function which we'll use to initialize all the
necessary members for training. In our case, we need:

* The network itself,
* The optimizer, and
* The loss function.

Following the tutorial, we'll use ``SGD`` for our optimizer and ``CrossEntropyLoss`` for
our loss function. The could would be as follows:

.. code-block:: python

    class ClassifierModel(hlm.Model):
        def __init__(self) -> None:
            super().__init__("classifier")

        def setup(self, fast_init: bool = False) -> None:
            self._net = Net().to(self.device)
            self._criterion = nn.CrossEntropyLoss().to(self.device)

            self._optimizer = hlo.create_optimizer(
                "SGD", self._net.parameters(), lr=0.001, momentum=0.9
            )

A few comments:

#. All classes that derive from :py:class:`~helios.model.model.Model` *must* provide a name to
   the base class. This is used to determine the name that will be given to the
   checkpoints when they are saved (more on this later).
#. Upon training start, the :py:class:`~helios.trainer.Trainer` will automatically set the
   correct ``torch.device`` into the model. This means that any classes that need to be
   moved to the device can do so through the :py:attr:`~helios.model.model.Model.device`
   property.

Registries
----------

One of the main features of Helios is the registry system that it ships with. The
registries can be used to write *re-usable* training code for different networks. The idea
is that a single model class can be written which can then create the necessary
optimizers, loss functions, etc. based on settings which can be provided externally
through a config file (for example). Helios ships with the following registries:

* :py:data:`~helios.data.datamodule.DATASET_REGISTRY`,
* :py:data:`~helios.data.samplers.SAMPLER_REGISTRY`,
* :py:data:`~helios.data.transforms.TRANSFORM_REGISTRY`,
* :py:data:`~helios.losses.utils.LOSS_REGISTRY`,
* :py:data:`~helios.metrics.metrics.METRICS_REGISTRY`,
* :py:data:`~helios.model.utils.MODEL_REGISTRY`,
* :py:data:`~helios.nn.utils.NETWORK_REGISTRY`,
* :py:data:`~helios.optim.utils.OPTIMIZER_REGISTRY`,
* :py:data:`~helios.scheduler.utils.SCHEDULER_REGISTRY`

Each registry comes with an associated ``create_`` function that will create the
corresponding type from the registry.

By default, the optimizer and scheduler registries ship with the classes that PyTorch
offers for each type. In our example, we could create the optimizer directly as follows:

.. code-block:: python

    from torch import optim

    self._optimizer = optim.SGD(self._net.parameters(), lr=0.001, momentum=0.9)

Alternatively, we can create it by directly through the registry as follows:

.. code-block:: python

    self._optimizer = hlo.create_optimizer(
        "SGD", self._net.parameters(), lr=0.001, momentum=0.9
    )

Note that here we're manually specifying the arguments to the optimizer, but we could have
just as easily stored the arguments in a dictionary (that were loaded from a file or
passed in as a command-line argument) and then passed them in as follows:

.. code-block:: python

    # These args are passed in externally.
    args = {"lr": 0.001, "momentum": 0.9}
    self._optimizer = hlo.create_optimizer("SGD", self._net.parameters(), **args)

This would allow us to re-use the same model with different combinations of networks and
optimizers, reducing code duplication and allowing the code to be standardised across
combinations of settings.

Checkpoints
-----------

Now that the loss and optimizer have been created, we turn our attention to checkpoints.
The :py:class:`~helios.trainer.Trainer` is designed to automatically save checkpoints at
predetermined intervals. The checkpoints store all the necessary state to ensure training
can be resumed. As part of the state stored, the model is able to add it's own state. In
our case, we would like to save the state of the network, optimizer, and loss function. To
do this, we need to override :py:meth:`~helios.model.model.Model.load_state_dict` and
:py:meth:`~helios.model.model.Model.state_dict`. The code is:

.. code-block:: python

    def load_state_dict(
        self, state_dict: dict[str, typing.Any], fast_init: bool = False
    ) -> None:
        self._net.load_state_dict(state_dict["net"])
        self._criterion.load_state_dict(state_dict["criterion"])
        self._optimizer.load_state_dict(state_dict["optimizer"])

    def state_dict(self) -> dict[str, typing.Any]:
        return {
            "net": self._net.state_dict(),
            "criterion": self._criterion.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

Similarly to the device, the model *should not* remap any weights from the loaded
checkpoint. Those will be automatically mapped by the :py:class:`~helios.trainer.Trainer`
when the checkpoint is loaded.

Training
--------

We can now focus on the training code itself. It is recommended that you read through the
documentation for the :py:class:`~helios.model.model.Model` so you are aware of all the
callbacks available for training, which can be identified by the prefix
``on_training_...``. For our purposes, we're going to need the following:

* We're going to trace the network and log it to tensorboard.
* We need to perform the forward and backward passes.
* We need to log the value of our loss function on each iteration.
* When training is done, we also want to log the final validation score as well as the
  final value of the loss function.

To start, let's add the code to switch our network into training mode:

.. code-block:: python

    def train(self) -> None:
        self._net.train()

Next, lets add the code to trace. Since we only need to do this once when training begins,
we're going to use :py:meth:`~helios.model.model.Model.on_training_start`:

.. code-block:: python

    def on_training_start(self) -> None:
        tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

        x = torch.randn((1, 3, 32, 32)).to(self.device)
        tb_logger.add_graph(self._net, x)

The Tensorboard writer is automatically created by the :py:class:`~helios.trainer.Trainer`
if requested to do so. As a result, :py:func:`~helios.core.logging.get_tensorboard_writer`
can return ``None``. We could ensure that it's valid by doing:

.. code-block:: python

    logger = logging.get_tensorboard_writer()
    if logger is not None:
        ...
    # Or alternatively:
    assert logger is not None

This is especially necessary when using linters like Mypy. Since this gets repetitive very
quickly, we can instead use :py:func:`~helios.core.utils.get_from_optional`, which ensures
that the provided value is not ``None`` and returns it in a way that Mypy correctly
identifies. Now to add the forward and backward passes. These are going to be kept in
:py:meth:`~helios.model.model.Model.train_step`:

.. code-block:: python

    def train_step(self, batch: typing.Any, state: hlt.TrainingState) -> None:
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self._optimizer.zero_grad()

        outputs = self._net(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()
        self._optimizer.step()

        self._loss_items["loss"] = loss

There's a few things to unpack here, so let's go one by one:

#. The type of the ``batch`` parameter is determined by our dataset. In the case of the
   CIFAR10 dataset, the batch is a tuple of tensors containing the inputs and labels. Note
   that the base model class imposes no restrictions on what the batch is.
#. Since the base model class makes no assumptions on the type of the batch, we need to
   move the components of the batch to the target device ourselves. This gives maximum
   flexibility since you can choose what (if anything) gets moved. Note that similarly to
   the creation of the network itself, we use the
   :py:attr:`~helios.model.model.Model.device` property.
#. We're going to store the returned loss into the ``_loss_items`` dictionary. This allows
   the model to automatically gather the tensors for us if we were doing distributed
   training.

Now let's look at the logging code:

.. code-block:: python

    def on_training_batch_end(
        self,
        state: hlt.TrainingState,
        should_log: bool = False,
    ) -> None:
        super().on_training_batch_end(state, should_log)

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

Let's examine each part independently:

#. The call to ``super().on_training_batch_end`` will automatically gather any tensors
   stored in the ``_loss_items`` dictionary if we're in distributed mode, so we don't have
   to manually do it ourselves.
#. When the :py:class:`~helios.trainer.Trainer` is created, we can specify the interval at
   which logging should occur. Since
   :py:meth:`~helios.model.model.Model.on_training_batch_end` is called on at the end of
   *every* batch, the ``should_log`` flag is used to indicate when logging should happen.

.. note::
   In our example, we're performing both the forward and backward passes in
   :py:meth:`~helios.model.model.Model.train_step`. That being said, it is possible to
   split the forward and backward passes and have them occur in
   :py:meth:`~helios.model.model.Model.train_step` and
   :py:meth:`~helios.model.model.Model.on_training_batch_end` if it makes sense for your
   workflow.

The rest of the code is pretty self-explanatory, with us just grabbing the Tensorboard
logger just like before. Note that we also call
:py:func:`~helios.core.logging.get_root_logger`, so let's discuss how Helios manages
logging.

Logging
-------

By default, Helios provides two loggers:

* :py:class:`~helios.core.logging.RootLogger`: logs to a file and to stdout.
* :py:class:`~helios.core.logging.TensorboardWriter`: wraps the PyTorch Tensorboard writer
  class.

.. note::
   The :py:class:`~helios.core.logging.RootLogger` will *always* be created with stream
   output by default. This behaviour *cannot* be changed, as it is used to correctly
   forward error messages that may occur during training. The logging to a file can be
   toggled on/off based on the arguments provided to the
   :py:class:`~helios.trainer.Trainer` upon construction.


The creation of these is handled by the :py:class:`~helios.trainer.Trainer`, and will be
performed before training starts. If training is distributed, both loggers are designed to
only log on the process whose rank is 0. In the event that training occurs over multiple
nodes, then logging is performed on the process whose *global* rank is 0. The loggers can
be obtained through :py:func:`~helios.core.logging.get_root_logger` and
:py:func:`~helios.core.logging.get_tensorboard_writer`.

.. warning::
   Only the :py:class:`~helios.core.logging.RootLogger` is guaranteed to exist. In the
   event that the trainer is created with Tensorboard logging disabled,
   :py:func:`~helios.core.logging.get_tensorboard_writer` will return ``None``.

Now that we have logged the training losses, let's add the code to log the final
validation result as well as the final loss value.

.. code-block:: python

    def on_training_end(self) -> None:
        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct // total
        writer = hlc.get_from_optional(logging.get_tensorboard_writer())
        writer.add_hparams(
            {"lr": 0.001, "momentum": 0.9, "epochs": 2},
            {"hparam/accuracy": accuracy, "hparam/loss": self._loss_items["loss"].item()},
        )

We will explain how validation works in the next section. The code itself is
self-explanatory: we compute the final accuracy and then log it to the Tensorboard writer.

Validation
----------

Similarly to the suite of callbacks used for training, the
:py:class:`~helios.model.model.Model` class has a set of functions for both validation and
testing. In our example, we want to perform validation, so let's first add a function to
switch our network to evaluation mode:

.. code-block:: python

    def eval(self) -> None:
        self._net.eval()

The :py:class:`~helios.model.model.Model` contains a dictionary for validation scores
similar to the one we used earlier for loss values. In our example, we need to keep track
of the number of labels we have seen, and how many of those labels have been correct. To
do this, we're going to assign these fields before validation starts:

.. code-block:: python

    def on_validation_start(self, validation_cycle: int) -> None:
        super().on_validation_start(validation_cycle)

        self._val_scores["total"] = 0
        self._val_scores["correct"] = 0

Calling :py:meth:`~helios.mode.model.Model.on_validation_start` on the base class
automatically clears out the ``_val_scores`` dictionary to ensure we don't accidentally
over-write or overlap values. After setting the fields we care about, let's perform the
validation step:

.. code-block:: python

    def valid_step(self, batch: typing.Any, step: int) -> None:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self._net(images)

        _, predicted = torch.max(outputs.data, 1)
        self._val_scores["total"] += labels.size(0)
        self._val_scores["correct"] += (predicted == labels).sum().item()

The :py:meth:`~helios.model.model.Model.valid_step` function is analogous to
:py:meth:`~helios.model.model.Model.train_step`. Like before, we receive the batch from
our dataset and we are responsible for moving the data into the appropriate device using
:py:attr:`~helios.model.model.Model.device`. The rest of the code is identical to the
PyTorch tutorial, with the only difference that we assign the results to the fields we
added before validation began.

Finally, we need to compute the final accuracy score and log it:

.. code-block:: python

    def on_validation_end(self, validation_cycle: int) -> None:
        root_logger = logging.get_root_logger()
        tb_logger = hlc.get_from_optional(logging.get_tensorboard_writer())

        total = self._val_scores["total"]
        correct = self._val_scores["correct"]
        accuracy = 100 * correct // total

        root_logger.info(f"[Validation {validation_cycle}] accuracy: {accuracy}")
        tb_logger.add_scalar("val", accuracy, validation_cycle)

Creating the Trainer
====================

Now that we have all of our training code ready, all that is left is to create the trainer
and train our network. For the sake of simplicity, we're going to be performing this in
the main block of our script. The trainer requires two things to train:

#. The model we want to use.
#. The datamodule with our datasets.

Let's make those first:

.. code-block:: python

    if __name__ == "__main__":
        datamodule = CIFARDataModule(pathlib.Path.cwd())
        model = ClassifierModel()

Now let's create the trainer itself:

.. code-block:: python

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

The :py:class:`~helios.trainer.Trainer` constructor takes a long list of arguments that
provide control over various aspects of training. You're encouraged to read through the
list of parameters for more details. Let's go over each of the arguments we set in our
example, starting with the training unit.

Training Units
--------------

The :py:class:`~helios.trainer.Trainer` provides two ways of training networks based on
the *training unit*. These are:

#. :py:attr:`~helios.trainer.TrainingUnit.ITERATION`: used when the network needs to be
   trained for :math:`N` iterations.
#. :py:attr:`~helios.trainer.TrainingUnit.EPOCH`: used when the network needs to be
   trained for :math:`N` epochs.

The choice of training unit determines the behaviour of certain portions of the training
loop, which we will discuss next.

Training by Epoch
^^^^^^^^^^^^^^^^^

This is the most common case for training. In this mode, the training loop will run until
the number of epochs specified by ``total_steps`` has been reached and it has the
following behaviour:

* ``valid_frequency`` and ``chkpt_frequency`` occur on epochs. For example, say that we
  want to train for 10 epochs and we want to perform validation every second epoch. This
  means that validation will occur on epochs 2, 4, 5, 8, and 10. Likewise, if we want to
  save checkpoints every second epoch, then checkpoints will be saved on epochs 2, 4, 5,
  8, and 10.
* Early stopping is performed on epochs. See :ref:`stopping-training`.
* Gradient accumulation has no effect on the number of epochs. See
  :ref:`gradient-accumulation`.

.. note::
   ``print_frequency`` **always** refers to the number of iterations that logging should
   occur in. This is *independent* of the training unit.

Training by Iteration
^^^^^^^^^^^^^^^^^^^^^

In this mode, the training loop will run until the number of iterations specified by
``total_steps`` has been reached *regardless* of how many epochs (complete or fractional)
are performed. It has the following behaviour:

* ``valid_frequency`` and ``chkpt_frequency`` occur on iterations. For example, say that
  we want to train for 10k iterations and we want to perform validation every 2k
  iterations. This means that validation will occur on iterations 2k, 4k, 6k, 8k, and 10k.
  Likewise, if we want to save checkpoints every 2k iterations, then checkpoints will be
  saved on iterations 2k, 4k, 6k, 8k, and 10k.
* Early stopping is performed on iterations. See :ref:`stopping-training`.
* Gradient accumulation multiplies the total number of iterations. See
  :ref:`gradient-accumulation`.

Enabling Logging and Checkpoints
--------------------------------

The next 3 arguments of the trainer cover the various kinds of logging that are available.
As mentioned previously, the :py:class:`~helios.trainer.Trainer` will *always* create the
:py:class:`~helios.core.logging.RootLogger` with output to stdout. That said, we can add
logging to a file and to Tensorboard by setting the corresponding flags:

* ``enable_tensorboard``: enables the Tensorboard writer.
* ``enable_file_logging``: adds a file stream to the log.

.. warning::
   If either of ``enable_tensorboard`` or ``enable_file_logging`` is set, then you
   **must** also set ``run_path`` or ``log_path`` respectively. These should be set to a
   directory where the logs will be saved. Note that if the directory doesn't exist, it
   will be created automatically.

The final logging flag determines whether a progress bar is displayed while training is
ongoing. See :ref:`logging` for more details.

Finally, since we want to save checkpoints, then we also assign the path that the
checkpoints are saved to using ``chkpt_root``.

.. warning::
   If ``chkpt_frequency`` is not 0, then you **must** set ``chkpt_root`` to the directory
   where checkpoints are saved. Note that if the directory doesn't exist, it will be
   created automatically.

See :ref:`checkpoint-saving`.

We also set ``enable_deterministic`` to indicate to PyTorch that we want to use
deterministic operations while training. This belongs to a set of flags that configure the
environment when the trainer is created. See .

Launching Training
==================

The final step is to start training. With the trainer created, all that we have to do is
this:

.. code-block:: python

    trainer.fit(model, datamodule)

And that's it! Helios will automatically configure the training environment and run the
training loop for the specified number of epochs. Every epoch validation will be performed
and a checkpoint will be saved.

Helios provides more functionality than what is shown here, so you are encouraged to read
through the quick reference guide for more details.
