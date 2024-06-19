Getting Started
===============

Installation
------------

You can install Helios using ``pip``::

    pip install -U helios-ml

If you require a specific version of CUDA, you can install with::

    pip install -U helios-ml --extra-index-url https://download.pytorch.org/whl/cu<version>

Overview
--------

Helios contains three main classes that handle most of the training code. These are:

* :py:class:`~helios.data.datamodule.DataModule`
* :py:class:`~helios.model.model.Model`
* :py:class:`~helios.trainer.Trainer`

DataModule
~~~~~~~~~~

This class bundles all of the logic for setting up and creating the datasets and
dataloaders used for testing. This allows multiple models to re-use datasets with
standardised settings without having to duplicate code.

As part of the creation of the dataloaders, Helios ensures that all worker processes are
correctly seeded to ensure reproducibility. In addition to this, samplers are used that
guarantee the ability to restart training if it is stopped at *any* time.

Model
~~~~~

This is the class that handles all of the training, validation, and testing logic for your
network(s). Specifically, it bundles the following:

* Creation, loading, and saving of network(s) and their associated training state. This
  includes things like schedulers, optimizers, loss functions, etc.
* The training, validation, and testing steps.

In addition to these, the :py:class:`~helios.model.model.Model` is also equipped with a
series of callbacks that happen at different times during training, validation and
testing. In summary, it contains callbacks for:

* When training starts/ends,
* When a training epoch starts/ends,
* When a training batch starts/ends,
* When validation starts/ends,
* When a validation batch starts/ends,
* When testing starts/ends,
* When a testing batch starts/ends.

These can be used to accomplish a variety of tasks such as logging, custom scheduler
steps, etc.

Trainer
~~~~~~~

The trainer is the main class that holds the main training, testing, and validation loops.
In addition, it also the one in charge of ensuring the
:py:class:`~helios.model.model.Model` and the
:py:class:`~helios.data.datamodule.DataModule` are correctly initialised. This involves
the following:

* Ensuring that the correct mappings for the current device(s) are set and are made
  available to the corresponding classes.
* Handling the correct initialisation for distributed training. This can be accomplished
  automatically or through ``torchrun``. If multiple GPUs are present, Helios is able to
  automatically detect them and start distributed training on all available devices.
  Alternatively, the user can manually specify which devices they wish to use.

In addition, it will also perform the following tasks:

* Automatically print and save checkpoints with the specified frequency,
* Capture any exceptions and log them before exiting,
* Stop training when requested.

The :py:class:`~helios.trainer.Trainer` also natively supports the following logging
schemes (which can be manually enabled):

* File logging,
* Tensorboard.

In the next section, we will go over a brief example on how to use Helios.
