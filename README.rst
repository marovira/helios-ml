.. -*- mode: rst -*-

.. image:: https://raw.githubusercontent.com/marovira/helios-ml/master/data/logo/logo-transparent.png
   :target: https://github.com/marovira/helios-ml

|Test| |Codecov| |CodeFactor| |Ruff| |PythonVersion| |PyPi| |License|

.. |Test| image:: https://github.com/marovira/helios-ml/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/marovira/helios-ml/actions/workflows/tests.yml

.. |Codecov| image:: https://codecov.io/github/marovira/helios-ml/graph/badge.svg?token=NFC0GBJC6J
   :target: https://codecov.io/github/marovira/helios-ml

.. |CodeFactor| image:: https://www.codefactor.io/repository/github/marovira/helios-ml/badge
   :target: https://www.codefactor.io/repository/github/marovira/helios-ml

.. |Ruff| image:: https://img.shields.io/badge/code%20style-ruff-red
   :target: https://github.com/astral-sh/ruff

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/helios-ml.svg
   :target: https://pypi.org/project/helios-ml/

.. |PyPi| image:: https://img.shields.io/pypi/v/helios-ml.svg
   :target: https://pypi.org/project/helios-ml/

.. |License| image:: https://img.shields.io/pypi/l/helios-ml.svg
   :target: https://opensource.org/license/bsd-3-clause

.. what_is_helios

What is Helios?
---------------

Named after Greek god of the sun, Helios is a light-weight package for training ML
networks built on top of PyTorch. It is designed to abstract all of the "boiler-plate"
code involved with training. Specifically, it wraps the following common patterns:

- Creation of the dataloaders.
- Initialisation of CUDA, PyTorch, and random number states.
- Initialisation for distributed training.
- Training, validation, and testing loops.
- Saving and loading checkpoints.
- Exporting to ONNX.

It is important to note that Helios is **not** a fully fledged training environment similar
to `PyTorch Lightning <https://github.com/Lightning-AI/pytorch-lightning>`__. Instead,
Helios is focused on providing a simple and straight-forward interface that abstracts most
of the common code patterns while retaining the ability to be easily overridden to suit
the individual needs of each training scheme.

.. quick_start

Quick Start
-----------

Training with Helios requires three steps:

1. Subclass ``DataModule`` to define your datasets and dataloaders.
2. Subclass ``Model`` to implement your training logic.
3. Create a ``Trainer`` and call ``fit()``.

.. code-block:: python

   import helios
   import helios.data as hld
   import helios.model as hlm

   class MyDataModule(hld.DataModule):
       def setup(self) -> None:
           dataset = MyDataset(...)
           params = hld.DataLoaderParams(batch_size=32)
           self._add_train_phase(dataset, params)

   class MyModel(hlm.Model):
       def setup(self, for_inference: bool = False) -> None:
           self._net = MyNetwork()
           self._optimizer = ...
           self._criterion = ...

       def train_step(self, batch, state: helios.TrainingState) -> None:
           inputs, labels = batch
           self._optimizer.zero_grad()
           loss = self._criterion(self._net(inputs), labels)
           loss.backward()
           self._optimizer.step()

   trainer = helios.Trainer()
   trainer.fit(MyModel("my_model"), MyDataModule())

.. main_features

Main Features
-------------

Helios offers the following functionality out of the box:

#. Resume training: Helios has been built with the ability to resume training if it is
   paused. Specifically, Helios will ensure that the behaviour of the trained model is
   *identical* to the one it would've had if it had been trained without pauses.
#. Automatic detection of multi-GPU environments for distributed training. In addition,
   Helios also supports training using ``torchrun`` and will automatically handle the
   initialisation and clean up of the distributed state. It will also correctly set the
   devices and maps to ensure weights are mapped to the correct location.
#. Registries for creation of arbitrary types. These include: networks, loss functions,
   optimisers, schedulers, etc.
#. Correct handling of logging when doing distributed training (even over multiple nodes).
#. Native integration with Optuna for hyper-parameter optimisation. Also supports resuming
   studies and generating checkpoints to ensure reproducibility.

.. why_helios

Why Helios?
-----------

Compared to larger frameworks, Helios prioritises explicitness and simplicity over
automation. The table below compares it against PyTorch Lightning and Ignite across the
most important areas for research and engineering work:

.. list-table::
   :header-rows: 1
   :widths: 30 23 23 24

   * - Feature
     - Helios
     - Lightning
     - Ignite
   * - Mixed precision
     - Helper functions
     - Automatic
     - Manual
   * - Reproducible resume
     - Built-in
     - Partial
     - Manual
   * - Distributed training
     - ``torchrun``
     - ``torchrun``
     - Manual
   * - Boilerplate style
     - Explicit
     - Magic hooks
     - Event-based
   * - Training unit
     - First-class (``EPOCH``/``ITERATION``)
     - Epoch-based
     - Epoch-based
   * - Gradient accumulation
     - Iteration-aware
     - Epoch-based
     - Manual
   * - Registry system
     - Built-in
     - None
     - None
   * - Learning curve
     - Low
     - Medium
     - High
   * - Debuggability
     - High
     - Medium
     - High

**Registry system.** Helios provides typed global registries (``MODEL_REGISTRY``,
``DATASET_REGISTRY``, ``OPTIMIZER_REGISTRY``, and others) that map string names to
types. Any component can be created by name, enabling config-file-driven experiments and
trivial component swaps without changing training code. Register a class with
``@REGISTRY.register`` and create an instance with ``create_model("MyModel", ...)``.

**Core philosophy.** Helios removes training boilerplate without hiding what is
happening. You can always read the code and know exactly what runs at every point in the
training loop.

.. installation

Installation
------------

Install Helios using ``pip``::

    pip install -U helios-ml

The following optional features are available as install extras:

- Optuna integration for hyper-parameter tuning: ``pip install -U helios-ml[tune]``
- Tensorboard logging: ``pip install -U helios-ml[tensorboard]``
- Weights & Biases logging: ``pip install -U helios-ml[wandb]``

If you require a specific version of CUDA, you can install with::

    pip install -U helios-ml --extra-index-url https://download.pytorch.org/whl/cu<version>

Documentation
-------------

Documentation available `here <https://marovira.github.io/helios-ml>`__.

Contributing
------------

There are three ways in which you can contribute to Helios:

- If you find a bug, please open an issue. Similarly, if you have a question
  about how to use it, or if something is unclear, please post an issue so it
  can be addressed.
- If you have a fix for a bug, or a code enhancement, please open a pull
  request. Before you submit it though, make sure to abide by the rules written
  below.
- If you have a feature proposal, you can either open an issue or create a pull
  request. If you are submitting a pull request, it must abide by the rules
  written below. Note that any new features need to be approved by me.

If you are submitting a pull request, the guidelines are the following:

1. Ensure that your code follows the standards and formatting of Helios. The coding
   standards and formatting are enforced through the Ruff Linter and Formatter. Any
   changes that do not abide by these rules will be rejected. It is your responsibility to
   ensure that both Ruff and Mypy linters pass.
2. Ensure that *all* unit tests are working prior to submitting the pull
   request. If you are adding a new feature that has been approved, it is your
   responsibility to provide the corresponding unit tests (if applicable).

License
-------

Helios is published under the BSD-3 license and can be viewed
`here <https://raw.githubusercontent.com/marovira/helios-ml/master/LICENSE>`__.
