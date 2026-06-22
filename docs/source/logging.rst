Logging
#######

Helios supports multiple logging backends. By default, Helios will use the :py:class:`~helios.core.loggers.RootLogger` for all logging operations. You can optionally enable two other backends:

* Tensorboard,
* Weights & Biases.

For each optional backend, we'll cover how to install, configure, and use the backend.

Tensorboard
===========

Tensorboard is an optional dependency that can be installed alongside Helios with:

    pip install -U helios-ml[tensorboard]

It can then be enabled by passing:

#. ``enable_tensorboard=True`` and
#. ``log_root``

when constructing the :py:class:`~helios.trainer.Trainer`:

.. code-block:: python

   import helios
   import pathlib

   trainer = helios.Trainer(
       log_root=pathlib.Path("logs"),
       enable_tensorboard=True,
   )


Each run creates a folder called ``<run_name>_<date_time>`` under
``<log_root>/tensorboard``. When a checkpoint is saved, this path is stored alongside the
training state. When Helios resumes training, the saved path is restored and new data is
appended to the existing run rather than starting a new one.

.. note::
   In distributed training, the Tensorboard writer is only initialised on rank 0. All
   other ranks skip Tensorboard entirely.

The writer is accessible via :py:func:`~helios.core.loggers.get_tensorboard_writer`.
It wraps :py:class:`torch.utils.tensorboard.SummaryWriter` and exposes the following
functions:

* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_scalar`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_scalars`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_image`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_images`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_figure`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_text`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_graph`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_pr_curve`
* :py:func:`~helios.core.loggers.tensorboard.TensorboardWriter.add_hparams`

Weights & Biases
================

W&B is an optional dependency that can be installed alongside Helios with

    pip install -U helios-ml[wandb]

It can then be enabled by passing a ``wandb_args`` dictionary to the
:py:class:`~helios.trainer.Trainer` constructor:

.. code-block:: python

   import helios

   trainer = helios.Trainer(
       log_root=pathlib.Path("logs"),
       wandb_args={
           "project": "my-project",
           "name": "run-1",
           "config": {"lr": 0.001, "batch_size": 32},
       },
   )

The ``wandb_args`` is represented by :py:class:`~helios.core.loggers.wandb.WandbArgs` and
contains the following fields:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Key
     - Required
     - Description
   * - ``project``
     - Yes
     - W&B project name.
   * - ``name``
     - No
     - Display name for the run shown in the W&B UI. If not provided, defaults to
       the ``run_name`` set on the :py:class:`~helios.trainer.Trainer`.
   * - ``config``
     - No
     - Hyper-parameter dictionary to associate with the run.
   * - ``extra_args``
     - No
     - Additional keyword arguments forwarded verbatim to :func:`wandb.init`.

When ``log_root`` is provided to the :py:class:`~helios.trainer.Trainer`, W&B data is
written under ``log_root/wandb``, otherwise W&B will use its own default directory.
Similarly to the Tensorboard writer, Helios will store the W&B id in the checkpoint. When
Helios resumes training, the saved id is restored and passed back to ``wandb.init()`` with
``resume="allow"`` so new data is appended to the existing run.

.. note::
   In distributed training, the W&B writer is only initialised on rank 0. All other
   ranks skip W&B entirely.
