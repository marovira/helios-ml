Tensorboard
###########

Helios supports Tensorboard as an optional logging backend. This page covers
installation, configuration, directory layout, run resumption, and the available
logging functions.

Installation
============

Tensorboard is an optional dependency. Install it alongside Helios with::

    pip install -U helios-ml[tensorboard]

Configuration
=============

To enable Tensorboard, set ``enable_tensorboard=True`` and provide a ``log_root`` when
constructing the :py:class:`~helios.trainer.Trainer`:

.. code-block:: python

   import helios
   import pathlib

   trainer = helios.Trainer(
       log_root=pathlib.Path("logs"),
       enable_tensorboard=True,
   )

Directory Layout
================

Tensorboard data is written under ``log_root/tensorboard/``. Each run creates a
subdirectory named ``<run_name>_<date_time>``.

Run Resumption
==============

When a checkpoint is saved, the path to the current Tensorboard run directory is stored
inside it. On resume, Helios restores the saved path so that new data is appended to the
existing run rather than starting a fresh one.

.. note::
   In distributed training, the Tensorboard writer is only initialised on rank 0. All
   other ranks skip Tensorboard entirely.

Logging Functions
=================

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
