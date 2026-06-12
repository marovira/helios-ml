Weights & Biases
################

Helios supports `Weights & Biases <https://wandb.ai>`__ (W&B) as an optional logging
backend. This page covers installation, configuration, directory layout, and run
resumption.

Installation
============

W&B is an optional dependency. Install it alongside Helios with::

    pip install -U helios-ml[wandb]

Configuration
=============

To enable W&B logging, pass a ``wandb_args`` dictionary to the
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

The ``wandb_args`` dictionary uses the
:py:class:`~helios.core.loggers.wandb.WandbArgs` TypedDict. Its fields are described
below:

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

Directory Layout
================

When ``log_root`` is provided to the :py:class:`~helios.trainer.Trainer`, W&B data is
written under ``log_root/wandb/``. If ``log_root`` is not set, W&B uses its own default
directory.

Run Resumption
==============

When a checkpoint is saved, the current W&B run ID is stored inside it. On resume,
Helios retrieves the saved run ID and passes it back to :func:`wandb.init` with
``resume="allow"``, so new data is appended to the original run rather than starting a
fresh one. No additional configuration is required.

.. note::
   In distributed training, the W&B writer is only initialised on rank 0. All other
   ranks skip W&B entirely.
