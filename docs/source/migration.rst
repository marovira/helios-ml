Migration Guide
===============

This guide covers the changes introduced in 2.0.0 that require updates to existing
1.x code. The changes are grouped by area and ordered by level of impact.

Checkpoint Migration
--------------------

The checkpoint format has changed in 2.0.0, making Helios unable to load checkpoints
created by 1.x. If you wish to load checkpoints, you must migrate them first using the
``chkpt_migrator`` tool::

    python -m helios.chkpt_migrator <checkpoint_path>

Run ``python -m helios.chkpt_migrator --help`` for the full list of options.

.. warning::
   Attempting to load a 1.x checkpoint in 2.0.0 without migrating it first will result
   in an error.

Tensorboard is Now Optional
---------------------------

Tensorboard has been turned into an optional dependency. If you already have it installed,
no changes are necessary. If you wish to continue using it, install the ``tensorboard``
extra::

    pip install -U helios-ml[tensorboard]

If Tensorboard is not installed, calls to create a
:py:class:`~helios.core.loggers.TensorboardWriter` will raise an import error at runtime.

Logging Module Replaced
------------------------

The ``helios.core.logging`` module has been removed and replaced by the
``helios.core.loggers`` subpackage. The new module provides the same functionality but
with a cleaner API and support for additional logging backends (Weights & Biases).

The affected functions and their replacements are listed below.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 1.x
     - 2.0.0
   * - ``helios.core.logging``
     - ``helios.core.loggers``
   * - ``create_default_loggers()``
     - :py:func:`~helios.core.loggers.create_loggers`
   * - ``setup_default_loggers()``
     - :py:func:`~helios.core.loggers.setup_loggers`
   * - ``restore_default_loggers()``
     - :py:func:`~helios.core.loggers.restore_loggers`
   * - ``flush_default_loggers()``
     - :py:func:`~helios.core.loggers.flush_loggers`
   * - ``close_default_loggers()``
     - :py:func:`~helios.core.loggers.close_loggers`

The following functions are unchanged and exist in the new module under the same names:
:py:func:`~helios.core.loggers.get_root_logger`,
:py:func:`~helios.core.loggers.get_tensorboard_writer`,
and :py:func:`~helios.core.loggers.is_root_logger_active`.

**Before:**

.. code-block:: python

   import helios.core.logging as hllog

   hllog.create_default_loggers(enable_tensorboard=True)
   hllog.setup_default_loggers(run_name, log_root)
   ...
   hllog.close_default_loggers()

**After:**

.. code-block:: python

   import helios.core.loggers as hllog

   hllog.create_loggers(enable_tensorboard=True)
   hllog.setup_loggers(run_name, log_root)
   ...
   hllog.close_loggers()

``CUDAPlugin`` Removed
----------------------

The functionality of the ``CUDAPlugin`` has been absorbed by the
:py:class:`~helios.model.model.Model` class via the
:py:meth:`~helios.model.model.Model.batch_to_device` function. The function is called
automatically by the :py:class:`~helios.trainer.Trainer` before each training, validation,
and testing step. As a result of this, the plugin has been removed.

The default implementation of :py:meth:`~helios.model.model.Model.batch_to_device` handles
tensors, as well as lists, tuples, and dictionaries that contain tensors, recursively. If
your batches use a standard structure, no action is required beyond removing the
``CUDAPlugin`` from your setup code.

If you previously overrode ``CUDAPlugin.process_training_batch()`` to perform custom
pre-processing, override
:py:meth:`~helios.model.model.Model.batch_to_device` in your
:py:class:`~helios.model.model.Model` subclass instead:

**Before:**

.. code-block:: python

   import helios.plugins as hlp

   class MyPlugin(hlp.CUDAPlugin):
       def process_training_batch(self, batch, ...):
           batch = super().process_training_batch(batch, ...)
           # custom processing
           return batch

   plugin = MyPlugin()
   plugin.configure_trainer(trainer)

**After:**

.. code-block:: python

   import helios
   import helios.model as hlm

   class MyModel(helios.Model):
       def batch_to_device(self, batch, phase: hlm.BatchPhase):
           batch = super().batch_to_device(batch, phase)
           # custom processing
           return batch

Plugin Registration Changed
----------------------------

The way plug-ins are registered with the trainer has been reworked. Previously, plug-ins
had to:

* Register themselves by calling ``_register_in_trainer`` inside ``configure_trainer``,
* Manually call ``configure_trainer`` and ``configure_model``.

This has been replaced by :py:meth:`~helios.trainer.Trainer.register_plugin`. In addition
to this, :py:meth:`~helios.plugins.plugin.Plugin.configure_trainer` and
:py:meth:`~helios.plugins.plugin.Plugin.configure_model` are now called automatically by
the trainer when :py:meth:`~helios.trainer.Trainer.fit` or
:py:meth:`~helios.trainer.Trainer.test` is called. They no longer need to be invoked
manually.

**Before:**

.. code-block:: python

   class MyPlugin(hlp.Plugin):
       def configure_trainer(self, trainer):
           self._register_in_trainer(trainer)

   plugin = MyPlugin()
   plugin.configure_trainer(trainer)
   plugin.configure_model(model)

**After:**

.. code-block:: python

   plugin = MyPlugin()
   trainer.register_plugin(plugin)
   # configure_trainer and configure_model are called automatically by the trainer.

Model State Dictionary API Changed
------------------------------------

In 1.x, the :py:class:`~helios.model.model.Model` users had to override
:py:meth:`~helios.model.model.Model.state_dict` and
:py:meth:`~helios.model.model.Model.load_state_dict` directly to save their training
state. In 2.0.0 these functions are reserved for internal use and manage the separation
between user state and Helios internal state (such as the AMP scaler).

Users must now override :py:meth:`~helios.model.model.Model.user_state_dict` and
:py:meth:`~helios.model.model.Model.load_user_state_dict` instead. The signatures and
responsibilities of these functions are identical to their 1.x counterparts.

**Before:**

.. code-block:: python

   def state_dict(self) -> dict:
       return {
           "net": self._net.state_dict(),
           "optimizer": self._optimizer.state_dict(),
       }

   def load_state_dict(self, state_dict: dict, fast_init: bool = False) -> None:
       self._net.load_state_dict(state_dict["net"])
       if not fast_init:
           self._optimizer.load_state_dict(state_dict["optimizer"])

**After:**

.. code-block:: python

   def user_state_dict(self) -> dict:
       return {
           "net": self._net.state_dict(),
           "optimizer": self._optimizer.state_dict(),
       }

   def load_user_state_dict(self, state_dict: dict, for_inference: bool) -> None:
       self._net.load_state_dict(state_dict["net"])
       if not for_inference:
           self._optimizer.load_state_dict(state_dict["optimizer"])

``fast_init`` Renamed to ``for_inference``
------------------------------------------

The ``fast_init`` parameter on :py:meth:`~helios.model.model.Model.setup` and
:py:meth:`~helios.model.model.Model.load_user_state_dict` has been renamed to
``for_inference``. The name better reflects its purpose: when ``True``, the model skips
loading training-only state (optimisers, schedulers, etc.) because it is being prepared
for inference rather than for continued training.

Update any overrides of ``setup()`` or ``load_state_dict()`` (now ``load_user_state_dict()``)
to use the new parameter name.

**Before:**

.. code-block:: python

   def setup(self, fast_init: bool = False) -> None:
       ...

**After:**

.. code-block:: python

   def setup(self, for_inference: bool = False) -> None:
       ...

Trainer Arguments Changed
--------------------------

The ``log_path`` and ``run_path`` constructor arguments on
:py:class:`~helios.trainer.Trainer` have been replaced by the single ``log_root``
argument. Previously, ``log_path`` controlled where the log file was written and
``run_path`` controlled where the Tensorboard run data was written. Both are now derived
automatically from ``log_root``.

**Before:**

.. code-block:: python

   trainer = Trainer(
       ...,
       log_path=pathlib.Path("logs"),
       run_path=pathlib.Path("runs"),
   )

**After:**

.. code-block:: python

   trainer = Trainer(
       ...,
       log_root=pathlib.Path("logs"),
   )

``DataLoaderParams.debug_mode`` Removed
----------------------------------------

The ``debug_mode`` field on
:py:class:`~helios.data.datamodule.DataLoaderParams` has been removed. Its only effect
was to set ``num_workers`` to ``0`` when enabled. Set ``num_workers=0`` directly instead.

**Before:**

.. code-block:: python

   params = DataLoaderParams(..., debug_mode=True)

**After:**

.. code-block:: python

   params = DataLoaderParams(..., num_workers=0)

``DataLoaderParams.pin_memory`` Default Changed
------------------------------------------------

The default value of ``pin_memory`` in
:py:class:`~helios.data.datamodule.DataLoaderParams` has changed from ``True`` to
``False``. This matches the PyTorch default and avoids unexpected behaviour on machines
with limited RAM or in CPU-only environments.

If you rely on pinned memory for GPU training performance, set ``pin_memory=True``
explicitly in your :py:class:`~helios.data.datamodule.DataLoaderParams`.

``get_default_numpy_rng()`` Return Type Changed
-------------------------------------------------

:py:func:`~helios.core.rng.get_default_numpy_rng` previously returned a
``DefaultNumpyRNG`` wrapper object. It now returns a
:py:class:`numpy.random.Generator` directly. Update any code that was accessing the
generator through the wrapper:

**Before:**

.. code-block:: python

   rng = helios.core.rng.get_default_numpy_rng()
   value = rng.generator.integers(0, 10)

**After:**

.. code-block:: python

   rng = helios.core.rng.get_default_numpy_rng()
   value = rng.integers(0, 10)

New Features in 2.0.0
-----------------------

The following features are new in 2.0.0. They do not require changes to existing code
but are worth reviewing as they may simplify your training setup.

Mixed Precision Training (AMP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`~helios.model.model.Model` class now includes built-in support for
Automatic Mixed Precision (AMP) training on both GPU (``float16`` and ``bfloat16``) and
CPU (``bfloat16`` only). See
:ref:`amp` in the Quick Reference for full details.

Gradient Clipping
~~~~~~~~~~~~~~~~~

A :py:meth:`~helios.model.model.Model.clip_gradients` function is now available directly
on the model, with correct integration with the AMP scaler when AMP is active. See
:ref:`grad-clipping` in the Quick Reference.

Weights & Biases Support
~~~~~~~~~~~~~~~~~~~~~~~~~

Weights & Biases logging is now supported natively. Pass a ``wandb_args`` dictionary to
the :py:class:`~helios.trainer.Trainer` constructor to enable it. See the
:doc:`logging` page for full details.

Multi-phase Training
~~~~~~~~~~~~~~~~~~~~

The :py:class:`~helios.data.datamodule.DataModule` now supports multi-phase training,
allowing successive datasets to be registered and advanced automatically during training.
See :ref:`multi-phase-training` in the Quick Reference for details.

``should_save_checkpoint()`` Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override :py:meth:`~helios.model.model.Model.should_save_checkpoint` to control whether
the trainer writes a checkpoint at a given point. This is useful for implementing
save-best logic.

``get_train_steps_per_epoch()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`~helios.model.model.Model.get_train_steps_per_epoch` (delegating to
:py:meth:`~helios.data.datamodule.DataModule.get_train_steps_per_epoch`) is now
available to obtain the number of training steps per epoch. This is useful for
initialising schedulers in :py:meth:`~helios.model.model.Model.setup`.

Linear Warmup Scheduler
~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~helios.scheduler.LinearWarmupScheduler` wraps any existing scheduler and
applies a linear warmup phase before handing off to the wrapped scheduler.

Expanded Metrics
~~~~~~~~~~~~~~~~

The :py:mod:`helios.metrics` module now includes additional metrics beyond accuracy:
precision, recall, F1 score, RMSE, SSIM, PSNR, mAP, and MAE.

Expanded ONNX Support
~~~~~~~~~~~~~~~~~~~~~~

:py:func:`~helios.onnx.export_to_onnx` now supports multiple inputs, multiple outputs,
and dictionary outputs. It also automatically selects between the legacy and dynamo export
paths based on the installed PyTorch version.
