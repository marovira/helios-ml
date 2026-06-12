Quick Reference
######################

Here is where you will find detailed explanations regarding the various modules that
Helios offers. The topics are not sorted in any particular order, and are meant to be used
as reference for developers.

.. _repro:

Reproducibility
===============

One of the largest features in Helios is the ability to maintain reproducibility even if
training runs are stopped and restarted. The mechanisms through which reproducibility is
ensured are split in several groups, which we will discuss in more detail below. At the
end is a short summary explaining how to use Helios correctly to ensure reproducibility.

.. warning::
   While every effort is done to ensure reproducibility, there are some limitations to
   what Helios can do. As Helios depends on PyTorch, it is bound to the same
   reproducibility limitations. For more information, see the
   `reproducibility documentation from PyTorch <https://github.com/Lightning-AI/pytorch-lightning>`__.

Random Number Generation
------------------------

To ensure that sequences of random numbers are maintained, Helios provides an automatic
seeding system that is invoked as part of the start up process by the
:py:class:`~helios.trainer.Trainer`. The seed value used to initialise the random number
generators can be assigned by setting the ``random_seed`` parameter of the trainer.

.. note::
   If no value is assigned, the default seed is used. Helios has a default value of 6691
   for seeding RNGs.

Random numbers may be required throughout the training process, so Helios will
automatically seed the following generators:

* PyTorch: through ``torch.manual_seed``.
* PyTorch CUDA: through ``torch.cuda.manual_seed_all``. Note that this is only done if
  CUDA is available.
* Python's builtin random module through ``random.seed``.

NumPY RNG
^^^^^^^^^

Starting with NumPY 1.16, ``numpy.random.rand`` is considered legacy and will receive no
further updates. Their documentation states that newer code should rely instead on their
new ``Generator`` class. In order to facilitate the seeding, saving and restoring of the
NumPY generators, Helios provides a wrapper class called
:py:class:`~helios.core.rng.DefaultNumpyRNG`. This class is automatically created by the
trainer and initialised with the default seed (unless a different seed is specified).
The generator can be accessed through :py:func:`~helios.core.rng.get_default_numpy_rng` as
seen below:

.. code-block:: python

   from helios.core import rng

   generator = rng.get_default_numpy_rng().generator

   # Use the generator as necessary. For example, we can retrieve a uniform random float
   # between 0 and 1.
   generator.uniform(0, 1)

.. warning::
   Helios **does not** initialise the legacy random generator from NumPY at **any** point.
   You are strongly encouraged to use the provided NumPY generator instead.

The state of the RNGs that Helios seeds is automatically stored whenever checkpoints are
saved, so model classes do not have to handle this themselves. Likewise, whenever
checkpoints are loaded, the RNG state is automatically restored.

DataLoaders and Samplers
------------------------

The next major block pertains to how the dataloaders and samplers are handled by Helios.
This is split into two sets: the worker processes and the way datasets are sampled.

Worker Processes
^^^^^^^^^^^^^^^^

When the dataloader for a dataset is created, Helios passes in a custom function to seed
each worker process so the random sequences remain the same. The code is adapted from the
official PyTorch documentation as shown here:

.. code-block:: python

    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        rng.seed_rngs(worker_seed, skip_torch=True)

This ensures that all the RNGs that Helios supports are correctly seeded in the worker
processes.

.. note::
   This function is passed in internally as an argument to ``worker_init_fn`` of the
   PyTorch ``DataLoader`` class. At this time it is not possible to override this
   function, though it may be considered for a future release.

Samplers
^^^^^^^^

A critical component of ensuring reproducibility is to have a way for the order in which
batches are retrieved from the dataset stays the same even if a training run is stopped.
PyTorch does not provide a built-in system to allow this, so Helios implements this
through the :py:class:`~helios.data.samplers.ResumableSampler` base class. The goal is to
provide a way to do the following:

#. The sampler must have a way of setting the starting iteration. For example, suppose
   that the sampler would've produced for a given epoch a sequence of :math:`N` batches
   numbered :math:`0, 1, \ldots, N`. We need the sampler to provide way for us to set the
   starting batch to a given number :math:`n_i` such that the sequence of batches
   continues from that starting point.
#. The sampler must have a way of setting the current epoch. This is to allow the samplers
   to re-shuffle between epochs (if shuffling is used) and to guarantee that the resulting
   shuffled list is consistent.

Helios contains 3 samplers that provide this functionality. These are:

* :py:class:`~helios.data.samplers.ResumableRandomSampler`
* :py:class:`~helios.data.samplers.ResumableSequentialSampler`
* :py:class:`~helios.data.samplers.ResumableDistributedSampler`

By default, the sampler is automatically selected using the following logic:

* If training is distributed, use
  :py:class:`~helios.data.samplers.ResumableDistributedSampler`
* If training is not distributed, then check if shuffling is required. If it is, use
  :py:class:`~helios.data.samplers.ResumableRandomSampler`. Otherwise use
  :py:class:`~helios.data.samplers.ResumableSequentialSampler`.

It is possible to override this by providing your own sampler, in which case you should
set the :py:attr:`~helios.data.datamodule.DataLoaderParams.sampler` field of the
:py:class:`~helios.data.datamodule.DataLoaderParams`.

.. warning::
   The sampler **must** derive from either
   :py:class:`~helios.data.samplers.ResumabeSampler` or
   :py:class:`~helios.data.samplers.ResumableDistributedSampler`

.. note::
   :py:attr:`~helios.data.datamodule.DataLoaderParams.pin_memory` defaults to ``False``,
   matching PyTorch's default. If you are training on a GPU, set ``pin_memory=True`` in
   your :py:class:`~helios.data.datamodule.DataLoaderParams` to enable pinned memory
   transfers.

Checkpoints
-----------

The final mechanism Helios has to ensure reproducibility is in the way checkpoints are
saved. Specifically, the data that is stored in the checkpoints when they are created. By
default, the trainer will write the following data:

* The state of all supported RNGs.
* The current :py:class:`~helios.trainer.TrainingState`.
* The state of the model (if any).
* The state of any active loggers (if any).

If training is stopped and restarted, then Helios will look in the folder where
checkpoints are stored and load the last checkpoint. This checkpoint is found by finding
the file with the highest epoch and/or iteration number. Upon loading, the trainer will do
the following:

* Restore the state of all supported RNGs.
* Load the saved training state.
* Provide the loaded state to the model (if any).
* Restore any active loggers to continue writing to their original locations (if any).

.. note::
   Any weights contained in the saved checkpoint are automatically mapped to the correct
   device when the checkpoint is loaded.

TL;DR
-----

Below is a quick summary to ensure you use Helios' reproducibility system correctly:

#. Helios provides a default random seed, but you can override it by setting
   ``random_seed`` in the :py:class:`~helios.trainer.Trainer`.
#. If you need RNG, you can use the built-in ``random`` module from Python,
   ``torch.random`` and ``torch.cuda.random``. If you need to use a NumPY RNG, use
   :py:func:`~helios.core.rng.get_default_numpy_rng`.
#. Seeding of workers for dataloaders is automatically handled by Helios, so you don't
   have to do extra work.
#. Helios ships with custom samplers that ensure reproducibility in the event training
   stops. The choice of sampler is automatically handled, but you may override it by
   setting :py:attr:`~helios.data.datamodule.DataLoaderParams.sampler`.
#. Checkpoints created by Helios automatically store the RNG state alongside training
   state. No more work is required on your part beyond saving the state of your model.


.. _stopping-training:

Stopping Training
=================

In certain cases, it is desirable for training to halt under specific conditions. For
example,

* Either the validation metric or loss function have reached a specific threshold after
  which training isn't necessary.
* The loss function is returning invalid values.
* The validation metric has not improved in the last :math:`N` validation cycles.

Helios provides a way for training to halt if a condition is met. The behaviour is
dependent on the choice of training unit, but in general, the following options are
available:

* If you wish to stop training after :math:`N` validation cycles because the metric hasn't
  improved, then you can use :py:meth:`~helios.model.model.Model.have_metrics_improved` in
  conjunction with the ``early_stop_cycles`` argument of the
  :py:class:`~helios.trainer.Trainer`.
* If you wish to stop training for any other reason, you can use
  :py:meth:`~helios.model.model.Model.should_training_stop`.

We will now discuss each of these in more detail.

Stopping After :math:`N` Validation Cycles
------------------------------------------

Helios will perform validation cycles based on the frequency assigned to
``valid_frequency`` in the :py:class:`~helios.trainer.Trainer`. The value specifies:

* The number of epochs between each cycle if the training unit is set to
  :py:attr:`~helios.trainer.TrainingUnit.EPOCH` or,
* The number of iterations between each cycle if the training unit is set to
  :py:attr:`~helios.trainer.TrainingUnit.ITERATION`.

After the validation cycle is completed, the trainer will call
:py:meth:`~helios.model.model.Model.have_metrics_improved`. If ``early_stop_cycles`` has
been assigned when the trainer was created, then the following logic applies:

* If the function returns true, then the early stop counter resets to 0 and training
  continues.
* If the function returns false, then the early stop counter increases by one. If the
  counter is greater than or equal to the value given to ``early_stop_cycles``, then
  training stops.

.. note::
   If you wish to use the early stop system, you **must** assign ``early_stop_cycles``.

.. note::
   The call to :py:meth:`~helios.model.model.Model.have_metrics_improved` is performed
   after checking if a checkpoint should be saved. If your validation and checkpoint
   frequencies are the same, then you're guaranteed that a checkpoint will be saved
   *before* the early stop check happens.

Stopping on a Condition
-----------------------

The function used to determine if training should stop for reasons that are not related to
the early stop system is :py:meth:`~helios.model.model.Model.should_training_stop`. As
there are various places in which it would be desirable for training to halt, Helios
checks this function at the following times:

* After a training batch is complete. Specifically, this check will be done after
  :py:meth:`~helios.model.model.Model.on_training_batch_start`,
  :py:meth:`~helios.model.model.Model.train_step`, and
  :py:meth:`~helios.model.model.Model.on_training_batch_end` have been called.
* After a validation cycle has been completed.

.. note::
   The behaviour of the training batch is consistent regardless of the training unit.

.. note::
   Remember: the choice of training unit affects the place where validation cycles are
   performed:

   * If training by epochs, then validation cycles occur at the end of every epoch.
   * If training by iterations, then validation cycles will occur after the training batch
     finishes on an iteration number that is a multiple of the validation frequency. In
     this case, the early stop checks would occur after the check to see if training
     should halt.

.. _multi-phase-training:

Multi-Phase Training
====================

Multi-phase training allows the training dataset to change during a single run. The most
common use case is to apply different data augmentations or configurations at different
points during training.

Phases are registered in :py:meth:`~helios.data.datamodule.DataModule.setup` by calling
:py:meth:`~helios.data.datamodule.DataModule._add_train_phase` once per phase:

.. code-block:: python

   class MyDataModule(data.DataModule):
       def setup(self) -> None:
           # Phase 1: light augmentation
           self._add_train_phase(
               MyDataset(transform=LightAugment()),
               DataLoaderParams(batch_size=32),
           )
           # Phase 2: stronger augmentation
           self._add_train_phase(
               MyDataset(transform=StrongAugment()),
               DataLoaderParams(batch_size=32),
           )

The first call to :py:meth:`~helios.data.datamodule.DataModule._add_train_phase`
automatically sets the active dataset to phase 0. If only one phase is registered,
the behaviour is identical to a regular single-phase setup.

Advancing Phases
----------------

To signal that the trainer should move to the next phase, override
:py:meth:`~helios.model.model.Model.should_advance_dataset_phase` in your model and
return ``True`` when the condition is met:

.. code-block:: python

   def should_advance_dataset_phase(self) -> bool:
       if not self._phase_advanced and self._current_epoch >= 10:
           self._phase_advanced = True
           return True
       return False

.. important::
   This function should return ``True`` if and only if the phase should advance. It is
   the model's responsibility to ensure the function does not keep returning ``True``
   after the phase has already been advanced (for example, by using a flag as shown
   above).

When :py:meth:`~helios.model.model.Model.should_advance_dataset_phase` returns ``True``,
the trainer calls
:py:meth:`~helios.data.datamodule.DataModule.advance_train_phase` automatically and
rebuilds the training dataloader for the new phase. The check is performed:

* At the end of each epoch, if the training unit is
  :py:attr:`~helios.trainer.TrainingUnit.EPOCH`.
* At the end of each iteration, if the training unit is
  :py:attr:`~helios.trainer.TrainingUnit.ITERATION`.

Checkpoint Behaviour
--------------------

The current phase index is saved and restored automatically with each checkpoint. When
training resumes from a checkpoint, the datamodule restores the correct phase so that
the active dataset matches the state at the time the checkpoint was written.

Interaction with ``get_train_steps_per_epoch()``
-------------------------------------------------

:py:meth:`~helios.model.model.Model.get_train_steps_per_epoch` returns the step count
for the *current* active phase. If you cache this value in
:py:meth:`~helios.model.model.Model.setup`, be aware that it reflects phase 0. Call it
again after advancing the phase if the step count may differ between phases.

.. _gradient-accumulation:

Gradient Accumulation
=====================

The :py:class:`~helios.trainer.Trainer` provides native support for performing gradient
accumulation while training. The behaviour is dependent on the choice of training unit,
and the logic is the following:

* If :py:attr:`~helios.trainer.TrainingUnit.EPOCH` is used, then gradient accumulation has
  no effect on the trainer. Specifically, the iteration count does not change, and neither
  do the total number of epochs.
* If :py:attr:`~helios.trainer.TrainingUnit.ITERATION` is used, then accumulating by
  :math:`N_g` steps with a total number of iterations :math:`N_i` will result in
  :math:`N_g \cdot N_i` total training iterations.

Training by Epoch
-----------------

To better understand the behaviour of each unit type, let's look at an example. First, lets
set the training unit to be epochs. Then, suppose that we want to train a network for 5
epochs and the batch size results in 10 iterations per epoch. We want to accumulate
gradients for 2 iterations, effectively emulating a batch size that results in 5
iterations per epoch. In this case, the total number of iterations that the dataset loop
has to run for remains unchanged. We're still going to go through all 10 batches, but the
difference is that we only want to compute backward passes on batches 2, 4, 6, 8, and 10.
Since this is the responsibility of the model, the trainer doesn't have to do any special
handling, which results in the following data being stored in the
:py:class:`~helios.trainer.TrainingState`:

* :py:attr:`~helios.trainer.TrainingState.current_iteration` and
  :py:attr:`~helios.trainer.TrainingState.global_iteration` will both have the same value,
  which will correspond to :math:`n_e \cdot n_i` where :math:`n_e` is the current epoch
  number and :math:`n_i` is the batch number in the dataset.
* :py:attr:`~helios.trainer.TrainingState.global_epoch` will contain the current epoch
  number.

Lets suppose that we want to perform the backward pass in the
:py:meth:`~helios.model.model.Model.on_training_batch_end` function of the model. Then we
would do something like this:

.. code-block:: python

    def on_training_batch_end(
        self, state: TrainingState, should_log: bool = False
    ) -> None:
        # Suppose that our loss tensor is stored in self._loss_items and the number of
        # accumulation steps is stored in self._accumulation_steps
        if state.current_iteration % self._accumulation_steps == 0:
            self._loss_items["loss"].backward()
            self._optimizer.step()
        ...

.. note::
   In the example above, we could've just as easily used ``state.global_iteration`` as
   they both have the same value.

Training by Iteration
---------------------

Now let's see what happens when we switch to training by iterations. In this case, suppose
we want to train a network for 10k iterations. We want to emulate a batch size that is
twice our current size, so we want to accumulate by 2. If we were to run the training loop
for 10k iterations performing backward passes every second iteration, we would've
performed at total of 5k backward passes, which is half of what we want. Remember: we want
to train for *10k* iterations at *double* the batch size that we have. This means that, in
order to get the same number of backward passes, we need to double the total iteration
count to 20k. This way, we would get the 10k backward passes that we want.

In order to simplify things, the trainer will automatically handle this calculation for
you, which results in the following data being stored in the
:py:class:`~helios.trainer.TrainingState`:

* :py:attr:`~helios.trainer.TrainingState.current_iteration` is the *real* iteration count
  that accounts for gradient accumulation. In our example, this number would only increase
  every *second* iteration, and it is used to determine when training should stop.
* :py:attr:`~helios.trainer.TrainingState.global_iteration`: is the *total* number of
  iterations that have been performed. In our example, this would be twice the value
  of the current iteration.
* :py:attr:`~helios.trainer.TrainingState.global_epoch` is the current epoch number.

Like before, suppose that we want to perform the backward pass in the
:py:meth:`~helios.model.model.Model.on_training_batch_end` function of the model. Then we
would do something like this:

.. code-block:: python

    def on_training_batch_end(
        self, state: TrainingState, should_log: bool = False
    ) -> None:
        # Suppose that our loss tensor is stored in self._loss_items and the number of
        # accumulation steps is stored in self._accumulation_steps
        if state.global_iteration % self._accumulation_steps == 0:
            self._loss_items["loss"].backward()
            self._optimizer.step()
        ...

.. warning::
   Unlike the epoch case, we **cannot** use ``state.current_iteration`` as that keeps
   track of the number of *complete* iterations we have done.

.. _amp:

Automatic Mixed Precision
=========================

The :py:class:`~helios.model.model.Model` class provides helper functions for Automatic
Mixed Precision (AMP) training.

To enable AMP, call :py:func:`~helios.model.model.Model.create_scaler` in your
:py:func:`~helios.model.model.Model.setup` function:

.. code-block:: python

   def setup(self, for_inference: bool = False) -> None:
       self._net = ...
       self._optimizer = ...
       self.create_scaler(dtype=torch.float16)

Once enabled, use :py:func:`~helios.model.model.Model.autocast` to wrap the forward
pass in your training step:

.. code-block:: python

   def train_step(self, batch, state: TrainingState) -> None:
       inputs, labels = batch
       self._optimizer.zero_grad()

       with self.autocast():
           loss = self._criterion(self._net(inputs), labels)

       self._loss_items["loss"] = loss
       scaler = self.amp_context.scaler
       scaler.scale(loss).backward()
       scaler.step(self._optimizer)
       scaler.update()

:py:func:`~helios.model.model.Model.autocast` returns
:py:class:`torch.amp.autocast` when AMP is active and a null context otherwise, so the
same training step code works regardless of whether AMP is enabled.

The AMP state can be accessed directly through the
:py:attr:`~helios.model.model.Model.amp_context` property, which returns an
:py:class:`~helios.model.model.AMPContext` dataclass containing the scaler and the
dtype. When AMP is disabled, :py:attr:`~helios.model.model.Model.amp_context` is
``None``.

AMP state is saved and restored automatically with each checkpoint.

.. note::
   On CPU, only ``torch.bfloat16`` is supported. Passing any other dtype when the
   device is CPU will raise a :py:exc:`ValueError`.

.. _grad-clipping:

Gradient Clipping
=================

:py:func:`~helios.model.model.Model.clip_gradients` handles gradient clipping correctly
regardless of whether AMP is active. When AMP is enabled, gradients must be unscaled
before the norm is computed, otherwise the threshold is applied to scaled values and the
effective clip is wrong. The function handles this automatically:

1. When AMP is active, call :py:meth:`torch.amp.GradScaler.unscale_` on the optimizer.
2. Call :py:func:`torch.nn.utils.clip_grad_norm_`.

Call it between the backward pass and the optimizer step:

.. code-block:: python

   def train_step(self, batch, state: TrainingState) -> None:
       inputs, labels = batch
       self._optimizer.zero_grad()

       with self.autocast():
           loss = self._criterion(self._net(inputs), labels)

       self._loss_items["loss"] = loss
       if self.amp_context is not None:
           scaler = self.amp_context.scaler
           scaler.scale(loss).backward()
           self.clip_gradients(self._net.parameters(), self._optimizer, max_norm=1.0)
           scaler.step(self._optimizer)
           scaler.update()
       else:
           loss.backward()
           self.clip_gradients(self._net.parameters(), self._optimizer, max_norm=1.0)
           self._optimizer.step()

.. _linear-warmup-scheduler:

Linear Warmup Scheduler
=======================

:py:class:`~helios.scheduler.schedulers.LinearWarmupScheduler` implements a two-phase
learning rate schedule:

#. For the first :math:`N_w` steps (``warmup_steps``), the learning rate increases
   linearly from :math:`f_0 \cdot \eta_0` to :math:`\eta_0`, where :math:`\eta_0` is the
   base learning rate and :math:`f_0` is ``warmup_start_factor``.
#. After :math:`N_w` steps, the learning rate is controlled by the wrapped ``scheduler``.

Typical setup consists of using
:py:meth:`~helios.model.model.Model.get_train_steps_per_epoch` to derive the warmup
duration from the dataset size:

.. code-block:: python

   from helios.scheduler import schedulers as hls

   def setup(self, for_inference: bool = False) -> None:
       self._optimizer = ...
       steps_per_epoch = self.get_train_steps_per_epoch()
       base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
           self._optimizer, T_max=total_steps
       )
       self._scheduler = hls.LinearWarmupScheduler(
           optimizer=self._optimizer,
           warmup_steps=5 * steps_per_epoch,  # warm up for 5 epochs
           scheduler=base_scheduler,
           warmup_start_factor=0.1,
       )

The ``warmup_start_factor`` argument controls :math:`f_0`, the fraction of
:math:`\eta_0` to begin from. For :math:`f_0 \in [0.0, 1.0)`, a value of 0.0 starts
the warmup from zero. The default is 0.0.

.. _checkpoint-saving:

Checkpoint Saving
=================

As mentioned in :ref:`repro`, Helios will automatically save checkpoints whenever both
``chkpt_frequency`` and ``chkpt_root`` are set in the :py:class:`~helios.trainer.Trainer`.
The data for checkpoints is stored in a dictionary that *always* contains the following
keys:

* ``training_state``: contains the current :py:class:`~helios.trainer.TrainingState`
  object.
* ``model``: contains the state of the model as returned by
  :py:meth:`~helios.model.model.Model.state_dict`. Note that by default this is an empty
  dictionary.
* ``rng``: contains the state of the supported RNGs.
* ``version``: contains the version of Helios used to generate the checkpoint.

The following key may optionally appear in the dictionary:

* ``loggers``: appears when at least one logger is active. It maps logger type names to
  their respective state dictionaries:

  * ``root`` → ``{"log_file": <path>}`` (present when file logging is enabled)
  * ``tensorboard`` → ``{"run_path": <path>}`` (present when Tensorboard logging is enabled)
  * ``wandb`` → ``{"run_id": <run-id>}`` (present when W&B logging is enabled)

The name of the checkpoint is determined as follows:

.. code-block:: text

   <run-name>_<epoch>_<iteration>_<additional-metadata>.pth

Where:

* ``<run-name>`` is the value assigned to ``run_name`` in the trainer.
* ``<epoch>`` and ``<iteration>`` are the values stored in
  :py:attr:`~helios.trainer.TrainingState.global_epoch` and
  :py:attr:`~helios.trainer.TrainingState.global_iteration`, respectively.

The ``<additional-metadata>`` field is used to allow users to append additional
information to the checkpoint name for easier identification later on. This data is
retrieved from the :py:meth:`~helios.model.model.Model.append_metadata_to_chkpt_name`
function from the model. For example, suppose we want to add the value of the accuracy
metric we computed for validation. Then we would do something like this:

.. code-block:: python

    def append_metadata_to_chkpt_name(self, chkpt_name: str) -> str:
        # Suppose the accuracy is stored in self._val_scores
        accuracy = round(self._val_scores["accuracy"], 4)
        return "accuracy_{accuracy}"

This will append the string to the end of the checkpoint name. Say our run name is
``cifar10`` and we're saving on iteration 100 and epoch 3. Then the checkpoint name would
be:

.. code-block:: text

   cifar10_epoch_3_iter_100_accuracy_0.89.pth

.. note::
   You do not have to add the ``pth`` extension to the name when you append metadata. This
   will be automatically handled by the trainer.

.. note::
   If distributed training is used, then only the process with *global rank* 0 will save
   checkpoints.

Controlling When Checkpoints Are Saved
---------------------------------------

By default, the trainer writes a checkpoint whenever ``chkpt_frequency`` is reached. You
can control this by overriding :py:meth:`~helios.model.model.Model.should_save_checkpoint`
in your model:

.. code-block:: python

   def should_save_checkpoint(self) -> bool:
       # Only save when the validation score has improved
       return self._val_score > self._best_score

When this function returns ``False``, the trainer skips writing a checkpoint for that
cycle. This is useful when you want to only want to save checkpoints based on metrics or
other criteria.

Migrating Checkpoints
---------------------

The ``version`` key stored in the checkpoints generated by Helios acts as a fail-safe to
prevent future changes from breaking previously generated checkpoints. Helios *guarantees*
compatibility between checkpoints generated within the same major revision. In other
words, checkpoints generated by version 1.0 will be compatible with version 1.1.
Compatibility between major versions is **not** guaranteed. Should you wish to migrate
your checkpoints to a newer version of Helios, you may do so by either manually calling
:py:func:`~helios.chkpt_migrator.migrate_checkpoints_to_current_version` or by using the
script directly from the command line as follows:

.. code-block:: sh

   python -m helios.chkpt_migrator <chkpt-root>


.. _logging:

Logging
=======

The :py:class:`~helios.trainer.Trainer` supports three logging options controlled by
flags:

* ``enable_file_logging``: writes a log file under ``log_root``.
* ``enable_tensorboard``: enables Tensorboard logging. See :doc:`tensorboard` for full
  details.
* ``enable_progress_bar``: displays a progress bar during training.

W&B logging is not toggled by a flag; it is enabled by passing ``wandb_args`` to the
:py:class:`~helios.trainer.Trainer` constructor. See :doc:`wandb` for full details.

All file-based loggers write under the single ``log_root`` directory. This covers file
logging, Tensorboard, and W&B.

.. warning::
   If any file-based logging is enabled (``enable_file_logging``, ``enable_tensorboard``,
   or ``wandb_args``), you **must** also provide ``log_root``.

.. note::
   If the ``log_root`` path does not exist, it will be created automatically.

The way the names for logs is determined as follows:

.. code-block:: text

   <run-name>_<current-date/time>

Where ``<run-name>`` is the value assigned to the ``run_name`` argument of the
:py:class:`~helios.trainer.Trainer` and ``<current-date/time>`` is the string
representation of the current date and time with the format
``MonthDay_Hour-Minute-Second``. This allows multiple training runs with the same names to
save to different logs, which can be useful when tweaking hyper-parameters.

The ``enable_progress_bar`` flag determines whether a progress bar is shown on the screen
while training is ongoing. The progress bar is *only* shown on the screen and does not
appear in the file log (if enabled). The behaviour of the progress bar depends on the
choice of training unit:

* If epochs are used, then two progress bars are displayed: one that tracks the number of
  epochs and another that tracks the iterations within the current epoch.
* If iterations are used, then a single progress bar is shown that tracks the number of
  iterations.

The progress bar is also shown during validation, in which case it tracks the number of
iterations in the validation set.

.. _tensorboard-ref:

Tensorboard
===========

Tensorboard logging is enabled by setting ``enable_tensorboard=True`` and providing a
``log_root`` when constructing the :py:class:`~helios.trainer.Trainer`:

.. code-block:: python

   import helios
   import pathlib

   trainer = helios.Trainer(
       log_root=pathlib.Path("logs"),
       enable_tensorboard=True,
   )

The writer is accessible via :py:func:`~helios.core.loggers.get_tensorboard_writer`.
For full details including run resumption, directory layout, and available logging
functions, see :doc:`tensorboard`.

.. _wandb-ref:

Weights & Biases
================

W&B logging is enabled by passing a ``wandb_args`` dictionary to the
:py:class:`~helios.trainer.Trainer` constructor:

.. code-block:: python

   import helios

   trainer = helios.Trainer(
       ...,
       wandb_args={"project": "my-project", "name": "run-1"},
   )

The ``wandb_args`` dictionary accepts the fields defined by
:py:class:`~helios.core.loggers.wandb.WandbArgs`. The ``project`` key is the only
required field; all others are optional.

For full details including run resumption and directory layout, see :doc:`wandb`.

CUDA
====

Helios provides several conveniences for handling training on GPUs through CUDA as well as
distributed training. These are:

* Automatic detection and selection of GPUs to train in,
* Automatic mapping of checkpoints to the correct device,
* Support for ``torchrun``,
* Ability to set certain CUDA flags.

The :py:class:`~helios.trainer.Trainer` has two flags that can be used to control the
behaviour when using CUDA. These are:

* ``enable_deterministic``: uses deterministic training.
* ``enable_cudnn_benchmark``: enables the use of CuDNN benchmarking for faster training.

.. note::
   ``enable_deterministic`` can also be used when training on the CPU.

.. note::
   CuDNN is enabled *only* during training. It is disabled automatically during validation
   to avoid non-deterministic issues.

Device Selection
----------------

When the trainer is created, there are a two arguments that can be used to determine which
device(s) will be used for training: ``gpus`` and ``use_cpu``. The logic for determining
the device is the following:

* If ``use_cpu`` is true, then the CPU will be used for training.
* Otherwise, the choice of devices is determined by ``gpus``. If no value is assigned, and
  CUDA is not available, then the CPU will be used.
* If ``gpus`` is not assigned and CUDA is available, then Helios will automatically use
  all available GPUs in the system, potentially triggering distributed training if more
  than one is found.
* If ``gpus`` is set, then the indices it contains determine the devices that will be used
  for training.

.. note::
   If ``torchrun`` is used, then Helios will automatically detect the GPU assigned to the
   process as if it was assigned to ``gpus``.

.. note::
   If multiple GPUs are found, or if more than one index is provided to ``gpus``, then
   Helios will automatically launch distributed training.

.. warning::
   ``gpus`` must be set to a list of indices that represent the IDs of the GPU(s) to use.

Model Functions
===============

The :py:class:`~helios.model.model.Model` class provides several callbacks that can be
used for training, validation, and testing. Below is a list of all available callbacks
alongside with their use in the training loops.

Training Functions
------------------

The order in which the training functions are called roughly corresponds to the following
code:

.. code-block:: python

   model.on_training_start()
   model.train()
   for epoch in epoch:
        model.on_training_epoch_start()

        for batch in dataloader:
            model.on_training_batch_start()
            model.train_step()
            model.on_training_batch_end()

        model.on_training_epoch_end()
   model.on_training_end()

.. note::
   :py:meth:`~helios.model.model.Model.train` is a no-op by default. You must override
   it and call ``.train()`` on your network(s) manually. The trainer calls this function
   at the correct point in the loop.

Validation Functions
--------------------

The order in which the validation functions are called roughly corresponds to the
following code:

.. code-block:: python

   model.eval()
   model.on_validation_start()
        for batch in dataloader:
            model.on_validation_batch_start()
            model.valid_step()
            model.on_validation_batch_end()

   model.on_validation_end()

.. note::
   :py:meth:`~helios.model.model.Model.eval` is a no-op by default. You must override it
   and call ``.eval()`` on your network(s) manually. The trainer calls this function at
   the correct point in the loop.

Testing Functions
-----------------

The order of the testing functions is identical to the one shown for validation:

.. code-block:: python

   model.eval()
   model.on_testing_start()
        for batch in dataloader:
            model.on_testing_batch_start()
            model.test_step()
            model.on_testing_batch_end()

   model.on_testing_end()

Getting the Number of Training Steps per Epoch
-----------------------------------------------

:py:meth:`~helios.model.model.Model.get_train_steps_per_epoch` returns the number of
training batches in a single epoch. This wraps around
:py:meth:`~helios.data.datamodule.DataModule.get_train_steps_per_epoch`, which constructs
the training dataloader internally and returns its length.

This is particularly useful for initialising schedulers in
:py:meth:`~helios.model.model.Model.setup`:

.. code-block:: python

   def setup(self, for_inference: bool = False) -> None:
       steps_per_epoch = self.get_train_steps_per_epoch()
       self._scheduler = ...  # use steps_per_epoch to configure warmup or decay

.. note::
   Calling this function constructs the dataloader each time it is called. Cache the
   result if you need it more than once.

Batch Phase
------------

:py:class:`~helios.model.model.BatchPhase` is an enum with three values:

* :py:attr:`~helios.model.model.BatchPhase.TRAIN`,
* :py:attr:`~helios.model.model.BatchPhase.VALID`, and
* :py:attr:`~helios.model.model.BatchPhase.TEST`.

It is passed by the trainer to :py:meth:`~helios.model.model.Model.batch_to_device` to
indicate which phase the current batch belongs to. This can be useful if you require
different implementations for each phase.

Exception Handling
==================

By default, the main functions of :py:class:`~helios.trainer.Trainer` (those being
:py:meth:`~helios.trainer.Trainer.fit` and :py:meth:`~helios.trainer.Trainer.test`) will
automatically catch any unhandled exceptions and re-raise them. Depending on the
situation, it may be desirable for certain exceptions to be passed through untouched. In
order to accommodate this, the trainer has two sets of lists of exception types:

* :py:attr:`~helios.trainer.Trainer.train_exceptions` and
* :py:attr:`~helios.trainer.Trainer.test_exceptions`.

If an exception is raised and said exception is found in the training list (for
:py:meth:`~helios.trainer.Trainer.fit`) or testing list (for
:py:meth:`~helios.trainer.Trainer.test`), then the exception is passed through unchanged.
Any other exceptions use the default behaviour.

For example, suppose we had a custom exception called ``MyException`` and we wanted that
exception to be passed through when training because we're going to handle it ourselves.
We would then do the following:

.. code-block:: python

   import helios.trainer as hlt

   trainer = hlt.Trainer(...)
   trainer.train_exceptions.append(MyException)

   try:
       trainer.fit(...)
   except MyException as e:
       ...

The same logic applies for testing. This functionality is particularly useful when paired
with plug-ins.

Synchronization
===============

Helios provides some synchronization wrappers found in the
:py:mod:`~helios.core.distributed` module:

* :py:func:`~helios.core.distributed.gather_into_tensor`,
* :py:func:`~helios.core.distributed.all_reduce_tensors`.

The trainer also provides another way to synchronize values through the multi-processing
queue. When using distributed training that isn't through ``torchrun``, Helios uses
``spawn`` to create the processes for each GPU. This triggers a copy of the arguments
passed in to the handler, which in this case are the trainer, model, and datamodule. This
presents a problem in the event that we need to return values back to the main process
once training is complete. To facilitate this task, the trainer will create a `queue
<https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue>`__ that can
be accessed through :py:attr:`~helios.trainer.Trainer.queue`.

.. note::
   If training isn't distributed or if it was started through ``torchrun``, then the
   :py:attr:`~helios.trainer.Trainer.queue` is set to ``None``.

The queue can then be used by either the :py:class:`~helios.model.model.Model`, the
:py:class:`~helios.data.datamodule.DataModule`, or any plug-in through their reference to
the trainer.
