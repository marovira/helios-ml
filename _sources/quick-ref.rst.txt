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

Checkpoints
-----------

The final mechanism Helios has to ensure reproducibility is in the way checkpoints are
saved. Specifically, the data that is stored in the checkpoints when they are created. By
default, the trainer will write the following data:

* The state of all supported RNGs.
* The current :py:class:`~helios.trainer.TrainingState`.
* The state of the model (if any).
* The paths to the log file and Tensorboard folder (if using).

If training is stopped and restarted, then Helios will look in the folder where
checkpoints are stored and load the last checkpoint. This checkpoint is found by finding
the file with the highest epoch and/or iteration number. Upon loading, the trainer will do
the following:

* Restore the state of all supported RNGs.
* Load the saved training state.
* Provide the loaded state to the model (if any).
* Restore the file and Tensorboard loggers to continue writing to their original locations
  (if using).

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

To better understand the behaviour of each unit type, lets look at an example. First, lets
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

The following keys may optionally appear in the dictionary:

* ``log_path``: appears only when file logging is enabled and contains the path to the log
  file.
* ``run_path``: appears only when Tensorboard logging is enabled and contains the path to
  the directory where the data is stored.

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

The :py:class:`~helios.trainer.Trainer` has several sets of flags that control logging.
These are:

* ``enable_tensorboard`` which is paired with ``run_path``,
* ``enable_file_logging`` which is paired with ``log_path``, and
* ``enable_progress_bar``.

The ``*_path`` arguments determine the root directories where the corresponding logs will
be saved.

.. warning::
   If a flag is paired with a path, then you **must** provide the corresponding path if
   the flag is enabled. In other words, if you set ``enable_tensorboard``, then you must
   also provide ``run_path``.

.. note::
   If the given path doesn't exist, it will be created automatically.

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
