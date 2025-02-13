Plug-Ins
############

Plug-in API
===========

Helios offers a plug-in system that allows users to override certain elements of the
training loops. All plug-ins *must* derive from the main
:py:class:`~helios.plugins.plugin.Plugin` interface. The list of functions that are
available is similar to the ones offered by the :py:class:`~helios.model.model.Model`
class and follow the same call order. For example, the training loop would look something
like this:

.. code-block:: python

   plugin.on_training_start()
   model.on_training_start()
   model.train()
   for epoch in epoch:
        model.on_training_epoch_start()

        for batch in dataloader:
            plugin.process_training_batch(batch)
            model.on_training_batch_start()
            model.train_step()
            model.on_training_batch_end()

        model.on_training_epoch_end()
   plugin.on_training_end()
   model.on_training_end()

Notice that the plug-in functions are always called *before* the corresponding model
functions. This is to allow the plug-ins to override the model if necessary or to set
state that can be later accessed by the model. The model (and the dataloader) can access
the plug-ins through the reference to the trainer:

.. code-block:: python

   def train_step(...):
      model.trainer.plugins["foo"] # <- Access plug-in with name "foo"

Batch Processing
----------------

The major difference between the functions of the model and the plug-ins is the lack of a
:py:meth:`~helios.model.model.Model.train_step` function (and similarly for validation and
testing). Instead, the plug-ins have 3 equivalent functions:

* :py:meth:`~helios.plugins.plugin.Plugin.process_training_batch`
* :py:meth:`~helios.plugins.plugin.Plugin.process_validation_batch`
* :py:meth:`~helios.plugins.plugin.Plugin.process_testing_batch`

These functions receive the batch as an argument and return the processed batch. They can
be used for a variety of tasks such as moving tensors to a given device, filtering batch
entries, converting values, etc. For example, suppose we wanted to reduce the training
batch size by removing elements. We could do this as follows:

.. code-block:: python

   def process_training_batch(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
       return batch[:2] # <- Take the first two elements of the batch.

When the model's :py:meth:`~helios.model.model.Model.train_step` function is called, it
will only receive the first 2 tensors of the original batch.

Plug-in Registration
--------------------

The trainer contains the :py:attr:`~helios.trainer.Trainer.plugins` table in which all
plug-ins must be registered. To facilitate this, the plug-in base class requires a string
to act as the key with which it will be added to the table. In addition, it provides a
function that automatically registers the plug-in itself into the plug-in table. The
function can be easily invoked from the
:py:meth:`~helios.plugins.plugin.Plugin.configure_trainer` function as follows:

.. code-block:: python

   import helios.plugins as hlp
   import helios.trainer as hlt

   class MyPlugin(hlp.Plugin):
       def __init__(self):
           super().__init__("my_plugin")

       def configure_trainer(self, trainer: hlt.Trainer) -> None:
           self._register_in_trainer(trainer) # <- Automatically registers the plug-in.

.. note::
   All plug-ins that are shipped with Helios contain a ``plugin_id`` field as a class
   variable that can be used to easily access them from the trainer table. You are
   *encouraged* to always use this instead of manually typing in the key. For example,
   with the :py:class:`~helios.plugins.plugin.CUDAPlugin`, you could access it like this:

   .. code-block:: python

       import helios.plugins as hlp
       import helios.trainer as hlt

       trainer = hlt.Trainer(...)
       plugin = hlp.CUDAPlugin()
       plugin.configure_trainer(trainer)
       trainer.plugins[hlp.CUDAPlugin.plugin_id] # <- Access the plug-in like this.

Unique Traits
-------------

In order to avoid conflicts, the plug-in API designates certain functions as *unique*. In
this context, a plug-in with a *unique* override may only appear exactly *once* in the
:py:attr:`~helios.trainer.Trainer.plugins` table from the trainer. If a second plug-in
with that specific override is added, an exception is raised. The full list of overrides
can be found in the :py:class:`~helios.plugins.plugin.UniquePluginOverrides` struct. Each
plug-in has a copy found under :py:attr:`~helios.plugins.plugin.Plugin.unique_overrides`
and *must* be filled in with the corresponding information for each plug-in.

For example, suppose we want to build a new plug-in that can modify the training batch and
cause training to stop early. We would then set the structure as follows:

.. code-block:: python

   import helios.plugins as hlp

   class MyPlugin(hlp.Plugin):
       def __init__(self):
           super().__init__("my_plugin")

           self.unique_overrides.training_batch = True
           self.unique_overrides.should_training_stop = True

       def process_training_batch(...):
           ...

       def should_training_stop(...):
           ...

.. warning::
   Attempting to add two plug-ins with the same overrides **will** result in an exception
   being raised.


Built-in Plug-ins
=================

Helios ships with the following built-in plug-ins, which will be discussed in the
following sections:

* :py:class:`~helios.plugins.plugin.CUDAPlugin`
* :py:class:`~helios.plugins.optuna.OptunaPlugin`

CUDA Plug-in
------------

The :py:class:`~helios.plugins.plugin.CUDAPlugin` is designed to move tensors from the
batches returned by the datasets to the current CUDA device. The device is determined by
the trainer when training starts with the same logic used to assign the device to the
model. Specifically:

* If training isn't distributed, the device is the GPU that is used for training.
* If training is distributed, then the device corresponds to the GPU assigned to the given
  process (i.e. the local rank).

.. warning::
   As its name implies, the :py:class:`~helios.plugins.plugin.CUDAPlugin` **requires**
   CUDA to be enabled to function. If it isn't, an exception is raised.

The plug-in is designed to handle the following types of batches:

* :py:class:`torch.Tensor`,
* Lists of :py:class:`torch.Tensor`,
* Tuples of :py:class:`torch.Tensor`, and
* Dictionaries whose values are :py:class:`torch.Tensor`.

.. note::
   The contents of the containers need not be homogeneous. In other words, it is perfectly
   valid some entries in a dictionary to *not* be tensors. The plug-in will automatically
   recognise tensors and move them to the device.

.. warning::
   The plug-in is **not** designed to handle nested containers. For instance, if your
   batch is a dictionary containing arrays of tensors, then the plug-in will **not**
   recognise the tensors contained in the arrays and move them.

In the event that your batch requires special handling, you can easily derive the class
and override the function that moves the tensors to the device. For example, suppose that
our batch consists of a dictionary of arrays of tensors. Then we would do the following:

.. code-block:: python

   import helios.plugins as hlp
   import torch

   class MyCUDAPlugin(hlp.CUDAPlugin):
       # Only need to override this function. Everything else will work automatically.
       def _move_collection_to_device(
           self, batch: dict[str, list[torch.Tensor]]
       ) -> dict[str, list[torch.Tensor]]:
           for key, value in batch.items():
               for i in range(len(value)):
                   value[i] = value[i].to(self.device)
               batch[key] = value

           return batch

.. note::
   The :py:class:`~helios.plugins.plugin.CUDAPlugin` is automatically registered in the
   plug-in registry and can therefore be created through the
   :py:func:`~helios.plugins.plugin.create_plugin` function.

Optuna Plug-in
--------------

In order to use the Optuna plugin, we first need to install `optuna
<https://optuna.readthedocs.io/en/stable/>`__::

    pip install -U optuna

.. warning::
   Optuna is a **required** dependency for this plug-in. If it isn't installed, an
   exception is raised.

The plug-in will automatically integrate with Optuna for hyper-parameter optimisation by
performing the following tasks:

* Register the ``optuna.TrialPruned`` exception type with the trainer for correct trial
  pruning.
* Automatically update the :py:class:`~helios.model.model.Model` so the save name is
  consistent and allow trials to continue if they're interrupted.
* Correctly handle reporting and pruning for regular and distributed training.

A full example for how to use this plug-in can be found `here
<https://github.com/marovira/helios-ml/blob/master/examples/optuna_tutorial.py>`__, but we
will discuss the basics below. For the sake of simplicity, the code is identical to the
`cifar10 <https://github.com/marovira/helios-ml/blob/master/examples/classifier_tutorial.py>`__
example, so we will only focus on the necessary code to use the plug-in.

Plug-in Registration
^^^^^^^^^^^^^^^^^^^^

After the creation of the :py:class:`~helios.model.model.Model`,
:py:class:`~helios.data.datamodule.DataModule`, and the
:py:class:`~helios.trainer.Trainer`, we can create the plug-in and do the following:

.. code-block:: python

   import helios.plugins.optuna as hlpo
   import optuna

def objective(trial: optuna.Trial) -> float:
       model = ...
       datamodule = ...
       trainer = ...

       plugin = hlpo.OptunaPlugin(trial, "accuracy")
       plugin.configure_trainer(trainer)
       plugin.configure_model(model)

The two ``configure_`` functions will do the following:

#. Configure the trainer so the plug-in is registered into the plug-in table and ensure
   that ``optuna.TrialPruned`` is registered as a valid exception.
#. Configure the name of the model to allow cancelled trials to continue. Specifically, it
   will append ``_trial-<trial-numer>`` to the model name.

.. note::
   The call to :py:meth:`~helios.plugins.optuna.OptunaPlugin.configure_model` is
   completely optional and only impacts the ability to resume trials. You may choose to
   handle this yourself if it makes sense for your use-case.

Using the Trial
^^^^^^^^^^^^^^^

The trial instance is held by the plugin and can be easily accessed through the trainer.
For example, we can use it to configure the layers in the classifier network within the
:py:meth:`~helios.model.model.Model.setup` function like this:

.. code-block:: python

    def setup(self, fast_init: bool = False) -> None:
        plugin = self.trainer.plugins[0]
        assert isinstance(plugin, OptunaPlugin)

        # Assign the tunable parameters so we can log them as hyper-parameters when
        # training ends.
        self._tune_params["l1"] = plugin.trial.suggest_categorical(
            "l1", [2**i for i in range(9)]
        )
        self._tune_params["l2"] = plugin.trial.suggest_categorical(
            "l2", [2**i for i in range(9)]
        )
        self._tune_params["lr"] = plugin.trial.suggest_float("lr", 1e-4, 1e-1, log=True)

        self._net = Net(
            l1=self._tune_params["l1"],  # type: ignore[arg-type]
            l2=self._tune_params["l2"],  # type: ignore[arg-type]
        ).to(self.device)

Reporting Metrics
^^^^^^^^^^^^^^^^^

In order to allow the model to compute the metrics whenever it fits best within the
training workflow, the plug-in does *not* automatically report metrics to the trial. As a
result, it is the responsibility of the model to call
:py:meth:`~helios.plugins.optuna.OptunaPlugin.register_metrics` whenever the metrics are
ready. In order for the plug-in to correctly report the metrics, the plug-in relies on the
following things:

#. The :py:attr:`~helios.model.model.Model.metrics` table and
#. The value of ``metric_name`` in the constructor of
   :py:class:`~helios.plugins.optuna.OptunaPlugin`.

In order for the plug-in to work properly, the plug-in assumes that the ``metric_name``
key exists in the :py:attr:`~helios.model.model.Model.metrics` table. If it doesn't,
nothing is reported to the trial. The plug-in will automatically handled distributed
training correctly, so there's no need for the model to do extra work.

.. warning::
   In distributed training, it is your responsibility to ensure that the value of the
   metric is correctly synced across processess (if applicable).

For example, we can report the metrics at the end of
:py:meth:`~helios.model.model.Model.on_validation_end` like this:

.. code-block:: python

   def on_validation_end(self, validation_cycle: int) -> None:
        ...
        plugin = self.trainer.plugins[OptunaPlugin.plugin_id]
        assert isinstance(plugin, OptunaPlugin)
        plugin.report_metrics(validation_cycle)


Trial Pruning and Returning Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The plug-in will automatically detect if a trial is pruned by optuna and gracefully
request that training end. The exact behaviour depends on whether training is distributed
or not. Specifically:

* If training is not distributed, then the plug-in will raise a
  :py:class:`optuna.TrialPruned` exception *after* calling
  :py:meth:`~helios.model.model.Model.on_training_end` on the model. This ensures that if
  any metrics are logged when training ends, they get logged if the trial is pruned.
* If training is distributed, then the plug-in requests that training terminate early. The
  normal execution flow occurs when training is terminated early. Once the code exits the
  :py:meth:`~helios.trainer.Trainer.fit` function, the user should call
  :py:meth:`~helios.plugins.optuna.OptunaPlugin.check_pruned` to ensure that the
  corresponding exception is correctly raised.

In code, this can be handled as follows:

.. code-block:: python

   def objective(trial: optuna.Trial) -> float:
        ...
        plugin.configure_trainer(trainer)
        plugin.configure_model(model)
        trainer.fit(model, datamodule)
        plugin.check_pruned()

To correctly return metrics, there are two cases that need to be handled. If training
isn't distributed, then the metrics can be grabbed directly from the
:py:attr:`~helios.model.model.Model.metrics` table. If training is distributed, then the
model needs to do a bit more work to ensure things get synchronized correctly. For our
example, we will place the synchronization of the metrics on
:py:meth:`~helios.model.model.Model.on_training_end`, but you may place it elsewhere if
it's convenient for you:

.. code-block:: python

    def on_training_end(self) -> None:
        ...
        # Push the metrics we want to save into the multi-processing queue.
        if self.is_distributed and self.rank == 0:
            assert self.trainer.queue is not None
            self.trainer.queue.put(
                {"accuracy": accuracy, "loss": self._loss_items["loss"].item()}
            )

The :py:attr:`~helios.trainer.Trainer.queue` ensures that the values get transferred to
the primary process. Once that's done, we just need to add the following to our
``objective`` function:

.. code-block:: python

   def objective(trial: optuna.Trial) -> float:
        ...
        plugin.configure_trainer(trainer)
        plugin.configure_model(model)
        trainer.fit(model, datamodule)
        plugin.check_pruned()

        if trainer.queue is None:
            return model.metrics["accuracy"]

        metrics = trainer.queue.get()
        return metrics["accuracy"]

Generic Suggestion of Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The plug-in comes equipped with a function that wraps the ``suggest_`` family of functions
from the :py:class:`optuna.Trial` instance it holds. This function is designed to allow
the suggestion of parameters to be controlled by an outside source (such as command line
arguments or a config file). The goal is to allow code re-usability by not having the
parameters be hard-coded. The function is called
:py:meth:`~helios.plugins.optuna.OptunaPlugin.suggest` and can be used as follows:

.. code-block:: python

   def objective(trial: optuna.Trial) -> float:
        val1 = plugin.suggest("categorical", "val1", choices=[1, 2, 3])
        val2 = plugin.suggest("float", "val2", low=0, high=1, log=True)

The exact arguments for each ``suggest_`` function can be found `here
<https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical>`__.

.. warning::
   The plug-in does *not* provide wrappers for any function that is marked as deprecated.

Resuming Optuna Studies
^^^^^^^^^^^^^^^^^^^^^^^^

Similar to the ability of the :py:class:`~helios.trainer.Trainer` to resume training in the
event of a failure or cancellation, Helios ships with a system to allow Optuna studies to
resume. The system is split into two halves: restoring the state of the samplers and
restoring trials that failed while preserving trials that completed. Let's start by
looking at how we can recover samplers.

As per the official Optuna documentation, the recommended way of ensuring samplers are
reproducible is to:

#. Seed them,
#. Save them periodically to ensure they can be restored.

Seeding of samplers is left to the user, though the use of
:py:func:`~helios.core.rng.get_default_seed` is recommended to ensure consistency. To
checkpoint the samplers, you can add do:

.. code-block:: python

   def objective(trial: optuna.Trial) -> float:
        checkpoint_sampler(trial, root)

This will create a checkpoint stored in the folder specified by ``root``. If the study
is stopped and you wish to restore the last checkpoint, you may do so like this:

.. code-block:: python

   # In your setup code
   sampler = restore_sampler(root)

Note that :py:func:`~helios.plugins.optuna.restore_sampler` can return ``None`` if no
checkpoint is found, in which case you should construct the sampler manually.

To resume studies, you may use :py:func:`~helios.plugins.optuna.resume_study`. The
function can be used as follows:

.. code-block:: python

   # In your setup code
   study_args = {
        "study_name": "test_study",
        "storage": "sqlite:///study.db", # <- This is required
        "load_if_exists": True,
   }

   study = resume_study(study_args)

The call to :py:func:`~helios.plugins.optuna.resume_study` will perform the following
operations:

#. It will create the study using the provided ``study_args``. If the study contains no
   previously run trials, it will return immediately.
#. If trials are found, then they are split into two groups: trials that completed (this
   includes pruned trials) and trials that failed.
#. The old study will be backed up and a new one will be made. Trials that completed
   successfully will be transferred over intact, whereas trials that failed will be
   re-enqueued. Once this is done, the study is returned.

.. warning::
   The ``study_args`` dictionary **must** contain the ``storage`` key set to a path
   beginning with ``sqlite``. Likewise, ``load_if_exists`` **must** be set to true.

The function offers some points of control, mainly:

* A list of states to be considered as failures can be passed in to ``failed_states``. By
  default only ``optuna.trial.TrialState.FAIL`` is considered, but you may add additional
  states such as ``optuna.trial.TrialState.RUNNING`` to handle cases where the study was
  killed by an external source.
* The function will automatically backup the storage of the study before creating the new
  one as a fail-safe in the event that something goes wrong. If you don't wish for a
  backup to be made, you may set ``backup_study`` to false.
