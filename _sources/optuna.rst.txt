Optuna Integartion
##################

Helios ships with first-class integration for `optuna
<https://optuna.readthedocs.io/en/stable/>`__. The integration is split into two blocks:

* The :py:class:`~helios.plugins.optuna.OptunaPlugin`.
* The system to create and resume studies.

This page will focus on the latter.

Sampler Checkpoints
===================

Similar to the ability of the :py:class:`~helios.trainer.Trainer` to resume training in
the event of a failure or cancellation, Helios ships with a system to allow Optuna studies
to resume. The system is split into two halves: restoring the state of the samplers and
resuming studies. Let's start by looking at how we can recover samplers.

Per the official Optuna documentation, the recommended way of ensuring samplers are
reproducible is to:

#. Seed them,
#. Save them periodically to ensure they can be restored.

The seeding of samplers is left to the user, though the use of
:py:func:`~helios.core.rng.get_default_seed` is recommended to ensure consistency. To
checkpoint samplers you can do:

.. code-block:: python

   def objective(trial: optuna.Trial) -> float:
        checkpoint_sampler(trial, root)

This will create a checkpoint stored in the folder specified by ``root``. It is
recommended that this function be called at the start and end of the ``objective``
function. The reasoning is to allow recovery in the following two cases:

#. The ``objective`` function fails before reaching the end. If the study is then resumed,
   the sampler can be restored to the state *before* values were obtained from it, thereby
   guaranteeing that the same values are obtained upon resuming.
#. The study is stopped (either cancelled or natural termination) and we wish to resume
   with a new trial. This ensures the sampler can be restored and that the values obtained
   by the new trial match those that would've been obtained if the study hadn't stopped.

.. note::
   When a checkpoint is created, it will overwrite an existing checkpoint with the same
   name. Therefore the recommendation stated above will result in a single checkpoint
   being stored per trial.

If you wish to manually restore the sampler, you can do so by calling
:py:func:`~helios.plugins.optuna.utils.restore_sampler` like this:

.. code-block:: python

   # In your setup code.
   sampler = restore_sampler(root)

.. note::
   This function **must** be called before the creation of the study so the sampler can be
   passed in.

Note that :py:func:`~helios.plugins.optuna.utils.restore_sampler` can return ``None`` if
no sampler is found.

Creating and Restarting Studies
===============================

Helios offers two ways of creating studies:

#. :py:func:`~helios.plugins.optuna.utils.create_or_load_study`
#. :py:func:`~helios.plugins.optuna.utils.create_study_starting_from_trial`

The choice of function depends on the desired behaviour. That said, both functions
**require** that the following conditions are met:

#. The study uses a string that starts with ``sqlite:///`` to specify ``RDBStorage`` as an
   argument to ``storage``.
#. The ``load_if_exists`` flag is set to ``True``.

You can find more details of these arguments in the
:py:class:`~helios.plugins.optuna.utils.StudyArgs` or in the official `docs
<https://optuna.readthedocs.io/en/latest/reference/generated/optuna.create_study.html>`__.

Creating or Reloading Studies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`~helios.plugins.optuna.utils.create_or_load_study` function can be used in
the following cases:

#. You wish to create a study that is configured so it can be resumed.
#. You wish to continue a study that has stopped.

In order to use this function, it is recommended that you:

#. Pass in ``sampler_path`` so samplers can be saved/restored.
#. Ensure that you use the checkpoint functions for samplers as described above.

If you wish to resume a study, you can also pass in ``backup_study`` and
``backup_samplers`` to create backups of the original study and checkpoints.

.. note::
   This function will not try to re-run failed trials. It will simply continue the study
   with a new blank trial.

To use this function, you can do:

.. code-block:: python

   import helios.plugins.optuna as hlpo

   args: hlpo.StudyArgs = {}
   args["storage"] = "sqlite:///test.db"
   args["load_if_exists"] = True
   # Remaining arguments can go here.

   study = hlpo.create_or_load_study(sampler_path, **args)

   # Later on, if we wish to resume the study, we can do:
   study = hlpo.create_or_load_study(sampler_path, **args)

The function will automatically restore the sampler for you (if it exists) so you don't
have to specify it manually.

Restarting a Study from a Given Trial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`~helios.plugins.optuna.utils.create_study_starting_from_trial` can be used
to restart a study by specifying the trial number of the next trial that should be run.
This is particularly useful in the event that a study fails in an unexpected way and you
wish to continue the study prior to the failure while ensuring the results remain
reproducible. To better explain this, let's suppose that we have a study that is meant to
run for 10 trials and we encounter a failure at trial 5 which stops the study. What we'd
want to do is to restart the study such that trial 5 is re-run. Assuming we have been
creating checkpoints for samplers, then we can do the following:

.. code-block:: python

   import helios.plugins.optuna as hlpo

    args: hlpo.StudyArgs = {}
    args["storage"] = "sqlite:///study.db"
    args["load_if_exists"] = True
    args["study_name"] = "study"

    study = hlpo.create_or_load_study(sampler_path, **args)

    study.optimize(...) # This call fails.

    # We want to re-run trial 5, which maps to index 4:
    study = hlpo.create_study_starting_from_trial(4, sampler_path=sampler_path, **args)

    # We can now resume from trial 5.

The way the function works is as follows:

#. It will first (by default) backup the study and the sampler checkpoints in case
   something goes wrong so they can be restored.
#. It loads the study and grabs all the trials whose number is less than the given index
   (in our example, all trials whose number is less than 4).
#. If the sampler path was given, then it will also delete all sampler checkpoints
   corresponding to trials whose number is greater than or equal to the given number. This
   ensures that if the study succeeds, the sampler checkpoints are left in a valid state.
#. It will then create a new study and transfer all of the trials it grabbed in step 2 so
   the new study can start on the given trial number.

If you wish to skip the backups (not recommended), then you can set ``backup_study`` and
``backup_samplers`` to false.

.. note::
   While it isn't a requirement, it is *highly* recommended that you backup your samplers
   to ensure reproducibility.

Once the study is returned, you can then run the ``optimize`` function and continue the
study as usual. If you are using the checkpoint system from
:py:class:`~helios.model.Model`, then the trial can be restarted from the last saved
checkpoint.
