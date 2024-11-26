try:
    import optuna
except ImportError as e:
    raise ImportError("error: OptunaPlugin requires Optuna to be installed") from e
import gc
import pathlib
import pickle
import shutil
import typing

import torch

import helios.core.distributed as dist
import helios.model as hlm
import helios.plugins as hlp
import helios.trainer as hlt

# Ignore private member access.
# ruff: noqa: SLF001

_PRUNED_KEY = "ddp_hl:pruned"
_CYCLE_KEY = "ddp_hl:cycle"
_ORIG_NUMBER_KEY = "hl:id"


def _backup_study(storage_path: pathlib.Path) -> None:
    if not storage_path.exists():
        return

    name = storage_path.stem
    root = storage_path.parent

    i = -1
    for path in root.iterdir():
        if not path.is_file():
            continue
        if "_bkp-" in path.stem:
            elems = path.stem.split("_")
            idx = [i for i, e in enumerate(elems) if "bkp-" in e][0]
            n = int(elems[idx].split("-")[-1])
            i = max(i, n)

    i = i + 1
    bkp_path = root / f"{name}_bkp-{i}.db"
    shutil.copy(str(storage_path), str(bkp_path))


def resume_study(
    study_args: dict[str, typing.Any],
    failed_states: typing.Sequence = (optuna.trial.TrialState.FAIL,),
    backup_study: bool = True,
) -> optuna.Study:
    """
    Resume a study that stopped because of a failure.

    The goal of this function is to allow studies that failed due to an error (either an
    exception, system error etc.) to continue utilising the built-in checkpoint system
    from Helios. To accomplish this, the function will do the following:

    #. Grab all the trials from the study created by the given ``study_args``, splitting
        them into three groups: completed/pruned, failed, and failed but completed.
    #. Create a new study with the same name and storage. This new study will get all of
        the completed trials of the original, and will have the failed trials re-enqueued.

    .. warning::
        This function **requires** the following conditions to be true:
            #. It is called **before** the trials are started.
            #. The study uses ``RDBStorage`` as the storage argument for
            ``optuna.create_study``.
            #. ``load_if_exists`` is set to True in ``study_args``.
            #. ``TrialState.PRUNED`` **cannot** be in the list of ``failed_states``.

    The ``failed_states`` argument can be used to set additional trial states to be
    considered as "failures". This can be useful when dealing with special cases where
    trials were either completed or pruned but need to be re-run.

    By default, the original study (assuming there is one) will be backed up with the name
    ``<study-name>_backup-#`` where ``<study-name>`` is the name of the database of the
    original study, and ``#`` is an incremental number starting at 0. This behaviour can
    be disabled by setting ``backup_study`` to False.

    This function works in tandem with
    :py:meth:`~helios.plugins.optuna.OptunaPlugin.configure_model` to ensure that when
    the failed trial is re-run, the original save name is restored so any saved
    checkpoints can be re-used so the trial can continue instead of starting from
    scratch.

    .. note::
        Only trials that fail but haven't been completed will be enqueued by this
        function. If a trial fails and is completed later on, it will be treated as if it
        had finished successfully.

    Args:
        study_args: dictionary of arguments for ``optuna.create_study``.
        failed_states: the trial states that are considered to be failures and should
            be re-enqueued.
        backup_study: if True, the original study is backed up so it can be re-used later
            on.
    """
    if "storage" not in study_args:
        raise TypeError("error: RDB storage is required for resuming studies")
    if "load_if_exists" not in study_args or not study_args["load_if_exists"]:
        raise KeyError("error: study must be created with 'load_if_exists' set to True")
    if optuna.trial.TrialState.PRUNED in failed_states:
        raise ValueError("error: pruned trials cannot be considered as failed")

    storage_str: str = study_args["storage"]
    if not isinstance(storage_str, str):
        raise TypeError("error: only strings are supported for 'storage'")

    storage = pathlib.Path(storage_str.removeprefix("sqlite:///")).resolve()
    if backup_study:
        _backup_study(storage)

    # Step 1: create the study with the current DB and grab all the trials.
    study = optuna.create_study(**study_args)

    # Fast exit: if there are no trials, return immediately.
    if len(study.trials) == 0:
        return study

    complete: list[optuna.trial.FrozenTrial] = []
    failed_but_completed: list[optuna.trial.FrozenTrial] = []
    failed: dict[int, optuna.trial.FrozenTrial] = {}

    for trial in study.trials:
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and _ORIG_NUMBER_KEY in trial.user_attrs
        ):
            failed_but_completed.append(trial)
        elif (
            trial.state == optuna.trial.TrialState.COMPLETE
            or trial.state == optuna.trial.TrialState.PRUNED
        ):
            complete.append(trial)
        elif trial.state in failed_states:
            failed[trial.number] = trial

    # Make sure that any trials that failed but were completed are pruned from the failed
    # trials list.
    for trial in failed_but_completed:
        trial_num = trial.user_attrs[_ORIG_NUMBER_KEY]
        failed.pop(trial_num, None)

    # Make sure that the study is cleared out before we attempt to rename the storage
    del study
    gc.collect()

    # Step 2: rename the DB and create a new empty study.
    tmp_storage = storage.parent / (storage.stem + "_tmp" + storage.suffix)
    storage.rename(tmp_storage)
    study = optuna.create_study(**study_args)

    # Step 3: move all the trials into the new study, re-setting all trials that failed.
    for trial in complete:
        study.add_trial(trial)

    for _, trial in failed.items():
        study.enqueue_trial(trial.params, {_ORIG_NUMBER_KEY: trial.number})

    # Once everything's done, clean up the temp storage.
    tmp_storage.unlink()

    return study


def checkpoint_sampler(trial: optuna.Trial, chkpt_root: pathlib.Path) -> None:
    """
    Create a checkpoint with the state of the sampler.

    This function can be used to ensure that if a study is restarted, the state of the
    sampler is recovered so trials can be reproducible. The function will automatically
    create a checkpoint using ``torch.save``.

    .. note::
        It is recommended that this function be called at the start of the objective
        function to ensure the checkpoint is made correctly, but it can be called at any
        time.

    Args:
        trial: the current trial.
        chkpt_root: the root where the checkpoints will be saved.
    """
    chkpt_path = chkpt_root / (f"sampler_trial-{trial.number}.pkl")

    sampler = trial.study.sampler
    with chkpt_path.open("wb") as outfile:
        pickle.dump(sampler, outfile)


def restore_sampler(chkpt_root: pathlib.Path) -> optuna.samplers.BaseSampler | None:
    """
    Restore the sampler from a previously saved checkpoint.

    This function can be used in tandem with
    :py:func:`~helios.plugins.optuna.checkpoint_sampler` to ensure that the last
    checkpoint is loaded and the correct state is restored for the sampler. This function
    **needs** to be called before ``optuna.create_study`` is called.

    Args:
        chkpt_root: the root where the checkpoints are stored.

    Returns:
        The restored sampler.
    """

    def key(path: pathlib.Path) -> int:
        return int(path.stem.split("-")[-1])

    chkpts = list(chkpt_root.glob("*.pkl"))
    chkpts.sort(key=key)
    if len(chkpts) == 0:
        return None

    sampler = pickle.load(chkpts[-1].open("rb"))
    return sampler


@hlp.PLUGIN_REGISTRY.register
class OptunaPlugin(hlp.Plugin):
    """
    Plug-in to do hyper-parameter tuning with Optuna.

    This plug-in integrates `Optuna <https://optuna.readthedocs.io/en/stable/>`__ into the
    training system in order to provide hyper-parameter tuning. The plug-in provides the
    following functionality:

    #. Automatic handling of trial pruning.
    #. Automatic reporting of metrics.
    #. Exception registration for trial pruning.
    #. Easy integration with Helios' checkpoint system to continue stopped trials.

    .. warning::
        This plug-in **requires** Optuna to be installed before being used. If it isn't,
        then :py:exc:`ImportError` is raised.

    Example:
        .. code-block:: python

            import helios.plugins as hlp
            import optuna

            def objective(trial: optuna.Trial) -> float:
                datamodule = ...
                model = ...
                plugin = hlp.optuna.OptunaPlugin(trial, "accuracy")

                trainer = ...

                # Automatically registers the plug-in with the trainer.
                plugin.configure_trainer(trainer)

                # This can be skipped if you don't want the auto-resume functionality or
                # if you wish to manage it yourself.
                plugin.configure_model(model)

                trainer.fit(model, datamodule)
                plugin.check_pruned()
                return model.metrics["accuracy"]

            def main():
                # Note that the plug-in requires the storage to be persistent.
                study = optuna.create_study(storage="sqlite:///example.db", ...)
                study.optimize(objective, ...)

    Args:
        trial: the Optuna trial.
        metric_name: the name of the metric to monitor. This assumes the name will be
            present in the :py:attr:`~helios.model.model.Model.metrics` table.
    """

    plugin_id = "optuna"

    def __init__(self, trial: optuna.Trial, metric_name: str) -> None:
        """Create the plug-in."""
        super().__init__(self.plugin_id)
        self._trial = trial
        self._metric_name = metric_name
        self._last_cycle: int = 0

        self.unique_overrides.should_training_stop = True

    @property
    def trial(self) -> optuna.Trial:
        """Return the trial."""
        return self._trial

    @trial.setter
    def trial(self, t: optuna.Trial) -> None:
        self._trial = t

    def configure_trainer(self, trainer: hlt.Trainer) -> None:
        """
        Configure the trainer with the required settings.

        This will do two things:

        #. Register the plug-in itself with the trainer.
        #. Append the trial pruned exception to the trainer.

        Args:
            trainer: the trainer instance.
        """
        self._register_in_trainer(trainer)
        self._append_train_exceptions(optuna.TrialPruned, trainer)

    def configure_model(self, model: hlm.Model) -> None:
        """
        Configure the model to set the trial number into the save name.

        This will alter the :py:attr:`~helios.model.model.Model.save_name` property of the
        model by appending :code:`_trial-<trial-numer>`.

        Args:
            model: the model instance.
        """
        n_trial = self.trial.user_attrs.get(_ORIG_NUMBER_KEY, self.trial.number)
        model._save_name = model._save_name + f"_trial-{n_trial}"

    def suggest(self, type_name: str, name: str, **kwargs: typing.Any) -> typing.Any:
        """
        Generically Wrap the ``suggest_`` family of functions of the optuna trial.

        This function can be used to easily invoke the corresponding ``suggest_`` function
        from the Optuna trial held by the plug-in without having to manually type each
        individual function. This lets you write generic code that can be controlled by an
        external source (such as command line arguments or a config table). The function
        wraps the following functions:

        .. list-table:: Suggestion Functions
            :header-rows: 1

            * - Function
              - Name
            * - ``optuna.Trial.suggest_categorical``
              - categorical
            * - ``optuna.Trial.suggest_int``
              - int
            * - ``optuna.Trial.suggest_float``
              - float

        .. warning::
            Functions that are marked as deprecated by Optuna are *not* included in this
            wrapper.

        .. note::
            You can find the exact arguments for each function `here
            <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html>`__.

        Example:
            .. code-block:: python

                import helios.plugin as hlp
                import optuna

                def objective(trial: optuna.Trial) -> float:
                    plugin = hlp.optuna.OptunaPlugin(trial, "accuracy")
                    # ... configure model and trainer.

                    val1 = plugin.suggest("categorical", "val1", choices=[1, 2, 3])
                    val2 = plugin.suggest("int", "val2", low=0, high=10)
                    val3 = plugin.suggest("float", "val3", low=0, high=1)

        Args:
            type_name: the name of the type to suggest from.
            name: a parameter name
            **kwargs: keyword arguments to the corresponding suggest function.

        Raises:
            KeyError: if the value passed in to ``type_name`` is not recognised.
        """
        if type_name not in ("categorical", "float", "int"):
            raise KeyError(f"error: {type_name} is not a valid suggestion type.")

        fn = getattr(self._trial, f"suggest_{type_name}")
        return fn(name, **kwargs)

    def setup(self) -> None:
        """
        Configure the plug-in.

        Raises:
            ValueError: if the study wasn't created with persistent storage.
        """
        if self.is_distributed and not (
            isinstance(self.trial.study._storage, optuna.storages._CachedStorage)
            and isinstance(self.trial.study._storage._backend, optuna.storages.RDBStorage)
        ):
            raise ValueError(
                "error: optuna integration supports only optuna.storages.RDBStorage "
                "in distributed mode"
            )

    def report_metrics(self, validation_cycle: int) -> None:
        """
        Report metrics to the trial.

        This function should be called from the model once the corresponding metrics have
        been saved into the :py:attr:`~helios.model.model.Model.metrics` table.

        Example:
            .. code-block:: python

                import helios.model as hlm
                import helios.plugins.optuna as hlpo

                class MyModel(hlm.Model):
                    ...
                    def on_validation_end(self, validation_cycle: int) -> None:
                        # Compute metrics
                        self.metrics["accuracy"] = 10

                        plugin = self.trainer.plugins[hlpo.OptunaPlugin.plugin_id]
                        assert isinstance(plugin hlpo.OptunaPlugin)
                        plugin.report_metrics(validation_cycle)

        .. note::
            In distributed training, only rank 0 will report the metrics to the trial.

        Args:
            validation_cycle: the current validation cycle.
        """
        model = self.trainer.model
        if not model.metrics or self._metric_name not in model.metrics:
            return

        if self.rank == 0:
            self.trial.report(model.metrics[self._metric_name], validation_cycle)
        self._last_cycle = validation_cycle

    def should_training_stop(self) -> bool:
        """
        Handle trial pruning.

        Returns:
            True if the trial should be pruned, false otherwise.
        """
        should_stop = False
        if self.rank == 0:
            should_stop = self.trial.should_prune()

        # Sync the value across all processes (if using distributed training).
        if self.is_distributed:
            t = dist.all_reduce_tensors(torch.tensor(should_stop).to(self.device))
            should_stop = t.item()  # type: ignore[assignment]

        if should_stop and self.rank == 0:
            self.trial.set_user_attr(_PRUNED_KEY, True)
            self.trial.set_user_attr(_CYCLE_KEY, self._last_cycle)

        return should_stop

    def on_training_end(self) -> None:
        """
        Clean-up on training end.

        If training is non-distributed and the trial was pruned, then this function will
        do the following:

        #. Call :py:meth:`~helios.model.model.Model.on_training_end` to ensure metrics are
           correctly logged (if using).
        #. Raise :py:exc:`optuna.TrialPruned` exception to signal the trial was pruned.

        If training is distributed, this function does nothing.

        Raises:
            TrialPruned: if the trial was pruned.
        """
        if not self.is_distributed and self.trial.should_prune():
            self.trainer.model.on_training_end()
            raise optuna.TrialPruned(f"Pruned on validation cycle {self._last_cycle}")

    def check_pruned(self) -> None:
        """
        Ensure pruned distributed trials are correctly handled.

        Due to the way distributed training works, we can't raise an exception within the
        distributed processes, so we have to do it after we return to the main process.
        If the trial was pruned, this function will raise :py:exc:`optuna.TrialPruned`. If
        distributed training wasn't used, this function does nothing.

        .. warning::
            You *must* ensure this function is called after
            :py:meth:`~helios.trainer.Trainer.fit` to ensure pruning works correctly.

        Raises:
            TrialPruned: if the trial was pruned.
        """
        trial_id = self.trial._trial_id
        study = self.trial.study
        trial = study._storage._backend.get_trial(trial_id)  # type: ignore[attr-defined]
        is_pruned = trial.user_attrs.get(_PRUNED_KEY)
        val_cycle = trial.user_attrs.get(_CYCLE_KEY)

        if is_pruned is None or val_cycle is None:
            return

        if is_pruned:
            raise optuna.TrialPruned(f"Pruned on validation cycle {val_cycle}")

    def state_dict(self) -> dict[str, typing.Any]:
        """
        Get the state of the current trial.

        This will return the parameters to be optimised for the current trial.

        Returns:
            The parameters of the trial.
        """
        return self._trial.params
