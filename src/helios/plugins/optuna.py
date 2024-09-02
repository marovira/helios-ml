try:
    import optuna
except ImportError as e:
    raise ImportError("error: OptunaPlugin requires Optuna to be installed") from e
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
        Configure the model to allow trials to resume.

        This will alter the :py:attr:`~helios.model.model.Model.save_name` property of the
        model by appending :code:`_trial-<trial-numer>`. In the event that a trial with
        that number has already been attempted, it will be set to that number instead.
        This will allow the automatic checkpoint system of the trainer to resume the
        trial.

        Args:
            model: the model instance.
        """
        trial_number = optuna.storages.RetryFailedTrialCallback.retried_trial_number(
            self.trial  # type: ignore[arg-type]
        )
        trial_id = (
            f"_trial-{self.trial.number}"
            if trial_number is None
            else f"_trial-{trial_number}"
        )

        model._save_name = model._save_name + trial_id

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

    def on_validation_end(self, validation_cycle: int) -> None:
        """
        Report metrics to the trial.

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
