import enum
import gc
import pathlib
import pickle
import shutil
import typing
import warnings

import optuna


class _BackupType(enum.Enum):
    FILE = enum.auto()
    DIR = enum.auto()


class StudyArgs(typing.TypedDict):
    """
    Type-hint class for the arguments of ``optuna.create_study``.

    See the `docs
    <https://optuna.readthedocs.io/en/latest/reference/generated/optuna.create_study.html>`__
    for more information.

    Args:
        storage: Database URL. If this argument is set to None, ``InMemoryStorage`` is
            used, and the ``Study`` will not be persistent.
        sampler: A sampler object that implements background algorithm for value
            suggestion.
        pruner: A pruner object that decides early stopping for unpromising trials.
        study_name: Study's name.
        direction: Direction of optimization.
        load_if_exists: flag to control the behaviour to handle conflict of study names.
        directions: A sequence of directions during multi-objective optimization.
    """

    storage: typing.NotRequired[str | optuna.storages.BaseStorage | None]
    sampler: typing.NotRequired[optuna.samplers.BaseSampler | None]
    pruner: typing.NotRequired[optuna.pruners.BasePruner | None]
    study_name: typing.NotRequired[str | None]
    direction: typing.NotRequired[str | optuna.study.StudyDirection]
    load_if_exists: typing.NotRequired[bool]
    directions: typing.NotRequired[
        typing.Sequence[str | optuna.study.StudyDirection] | None
    ]


def _get_backup_name(root: pathlib.Path, name: str, bkp_type: _BackupType) -> str:
    i = -1
    for path in root.iterdir():
        if not path.is_file() and bkp_type == _BackupType.FILE:
            continue
        if not path.is_dir() and bkp_type == _BackupType.DIR:
            continue
        if "_bkp-" in path.stem:
            elems = path.stem.split("_")
            idx = [i for i, e in enumerate(elems) if "bkp-" in e][0]
            n = int(elems[idx].split("-")[-1])
            i = max(i, n)

    i = i + 1
    return f"{name}_bkp-{i}"


def _backup_study(storage_path: pathlib.Path) -> None:
    if not storage_path.exists():
        return

    name = storage_path.stem
    root = storage_path.parent
    bkp_name = _get_backup_name(root, name, _BackupType.FILE)
    bkp_path = root / f"{bkp_name}.db"
    shutil.copy(str(storage_path), str(bkp_path))


def _backup_sampler_chkpts(sampler_path: pathlib.Path) -> None:
    if not sampler_path.exists():
        return

    name = sampler_path.stem
    root = sampler_path.parent
    bkp_name = _get_backup_name(root, name, _BackupType.DIR)
    bkp_path = root / bkp_name
    bkp_path.mkdir()

    for chkpt in sampler_path.glob("*.pkl"):
        dst_path = bkp_path / chkpt.name
        shutil.copy2(str(chkpt), str(dst_path))


def _sampler_chkpt_key(path: pathlib.Path) -> int:
    return int(path.stem.split("-")[-1])


def _check_study_args(**study_args: typing.Unpack[StudyArgs]) -> None:
    if "storage" not in study_args or study_args["storage"] is None:
        raise RuntimeError("error: RDB storage is required for resuming studies")
    if "load_if_exists" not in study_args or not study_args["load_if_exists"]:
        raise RuntimeError(
            "error: study must be created with 'load_if_exists' set to true"
        )
    if not isinstance(study_args["storage"], str):
        raise TypeError("error: only strings are supported for 'storage'")

    if "sqlite:///" not in study_args["storage"]:
        raise RuntimeError("error: only sqlite storage is supported")


def _get_storage_path(**study_args: typing.Unpack[StudyArgs]) -> pathlib.Path:
    name = study_args["storage"]
    assert isinstance(name, str)
    return pathlib.Path(name.removeprefix("sqlite:///")).resolve()


def checkpoint_sampler(trial: optuna.Trial, chkpt_root: pathlib.Path) -> None:
    """
    Create a checkpoint with the state of the sampler.

    This function can be used to ensure that if a study is restarted, the state of the
    sampler is recovered so trials can be reproducible. The function will automatically
    create a checkpoint using ``torch.save``.

    .. note::
        It is recommended that this function be called at the start and end of the
        objective function to ensure the state of the sampler is correctly stored. This
        ensures that if a trial fails before it returns, then the sampler can be restored
        and the same values can be retrieved. Alternatively, if the trial completes and
        you wish to resume the study, then the sampler can be restored to the state it
        would've had if the study hadn't been stopped.

    Args:
        trial: the current trial.
        chkpt_root: the root where the checkpoints will be saved.
    """
    chkpt_path = chkpt_root / (f"sampler_trial-{trial.number}.pkl")

    sampler = trial.study.sampler
    with chkpt_path.open("wb+") as outfile:
        pickle.dump(sampler, outfile)


def restore_sampler(
    chkpt_root: pathlib.Path, trial_id: int | None = None
) -> optuna.samplers.BaseSampler | None:
    """
    Restore the sampler from a previously saved checkpoint.

    This function can be used in tandem with
    :py:func:`~helios.plugins.optuna.checkpoint_sampler` to ensure that the last
    checkpoint is loaded and the correct state is restored for the sampler. This function
    **needs** to be called before ``optuna.create_study`` is called.

    If you wish to restore the sampler for a specific trial, you can use ``trial_id`` to
    restore it.

    Args:
        chkpt_root: the root where the checkpoints are stored.
        trial_id: the trial ID for which to restore the sampler.

    Returns:
        The restored sampler.
    """
    chkpts = list(chkpt_root.glob("*.pkl"))
    chkpts.sort(key=_sampler_chkpt_key)
    if len(chkpts) == 0:
        return None

    chkpt = (
        chkpts[trial_id]
        if trial_id is not None and trial_id < len(chkpts)
        else chkpts[-1]
    )

    sampler = pickle.load(chkpt.open("rb"))
    return sampler


def create_or_load_study(
    sampler_path: pathlib.Path | None = None,
    backup_study: bool = False,
    backup_samplers: bool = False,
    **study_args: typing.Unpack[StudyArgs],
) -> optuna.Study:
    """
    Start or continue a study.

    This function can be used to either start a new study or resume one that has stopped.
    This leverages the existing built-in checkpoint system from Helios as well as
    :py:func:`~helios.plugins.optuna.restore_sampler` to ensure the study can be resumed
    properly.

    .. warning::
        This function **requires** the following conditions to be true:
            #. The study uses a string to specify the ``RDBStorage`` as the argument to
                ``storage`` for ``optuna.create_study``.
            #. The storage string starts with ``sqlite:///``.
            #. ``load_if_exists`` is set to true.

    If you have been using :py:func:`~helios.plugins.optuna.checkpoint_sampler` and wish
    to restore the sampler upon restarting the study, you can provide the root where the
    checkpoints are stored by passing in ``sampler_path``.

    You can use ``backup_study`` to create a backup of the original study. If there is
    one, the backup will be called ``<study-name>_backup-#`` where ``<study-name>`` is the
    name of the database of the original study and ``#`` is an incremental number starting
    at 0.

    .. note::
        This function **does not** retry failed trials, nor does it continue trials that
        were stopped. It will simply continue with the next (new) trial as determined by
        Optuna. To restart a study from a specific trial, please use
        :py:func:`~helios.plugins.optuna.create_study_starting_from_trial`.

    Args:
        sampler_path: the path where the sampler checkpoints are stored. Defaults to None.
        backup_study: if True, the original study is backed up so it can be re-used later
            on. Defaults to false.
        backup_samplers: if True, a backup will be made of the samplers checkpoint folder
            (if given). Defaults to false.
        study_args: arguments for ``optuna.create_study``.
    """
    _check_study_args(**study_args)

    storage = _get_storage_path(**study_args)
    if backup_study:
        _backup_study(storage)

    if backup_samplers and sampler_path is not None:
        _backup_sampler_chkpts(sampler_path)

    sampler = restore_sampler(sampler_path) if sampler_path is not None else None
    if "sampler" in study_args and sampler is not None:
        warnings.warn(
            "warning: existing sampler will be replaced with checkpointed sampler",
            stacklevel=2,
        )

    # Only set the sampler if we actually have one, otherwise we'll end up overriding the
    # user's sampler and that's not what we want.
    if sampler is not None:
        study_args["sampler"] = sampler
    study = optuna.create_study(**study_args)
    return study


def create_study_starting_from_trial(
    trial_id: int,
    backup_study: bool = True,
    backup_samplers: bool = True,
    sampler_path: pathlib.Path | None = None,
    **study_args: typing.Unpack[StudyArgs],
) -> optuna.Study:
    """
    Resume a study from a specific trial number.

    The goal of this function is to allow studies to resume from a specific trial number,
    effectively discarding all trials after that point. This can be used to recover from
    certain types of unexpected errors that result in trials being invalid but still
    marked as completed, thereby making :py:func:`~helios.plugins.optuna.resume_study`
    unable to properly resume. To accomplish this, the function will do the following:

    #. Grab all the trials from the study created by the given ``study_args`` and grab all
        trials whose numbers are *less than or equal to* ``last_trial_number``.
    #. Create a new study with the same name and storage. This new study will get all of
        the trials that were obtained in the previous step.

    If ``sampler_path`` is provided, then the samplers for trials that occur after
    ``trial_id`` will be pruned. The sampler that corresponds to ``trial_id`` will be
    preserved so it can be restored in to the new study. If you don't wish this to happen,
    simply pass in the sampler in ``study_args`` and set ``sampler_path`` to None.

    .. note::
        Trials are 0-indexed. This means that trial ``N`` maps to ID ``N - 1``.

    .. warning::
        This function **requires** the following conditions to be true:

        #. It is called **before** the trials are started.
        #. The study uses ``RDBStorage`` in ``study_args``.
        #. ``load_if_exists`` is set to True in ``study_args``.

    By default, the original study and samplers checkpoint folder will be backed up
    (assuming they exist). The backups are named ``<name>_backup-#`` where ``<name>`` is
    either the name of the database of the original study or the sampler folder, and ``#``
    is an incremental number starting at 0. This behaviour can be disabled by setting
    ``backup_study`` or ``backup_samplers`` to false.

    In addition, this function can also remove the saved samplers from the discarded
    trials if ``sampler_root`` is provided.

    Args:
        trial_id: the ID of the last trial from which the study should resume.
        backup_study: if True, the original study is backed up so it can be re-used later
            on.
        sampler_path: the path where the sampler checkpoints are stored.
        study_args: arguments for ``optuna.create_study``.

    Returns:
        The study with trials up to ``last_trial_number``.
    """
    _check_study_args(**study_args)

    storage = _get_storage_path(**study_args)
    if backup_study:
        _backup_study(storage)

    if backup_samplers and sampler_path is not None:
        _backup_sampler_chkpts(sampler_path)

    study = optuna.create_study(**study_args)

    # Early exit: if there are no trials, return immediately.
    if len(study.trials) == 0:
        return study

    valid_trials: list[optuna.trial.FrozenTrial] = []
    for trial in study.trials:
        if trial.number < trial_id:
            valid_trials.append(trial)

    if sampler_path is not None:
        chkpts = list(sampler_path.glob("*.pkl"))
        chkpts.sort(key=_sampler_chkpt_key)
        for i, path in enumerate(chkpts):
            if i < trial_id:
                continue
            path.unlink()

        if "sampler" in study_args:
            warnings.warn(
                "warning: existing sampler will be replaced with checkpointed sampler",
                stacklevel=2,
            )

        study_args["sampler"] = restore_sampler(sampler_path, trial_id)

    del study
    gc.collect()

    tmp_storage = storage.parent / (storage.stem + "_tmp" + storage.suffix)
    storage.rename(tmp_storage)
    study = optuna.create_study(**study_args)

    for trial in valid_trials:
        study.add_trial(trial)

    tmp_storage.unlink()
    return study
