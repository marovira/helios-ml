"""
Optuna module for Helios.

This contains the classes and utilities related to the Optuna integration.

.. warning::
    This module **requires** that Optuna be installed.
"""

import importlib.util

if importlib.util.find_spec("optuna") is None:
    raise ImportError("error: Optuna is required to access this module")

from .plugin import OptunaPlugin
from .utils import (
    StudyArgs,
    checkpoint_sampler,
    create_or_load_study,
    create_study_starting_from_trial,
    restore_sampler,
)

__all__ = [
    "OptunaPlugin",
    "StudyArgs",
    "checkpoint_sampler",
    "create_or_load_study",
    "create_study_starting_from_trial",
    "restore_sampler",
]
