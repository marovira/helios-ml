from __future__ import annotations

import pathlib
import typing

from helios import core

if typing.TYPE_CHECKING:
    from .model import Model

MODEL_REGISTRY = core.Registry("model")
"""
Global instance of the registry for models.

Example:
    .. code-block:: python

        import helios.model as hlm

        # This automatically registers your model.
        @hlm.MODEL_REGISTRY.register
        class MyModel:
            ...

        # Alternatively you can manually register a model like this:
        hlm.MODEL_REGISTRY.register(MyModel)
"""


def create_model(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> Model:
    """
    Create the model for the given type.

    Args:
        type_name: the type of the model to create.
        args: positional arguments to pass into the model.
        kwargs: keyword arguments to pass into the model.

    Returns:
        The model.
    """
    return MODEL_REGISTRY.get(type_name)(*args, **kwargs)


def find_pretrained_file(root: pathlib.Path, name: str) -> pathlib.Path:
    """
    Find the pre-trained file in the given root.

    The assumption is the following:
    Given a root ``/models/cifar`` and a name ``resnet-50``, then the name of the
    pre-trained file will contain ``cifar_resnet-50_`` as a prefix. If no file is
    found, an exception is raised.

    Args:
        root: the root where the file is stored.
        net_name: the save name of the file.

    Returns:
        The path to the file.

    Raises:
        RuntimeError: if no pre-trained network was found.
    """
    for path in root.glob("*.pth"):
        file_name = str(path.stem)
        base_name = f"{str(root.stem)}_{name}_"
        if base_name in file_name:
            return path

    raise RuntimeError(
        f"error: unable to find a pretrained network named {name} at {str(root)}"
    )
