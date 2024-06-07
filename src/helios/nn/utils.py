import typing

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules import batchnorm as bn

from helios import core

NETWORK_REGISTRY = core.Registry("network")
"""
Global instance of the registry for networks.

Example:
    .. code-block:: python

        import helios.nn as hln

        # This automatically registers your network.
        @hln.NETWORK_REGISTRY.register
        class MyNetwork:
            ...

        # Alternatively you can manually register a network like this:
        hln.NETWORK_REGISTRY.register(MyNetwork)
"""


def create_network(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> nn.Module:
    """
    Create the network for the given type.

    Args:
        type_name: the type of the network to create.
        args: positional arguments to pass into the network.
        kwargs: keyword arguments to pass into the network.
    Returns:
        The network.
    """
    return NETWORK_REGISTRY.get(type_name)(*args, **kwargs)


@torch.no_grad()
def default_init_weights(
    module_list: list[nn.Module] | nn.Module,
    scale: float = 1,
    bias_fill: float = 0,
    **kwargs: typing.Any,
) -> None:
    """
    Initialize network weights.

    Specifically, this function will default initialize the following types of blocks:

    * ``torch.nn.Conv2d``,
    * ``torch.nn.Linear``,
    * ``torch.nn.modules.batchnorm._BatchNorm``

    Args:
        module_list: the list of modules to initialize.
        scale: scale initialized weights, especially for residual blocks. Defaults to 1.
        bias_fill: bias fill value. Defaults to 0.
        kwargs: keyword arguments for the ``torch.nn.init.kaiming_normal_`` function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, bn._BatchNorm):  # noqa: SLF001
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
