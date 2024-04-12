import typing

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules import batchnorm as bn

from helios import core

NETWORK_REGISTRY = core.Registry("network")


def create_network(type_name: str, *args: typing.Any, **kwargs: typing.Any) -> nn.Module:
    """
    Create the network for the given type.

    Args:
        type_name (str): the type of the network to create.
        args: positional arguments to pass into the network.
        kwargs: keyword arguments to pass into the network.
    Returns:
        nn.Module: the network.
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

    Args:
        module_list (list[nn.Module] | nn.Module): the list of modules to initialize.
        scale (float): scale initialized weights, especially for residual blocks.
        bias_fill (float): bias fill value.
        kwargs (Any): keyword arguments for the torch.nn.init.kaiming_normal_ function.
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
