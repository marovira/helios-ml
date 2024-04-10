import copy
import typing

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules import batchnorm as bn

from pyro import core

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


class EMA(nn.Module):
    """
    Implements Exponential Moving Average (EMA).

    Args:
        net (nn.Module): the bare network on which EMA will be performed.
        decay (float): decay rate.
        device (torch.device | None): the device to be used.
    """

    def __init__(
        self, net: nn.Module, decay: float = 0.9997, device: torch.device | None = None
    ):
        """Create the EMA wrapper."""
        super().__init__()

        self._module = copy.deepcopy(net)
        self._module = self._module.eval()
        self._decay = decay
        self._device = device

        if self._device is not None:
            self._module.to(device=device)

    @torch.no_grad()
    def _update(self, net: nn.Module, update_fn: typing.Callable) -> None:
        for ema_v, net_v in zip(
            self._module.state_dict().values(),
            net.state_dict().values(),
            strict=True,
        ):
            if self._device:
                net_v = net_v.to(device=self.device)
            ema_v.copy_(update_fn(ema_v, net_v))

    def update(self, net: nn.Module) -> None:
        """Update the weights using EMA from the given network."""
        self._update(net, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, net: nn.Module) -> None:
        """Re-set the base weights."""
        self._update(net, update_fn=lambda e, m: m)

    def forward(self, *args: typing.Any, **kwargs: typing.Any):
        """
        Evaluate the EMA wrapper on the network inputs.

        Args:
            args (Any): named parameters for your network's forward function.
            kwargs (Any): keyword arguments for your network's forward function.
        """
        return self.module(*args, **kwargs)
