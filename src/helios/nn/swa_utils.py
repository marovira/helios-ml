import copy
import typing

import torch
from torch import nn


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

    @property
    def module(self) -> nn.Module:
        """Get the underlying network."""
        return self._module

    @torch.no_grad()
    def _update(self, net: nn.Module, update_fn: typing.Callable) -> None:
        for ema_v, net_v in zip(
            self._module.state_dict().values(),
            net.state_dict().values(),
            strict=True,
        ):
            if self._device:
                net_v = net_v.to(device=self._device)
            ema_v.copy_(update_fn(ema_v, net_v))

    def update(self, net: nn.Module) -> None:
        """Update the weights using EMA from the given network."""
        self._update(
            net, update_fn=lambda e, m: self._decay * e + (1.0 - self._decay) * m
        )

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
        return self._module(*args, **kwargs)
