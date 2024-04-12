import dataclasses
import typing

import pytest
from torch import nn

from helios import core


class SampleModuleNoArgs(nn.Module):
    def forward(self, x: int) -> int:
        return x


class SampleModuleArgs(nn.Module):
    def __init__(self, val: int):
        super().__init__()

        self._val = val

    def forward(self, x: int) -> int:
        return x * self._val


class SampleModuleKwargs(nn.Module):
    def __init__(self, val: int = 1):
        super().__init__()

        self._val = val

    def forward(self, x: int) -> int:
        return x * self._val


class SampleModuleArgsAndKwargs(nn.Module):
    def __init__(self, val: int, opt_val: int = 1):
        super().__init__()

        self._val = val
        self._opt_val = opt_val

    def forward(self, x: int) -> int:
        return x * self._opt_val + self._val


@dataclasses.dataclass
class SampleEntry:
    sample_type: type
    exp_ret: int
    args: list[typing.Any] = dataclasses.field(default_factory=list)
    kwargs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@pytest.fixture
def check_registry() -> typing.Callable[[core.Registry, list[str]], None]:
    def impl_fun(registry: core.Registry, registered_names: list[str]) -> None:
        assert len(typing.cast(typing.Sized, registry.keys())) != 0

        for name in registered_names:
            assert name in registry

    return impl_fun


@pytest.fixture
def check_create_function() -> typing.Callable[[core.Registry, typing.Callable], None]:
    def impl_fun(registry: core.Registry, create_fun: typing.Callable) -> None:
        in_val = 4
        test_table = [
            SampleEntry(SampleModuleNoArgs, exp_ret=in_val),
            SampleEntry(SampleModuleArgs, exp_ret=in_val * 2, args=[2]),
            SampleEntry(SampleModuleKwargs, exp_ret=in_val * 2, kwargs={"val": 2}),
            SampleEntry(
                SampleModuleArgsAndKwargs,
                exp_ret=in_val * 2 + 2,
                args=[2],
                kwargs={"opt_val": 2},
            ),
        ]

        for entry in test_table:
            registry.register(entry.sample_type)

        for entry in test_table:
            ret = create_fun(entry.sample_type.__name__, *entry.args, **entry.kwargs)
            assert isinstance(ret, entry.sample_type)
            val = ret(in_val)  # type: ignore[operator]
            assert val == entry.exp_ret

    return impl_fun
