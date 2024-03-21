import typing
import pytest

from pyro import core


class TestFunctional:
    def test_get_from_optional(self) -> None:
        opt: typing.Optional[int] = None

        with pytest.raises(RuntimeError):
            core.get_from_optional(opt, raise_on_empty=True)

        opt = 1
        assert core.get_from_optional(opt) == 1
