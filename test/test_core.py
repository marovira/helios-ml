import math
import pathlib
import time
import typing

import pytest
import torch

from pyro import core
from pyro.core import cuda


class TestUtils:
    def test_get_from_optional(self) -> None:
        opt: int | None = None

        with pytest.raises(RuntimeError):
            core.get_from_optional(opt, raise_on_empty=True)

        opt = 1
        assert core.get_from_optional(opt) == 1

    def test_convert_to_list(self) -> None:
        value = 0
        x: int | list[int] | tuple[int, ...] = value

        res = core.convert_to_list(x)
        assert isinstance(res, list)
        assert len(res) == 1
        assert res[0] == value

        x = [value]
        res = core.convert_to_list(x)
        assert res == x

        x = (1, 3, 4)
        res = core.convert_to_list(x)
        assert len(x) == len(res)
        assert list(x) == res

    def test_chdir_context(self, tmp_path: pathlib.Path) -> None:
        cur_dir = pathlib.Path.cwd()
        with core.ChdirContext(tmp_path) as cwd:
            assert cwd == cur_dir
            assert pathlib.Path.cwd() == tmp_path

        assert pathlib.Path.cwd() == cur_dir

    def test_average_timer(self) -> None:
        timer = core.AverageTimer()

        timer.start()
        for _ in range(10):
            time.sleep(0.1)
            timer.record()

        avg_time = timer.get_average_time()
        assert math.isclose(avg_time, 0.1, abs_tol=1e-3)

    def test_register(self) -> None:
        test_registry = core.Registry("test")

        test_registry.register(sample_fun)

        assert str(test_registry) == "test"
        assert len(typing.cast(typing.Sized, test_registry.keys())) == 1
        assert "sample_fun" in test_registry
        assert test_registry.get("sample_fun") == sample_fun


def sample_fun() -> int:
    return 1


class TestCUDA:
    def test_functions(self) -> None:
        if torch.cuda.is_available():
            cuda.requires_cuda_support()
        else:
            with pytest.raises(RuntimeError):
                cuda.requires_cuda_support()
