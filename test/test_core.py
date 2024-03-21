import math
import pathlib
import time
import typing

import pytest

from pyro import core


class TestUtils:
    def test_get_from_optional(self) -> None:
        opt: typing.Optional[int] = None

        with pytest.raises(RuntimeError):
            core.get_from_optional(opt, raise_on_empty=True)

        opt = 1
        assert core.get_from_optional(opt) == 1

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
