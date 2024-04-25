import dataclasses
import math
import pathlib
import random
import sys
import time
import typing

import numpy as np
import numpy.typing as npt
import pytest
import torch

from helios import core
from helios.core import cuda, rng


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

        with pytest.raises(KeyError):
            test_registry.get("foo")

    def test_update_registries(self) -> None:
        root = pathlib.Path(__file__)
        root = root.parent
        # HACK: Ensure that the current path gets added to sys path, otherwise imports
        # won't work.
        sys.path.append(str(root))
        core.update_all_registries(root / "registry_test", recurse=True)

        import registry_test as rt  # type: ignore[import-not-found]

        assert len(rt.FUNC_REGISTRY.keys()) == 2
        assert all(var in rt.FUNC_REGISTRY for var in ("foo", "bar"))


def sample_fun() -> int:
    return 1


class TestCUDA:
    def test_functions(self) -> None:
        if torch.cuda.is_available():
            cuda.requires_cuda_support()
        else:
            with pytest.raises(RuntimeError):
                cuda.requires_cuda_support()

    def _check_cudnn(self, val: bool) -> None:
        torch.backends.cudnn.benchmark = val
        assert torch.backends.cudnn.benchmark == val
        with cuda.DisableCuDNNBenchmarkContext():
            assert not torch.backends.cudnn.benchmark

        assert torch.backends.cudnn.benchmark == val

    def test_disable_cudnn_context(self) -> None:
        if torch.cuda.is_available():
            self._check_cudnn(True)
            self._check_cudnn(False)


@dataclasses.dataclass
class ExpectedRNG:
    torch_vals = torch.tensor([0, 1, 8, 6, 5, 7, 3, 9, 6, 9])
    rand_vals = [0, 9, 2, 3, 6, 6, 2, 6, 7, 4]
    np_vals = [3, 2, 3, 4, 4, 3, 1, 3, 3, 1]


class TestRNG:
    def check_torch(self, a: torch.Tensor, b: torch.Tensor) -> None:
        assert torch.all(a == b)

    def check_rand(self, a: list[int], b: list[int]) -> None:
        assert a == b

    def check_np(self, a: list[int], b: npt.NDArray) -> None:
        assert np.all(a == b)

    def test_seed_rngs(self) -> None:
        exp = ExpectedRNG()
        rng.seed_rngs()

        np_gen = rng.get_default_numpy_rng().generator

        self.check_torch(exp.torch_vals, torch.randint(10, [10]))
        self.check_rand(exp.rand_vals, [random.randint(0, 9) for _ in range(10)])
        self.check_np(exp.np_vals, np_gen.integers(0, 10, 10))

    def test_rng_restore(self) -> None:
        exp = ExpectedRNG()
        rng.seed_rngs()
        np_gen = rng.get_default_numpy_rng().generator

        # Check the first 5 entries.
        self.check_torch(exp.torch_vals[:5], torch.randint(10, [5]))
        self.check_rand(exp.rand_vals[:5], [random.randint(0, 9) for _ in range(5)])
        self.check_np(exp.np_vals[:5], np_gen.integers(0, 10, 5))

        # Grab the states and re-seed the generators.
        state = rng.get_rng_state_dict()
        rng.seed_rngs(0)
        np_gen = rng.get_default_numpy_rng().generator

        # Generate a few numbers to move the generators along.
        torch.randint(10, [10])
        [random.randint(0, 9) for _ in range(10)]
        np_gen.integers(0, 10, 10)

        # Now restore the RNG state and read the final 5 numbers.
        rng.load_rng_state_dict(state)
        np_gen = rng.get_default_numpy_rng().generator

        self.check_torch(exp.torch_vals[5:], torch.randint(10, [5]))
        self.check_rand(exp.rand_vals[5:], [random.randint(0, 9) for _ in range(5)])
        self.check_np(exp.np_vals[5:], np_gen.integers(0, 10, 5))
