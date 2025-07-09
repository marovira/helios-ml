import pathlib
import typing

import cv2
import numpy as np
import numpy.typing as npt
import PIL
import torch
from torch.utils import data as tud

from helios import core, data
from helios.core import rng
from helios.data import functional as hldf
from helios.data import samplers as hlds
from helios.data import transforms as hldt

DATASET_SIZE: int = 16


@data.DATASET_REGISTRY.register
class RandomDataset(tud.Dataset):
    def __getitem__(self, index):
        gen = rng.get_default_numpy_rng().generator
        return gen.integers(0, 1000, 3)

    def __len__(self):
        return DATASET_SIZE


@data.DATASET_REGISTRY.register
class SequentialDataset(tud.Dataset):
    def __init__(self):
        super().__init__()

        self._samples = [
            torch.tensor([359, 639, 636]),
            torch.tensor([963, 738, 996]),
            torch.tensor([766, 97, 292]),
            torch.tensor([500, 324, 687]),
            torch.tensor([15, 841, 180]),
            torch.tensor([553, 266, 928]),
            torch.tensor([269, 806, 472]),
            torch.tensor([694, 632, 678]),
            torch.tensor([92, 228, 424]),
            torch.tensor([421, 568, 705]),
            torch.tensor([686, 769, 903]),
            torch.tensor([440, 610, 126]),
            torch.tensor([809, 833, 629]),
            torch.tensor([340, 810, 512]),
            torch.tensor([23, 322, 577]),
            torch.tensor([757, 651, 429]),
        ]

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return DATASET_SIZE


class SampleDataModule(data.DataModule):
    def setup(self) -> None:
        params = data.DataLoaderParams(
            batch_size=2, num_workers=2, random_seed=rng.get_default_seed()
        )

        params.shuffle = True
        self._train_dataset = self._create_dataset(RandomDataset(), params)

        params.num_workers = 2
        self._valid_dataset = self._create_dataset(SequentialDataset(), params)

        params.shuffle = False
        self._test_dataset = self._create_dataset(SequentialDataset(), params)


class TestTransforms:
    def check_type(self, x, exp_type: type) -> None:
        assert isinstance(x, exp_type)

        if isinstance(exp_type, list | tuple):
            for elem in x:  # type: ignore[attr-defined]
                assert isinstance(elem, torch.Tensor)

    def check_to_tensor(self, img, exp_type: type, exp_shape: tuple[int, ...]) -> None:
        to_tensor = hldt.create_transform("ToImageTensor")
        as_tens: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] = to_tensor(
            img
        )

        self.check_type(as_tens, exp_type)

        if isinstance(as_tens, list | tuple):
            for t in as_tens:
                assert t.shape == exp_shape
        else:
            assert as_tens.shape == exp_shape

    def generate_sample(
        self,
        size: tuple[int, ...],
        num_elems: int,
        as_tuple: bool = False,
        as_pil: bool = False,
    ):
        gen = rng.get_default_numpy_rng().generator

        samples: list[npt.NDArray | PIL.Image.Image] = []
        for _ in range(num_elems):
            base = np.uint8(gen.random(size=size)) * 255
            if as_pil:
                samples.append(PIL.Image.fromarray(base))
            else:
                samples.append(base)  # type: ignore[arg-type]

        if as_tuple:
            return tuple(samples)
        return samples

    def test_to_tensor(self) -> None:
        assert "ToImageTensor" in hldt.TRANSFORM_REGISTRY

        rng.seed_rngs()
        size = (32, 32, 3)
        exp_size = (3, 32, 32)

        # Single image (numpy)
        self.check_to_tensor(self.generate_sample(size, 1)[0], torch.Tensor, exp_size)

        # Multi-image list (numpy)
        self.check_to_tensor(self.generate_sample(size, 2), list, exp_size)

        # Multi-image tuple (numpy)
        self.check_to_tensor(
            self.generate_sample(size, 2, as_tuple=True),
            tuple,
            exp_size,
        )

        # Single image (PIL)
        self.check_to_tensor(
            self.generate_sample(size, 1, as_pil=True)[0], torch.Tensor, exp_size
        )

        # Multi-image list (PIL)
        self.check_to_tensor(self.generate_sample(size, 2, as_pil=True), list, exp_size)

        # Multi-image tuple (PIL)
        self.check_to_tensor(
            self.generate_sample(size, 2, as_tuple=True, as_pil=True),
            tuple,
            exp_size,
        )


class TestFunctional:
    def check_image(self, gt: npt.NDArray, out_path: pathlib.Path, **kwargs) -> None:
        # Ensure that we convert to "BGR" so the output comes out the way we expect it to
        # come out.
        if kwargs.get("as_rgb", True):
            if len(gt.shape) == 2:
                tmp = gt
            elif len(gt.shape) == 3 and gt.shape[2] == 3:
                tmp = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            elif len(gt.shape) == 3 and gt.shape[2] == 4:
                tmp = cv2.cvtColor(gt, cv2.COLOR_RGBA2BGRA)
        else:
            tmp = gt

        cv2.imwrite(str(out_path), tmp)

        ret = hldf.load_image(out_path, **kwargs)
        assert isinstance(ret, np.ndarray)
        assert np.array_equal(gt, ret)

    def test_load_image(self, tmp_path: pathlib.Path):
        rng.seed_rngs()
        gen = rng.get_default_numpy_rng().generator
        out_path = tmp_path / "tmp.png"
        max_8 = 255
        max_16 = 65535

        # 8-bit grayscale.
        self.check_image(
            (gen.random(size=(32, 32)) * max_8).astype(np.uint8),
            out_path,
            flags=cv2.IMREAD_GRAYSCALE,
        )

        # 16-bit grayscale
        self.check_image(
            (gen.random(size=(32, 32)) * max_16).astype(np.uint16),
            out_path,
            flags=cv2.IMREAD_UNCHANGED,
        )

        # 8-bit RGB
        self.check_image(
            (gen.random(size=(32, 32, 3)) * max_8).astype(np.uint8),
            out_path,
        )

        # 16-bit RGB
        self.check_image(
            (gen.random(size=(32, 32, 3)) * max_16).astype(np.uint16),
            out_path,
            flags=cv2.IMREAD_UNCHANGED,
        )

        # 8-bit RGBA
        self.check_image(
            (gen.random(size=(32, 32, 4)) * max_8).astype(np.uint8),
            out_path,
            flags=cv2.IMREAD_UNCHANGED,
        )

        # 16-bit RGB
        self.check_image(
            (gen.random(size=(32, 32, 4)) * max_16).astype(np.uint16),
            out_path,
            flags=cv2.IMREAD_UNCHANGED,
        )

        # Load image as is.
        self.check_image(
            (gen.random(size=(32, 32, 4)) * max_8).astype(np.uint8),
            out_path,
            flags=cv2.IMREAD_UNCHANGED,
            as_rgb=False,
        )

    def test_load_image_pil(self, tmp_path: pathlib.Path):
        rng.seed_rngs()
        gen = rng.get_default_numpy_rng().generator
        img = (gen.random(size=(32, 32, 3)) * 255).astype(np.uint8)
        as_pil = PIL.Image.fromarray(img)

        out_path = tmp_path / "tmp.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        ret = hldf.load_image_pil(out_path)
        assert isinstance(ret, np.ndarray)
        assert np.array_equal(img, ret)

        ret = hldf.load_image_pil(out_path, as_numpy=False)
        assert isinstance(ret, PIL.Image.Image)
        diff = PIL.ImageChops.difference(as_pil, ret)  # type: ignore[attr-defined]
        assert diff.getbbox() is None


class TestTensorToNumpy:
    def get_default_tensor_args(self) -> dict[str, typing.Any]:
        return {
            "squeeze": False,
            "clamp": None,
            "transpose": False,
            "dtype": None,
            "as_uint8": False,
        }

    def test_clamp(self) -> None:
        # Base case: no clamping.
        args = self.get_default_tensor_args()
        x = torch.FloatTensor([-1.0, 1.0])
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert np.max(ret) == 1.0
        assert np.min(ret) == -1.0

        # Clamp to 0.5
        args["clamp"] = (0, 0.5)
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert np.max(ret) == 0.5
        assert np.min(ret) == 0.0

    def test_squeeze(self) -> None:
        # Base case: no squeezing
        args = self.get_default_tensor_args()
        x = torch.randn((1, 32, 1, 32))
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.shape == (1, 32, 1, 32)

        # Squeeze
        args["squeeze"] = True
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.shape == (32, 32)

    def test_transpose(self) -> None:
        # Base case: no transposition
        args = self.get_default_tensor_args()
        x = torch.randn((1, 3, 32, 32))
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.shape == (1, 3, 32, 32)

        # Transpose (3 dimensions)
        args["transpose"] = True
        x = torch.randn((3, 32, 32))
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.shape == (32, 32, 3)

        # Transpose (4 dimensions)
        x = torch.randn((1, 3, 32, 32))
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.shape == (1, 32, 32, 3)

    def test_dtype(self) -> None:
        # Base case: no change.
        args = self.get_default_tensor_args()
        x = torch.randint(0, 10, (32, 32), dtype=torch.int32)
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.int32

        # Convert
        args["dtype"] = torch.int32
        x = torch.randn((32, 32), dtype=torch.float)
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.int32

    def test_as_uint8(self) -> None:
        # Base case: no change.
        args = self.get_default_tensor_args()
        x = torch.FloatTensor((0.0, 1.0))
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.float32
        assert np.max(ret) == 1.0
        assert np.min(ret) == 0.0

        # Change to uint8
        args["as_uint8"] = True
        ret = hldf.tensor_to_numpy(x, **args)
        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.uint8
        assert np.max(ret) == 255
        assert np.min(ret) == 0

    def test_defaults(self) -> None:
        x = torch.rand(1, 3, 32, 32)
        ret = hldf.tensor_to_numpy(x)
        assert isinstance(ret, np.ndarray)
        assert ret.dtype == np.uint8
        assert ret.shape == (32, 32, 3)
        assert np.min(ret) == 0
        assert np.max(ret) <= 255


class TestDataModule:
    def get_expected_train_batches(self) -> list[torch.Tensor]:
        return [
            torch.tensor([[359, 639, 636], [963, 738, 996]]),
            torch.tensor([[766, 97, 292], [500, 324, 687]]),
            torch.tensor([[15, 841, 180], [553, 266, 928]]),
            torch.tensor([[269, 806, 472], [694, 632, 678]]),
            torch.tensor([[92, 228, 424], [421, 568, 705]]),
            torch.tensor([[686, 769, 903], [440, 610, 126]]),
            torch.tensor([[809, 833, 629], [340, 810, 512]]),
            torch.tensor([[23, 322, 577], [757, 651, 429]]),
        ]

    def get_expected_valid_batches(self) -> list[torch.Tensor]:
        return [
            torch.tensor([[809, 833, 629], [694, 632, 678]]),
            torch.tensor([[359, 639, 636], [553, 266, 928]]),
            torch.tensor([[500, 324, 687], [963, 738, 996]]),
            torch.tensor([[421, 568, 705], [15, 841, 180]]),
            torch.tensor([[766, 97, 292], [23, 322, 577]]),
            torch.tensor([[269, 806, 472], [757, 651, 429]]),
            torch.tensor([[340, 810, 512], [440, 610, 126]]),
            torch.tensor([[92, 228, 424], [686, 769, 903]]),
        ]

    def get_expected_test_batches(self) -> list[torch.Tensor]:
        return [
            torch.tensor([[359, 639, 636], [963, 738, 996]]),
            torch.tensor([[766, 97, 292], [500, 324, 687]]),
            torch.tensor([[15, 841, 180], [553, 266, 928]]),
            torch.tensor([[269, 806, 472], [694, 632, 678]]),
            torch.tensor([[92, 228, 424], [421, 568, 705]]),
            torch.tensor([[686, 769, 903], [440, 610, 126]]),
            torch.tensor([[809, 833, 629], [340, 810, 512]]),
            torch.tensor([[23, 322, 577], [757, 651, 429]]),
        ]

    def check_batches(self, exp: list[torch.Tensor], val: list[torch.Tensor]) -> None:
        for b, exp_b in zip(exp, val, strict=True):
            assert torch.all(b == exp_b)

    def prepare(
        self, split: data.DatasetSplit, skip_seed: bool = False
    ) -> tuple[tud.DataLoader, tud.Sampler]:
        if not skip_seed:
            rng.seed_rngs()

        datamodule = SampleDataModule()
        datamodule.setup()

        if split == data.DatasetSplit.TRAIN:
            return core.get_from_optional(datamodule.train_dataloader())
        if split == data.DatasetSplit.VALID:
            return core.get_from_optional(datamodule.valid_dataloader())
        return core.get_from_optional(datamodule.test_dataloader())

    def test_worker_seeding(self) -> None:
        dataloader, _ = self.prepare(data.DatasetSplit.TRAIN)

        batches = []
        for batch in dataloader:
            batches.append(batch)

        exp_batches = self.get_expected_train_batches()
        self.check_batches(exp_batches, batches)

    def test_resume_random(self) -> None:
        dataloader, sampler = self.prepare(data.DatasetSplit.VALID)
        assert isinstance(sampler, hlds.ResumableRandomSampler)

        half_step = len(dataloader) // 2
        exp_batches = self.get_expected_valid_batches()

        batches = []
        sampler.set_epoch(0)
        for step, batch in enumerate(dataloader):
            if step >= half_step:
                break
            batches.append(batch)

        self.check_batches(exp_batches[:half_step], batches)

        # Grab the state and clear everything.
        rng_state = rng.get_rng_state_dict()

        del dataloader, sampler
        batches = []
        rng.seed_rngs(0)

        # Restore everything.
        rng.load_rng_state_dict(rng_state)
        dataloader, sampler = self.prepare(data.DatasetSplit.VALID, skip_seed=True)
        assert isinstance(sampler, hlds.ResumableRandomSampler)
        sampler.set_epoch(0)
        sampler.start_iter = half_step

        # Continue.
        for batch in dataloader:
            batches.append(batch)

        self.check_batches(exp_batches[half_step:], batches)

    def test_resume_sequential(self) -> None:
        dataloader, sampler = self.prepare(data.DatasetSplit.TEST)
        assert isinstance(sampler, hlds.ResumableSequentialSampler)

        half_step = len(dataloader) // 2
        exp_batches = self.get_expected_test_batches()

        batches = []
        sampler.set_epoch(0)
        for step, batch in enumerate(dataloader):
            if step >= half_step:
                break
            batches.append(batch)

        self.check_batches(exp_batches[:half_step], batches)

        # Grab the state and clear everything.
        rng_state = rng.get_rng_state_dict()

        del dataloader, sampler
        batches = []
        rng.seed_rngs(0)

        # Restore everything.
        rng.load_rng_state_dict(rng_state)
        dataloader, sampler = self.prepare(data.DatasetSplit.TEST, skip_seed=True)
        assert isinstance(sampler, hlds.ResumableSequentialSampler)
        sampler.set_epoch(0)
        sampler.start_iter = half_step

        # Continue.
        for batch in dataloader:
            batches.append(batch)

        self.check_batches(exp_batches[half_step:], batches)

    def test_sampler_registry(self, check_registry) -> None:
        check_registry(
            hlds.SAMPLER_REGISTRY,
            [
                "ResumableRandomSampler",
                "ResumableSequentialSampler",
                "ResumableDistributedSampler",
            ],
        )

    def test_sampler_create(self, check_create_function) -> None:
        check_create_function(hlds.SAMPLER_REGISTRY, hlds.create_sampler)

    def test_dataset_registry(self, check_registry) -> None:
        check_registry(
            data.DATASET_REGISTRY,
            [
                "RandomDataset",
                "SequentialDataset",
            ],
        )

    def test_dataset_create(self, check_create_function) -> None:
        check_create_function(data.DATASET_REGISTRY, data.create_dataset)

    def test_collate_fn_registry(self, check_registry) -> None:
        check_registry(
            data.COLLATE_FN_REGISTRY,
            [
                "default_collate",
            ],
        )

    def test_collate_fn_create(self, check_create_function) -> None:
        check_create_function(data.COLLATE_FN_REGISTRY, data.create_collate_fn)

    def test_dataset_getters(self) -> None:
        datamodule = SampleDataModule()
        datamodule.setup()

        assert datamodule.train_dataset is not None
        assert datamodule.valid_dataset is not None
        assert datamodule.test_dataset is not None

        assert isinstance(datamodule.train_dataset, RandomDataset)
        assert isinstance(datamodule.valid_dataset, SequentialDataset)
        assert isinstance(datamodule.test_dataset, SequentialDataset)


if __name__ == "__main__":
    t = TestDataModule()
    t.test_resume_sequential()
