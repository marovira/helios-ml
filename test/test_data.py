import typing

import torch
from torch.utils import data as tud

from helios import core, data
from helios.core import rng
from helios.data import samplers as hlds
from helios.data import transforms as hldt

DATASET_SIZE: int = 16


class RandomDataset(tud.Dataset):
    def __getitem__(self, index):
        gen = rng.get_default_numpy_rng().generator
        return gen.integers(0, 1000, 3)

    def __len__(self):
        return DATASET_SIZE


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
    def test_to_tensor(self) -> None:
        assert "ToTensor" in hldt.TRANSFORM_REGISTRY

        rng.seed_rngs()
        gen = rng.get_default_numpy_rng().generator
        img = gen.random(size=(32, 32, 3))

        to_tensor = hldt.create_transform("ToTensor")
        as_tens = to_tensor(img)

        assert isinstance(as_tens, torch.Tensor)
        assert as_tens.shape == (3, 32, 32)


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

    def test_sampler_registry(self) -> None:
        registered_names = [
            "ResumableRandomSampler",
            "ResumableSequentialSampler",
            "ResumableDistributedSampler",
        ]
        assert len(typing.cast(typing.Sized, hlds.SAMPLER_REGISTRY.keys())) == len(
            registered_names
        )
        for name in registered_names:
            assert name in hlds.SAMPLER_REGISTRY


if __name__ == "__main__":
    t = TestDataModule()
    t.test_resume_sequential()
