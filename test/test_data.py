import torch
from torch.utils import data as tud

from pyro import core, data
from pyro.core import rng
from pyro.data import transforms as pdt


class RandomDataset(tud.Dataset):
    def __getitem__(self, index):
        gen = rng.get_default_numpy_rng().generator
        return gen.integers(0, 1000, 3)

    def __len__(self):
        return 16


class RandomDataModule(data.PyroDataModule):
    def setup(self) -> None:
        params = {
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 4,
            "random_seed": rng.get_default_seed(),
        }
        self._train_dataset = self._create_dataset(RandomDataset(), params)


class TestTransforms:
    def test_to_tensor(self) -> None:
        assert "ToTensor" in pdt.TRANSFORM_REGISTRY

        rng.seed_rngs()
        gen = rng.get_default_numpy_rng().generator
        img = gen.random(size=(32, 32, 3))

        to_tensor = pdt.create_transform("ToTensor")
        as_tens = to_tensor(img)

        assert isinstance(as_tens, torch.Tensor)
        assert as_tens.shape == (3, 32, 32)


class TestDataModule:
    def get_expected_train_batches(self) -> list[torch.Tensor]:
        return [
            torch.tensor([[359, 639, 636], [963, 738, 996]]),
            torch.tensor([[766, 97, 292], [500, 324, 687]]),
            torch.tensor([[624, 693, 364], [815, 519, 804]]),
            torch.tensor([[60, 251, 823], [327, 945, 138]]),
            torch.tensor([[15, 841, 180], [553, 266, 928]]),
            torch.tensor([[269, 806, 472], [694, 632, 678]]),
            torch.tensor([[459, 656, 19], [620, 700, 454]]),
            torch.tensor([[890, 85, 522], [739, 134, 839]]),
        ]

    def check_batches(self, exp: list[torch.Tensor], val: list[torch.Tensor]) -> None:
        for b, exp_b in zip(exp, val, strict=True):
            assert torch.all(b == exp_b)

    def prepare_dataloader(
        self, skip_seed: bool = False, get_valid: bool = False
    ) -> tud.DataLoader:
        if not skip_seed:
            rng.seed_rngs()

        datamodule = RandomDataModule()
        datamodule.setup()

        ret = (
            datamodule.train_dataloader()
            if not get_valid
            else datamodule.valid_dataloader()
        )

        dataloader, _ = core.get_from_optional(ret)
        assert dataloader is not None
        return dataloader

    def test_worker_seeding(self) -> None:
        dataloader = self.prepare_dataloader()
        batches = []
        for batch in dataloader:
            print(batch)
            batches.append(batch)

        exp_batches = self.get_expected_train_batches()
        self.check_batches(exp_batches, batches)
