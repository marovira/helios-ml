import numpy as np
import torch

from pyro.data import transforms as pdt


class TestTransforms:
    def test_to_tensor(self) -> None:
        assert "ToTensor" in pdt.TRANSFORM_REGISTRY

        rng = np.random.default_rng(0)
        img = rng.random(size=(32, 32, 3))

        to_tensor = pdt.create_transform("ToTensor")
        as_tens = to_tensor(img)

        assert isinstance(as_tens, torch.Tensor)
        assert as_tens.shape == (3, 32, 32)
