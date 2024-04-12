import torch

from helios import losses


@losses.LOSS_REGISTRY.register
class SampleLoss(losses.WeightedLoss):
    def __init__(self):
        super().__init__(loss_weight=0.5)

    def _eval(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestLosses:
    def test_registry(self, check_registry) -> None:
        check_registry(losses.LOSS_REGISTRY, ["SampleLoss"])

    def test_create(self, check_create_function) -> None:
        check_create_function(losses.LOSS_REGISTRY, losses.create_loss)

    def test_weighted_loss(self) -> None:
        loss = losses.create_loss("SampleLoss")
        assert isinstance(loss, SampleLoss)

        x = torch.tensor(10)
        y = loss(x)
        assert y == (x * 0.5)
