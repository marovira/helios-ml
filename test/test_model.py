import typing

import torch

import helios.model as hlm
import helios.trainer as hlt


@hlm.MODEL_REGISTRY.register
class MockModel(hlm.Model):
    def __init__(self) -> None:
        super().__init__("mock-model")

    @property
    def loss_items(self) -> dict[str, torch.Tensor]:
        return self._loss_items

    @property
    def val_scores(self) -> dict[str, float]:
        return self._val_scores

    @property
    def test_scores(self) -> dict[str, float]:
        return self._test_scores

    def setup(self, fast_init: bool = False) -> None:
        pass

    def train_step(self, batch: typing.Any, state) -> None:
        pass

    def populate_loss(self) -> None:
        self._loss_items["foo"] = torch.tensor(1)
        self._loss_items["bar"] = torch.tensor(2)

    def populate_val_scores(self) -> None:
        self._val_scores["foo"] = 1
        self._val_scores["bar"] = 2

    def populate_test_scores(self) -> None:
        self._test_scores["foo"] = 1
        self._test_scores["bar"] = 2


class TestModel:
    def test_defaults(self) -> None:
        model = MockModel()
        state = hlt.TrainingState()

        assert model.save_name == "mock-model"
        assert model.state_dict() == {}

        chkpt_name = "chkpt"
        assert model.append_metadata_to_chkpt_name(chkpt_name) == chkpt_name

        state_dict = {"a": 1, "b": 2}
        assert model.strip_training_data(state_dict) == state_dict

        model.populate_loss()
        assert len(model.loss_items) != 0
        model.on_training_batch_start(state)
        assert len(model.loss_items) == 0

        model.populate_val_scores()
        assert len(model.val_scores) != 0
        model.on_validation_start(0)
        assert len(model.loss_items) == 0

        assert model.have_metrics_improved()

        model.populate_test_scores()
        assert len(model.test_scores) != 0
        model.on_testing_start()
        assert len(model.test_scores) == 0

    def test_create(self, check_registry) -> None:
        check_registry(hlm.MODEL_REGISTRY, ["MockModel"])
        hlm.create_model("MockModel")
