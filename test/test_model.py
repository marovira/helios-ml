import contextlib
import pathlib
import typing
import unittest.mock

import pytest
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
    def running_loss(self) -> dict[str, float]:
        return self._running_losses

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
        assert model.user_state_dict() == {}
        assert model.trained_state_dict() == {}

        chkpt_name = "chkpt"
        assert model.append_metadata_to_chkpt_name(chkpt_name) == chkpt_name

        model.on_training_batch_start(state)
        assert len(model.loss_items) == 0
        model.populate_loss()
        assert len(model.loss_items) != 0
        model.on_training_batch_end(state)
        assert len(model.running_loss) != 0
        model.on_training_batch_start(state)
        assert len(model.loss_items) == 0

        model.populate_val_scores()
        assert len(model.val_scores) != 0
        model.on_validation_start(0)
        assert len(model.loss_items) == 0
        model.on_validation_end(0)
        assert len(model.running_loss) == 0

        assert model.have_metrics_improved()

        model.populate_test_scores()
        assert len(model.test_scores) != 0
        model.on_testing_start()
        assert len(model.test_scores) == 0

    def test_create(self, check_registry) -> None:
        check_registry(hlm.MODEL_REGISTRY, ["MockModel"])
        hlm.create_model("MockModel")


class TestAMPHelpers:
    def test_autocast_disabled(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        ctx = model.autocast()
        assert isinstance(ctx, contextlib.nullcontext)

    def test_autocast_enabled(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        model.amp_state.enabled = True
        ctx = model.autocast()
        assert isinstance(ctx, torch.amp.autocast)

    def test_create_scaler_noop_on_cpu(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        model.create_scaler()
        assert model.amp_state.scaler is None
        assert not model.amp_state.enabled

    def test_create_scaler_noop_when_device_unset(self) -> None:
        model = MockModel()
        model.create_scaler()
        assert model.amp_state.scaler is None
        assert not model.amp_state.enabled

    def test_create_scaler_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = MockModel()
        model.device = torch.device("cuda:0")
        model.create_scaler()
        assert model.amp_state.scaler is not None
        assert model.amp_state.enabled
        assert model.amp_state.dtype == torch.float16

    def test_create_scaler_cuda_custom_dtype(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = MockModel()
        model.device = torch.device("cuda:0")
        model.create_scaler(dtype=torch.bfloat16)
        assert model.amp_state.dtype == torch.bfloat16


class TestStateDicts:
    def test_state_dict_empty_without_scaler(self) -> None:
        model = MockModel()
        assert model.state_dict() == {}

    def test_state_dict_includes_scaler(self) -> None:
        model = MockModel()
        mock_scaler = unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        mock_scaler.state_dict.return_value = {"scale": 65536.0}
        model.amp_state.scaler = mock_scaler

        sd = model.state_dict()
        assert "_helios_amp_scaler" in sd
        assert sd["_helios_amp_scaler"] == {"scale": 65536.0}

    def test_state_dict_raises_on_reserved_key(self) -> None:
        @hlm.MODEL_REGISTRY.register
        class ConflictingModel(hlm.Model):
            def __init__(self) -> None:
                super().__init__("conflicting")

            def setup(self, fast_init: bool = False) -> None:
                pass

            def user_state_dict(self) -> dict[str, typing.Any]:
                return {"_helios_amp_scaler": "oops"}

        model = ConflictingModel()
        model.amp_state.scaler = unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        with pytest.raises(KeyError, match="_helios_amp_scaler"):
            model.state_dict()

    def test_load_state_dict_forwards_user_keys(self) -> None:
        received: dict[str, typing.Any] = {}

        @hlm.MODEL_REGISTRY.register
        class RecordingModel(hlm.Model):
            def __init__(self) -> None:
                super().__init__("recording")

            def setup(self, fast_init: bool = False) -> None:
                pass

            def load_user_state_dict(
                self, state_dict: dict[str, typing.Any], fast_init: bool = False
            ) -> None:
                received.update(state_dict)

        model = RecordingModel()
        model.load_state_dict({"foo": 1, "bar": 2})
        assert received == {"foo": 1, "bar": 2}

    def test_load_state_dict_restores_scaler(self) -> None:
        model = MockModel()
        mock_scaler = unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        model.amp_state.scaler = mock_scaler

        model.load_state_dict({"_helios_amp_scaler": {"scale": 65536.0}, "x": 1})

        mock_scaler.load_state_dict.assert_called_once_with({"scale": 65536.0})

    def test_load_state_dict_strips_helios_key_before_forwarding(self) -> None:
        received: dict[str, typing.Any] = {}

        @hlm.MODEL_REGISTRY.register
        class StrippingModel(hlm.Model):
            def __init__(self) -> None:
                super().__init__("stripping")

            def setup(self, fast_init: bool = False) -> None:
                pass

            def load_user_state_dict(
                self, state_dict: dict[str, typing.Any], fast_init: bool = False
            ) -> None:
                received.update(state_dict)

        model = StrippingModel()
        mock_scaler = unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        model.amp_state.scaler = mock_scaler
        model.load_state_dict({"_helios_amp_scaler": {"scale": 65536.0}, "x": 1})
        assert "_helios_amp_scaler" not in received
        assert received == {"x": 1}

    def test_amp_state_dict_round_trip(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = MockModel()
        model.device = torch.device("cuda:0")
        model.create_scaler()
        assert model.amp_state.scaler is not None

        sd = model.state_dict()
        assert "_helios_amp_scaler" in sd

        model2 = MockModel()
        model2.device = torch.device("cuda:0")
        model2.create_scaler()
        model2.load_state_dict(sd)
        assert model2.amp_state.scaler is not None


class TestModelUtils:
    def test_find_pretrained_file_found(self, tmp_path: pathlib.Path) -> None:
        models_dir = tmp_path / "mymodel"
        models_dir.mkdir()
        expected = models_dir / "mymodel_resnet50_epoch10.pth"
        expected.touch()
        result = hlm.find_pretrained_file(models_dir, "resnet50")
        assert result == expected

    def test_find_pretrained_file_not_found(self, tmp_path: pathlib.Path) -> None:
        models_dir = tmp_path / "mymodel"
        models_dir.mkdir()
        with pytest.raises(RuntimeError):
            hlm.find_pretrained_file(models_dir, "resnet50")

    def test_find_pretrained_file_multiple_matches(self, tmp_path: pathlib.Path) -> None:
        models_dir = tmp_path / "mymodel"
        models_dir.mkdir()
        (models_dir / "mymodel_resnet50_epoch1.pth").touch()
        (models_dir / "mymodel_resnet50_epoch10.pth").touch()
        result = hlm.find_pretrained_file(models_dir, "resnet50")
        assert result.suffix == ".pth"
        assert "resnet50" in result.stem

    def test_find_pretrained_file_no_match_in_dir(self, tmp_path: pathlib.Path) -> None:
        models_dir = tmp_path / "mymodel"
        models_dir.mkdir()
        (models_dir / "mymodel_vgg_epoch5.pth").touch()
        with pytest.raises(RuntimeError):
            hlm.find_pretrained_file(models_dir, "resnet50")
