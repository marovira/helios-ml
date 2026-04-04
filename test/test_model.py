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
        model.amp_context = hlm.AMPContext(
            scaler=unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        )
        ctx = model.autocast()
        assert isinstance(ctx, torch.amp.autocast)

    def test_create_scaler_noop_on_cpu(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        model.create_scaler()
        assert model.amp_context is None

    def test_create_scaler_noop_when_device_unset(self) -> None:
        model = MockModel()
        model.create_scaler()
        assert model.amp_context is None

    def test_create_scaler_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = MockModel()
        model.device = torch.device("cuda:0")
        model.create_scaler()
        assert model.amp_context is not None
        assert model.amp_context.dtype == torch.float16

    def test_create_scaler_cuda_custom_dtype(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = MockModel()
        model.device = torch.device("cuda:0")
        model.create_scaler(dtype=torch.bfloat16)
        assert model.amp_context is not None
        assert model.amp_context.dtype == torch.bfloat16


class TestClipGradients:
    def test_clip_gradients_without_amp(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        params = list(torch.nn.Linear(2, 2).parameters())
        optimizer = unittest.mock.MagicMock(spec=torch.optim.Optimizer)

        with unittest.mock.patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            model.clip_gradients(params, optimizer, max_norm=1.0)

        mock_clip.assert_called_once_with(params, 1.0)
        optimizer.assert_not_called()

    def test_clip_gradients_with_amp(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        mock_scaler = unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        model.amp_context = hlm.AMPContext(scaler=mock_scaler)

        params = list(torch.nn.Linear(2, 2).parameters())
        optimizer = unittest.mock.MagicMock(spec=torch.optim.Optimizer)

        call_order: list[str] = []
        mock_scaler.unscale_.side_effect = lambda _: call_order.append("unscale")

        with unittest.mock.patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            mock_clip.side_effect = lambda _p, _n, **_kw: call_order.append("clip")
            model.clip_gradients(params, optimizer, max_norm=1.0)

        mock_scaler.unscale_.assert_called_once_with(optimizer)
        mock_clip.assert_called_once_with(params, 1.0)
        assert call_order == ["unscale", "clip"]

    def test_clip_gradients_forwards_kwargs(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        params = list(torch.nn.Linear(2, 2).parameters())
        optimizer = unittest.mock.MagicMock(spec=torch.optim.Optimizer)

        with unittest.mock.patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            model.clip_gradients(params, optimizer, max_norm=1.0, norm_type=1.0)

        mock_clip.assert_called_once_with(params, 1.0, norm_type=1.0)


class TestStateDicts:
    def test_state_dict_empty_without_scaler(self) -> None:
        model = MockModel()
        assert model.state_dict() == {}

    def test_state_dict_includes_scaler(self) -> None:
        model = MockModel()
        mock_scaler = unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        mock_scaler.state_dict.return_value = {"scale": 65536.0}
        model.amp_context = hlm.AMPContext(scaler=mock_scaler)

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
        model.amp_context = hlm.AMPContext(
            scaler=unittest.mock.MagicMock(spec=torch.amp.GradScaler)
        )
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
        model.amp_context = hlm.AMPContext(scaler=mock_scaler)

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
        model.amp_context = hlm.AMPContext(scaler=mock_scaler)
        model.load_state_dict({"_helios_amp_scaler": {"scale": 65536.0}, "x": 1})
        assert "_helios_amp_scaler" not in received
        assert received == {"x": 1}

    def test_amp_context_dict_round_trip(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = MockModel()
        model.device = torch.device("cuda:0")
        model.create_scaler()
        assert model.amp_context is not None

        sd = model.state_dict()
        assert "_helios_amp_scaler" in sd

        model2 = MockModel()
        model2.device = torch.device("cuda:0")
        model2.create_scaler()
        model2.load_state_dict(sd)
        assert model2.amp_context is not None


class TestBatchToDevice:
    def test_tensor(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        t = torch.tensor([1.0, 2.0])
        result = model.batch_to_device(t, hlm.BatchPhase.TRAIN)
        assert isinstance(result, torch.Tensor)
        assert result.device == model.device

    def test_list_of_tensors(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        batch = [torch.tensor(1.0), torch.tensor(2.0)]
        result = model.batch_to_device(batch, hlm.BatchPhase.TRAIN)
        assert isinstance(result, list)
        assert all(t.device == model.device for t in result)

    def test_tuple_of_tensors(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        batch = (torch.tensor(1.0), torch.tensor(2.0))
        result = model.batch_to_device(batch, hlm.BatchPhase.TRAIN)
        assert isinstance(result, tuple)
        assert all(t.device == model.device for t in result)

    def test_dict_of_tensors(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        batch = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        result = model.batch_to_device(batch, hlm.BatchPhase.TRAIN)
        assert isinstance(result, dict)
        assert all(v.device == model.device for v in result.values())

    def test_non_tensor_leaf_passthrough(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        batch = [torch.tensor(1.0), "label", 42]
        result = model.batch_to_device(batch, hlm.BatchPhase.TRAIN)
        assert result[0].device == model.device
        assert result[1] == "label"
        assert result[2] == 42

    def test_nested_structure(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        batch = {"inputs": [torch.tensor(1.0), torch.tensor(2.0)], "label": "cat"}
        result = model.batch_to_device(batch, hlm.BatchPhase.VALID)
        assert isinstance(result["inputs"], list)
        assert all(t.device == model.device for t in result["inputs"])
        assert result["label"] == "cat"

    def test_all_phases(self) -> None:
        model = MockModel()
        model.device = torch.device("cpu")
        t = torch.tensor(1.0)
        for phase in hlm.BatchPhase:
            result = model.batch_to_device(t, phase)
            assert result.device == model.device

    def test_phase_override(self) -> None:
        seen_phases: list[hlm.BatchPhase] = []

        @hlm.MODEL_REGISTRY.register
        class PhaseCaptureModel(hlm.Model):
            def __init__(self) -> None:
                super().__init__("phase-capture")

            def setup(self, fast_init: bool = False) -> None:
                pass

            def batch_to_device(
                self, batch: typing.Any, phase: hlm.BatchPhase
            ) -> typing.Any:
                seen_phases.append(phase)
                return batch

        pm = PhaseCaptureModel()
        pm.device = torch.device("cpu")
        pm.batch_to_device("x", hlm.BatchPhase.TRAIN)
        pm.batch_to_device("x", hlm.BatchPhase.TEST)
        assert seen_phases == [hlm.BatchPhase.TRAIN, hlm.BatchPhase.TEST]


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
