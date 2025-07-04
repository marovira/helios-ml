import pathlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from helios import nn as hln
from helios import onnx
from helios.nn import layers as hlnl


@hln.NETWORK_REGISTRY.register
class ClassifierNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestNewtorks:
    def test_registry(self, check_registry) -> None:
        check_registry(hln.NETWORK_REGISTRY, ["ClassifierNet"])

    def test_create(self, check_create_function) -> None:
        check_create_function(hln.NETWORK_REGISTRY, hln.create_network)

    def test_ema(self) -> None:
        net = nn.Conv2d(1, 20, 5)
        ema = hln.swa_utils.EMA(net)
        ema.update(net)
        n = ema.module
        assert net != n


class TestAdaptiveAvgPool2D:
    def check_input_exception(self, *args) -> None:
        with pytest.raises(RuntimeError):
            hlnl.AdaptiveAvgPool2d(*args)

    def test_invalid_inputs(self) -> None:
        self.check_input_exception((1, 2, 3))
        self.check_input_exception((1,))
        self.check_input_exception(1.0)
        self.check_input_exception((1, 1.0))
        self.check_input_exception((1.0, 1))
        self.check_input_exception(None)

    def test_shape_tuple(self) -> None:
        m_t = nn.AdaptiveAvgPool2d((5, 7))
        m_h = hlnl.AdaptiveAvgPool2d((5, 7))
        x = torch.randn(1, 64, 8, 9)

        base = m_t(x)
        ret = m_h(x)
        assert base.shape == ret.shape

    def test_shape_int(self) -> None:
        m_t = nn.AdaptiveAvgPool2d(7)
        m_h = hlnl.AdaptiveAvgPool2d(7)
        x = torch.randn(1, 64, 10, 9)

        base = m_t(x)
        ret = m_h(x)
        assert base.shape == ret.shape

    def test_none_tuple_x_axis(self) -> None:
        m_t = nn.AdaptiveAvgPool2d((None, 7))
        m_h = hlnl.AdaptiveAvgPool2d((None, 7))
        x = torch.randn(1, 64, 10, 9)

        base = m_t(x)
        ret = m_h(x)
        assert base.shape == ret.shape

    def test_none_tuple_y_axis(self) -> None:
        m_t = nn.AdaptiveAvgPool2d((7, None))
        m_h = hlnl.AdaptiveAvgPool2d((7, None))
        x = torch.randn(1, 64, 10, 9)

        base = m_t(x)
        ret = m_h(x)
        assert base.shape == ret.shape

    def check_export(
        self, model: nn.Module, x: torch.Tensor, out_path: pathlib.Path
    ) -> None:
        onnx.export_to_onnx(model, x, out_path, validate_output=True)
        assert out_path.exists()

    @pytest.mark.filterwarnings("ignore:Converting")
    def test_export(self, tmp_path: pathlib.Path) -> None:
        self.check_export(
            hlnl.AdaptiveAvgPool2d((5, 7)),
            torch.randn(1, 64, 8, 9),
            tmp_path / "5_7.onnx",
        )
        self.check_export(
            hlnl.AdaptiveAvgPool2d(7),
            torch.randn(1, 64, 10, 9),
            tmp_path / "7_square.onnx",
        )
        self.check_export(
            hlnl.AdaptiveAvgPool2d((None, 7)),
            torch.randn(1, 64, 10, 9),
            tmp_path / "none_x.onnx",
        )

        self.check_export(
            hlnl.AdaptiveAvgPool2d((7, None)),
            torch.randn(1, 64, 10, 9),
            tmp_path / "none_y.onnx",
        )
