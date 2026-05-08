import pathlib

import pytest
import torch
from torch import nn

from helios import onnx


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(8, 8, bias=True)
        self.fc1 = nn.Linear(8, 4, bias=True)
        self.fc2 = nn.Linear(4, 2, bias=True)
        self.fc3 = nn.Linear(2, 2, bias=True)

    def forward(self, tensor_x: torch.Tensor):
        tensor_x = self.fc0(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc1(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc2(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        output = self.fc3(tensor_x)
        return output


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([x, y], dim=1))


class MultiOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(8, 4, bias=True)
        self.fc_b = nn.Linear(8, 2, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc_a(x), self.fc_b(x)


class DictOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(8, 4, bias=True)
        self.fc_b = nn.Linear(8, 2, bias=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"a": self.fc_a(x), "b": self.fc_b(x)}


class TestONNX:
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_jit(self, tmp_path: pathlib.Path) -> None:
        model = MLPModel()
        tensor_x = torch.rand((97, 8), dtype=torch.float32)
        out_path = tmp_path / "mlp.onnx"
        onnx.export_to_onnx(model, tensor_x, out_path, validate_output=True)
        assert out_path.exists()

        files = [file for file in tmp_path.iterdir() if file.is_file()]
        assert len(files) == 1

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_multi_input(self, tmp_path: pathlib.Path) -> None:
        model = MultiInputModel()
        x = torch.rand((4, 8), dtype=torch.float32)
        y = torch.rand((4, 8), dtype=torch.float32)
        out_path = tmp_path / "multi_input.onnx"
        onnx.export_to_onnx(model, (x, y), out_path, validate_output=True)
        assert out_path.exists()

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_multi_output(self, tmp_path: pathlib.Path) -> None:
        model = MultiOutputModel()
        x = torch.rand((4, 8), dtype=torch.float32)
        out_path = tmp_path / "multi_output.onnx"
        onnx.export_to_onnx(model, x, out_path, validate_output=True)
        assert out_path.exists()

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_dict_output(self, tmp_path: pathlib.Path) -> None:
        model = DictOutputModel()
        x = torch.rand((4, 8), dtype=torch.float32)
        out_path = tmp_path / "dict_output.onnx"
        onnx.export_to_onnx(model, x, out_path, validate_output=True)
        assert out_path.exists()
