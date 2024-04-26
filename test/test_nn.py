import torch
import torch.nn.functional as F
from torch import nn

from helios import nn as hln


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
