from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()

        kernel_size_1 = hparams.get('kernel_size_1', 5)
        kernel_size_2 = hparams.get('kernel_size_2', 5)
        filters_num_1 = hparams.get('filters_num_1', 30)
        filters_num_2 = hparams.get('filters_num_2', 20)

        self.conv1 = nn.Conv2d(1, filters_num_1, kernel_size=kernel_size_1)
        self.conv2 = nn.Conv2d(filters_num_1, filters_num_2, kernel_size=kernel_size_2)
        self.conv2_drop = nn.Dropout2d()
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(121 * filters_num_2, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = x.size(0)
        x = x.float().view(batch_size, 1, 56, 56)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x
