import torch
from torch import nn
import torch.nn.functional as F
import math

class SimpleCNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=hparams['kernel_size_1'])
        self.conv2 = nn.Conv2d(10, 20, kernel_size=hparams['kernel_size_2'])
        self.conv2_drop = nn.Dropout2d()
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(2420, 50)
        self.fc2 = nn.Linear(50, 36)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.float()

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(batch_size, 2420)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x
