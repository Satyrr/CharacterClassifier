from typing import Dict, Any

import timm
import torch
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, model_arch: str, pretrained: bool, hparams: Dict[str, Any]):
        super().__init__()

        self.model = timm.create_model(model_arch, pretrained=pretrained, in_chans=1)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 36)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x.float())
