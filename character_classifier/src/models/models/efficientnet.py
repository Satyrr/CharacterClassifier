import timm
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, model_arch: str, pretrained: bool, hparams: dict):
        super().__init__()

        self.model = timm.create_model(model_arch, pretrained=pretrained, in_chans=1)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 36)

    def forward(self, x):
        return self.model(x.float())
