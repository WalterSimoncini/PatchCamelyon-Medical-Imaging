import torch.nn as nn

from .base_factory import ModelFactory
from torchvision.models import resnet18, ResNet18_Weights


class Resnet18Factory(ModelFactory):
    def input_size(self) -> int:
        return 224

    def base_model(self) -> nn.Module:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=512, out_features=2)

        return model
