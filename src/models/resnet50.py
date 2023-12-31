import torch.nn as nn

from typing import Tuple
from .base_factory import ModelFactory
from torchvision.models import resnet50, ResNet50_Weights
 

class Resnet50Factory(ModelFactory):
    def input_size(self) -> int:
        return 224

    def base_model(self) -> nn.Module:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=2048, out_features=2)

        return model

    def to_feature_extractor(self, model: nn.Module) -> Tuple[nn.Module, int]:
        model.fc = nn.Identity()

        return model, 2048
