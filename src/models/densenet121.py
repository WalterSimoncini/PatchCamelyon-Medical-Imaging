import torch.nn as nn

from typing import Tuple
from .base_factory import ModelFactory
from torchvision.models import densenet121, DenseNet121_Weights
 

class DenseNet121Factory(ModelFactory):
    def input_size(self) -> int:
        return 224

    def base_model(self) -> nn.Module:
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(in_features=1024, out_features=2)

        return model

    def to_feature_extractor(self, model: nn.Module) -> Tuple[nn.Module, int]:
        model.classifier = nn.Identity()

        return model, 1024
