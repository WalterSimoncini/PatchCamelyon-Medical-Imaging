import torch.nn as nn

from typing import Tuple
from .base_factory import ModelFactory
from torchvision.models import inception_v3,  Inception_V3_Weights


class InceptionV3Wrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training:
            return self.model(x).logits
        else:
            return self.model(x)

    def to_feature_extractor(self) -> None:
        self.model.fc = nn.Identity()

    def output_feature_size(self) -> int:
        return 2048


class InceptionV3Factory(ModelFactory):
    def input_size(self) -> int:
        return 299
    
    def base_model(self) -> nn.Module:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=2048, out_features=2)

        return InceptionV3Wrapper(model=model)

    def to_feature_extractor(self, model: InceptionV3Wrapper) -> Tuple[InceptionV3Wrapper, int]:
        model.to_feature_extractor()

        return model, model.output_feature_size()
