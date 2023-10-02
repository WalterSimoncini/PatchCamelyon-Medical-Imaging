import torch.nn as nn

from .base_factory import ModelFactory
from torchvision.models import inception_v3,  Inception_V3_Weights


class InceptionV3Factory(ModelFactory):
    def input_size(self) -> int:
        return 299
    
    def base_model(self) -> nn.Module:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Linear(in_features=2048, out_features=2)

        return InceptionV3Wrapper(model=model)


class InceptionV3Wrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        if self.training:
            return self.model(x).logits
        else:
            return self.model(x)
