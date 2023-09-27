import torch.nn as nn

from .base_factory import ModelFactory
from torchvision.models import vit_h_14, ViT_H_14_Weights


class ViT14HFactory(ModelFactory):
    def input_size(self) -> int:
        return 518

    def base_model(self) -> nn.Module:
        model = vit_h_14(weights=ViT_H_14_Weights.DEFAULT)
        model.heads = nn.Sequential(
          nn.Linear(in_features=768, out_features=2)
        )

        return model
