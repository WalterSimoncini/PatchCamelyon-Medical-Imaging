import torch.nn as nn

from typing import Tuple
from .base_factory import ModelFactory
from torchvision.models import vit_l_32, ViT_L_32_Weights


class ViT32LFactory(ModelFactory):
    def input_size(self) -> int:
        return 224

    def base_model(self) -> nn.Module:
        model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
        model.heads = nn.Sequential(
          nn.Linear(in_features=1024, out_features=2)
        )

        return model

    def to_feature_extractor(self, model: nn.Module) -> Tuple[nn.Module, int]:
        model.heads = nn.Identity()

        return model, 1024
