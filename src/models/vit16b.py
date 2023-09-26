import torch.nn as nn

from .base_factory import ModelFactory
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViT16BFactory(ModelFactory):
    def input_size(self) -> int:
        return 224

    def base_model(self) -> nn.Module:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Sequential(
          nn.Linear(in_features=768, out_features=2)
        )

        return model
