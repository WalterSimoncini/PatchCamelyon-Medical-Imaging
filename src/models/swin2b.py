import torch.nn as nn

from typing import Tuple
from .base_factory import ModelFactory
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class SwinV2BFactory(ModelFactory):
    def input_size(self) -> int:
        # Any input size is fine, but a bigger one seems
        # to be working better
        return 224

    def base_model(self) -> nn.Module:
        model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        model.head = nn.Linear(in_features=1024, out_features=2)

        return model

    def to_feature_extractor(self, model: nn.Module) -> Tuple[nn.Module, int]:
        model.head = nn.Identity()

        return model, 1024
