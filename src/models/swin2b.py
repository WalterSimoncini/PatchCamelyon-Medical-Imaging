import torch.nn as nn

from .base_factory import ModelFactory
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class SwinV2BFactory(ModelFactory):
    def input_size(self) -> int:
        # Any input size is fine, so with this size we avoid rescaling
        # FIXME: does this affect the accuracy?
        # FIXME: we tested with 96, now we are testing 224
        return 96

    def base_model(self) -> nn.Module:
        model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
        model.head = nn.Linear(in_features=1024, out_features=2)

        return model
