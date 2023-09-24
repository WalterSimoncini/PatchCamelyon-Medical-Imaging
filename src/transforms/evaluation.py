import torch.nn as nn

from torchvision import transforms
from .base_factory import TransformFactory


class EvaluationTransformFactory(TransformFactory):
    def transform(self, input_size: int) -> nn.Module:
        return transforms.Compose([
            transforms.Resize(input_size, antialias=True)
        ])
