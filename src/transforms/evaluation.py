import torch.nn as nn

from torchvision import transforms
from .base_factory import TransformFactory


class EvaluationTransformFactory(TransformFactory):
    def transform(self, input_size: int) -> nn.Module:
        return transforms.Compose([
            # FIXME: Remove me if this does not work
            # transforms.Normalize(mean=[0.7007, 0.5384, 0.6916], std=[0.1818, 0.2008, 0.1648]),
            transforms.Resize(input_size, antialias=True)
        ])
