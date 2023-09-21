import torch.nn as nn

from typing import Tuple
from src.enums import ModelType

from .resnet18 import Resnet18Factory
from .resnet50 import Resnet50Factory


def get_model(type_: ModelType, weights_path: str = None) -> Tuple[nn.Module, int]:
    """Returns an initialized model of the given kind and its input size"""
    factory = {
        ModelType.RESNET_18: Resnet18Factory,
        ModelType.RESNET_50: Resnet50Factory
    }[type_]()

    if weights_path is not None:
        return factory.trained_model(weights_path=weights_path)

    return factory.base_model(), factory.input_size()
