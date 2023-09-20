import torch.nn as nn

from src.enums import ModelType
from .resnet18 import Resnet18Factory


def get_model(type_: ModelType, weights_path: str = None) -> nn.Module:
    factory = {
        ModelType.RESNET_18: Resnet18Factory
    }[type_]()

    if weights_path is not None:
        return factory.trained_model(weights_path=weights_path)

    return factory.base_model()
