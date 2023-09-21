import torch.nn as nn

from src.enums import TransformType

from .base import BaseTransformFactory
from .evaluation import EvaluationTransformFactory
from .reg_shape_color import RegularShapeAndColorTransform


def get_transform(type_: TransformType, input_size: int) -> nn.Module:
    factory = {
        TransformType.BASE: BaseTransformFactory,
        TransformType.EVALUATION: EvaluationTransformFactory,
        TransformType.REGULAR_SHAPE_COLOR: RegularShapeAndColorTransform
    }[type_]()

    return factory.transform(input_size=input_size)
