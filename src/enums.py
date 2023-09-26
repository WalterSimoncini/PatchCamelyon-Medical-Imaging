from enum import Enum


class PatchCamelyonSplit(Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


class ModelType(Enum):
    RESNET_18 = "resnet18"
    RESNET_50 = "resnet50"
    DENSENET_121 = "densenet121"
    VIT_16_B = "vit-16-b"


class TransformType(Enum):
    BASE = "base"
    EVALUATION = "evaluation"
    REGULAR_SHAPE_COLOR = "regular-shape-color"
