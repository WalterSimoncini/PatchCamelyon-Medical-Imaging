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
    INCEPTION_V_3 = "inception3"
    VIT_32_L = "vit-32-l"
    SWIN_V2_B = "swin-v2-b"


class TransformType(Enum):
    BASE = "base"
    EVALUATION = "evaluation"
    REGULAR_SHAPE_COLOR = "regular-shape-color"


class EnsembleStrategy(Enum):
    MAJORITY = "majority"
    AVERAGE = "average"
