from enum import Enum


class PatchCamelyonSplit(Enum):
    TRAIN = "train"
    VALIDATION = "valid"
    TEST = "test"


class ModelType(Enum):
    RESNET_18 = "resnet18"
    RESNET_50 = "resnet50"
