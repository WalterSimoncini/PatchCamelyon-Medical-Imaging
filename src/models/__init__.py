import torch.nn as nn

from typing import Tuple
from src.enums import ModelType

from .resnet18 import Resnet18Factory
from .resnet50 import Resnet50Factory
from .densenet121 import DenseNet121Factory
from .vit16b import ViT16BFactory
from .inception3 import InceptionV3Factory
from .vit32l import ViT32LFactory
from .swin2b import SwinV2BFactory
from .vit_hf import HuggingFaceViT16BFactory

from .stain_ensemble import StainFusionModule


def get_model(type_: ModelType, weights_path: str = None, as_feature_extractor: bool = False) -> Tuple[nn.Module, int]:
    """Returns an initialized model of the given kind and its input size"""
    factory = {
        ModelType.RESNET_18: Resnet18Factory,
        ModelType.RESNET_50: Resnet50Factory,
        ModelType.DENSENET_121: DenseNet121Factory,
        ModelType.VIT_16_B: ViT16BFactory,
        ModelType.INCEPTION_V_3: InceptionV3Factory,
        ModelType.VIT_32_L: ViT32LFactory,
        ModelType.SWIN_V2_B: SwinV2BFactory,
        ModelType.HF_VIT_16_B: HuggingFaceViT16BFactory
    }[type_]()

    feature_size = None

    if weights_path is not None:
        model = factory.trained_model(weights_path=weights_path)
    else:
        model = factory.base_model()

    if as_feature_extractor:
        model, feature_size = factory.to_feature_extractor(model=model)

    return model, factory.input_size(), feature_size


def get_stain_fusion_model(config: dict) -> Tuple[nn.Module, int]:
    image_branch, _, image_feature_size = get_model(
        type_=ModelType(config["models"]["image"]["model"]),
        weights_path=config["models"]["image"]["model_path"],
        as_feature_extractor=True
    )

    norm_branch, _, norm_feature_size = get_model(
        type_=ModelType(config["models"]["norm"]["model"]),
        weights_path=config["models"]["norm"]["model_path"],
        as_feature_extractor=True
    )

    # Here we assume all models have the same input size
    H_branch, input_size, H_feature_size = get_model(
        type_=ModelType(config["models"]["H"]["model"]),
        weights_path=config["models"]["H"]["model_path"],
        as_feature_extractor=True
    )

    model = StainFusionModule(
        image_branch=image_branch,
        image_branch_feature_size=image_feature_size,
        norm_branch=norm_branch,
        norm_branch_feature_size=norm_feature_size,
        H_branch=H_branch,
        H_feature_size=H_feature_size,
        hidden_sizes=config["hidden_sizes"]
    )

    return model, input_size
