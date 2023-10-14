import torch
import torch.nn as nn

from .base_factory import ModelFactory
from transformers import ViTImageProcessor, ViTForImageClassification


class HuggingFaceViT16BWrapper(nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()

        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.processor = ViTImageProcessor.from_pretrained(model_path, do_rescale=False)

    def forward(self, images: torch.tensor) -> torch.tensor:
        inputs = self.processor(images.unbind(dim=0), return_tensors="pt")
        inputs = inputs["pixel_values"].to(self.model.device)

        return self.model(inputs).logits


class HuggingFaceViT16BFactory(ModelFactory):
    def input_size(self) -> int:
        return 224

    def base_model(self) -> nn.Module:
        return HuggingFaceViT16BWrapper(model_path="facebook/vit-mae-base")

    def trained_model(self, weights_path: str) -> nn.Module:
        return HuggingFaceViT16BWrapper(model_path=weights_path)
