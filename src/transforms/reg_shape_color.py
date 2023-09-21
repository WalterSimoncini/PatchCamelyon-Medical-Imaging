import torch.nn as nn

from torchvision import transforms
from .base_factory import TransformFactory


class RegularShapeAndColorTransform(TransformFactory):
    """
        Transform to be used on raw dataset images (i.e. without super resolution)

        Transformations partially inspired by "Automatic Tumor Identification
        from Scans of Histopathological Tissues"

        Kundrotas, M., Mažonienė, E., & Šešok, D. (2023). Automatic Tumor
        Identification from Scans of Histopathological Tissues. Applied
        Sciences, 13(7), 4333.
    """
    def transform(self, input_size: int) -> nn.Module:
        return transforms.Compose([
            transforms.RandomCrop(size=(28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=360),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            # FIXME: Add normalization?
            transforms.Resize(input_size, antialias=True)
        ])
