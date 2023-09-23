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
            # FIXME: Maybe we can add this back?
            # FIXME: Verify if it's better to scale the image before applying transforms
            # transforms.Normalize(mean=[0.7007, 0.5384, 0.6916], std=[0.1818, 0.2008, 0.1648]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=360),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
            transforms.Resize(input_size, antialias=True)
        ])
