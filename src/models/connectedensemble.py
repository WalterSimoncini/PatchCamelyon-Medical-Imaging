import torch
import torch.nn as nn
from .inception3 import InceptionV3Wrapper
from .vit16b import ViT16BFactory
from .vit32l import ViT32LFactory
from torchvision.models.vision_transformer import VisionTransformer
import numpy as np

class EnsembleModel(nn.Module):
    def __init__(self, *models, freeze_pretrained: bool = True):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList([
             self._prepare_model(model, freeze_pretrained) for model in models
        ])

        # Create a dummy variable to infer the feature size
        # x_dummy = torch.zeros(64, 3, 299, 299)  # assuming input size is [1, 3, 299, 299]

        input_sizes = [model.image_size for model in models]
        dummies = [torch.zeros(64, 3, input_size, input_size) for input_size in input_sizes]

        z = zip(models, dummies)
        # Get the output feature size for each model and sum them up
        concatenated_size = sum(self._get_output_size(model, x_dummy) for (model, x_dummy) in z)

        # Creating new classification head
        self.classifier = nn.Sequential(
            nn.Linear(concatenated_size, 2, dtype=torch.float64)
        )

    def _prepare_model(self, model, freeze_pretrained):
        # Handle various model types accordingly.
        # Example: for InceptionV3, just use up to the fc layer.
        if isinstance(model, InceptionV3Wrapper):  # replace with actual class
            model.model.fc = nn.Identity()  # replace the FC layer with identity (does nothing, keeps tensor same)
            # model = model.model
        elif isinstance(model, VisionTransformer):
            model.heads = nn.Identity()
        # TODO Add similar handling for other model types as needed.
        # model = model.model
        if freeze_pretrained:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def _get_output_size(self, model, x):
        model.eval()
        with torch.no_grad():
            output = model(x)
        model.train()
        return output.view(output.size(0), -1).size(1)

    def forward(self, x):
        # Forward pass through each model, and concatenate along the feature dimension
        x = torch.cat([self._forward_single_model(model, x) for model in self.models], dim=1).type(torch.float64)
        x = self.classifier(x)
        return x
    
    def _forward_single_model(self, model, x):
        if model.image_size != x.size:
            x = torch.nn.functional.interpolate(x, size=(model.image_size, model.image_size), mode='bilinear')
        with torch.no_grad():
            x = model(x)
        # if isinstance(model, InceptionV3Wrapper):
        #     x = x.logits
        return x.view(x.size(0), -1)  # Flatten the output
