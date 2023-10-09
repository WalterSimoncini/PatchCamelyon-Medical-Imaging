import torch
import torch.nn as nn
from .inception3 import InceptionV3Wrapper

class EnsembleModel(nn.Module):
    def __init__(self, *models, freeze_pretrained: bool = True):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList([
             self._prepare_model(model, freeze_pretrained) for model in models
        ])

        # Create a dummy variable to infer the feature size
        x_dummy = torch.zeros(64, 3, 299, 299)  # assuming input size is [1, 3, 299, 299]

        # Get the output feature size for each model and sum them up
        concatenated_size = sum(self._get_output_size(model, x_dummy) for model in self.models)

        # Creating new classification head
        self.classifier = nn.Sequential(
            nn.Linear(concatenated_size, 2, dtype=torch.float64)
        )

    def _prepare_model(self, model, freeze_pretrained):
        # Handle various model types accordingly.
        # Example: for InceptionV3, just use up to the fc layer.
        if isinstance(model, InceptionV3Wrapper):  # replace with actual class
            model.model.fc = nn.Identity()  # replace the FC layer with identity (does nothing, keeps tensor same)
        # TODO Add similar handling for other model types as needed.
        model = model.model
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
        with torch.no_grad():
            x = model(x)
        return x.logits.view(x.logits.size(0), -1)  # Flatten the output
