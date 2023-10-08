import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, *models, freeze_pretrained: bool = True):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList([
             self._prepare_model(model, freeze_pretrained) for model in models
        ])

        # Create a dummy variable to infer the feature size
        x_dummy = torch.zeros(1, 3, 224, 224)  # assuming input size is [1, 3, 224, 224]

        # Get the output feature size for each model and sum them up
        concatenated_size = sum(self._get_output_size(model, x_dummy) for model in self.models)

        # Creating new classification head
        self.classifier = nn.Sequential(
            nn.Linear(concatenated_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def _prepare_model(self, model, freeze_pretrained):
        model = nn.Sequential(*list(model.children())[:-1])  # remove last fc layer
        
        if freeze_pretrained:
            for param in model.parameters():
                param.requires_grad = False

        return model


    def _get_output_size(self, model, x):
        with torch.no_grad():
            output = model(x)
        return output.view(output.size(0), -1).size(1)

    def forward(self, x):
        # Forward pass through each model, and concatenate along the feature dimension
        x = torch.cat([self._forward_single_model(model, x) for model in self.models], dim=1)
        x = self.classifier(x)
        return x
    
    def _forward_single_model(self, model, x):
        with torch.no_grad():
            x = model(x)
        return x.view(x.size(0), -1)  # Flatten the output
