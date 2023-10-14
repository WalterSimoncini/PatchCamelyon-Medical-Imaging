import torch
import torch.nn as nn

from typing import List
from torchvision.ops import MLP


class StainFusionModule(nn.Module):
    """
        This module takes as an input three feature extractors,
        one for regular images, one for Macenko-normalized images
        and one for the Hematoxylin image component and combines
        their output representations using an MLP
    """
    def __init__(
        self,
        image_branch: nn.Module,
        image_branch_feature_size: int,
        norm_branch: nn.Module,
        norm_branch_feature_size: int,
        H_branch: nn.Module,
        H_feature_size: int,
        hidden_sizes: List[int] = [1024, 512, 256, 128],
        nonlinearity: nn.Module = nn.ReLU,
        out_features: int = 2,
        dropout: float = 0.4
    ) -> None:
        super().__init__()

        self.image_branch = image_branch
        self.norm_branch = norm_branch
        self.H_branch = H_branch

        # Freeze the feature extractors
        self.__freeze_module(module=self.image_branch)
        self.__freeze_module(module=self.norm_branch)
        self.__freeze_module(module=self.H_branch)

        # Create an MLP to combine the extracted features
        self.head = MLP(
            in_channels=image_branch_feature_size + norm_branch_feature_size + H_feature_size,
            hidden_channels=hidden_sizes,
            activation_layer=nonlinearity,
            dropout=dropout
        )

        self.fc = nn.Linear(in_features=hidden_sizes[-1], out_features=out_features)

    def __freeze_module(self, module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, images: torch.tensor, norms: torch.tensor, Hs: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            embeddings = torch.cat([
                self.image_branch(images),
                self.norm_branch(norms),
                self.H_branch(Hs)
            ], dim=1).to(norms.device)

        mlp_output = self.head(embeddings)

        return self.fc(mlp_output)
