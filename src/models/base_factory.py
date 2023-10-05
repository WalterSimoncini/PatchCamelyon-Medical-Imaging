import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def input_size(self) -> int:
        """The input image size used by the model"""
        raise NotImplementedError

    @abstractmethod
    def base_model(self) -> nn.Module:
        """Returns a newly initialized model with random weights"""
        raise NotImplementedError

    def trained_model(self, weights_path: str) -> nn.Module:
        model = self.base_model()
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"))
        )

        return model, self.input_size()
