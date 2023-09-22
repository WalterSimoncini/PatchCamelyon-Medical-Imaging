import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def input_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def base_model(self) -> nn.Module:
        raise NotImplementedError

    def trained_model(self, weights_path: str) -> nn.Module:
        model = self.base_model()
        model.load_state_dict(torch.load(weights_path))

        return model, self.input_size()
