import torch.nn as nn

from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def base_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def trained_model(self, weights_path: str) -> nn.Module:
        raise NotImplementedError
