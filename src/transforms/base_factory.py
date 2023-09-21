import torch.nn as nn

from abc import ABC, abstractmethod


class TransformFactory(ABC):
    @abstractmethod
    def transform(self, input_size: int) -> nn.Module:
        raise NotImplementedError
