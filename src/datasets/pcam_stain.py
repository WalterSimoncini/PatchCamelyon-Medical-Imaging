import h5py
import torch
import numpy as np

from typing import Tuple
from torch.utils.data import Dataset


class PatchCamelyonStainNormalizedDataset(Dataset):
    """
        Patch Camelyion dataset with the original images
        and their Macenko-normalized counterparts if available
    """
    def __init__(self, data_path: str, transform=None) -> None:
        self.data = h5py.File(data_path)
        self.targets = self.data["y"]

        self.transform = transform

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        # Get the original image and its stain normalized counterparts
        E = self.__preprocess_sample(sample=self.data["E"][idx, :, :, :])
        H = self.__preprocess_sample(sample=self.data["H"][idx, :, :, :])

        image = self.__preprocess_sample(sample=self.data["x"][idx, :, :, :])
        norm = self.__preprocess_sample(sample=self.data["norm"][idx, :, :, :])

        # We need to squeeze the targets as they are
        # nested within multiple arrays
        target = torch.tensor(self.targets[idx].squeeze())

        if self.transform:
            E = self.transform(E)
            H = self.transform(H)

            image = self.transform(image)
            norm = self.transform(norm)

        # Norm, H, E may be zero-valued tensors if it
        # was not possible to normalize them during
        # preprocessing
        return image, norm, H, E, target

    def __preprocess_sample(self, sample: np.ndarray) -> torch.tensor:
        sample = torch.tensor(sample).float() / 255.0

        return torch.permute(sample, (2, 0, 1))
