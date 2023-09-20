import h5py
import torch

from typing import Tuple
from torch.utils.data import Dataset


class PatchCamelyonDataset(Dataset):
    def __init__(self, data_path: str, targets_path: str, transform=None) -> None:
        self.data = h5py.File(data_path)["x"]
        self.targets = h5py.File(targets_path)["y"]
        self.transform = transform

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        sample = torch.tensor(self.data[idx, :, :, :]).float() / 255.0
        # [channels, x, y] to [x, y, channels]
        sample = torch.permute(sample, (2, 0, 1))

        # We need to squeeze the targets as they are
        # nested within multiple arrays
        target = torch.tensor(self.targets[idx].squeeze())

        if self.transform:
            sample = self.transform(sample)
        
        return sample, target
