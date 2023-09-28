import os
import platform

from torch.utils.data import DataLoader

from .pcam import PatchCamelyonDataset
from src.enums import PatchCamelyonSplit


def get_data_loader(
    split: PatchCamelyonSplit,
    transform=None,
    batch_size: int = 64,
    shuffle: bool = True,
    data_dir: str = "data",
    data_key: str = "x"
):
    dataset = PatchCamelyonDataset(
        data_path=os.path.join(data_dir, f"camelyonpatch_level_2_split_{split.value}_x.h5"),
        targets_path=os.path.join(data_dir, f"camelyonpatch_level_2_split_{split.value}_y.h5"),
        transform=transform,
        data_key=data_key
    )

    num_workers = 2

    if platform.system() == "Darwin":
        # Workaround for macos
        num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
