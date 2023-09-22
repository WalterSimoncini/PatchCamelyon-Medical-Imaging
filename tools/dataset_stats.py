"""
    Calculate the dataset statistics (mean and std) for
    PatchCamelyon. The statistics are:

    mean: [0.7007, 0.5384, 0.6916]
    std: [0.1818, 0.2008, 0.1648]
"""
import torch

from tqdm import tqdm

from src.enums import PatchCamelyonSplit
from src.datasets.loaders import get_data_loader


train_loader = get_data_loader(split=PatchCamelyonSplit.TRAIN, batch_size=1, transform=None)

std = torch.zeros(3)
mean = torch.zeros(3)

for image, _ in tqdm(train_loader):
    image = image.squeeze()

    for channel in range(3):
        # Calculate the per channel mean and std for each image
        std[channel] += image[channel, :, :].std()
        mean[channel] += image[channel, :, :].mean()

# Calculate the overall mean and std
mean = mean / len(train_loader.dataset)
std = std / len(train_loader.dataset)

print(f"mean: {mean.tolist()}, std: {std.tolist()}")
