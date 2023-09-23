import cv2
import torch
import logging
import torchstain

from tqdm import tqdm
from torchvision import transforms

from src.enums import PatchCamelyonSplit
from src.utils.logging import configure_logging
from src.datasets.loaders import get_data_loader


configure_logging()

# Load the reference image used for normalization
reference_image = cv2.cvtColor(
    cv2.imread("tools/stain_normalization_target.tif"),
    cv2.COLOR_BGR2RGB
)

# Transform for the reference image
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
normalizer.fit(data_transform(reference_image))

dataset = get_data_loader(
    split=PatchCamelyonSplit.TRAIN,
    transform=None,
    batch_size=1,
    shuffle=False
).dataset

normalization_errors = 0

for i in tqdm(range(len(dataset))):
    image, _ = dataset[i]
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8)

    try:
        norm, H, E = normalizer.normalize(
            I=data_transform(image.numpy()),
            stains=True
        )
    except Exception as ex:
        normalization_errors += 1
        logging.warning(f"could not process sample {i}: {ex}")

logging.info(f"done normalizing images. Encountered {normalization_errors} errors")
