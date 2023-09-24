import os
import cv2
import h5py
import torch
import logging
import argparse
import torchstain

from tqdm import tqdm
from torchvision import transforms

from src.enums import PatchCamelyonSplit
from src.utils.logging import configure_logging
from src.datasets.loaders import get_data_loader


def main(args):
    # Create the folder if it does not exist
    os.makedirs(args.output_folder, exist_ok=True)

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

    # Fit the normalizer using the reference image
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    normalizer.fit(data_transform(reference_image))

    dataset = get_data_loader(
        split=args.split,
        transform=None,
        batch_size=1,
        shuffle=False
    ).dataset

    # Create an output file with three datasets, one for the
    # normalized images, one for E and one for H
    output_file_path = os.path.join(
        args.output_folder,
        f"camelyonpatch_level_2_split_{args.split.value}_x.h5"
    )

    output_file = h5py.File(output_file_path, "w")

    E_dataset = output_file.create_dataset("E", (len(dataset), 96, 96, 3), dtype="uint8")
    H_dataset = output_file.create_dataset("H", (len(dataset), 96, 96, 3), dtype="uint8")

    norm_dataset = output_file.create_dataset("norm", (len(dataset), 96, 96, 3), dtype="uint8")

    # Track how many samples could not be normalized
    normalization_errors = 0

    for i in tqdm(range(len(dataset))):
        image, _ = dataset[i]
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8)

        try:
            norm, H, E = normalizer.normalize(
                I=data_transform(image.numpy()),
                stains=True
            )

            # Save the normalized outputs to the datasets
            E_dataset[i, :, :, :] = E.numpy()
            H_dataset[i, :, :, :] = H.numpy()
            norm_dataset[i, :, :, :] = norm.numpy()
        except Exception as ex:
            # For failed normalizations we keep the original
            # image, but leave the E and H datasets blank
            norm_dataset[i, :, :, :] = image
            normalization_errors += 1

            logging.warning(f"could not process sample {i}: {ex}")

    output_file.close()
    logging.info(f"done normalizing images. Encountered {normalization_errors} errors")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Patch Camelyon Training")

    parser.add_argument("--output-folder", default="normalized_data", type=str, help="Path to save the dataset to")
    parser.add_argument("--split", type=PatchCamelyonSplit, choices=list(PatchCamelyonSplit), required=True, help="The dataset split to test on")

    main(parser.parse_args())
