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
    max_dataset_shape = (len(dataset), 96, 96, 3)

    y_dataset = output_file.create_dataset("y", len(dataset), dtype="uint8", maxshape=len(dataset))

    E_dataset = output_file.create_dataset("E", max_dataset_shape, dtype="uint8", maxshape=max_dataset_shape)
    H_dataset = output_file.create_dataset("H", max_dataset_shape, dtype="uint8", maxshape=max_dataset_shape)

    norm_dataset = output_file.create_dataset("norm", max_dataset_shape, dtype="uint8", maxshape=max_dataset_shape)

    if args.include_original:
        # If this argument was specified also create a dataset with regular images
        regular_image_dataset = output_file.create_dataset(
            "x",
            max_dataset_shape,
            dtype="uint8",
            maxshape=max_dataset_shape
        )

    # As some samples may be skipped due to failures we
    # need to keep track of the last index we wrote to
    current_cell_index = 0

    # Track how many samples could not be normalized
    normalization_errors = 0

    for i in tqdm(range(len(dataset))):
        image, target = dataset[i]
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8)

        try:
            norm, H, E = normalizer.normalize(
                I=data_transform(image.numpy()),
                stains=True
            )

            # Save the normalized outputs to the datasets
            E_dataset[current_cell_index, :, :, :] = E.numpy()
            H_dataset[current_cell_index, :, :, :] = H.numpy()
            norm_dataset[current_cell_index, :, :, :] = norm.numpy()
            y_dataset[current_cell_index] = target

            if args.include_original:
                regular_image_dataset[current_cell_index, :, :, :] = image

            current_cell_index += 1
        except Exception as ex:
            normalization_errors += 1

            if args.include_original:
                # If the normalization failed add the regular image
                # anyway and leave the others blank
                regular_image_dataset[current_cell_index, :, :, :] = image
                y_dataset[current_cell_index] = target

                current_cell_index += 1

            logging.warning(f"could not process sample {i}: {ex}")

    # Resize datasets to remove empty cells
    E_dataset.resize(current_cell_index, axis=0)
    H_dataset.resize(current_cell_index, axis=0)

    y_dataset.resize(current_cell_index, axis=0)

    norm_dataset.resize(current_cell_index, axis=0)

    output_file.close()
    logging.info(f"done normalizing images. Encountered {normalization_errors} errors")


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Patch Camelyon Stain Normalization")

    parser.add_argument("--output-folder", default="normalized_data", type=str, help="Path to save the dataset to")
    parser.add_argument("--split", type=PatchCamelyonSplit, choices=list(PatchCamelyonSplit), required=True, help="The dataset split to test on")
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, help="Whether to include the original images as well in the output file and leave normalized ones blank if the process fails for an image")

    main(parser.parse_args())
