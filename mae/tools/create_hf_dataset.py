import os
import h5py
import argparse
import numpy as np

from PIL import Image
from typing import Tuple
from datasets import Dataset, DatasetDict


def main(args):
    print(f"reading dataset from {args.data_dir}")

    out_dataset = DatasetDict({
        "train": create_dataset(
            data_key=args.data_key,
            data_dir=args.data_dir,
            split="train",
            num_proc=args.num_proc
        ),
        "valid": create_dataset(
            data_key=args.data_key,
            data_dir=args.data_dir,
            split="valid",
            num_proc=args.num_proc
        ),
        "test": create_dataset(
            data_key=args.data_key,
            data_dir=args.data_dir,
            split="test",
            num_proc=args.num_proc
        )
    })

    out_dataset.save_to_disk(args.output_path)

    print(f"dataset saved to {args.output_path}")


def create_dataset(data_key: str, data_dir: str, split: str, num_proc: int) -> Dataset:
    data_array, targets_array = get_data_arrays(
        data_key=data_key,
        split=split,
        data_dir=data_dir
    )

    dataset = Dataset.from_generator(
        dataset_generator,
        num_proc=num_proc,
        gen_kwargs={
            "data_array": data_array,
            "targets_array": targets_array
        }
    )

    return dataset


def get_data_arrays(data_key: str, data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
        Returns the data and targets for the specified
        split as a numpy array
    """
    data_file = h5py.File(
        os.path.join(data_dir, f"camelyonpatch_level_2_split_{split}_x.h5")
    )

    targets_file = h5py.File(
        os.path.join(data_dir, f"camelyonpatch_level_2_split_{split}_y.h5")
    )

    if len(targets_file["y"].shape) == 1:
        # The targets for stain-normalized datasets are a simple array
        targets_array = targets_file["y"][:]
    else:
        # The original data targets shape is (samples_n, 1, 1, 1)
        targets_array = targets_file["y"][:, :, :, :]

    data_array = data_file[data_key][:, :, :, :]

    data_file.close()
    targets_file.close()

    return data_array, targets_array


def dataset_generator(data_array, targets_array):
    for i in range(len(targets_array)):
        image, target = data_array[i, :, :, :], targets_array[i].squeeze().item()

        yield {
            "image": Image.fromarray(image),
            "label": target,
            "index": i
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tranforms the HDF5 Patch Camelyon dataset to an HuggingFace one")

    parser.add_argument("--data-key", type=str, default="x", help="The key that holds the data")
    parser.add_argument("--data-dir", type=str, help="The folder containing the original dataset")
    parser.add_argument("--num-proc", type=int, default=2, help="Number of processes used to create the datasets")
    parser.add_argument("--output-path", type=str, help="The folder where the output dataset should be saved to")

    main(parser.parse_args())
