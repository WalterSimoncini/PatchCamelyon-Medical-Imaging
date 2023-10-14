"""
    Splits a WSL (whole slide image) into cells
"""
import os
import sys
import argparse
import tifffile
import numpy as np

from tqdm import tqdm
from PIL import Image
from itertools import product


def image_per_cell(image: np.ndarray, cell_size: int, x: int, y: int) -> np.ndarray:
    return image[
        (cell_size * x):(cell_size * (x + 1)),
        (cell_size * y):(cell_size * (y + 1)),
        :
    ]


def main(args):
    output_folder = os.path.join(
        args.output_folder,
        os.path.split(args.input_image)[-1].split(".")[0]
    )

    # Do not continue if the folder already exists
    if os.path.isdir(output_folder):
        print(f"{output_folder} already exists. Terminating...")
        sys.exit(-1)

    os.makedirs(output_folder)

    print(f"reading {args.input_image}...")

    wsl_image = tifffile.imread(args.input_image)

    # Remove the image borders
    wsl_image = wsl_image[
        args.padding:(wsl_image.shape[0] - args.padding),
        args.padding:(wsl_image.shape[1] - args.padding),
        :
    ]

    output_cell_size = args.cell_size

    n_rows = wsl_image.shape[0] // output_cell_size
    n_cols = wsl_image.shape[1] // output_cell_size

    print(f"splitting will result in {n_rows * n_cols} cells")

    for x, y in tqdm(product(range(n_rows), range(n_cols))):
        cell = image_per_cell(image=wsl_image, cell_size=output_cell_size, x=x, y=y)

        if cell.mean() > args.brightness_threshold:
            # Skip images that are too bright as they do not
            # contain enough relevant information
            continue

        cell_image = Image.fromarray(cell)
        cell_image.save(os.path.join(output_folder, f"cell_{x}_{y}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSL Splitter")

    parser.add_argument("--input-image", type=str, required=True, help="The image to split")
    parser.add_argument("--cell-size", type=int, default=224, help="The size of output cells")
    parser.add_argument("--brightness-threshold", type=int, default=150, help="The maximum average brightess allowed for picked images")
    parser.add_argument("--padding", type=int, default=11000, help="The borders to be removed from the input image before splitting")
    parser.add_argument("--output-folder", type=str, required=True, help="The root output folder (the program will create a sub-directory for the input image)")

    main(parser.parse_args())
