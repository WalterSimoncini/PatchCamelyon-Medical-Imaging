import os
import glob
import argparse

from datasets import Dataset, Image


def main(args):
    # Code based on this GitHub issue
    # https://github.com/huggingface/datasets/issues/5317
    print(f"reading images from {args.root_folder}")

    ds = Dataset.from_dict({
        "image": list(glob.glob(
            os.path.join(args.root_folder, "**/*.jpg")
        ))
    })

    ds = ds.cast_column("image", Image())

    # save as Arrow locally
    ds.save_to_disk(args.output_path, num_proc=18)

    print(f"dataset saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a folder containing image folders into an HF dataset (an Arrow file)")

    parser.add_argument("--root-folder", type=str, help="The folder containing the image folders")
    parser.add_argument("--output-path", type=str, help="The folder where the output dataset should be saved to")

    main(parser.parse_args())
