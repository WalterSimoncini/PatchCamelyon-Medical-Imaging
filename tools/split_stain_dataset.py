"""
    This script splits the stain-normalized datasets
    into data and targets
"""
import os
import h5py
import argparse


def main(args):
    # Make sure the output folder exists
    os.makedirs(args.out_folder, exist_ok=True)

    source_file = h5py.File(args.dataset)

    output_template = args.dataset.split("/")[-1]
    output_template = output_template.replace(".h5", "")
    output_template = output_template[:-1]

    out_x_path = os.path.join(args.out_folder, f"{output_template}x.h5")
    out_y_path = os.path.join(args.out_folder, f"{output_template}y.h5")

    if os.path.isfile(out_x_path):
        print(f"data file {out_x_path} already exists. Overwriting it...")
        os.remove(out_x_path)

    if os.path.isfile(out_y_path):
        print(f"targets file {out_y_path} already exists. Overwriting it...")
        os.remove(out_y_path)

    out_x_file = h5py.File(out_x_path, "w")
    out_y_file = h5py.File(out_y_path, "w")

    data_shape = source_file["norm"].shape

    E_dataset = out_x_file.create_dataset("E", data_shape, dtype="uint8")
    H_dataset = out_x_file.create_dataset("H", data_shape, dtype="uint8")
    norm_dataset = out_x_file.create_dataset("norm", data_shape, dtype="uint8")    

    E_dataset[:, :, :, :] = source_file["E"]
    H_dataset[:, :, :, :] = source_file["H"]
    norm_dataset[:, :, :, :] = source_file["norm"]

    y_dataset = out_y_file.create_dataset("y", data_shape[0], dtype="uint8")
    y_dataset[:] = source_file["y"]

    out_x_file.close()
    out_y_file.close()

    print(f"written data file to {out_x_path}")
    print(f"written targets file to {out_x_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Camelyon Stain Normalization Splitter")

    parser.add_argument("--dataset", required=True, help="Path to the stain normalized dataset")
    parser.add_argument("--out-folder", required=True, help="Path where the output datasets should be saved to")

    main(parser.parse_args())
