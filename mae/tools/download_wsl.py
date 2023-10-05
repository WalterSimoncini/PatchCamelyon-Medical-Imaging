import os
import boto3
import argparse

from tqdm import tqdm
from botocore import UNSIGNED
from botocore.config import Config


def main(args):
    BUCKET_NAME = "camelyon-dataset"
    IMAGES_FOLDER = os.path.join("CAMELYON17", "images")

    # Make sure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))

    bucket = s3.Bucket(BUCKET_NAME)
    camelyon17_images = bucket.objects.filter(Prefix=IMAGES_FOLDER)

    objects = [x for x in camelyon17_images][:args.max_images]

    for image_object in tqdm(objects):
        image_filename = image_object.key.split("/")[-1]
        image_path = os.path.join(args.output_folder, image_filename)

        if os.path.isfile(image_path):
            # Skip files which have been downloaded already
            continue

        bucket.download_file(
            image_object.key,
            image_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Camelyon17 WSL images")

    parser.add_argument("--output-folder", default="camelyon17_wsl", type=str, help="Path where the downloaded images should be saved to")
    parser.add_argument("--max-images", type=int, default=10, help="The maximum number of images to be downloaded")

    main(parser.parse_args())
