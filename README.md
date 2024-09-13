# PatchCamelyon (PCam) Project - Medical Imaging UvA 2023

This repository implements various techniques for tackling the [Patch Camelyon](https://github.com/basveeling/pcam) Grand Challenge, whose goal is detecting if at least one pixel in a patch of a WSL (whole slide image) for histopathology is marked as a tumor. The WSL images are split into 96x96 patches which have to be classified into the `normal` (0) and `tumor` (1) classes.

## Setup

The scripts in this repository were developed with Python 3.10. While they may work with other Python versions we do not guarantee functionality. First of all install the required dependencies with 

```sh
pip install -r requirements.txt
```

Download the Patch Camelyon data using the following command (they will be downloaded in the `data` folder)

```sh
sh download.sh
# If Google Drive returns an error you can use
# the following command to download the data
# from a mirror
# sh download_mirror.sh
```

And finally log into wandb

```sh
wandb login
```

You're now ready to go! If you're running scrips/developing on Snellius there's a tailored guide in `snellius.md`.

### Stain-Normalized data

If needed, you can download the stain normalize data using the following command

```sh
sh download_stain_normalized.sh
```

The `x` files will contain three keys:

- `norm`: Macenko-normalized images
- `E`: the eosin component from the Macenko normalization
- `H`: the hematoxylin component from the Macenko normalization

### Pretrained Models

Click the links below (or use `gdown`) to download the pretrained models

- [ResNet50](https://drive.google.com/file/d/1jZpr0TXG2SnLWYP7_ZLHj3Pao96qU6gc/view?usp=sharing)
- [DenseNet121](https://drive.google.com/file/d/1jZpr0TXG2SnLWYP7_ZLHj3Pao96qU6gc/view?usp=sharing)
- [Inception V3](https://drive.google.com/file/d/1wcetqyKtpJf_7yYAg82feWsLnYmDhQeg/view?usp=drive_link)
- [Swin V2 B](https://drive.google.com/file/d/1SioQ9vgVwOg2BGfXGuqE2wTiKtYruFAr/view?usp=drive_link)
- [ViT 16 B](https://drive.google.com/file/d/1GbuS4TQ21K9HtlGFyPnj20OupeajuW51/view?usp=drive_link)
- [ViT 32 L](https://drive.google.com/file/d/1xBR-3PS3UjH09IuKdOvaN-0XU5jyydhL/view?usp=drive_link)
- [ViT 16 B (fine-tuned after MAE pretraining)](https://drive.google.com/file/d/1MhsqCvItJocv1FDhXCmlSt5Mj_LLblQq/view?usp=drive_link)
  - You can't use this model directly with torchvision, but you have to load it with HuggingFace using the `hf-16-b` model type

#### Stain-Normalized

You can download the pretrained models for stain-normalized data using the following commands

```sh
# Normalized images
gdown https://drive.google.com/uc?id=1CLq9cvCezyhmHxwNNoXNknQMvDrxQi-M
# Original images
gdown https://drive.google.com/uc?id=1bIclV6RHi3RHs4HGlWOfI-8vhWKBjKm9
# Hematoxylin (H) images
gdown https://drive.google.com/uc?id=1WLfXUYveL8jrgz3AkYKhO7eyP4Kw-2sC
# Eosin (E) images
gdown https://drive.google.com/uc?id=18vSI0O1PhmU7hg49KM2atUBi6QLek3VF
```

## Vision Models and Data Augmentation

We trained several vision models (CNNs and Vision Transformers) over the Patch Camelyon dataset, and the best performing one seems to be the [SwinTransformer V2](https://pytorch.org/vision/main/models/swin_transformer.html). The models can be trained using the `train.py` script as follows:

```sh
python train.py \
  --epochs 5 \
  --model swin-v2-b \
  --transform regular-shape-color
```

All the available models (and transform pipelines) are visible in `src/enums.py`. There are more command line options available, that allow you to configure other training parameters (e.g. the batch size, the learning rate, weight decay, etc.), the seed and where to save the output models. Display them with `python train.py --help`.

NOTE: By default the test accuracy is calculated on the latest model weights, not on the best ones.

### Registering a new Model

To integrate a new model first add its enumeration value to `ModelType` in `src/enums.py`, then create a subclass of `ModelFactory` in `src/models` (you must only implement the `base_model` and `input_size` methods). Finally register the factory class in the `get_model` function in `src/models/__init__.py`

### Registering a new Transform

The procedure to integrate a new transform is similar to integrating a new model:

1. Add an enumeration value to `TransformType` in `src/enums.py`
2. Create a subclass of `TransformFactory` and implement the `transform` method. This method should return a transform from `torchvision.transform`. Don't forget to resize the image to `input_size`! This ensures that the transform is compatible with all models.
3. Register the transform in the `get_transform` function in `src/transforms/__init__.py`

## Evaluation

To evaluate a model on a particular split (train, valid, test) you can use `eval.py` as follows:

```sh
python eval.py \
  --model-path /path/to/model.pt \
  --model vit-16-b \
  --split test
```

This will evaluate the model on the given split and output the test accuracy and AUC.

### Test-Time Augmentation (TTA)

Using `eval.py` you can also run test-time augmentation (TTA), which predicts the class probabilities over $n$ transformed copies of the image (in addition to the original image) and averages the prediction. You can run it as follows:

```sh
python eval.py \
  --model-path /path/to/model.pt \
  --model vit-16-b \
  --split test \
  --tta \
  --tta-transform regular-shape-color \
  --tta-samples 5
```

We recommend using the same transformation pipeline used during training. You can control how many augmentations are used for each prediction using the `--tta-samples` flag. Finally, we also implement a weighted version of TTA that controls how much weight is given to the original image and its augmentations. Given:

- $x$ as the original image
- $z_n$ the $n$-th augmentation of the original image
- $f(x)$ the softmax class probabilities from the classifier
- $\lambda$ the TTA weight

The final class probabilities are calculated as follows:

$$
\lambda f(x) + \frac{1 - \lambda}{N}\sum^N_{n=1} f(z_n)
$$

While this does not seem to have a meaningful effect on the result (it only marginally improves the AUC) you can run it as follows (e.g. using $\lambda = 0.5$):

```sh
python eval.py \
  --model-path /path/to/model.pt \
  --model vit-16-b \
  --split test \
  --tta \
  --tta-transform regular-shape-color \
  --tta-samples 5 \
  --tta-original-weight 0.5
```

## Stain Normalization

We also trained a model ensemble on the outputs of the Macenko normalization, obtained via [torchstain](https://github.com/EIDOSLAB/torchstain). The normalization target (`tools/stain_normalization_target.tif`) is a WSL patch from the [Camelyon17](https://camelyon17.grand-challenge.org/) dataset. You can run the normalization pipeline with the `tools/normalize_dataset.py` (beware that the script assumes it's placed in the root directory!) as follows:

```sh
python normalize_dataset.py \
  --output-folder data/stain_tmp \
  --split test
```

This script will then output an HDF5 file with four datasets: `(norm, H, E, y)`, where the first three elements are the normalization outputs and `y` represents the targets. It's necessary to save the targets as well, as some images cannot be normalized (most likely due to them not being large enough) and are thus skipped. Not saving the targets would cause a mismatch between the targets dataset and the data. To split back the file into data and targets you can use `tools/split_stain_dataset.py` as follows

```sh
python tools/split_stain_dataset.py \
  --dataset path/to/normalize_dataset/output.h5 \
  --out-folder data/stain
```

The above is needed to that you can use the regular `train.py` script to train models on the normalized images. You can specify which dataset to use in the HDF5 files by using the `--data-key` flag.

### Evaluation

For evaluation purposes we need the whole dataset, including images which cannot be normalized. To do so run the `normalize_dataset.py` script with the `--include-original` flag. The original images will then be preserved under the `x` dataset. Assuming you've trained individual models for the original and normalized images, plus the H normalization component (we do not use the E component as it did not give good empirical results) you can evaluate the ensemble by running:

```sh
python eval_stain.py \
    --image-model vit-16-b \
    --image-model-path path/to/image/model.pt \
    --H-model vit-16-b \
    --H-model-path path/to/H/model.pt \
    --norm-model vit-16-b \
    --norm-model-path path/to/norm/model.pt \
    --dataset-path path/to/dataset.h5
```

In our experiment this ensemble gave the best results, reaching an accuracy of ~92%.

### Notes

You can find several `.job` files for Snellius in `jobs/stain_normalized` for training stain-normalized models and evaluating them.

## MAE pretraining and ViT fine-tuning

This repository also contains utilities to pretrain a ViT using the Masked Autoencoder strategy, as described in [this paper](https://arxiv.org/pdf/2111.06377.pdf). Due to the fact that there is plenty of data for Histopathology leveraging SSL approaches is an approach worth exploring. The steps to do so are as follows:

1. Retrieve (some) WSLs (whole slide images). [Camelyon17](https://camelyon17.grand-challenge.org/) has 1000 of them. Each measuring around 60000x70000 pixels.
2. Split these images into patches, so that they can be used for pretraining. Assuming 224x224 patches each WSL would result in around 270000 patches.
3. Build a dataset in arrow format, to speed up the data loading (this cuts the training time by 66%)
4. Pretrain the ViT using the masked autoencoding strategy
5. Fine-tune the ViT for classification

The subsections below show how each step can be performed. You can download a checkpoint trained for 3 epochs on 8 WSIs (~8% of the entire Camelyon17 dataset), which correspond to ~2M image patches, using the following command:

```sh
gdown https://drive.google.com/uc?id=1bMJNpdHVVH5e0dGo5Sj2qq_DQ7SsPpIQ
```

### Retrieving WSL images

Tne data for Camelyon17 (and 16) is stored on AWS S3. You can either [install the AWS cli](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and download individual files using the command line as follows:

```sh
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/images/patient_192_node_3.tif data
```

Or you can use the `mae/tools/download_wsl.py` script to download an arbitrary number of images as follows (change the `--max-images` parameter as appropriate)

```sh
python mae/tools/download_wsl.py \
  --output-folder /scratch-shared/$USER/wsl \
  --max-images 10
```

### Splitting the WSL images

To split WSL images you can use the `mae/tools/split_wsl.py` script as follows:

```sh
python mae/tools/split_wsl.py \
  --input-image /path/to/filename \
  --output-folder /scratch-shared/$USER/extracted_wsl \
  --cell-size 224 
```

This command will split the input image in 224x224 cells and save them in a fodler under the `--output-folder` path. Before splitting the image in cells we remove a border (by default 11000) to avoid extracting all-white cells (which represent areas of the sample without any cells). You can controll the border size with the `--padding` argument.

The job file `jobs/mae/split_wsls.job` might be useful for scheduling multiple splitting tasks in parallel.

### Building the dataset

While you can use an `imagefolder` dataset to pretrain the ViT this will be incredibly slow, and for ~2M images it will take around 3 hours to index the dataset and 45 hours to pretrain the model for three epochs. If you convert the dataset to the Apache Arrow format before pretraining indexing will take a few minutes at most and the training time will be reduced to 12-15 hours. You can convert the dataset as follows (this script will only take a few minutes):

```sh
python mae/tools/create_pretraining_dataset.py \
  --root-folder /scratch-shared/$USER/extracted_wsl/train \
  --output-path /scratch-shared/$USER/dataset
```

### Pretraining the ViT

Once the dataset is ready you can pretrain the ViT using the `mae/mae_pretraining.py` script as follows:

```sh
python mae/mae_pretraining.py \
    --model_name_or_path facebook/vit-mae-base \
    --train_dir /scratch-shared/$USER/dataset \
    --cache_dir /scratch-shared/$USER/cache \
    --output_dir /scratch-shared/$USER/mae-models \
    --remove_unused_columns False \
    --label_names pixel_values \
    --save_steps 10000 \
    --do_train \
    --do_eval
```

For all the available command line options see `mae/mae_pretraining.py`. The output models will then be saved in `/scratch-shared/$USER/mae-models`.

### Fine-tuning the ViT

Before fine-tuning, we have to convert the PCam dataset to the arrow format, so that it can be used by the HuggingFace's `Trainer`. To do so run:

```sh
python mae/tools/create_hf_dataset.py \
    --data-dir data/original \
    --output-path data/hf \
    --num-proc 18
```

Once you've created the HuggingFace dataset, pretrained the ViT and picked a checkpoint you can finally finetune it as follows:

```sh
python $HOME/vit_finetuning.py \
    --model-path /path/to/checkpoint/folder \
    --dataset-path data/hf \
    --output-dir /scratch-shared/$USER/vit-models
```

You will find the fine-tuned checkpoints in `/scratch-shared/$USER/vit-models`
