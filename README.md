# PatchCamelyon (PCam) Project - Medical Imaging UvA 2023

For information about the base repository consult [this link](https://github.com/basveeling/pcam)

## TODOs

- We should look into stain normalization -- Partial (running pipeline)
- Ensemble different models
- Limit the snellius RAM requirements
- MAE pretraining? VAE pretraining?
  - Create a script to split an image into chunks (say 224x224)
  - See what would be the output size of these images as PNGs (or jpegs? even tho we would lose out on quality)
  - See the runtime
  - If runtime/dataset size are decent run the pipeline serially and retrieve the data (possibly directly on snellius, otherwise move data via GCP)

## Experiments to be run

- Train the model ensemble over stains. But not all images can be normalized. What can we do instead?
  - Train a backup model over the regular data, to be used if it was not possible to normalize an image -- OK
  - Create training and validation sets with only normalized images (including the H and E images)
    - Run pipeline for all images, save in incremental cells
    - Keep track of current cell index
    - Resize datasets as needed once processing is done
  - Train a model on the normalized data
  - Train a model on the H data
  - Train a model on the E data
- Maybe we can use an [LR scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)? Or AdamW as the optimizer?