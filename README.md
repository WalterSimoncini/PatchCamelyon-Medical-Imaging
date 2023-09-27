# PatchCamelyon (PCam) Project - Medical Imaging UvA 2023

For information about the base repository consult [this link](https://github.com/basveeling/pcam)

## TODOs

- We should look into stain normalization -- Partial (running pipeline)
- Maybe we can weight TTA over the generated samples (e.g. 80% weight to the original image, 20% divided on the others) and tune it on the validation set? Or use different transforms? -- OK, Testing (a weight of 0.5 seems the best, but we should do a more proper search)
  - Seems like it's not working. has some effect on validation but not on much else
- Maybe also save the optimizer state dict? So we can resume training?
- Ensemble different models
- Limit the snellius RAM requirements
- MAE pretraining
  - Create a script to split an image into chunks (say 224x224)
  - See what would be the output size of these images as PNGs (or jpegs? even tho we would lose out on quality)
  - See the runtime
  - If runtime/dataset size are decent run the pipeline serially and retrieve the data (possibly directly on snellius, otherwise move data via GCP)

## Experiments to be run

- Run an evaluation for weighted TTA on a pretrained model. Use 5 samples and linearly separated TTA weights (e.g. 0.1, 0.2, ... 0.9)
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