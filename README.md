# PatchCamelyon (PCam) Project - Medical Imaging UvA 2023

For information about the base repository consult [this link](https://github.com/basveeling/pcam)

## TODOs

- We should look into stain normalization -- Partial (running pipeline)
- Maybe we can weight TTA over the generated samples (e.g. 80% weight to the original image, 20% divided on the others) and tune it on the validation set? Or use different transforms? -- OK, Testing (a weight of 0.5 seems the best, but we should do a more proper search)
- Maybe also save the optimizer state dict? So we can resume training?

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