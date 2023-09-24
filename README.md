# PatchCamelyon (PCam) Project - Medical Imaging UvA 2023

For information about the base repository consult [this link](https://github.com/basveeling/pcam)

## TODOs

- We should look into stain normalization -- Partial
- Calculate the dataset statistics -- OK (does not work)
- Create a eval.py script -- OK
- Add AUC in eval.py -- OK
- TTA -- OK (to test)
- Maybe we can weight TTA over the generated samples (e.g. 80% weight to the original image, 20% divided on the others) and tune it on the validation set? Or use different transforms? -- OK, Testing (a weight of 0.5 seems the best, but we should do a more proper search)
- Maybe also save the optimizer state dict? So we can resume training?
- Maybe preprocess images for scaling, so it'll be faster to train models