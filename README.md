# PatchCamelyon (PCam) Project - Medical Imaging UvA 2023

For information about the base repository consult [this link](https://github.com/basveeling/pcam)

## TODOs

- We should look into stain normalization
- Calculate the dataset statistics -- OK (try training a model with it)
- Create a eval.py script -- OK
- Add AUC in eval.py -- OK
- TTA -- OK (to test)
- Try normalization before applying rotations
- Maybe we can weight TTA over the generated samples (e.g. 80% weight to the original image, 20% divided on the others) and tune it on the validation set? Or use different transforms?