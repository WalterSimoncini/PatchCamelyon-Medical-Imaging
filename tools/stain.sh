# Run the stain normalization pipeline for the
# validation, test and training sets
python normalize_dataset.py \
    --output-folder stain_normalized \
    --split valid

python normalize_dataset.py \
    --output-folder stain_normalized \
    --split test

python normalize_dataset.py \
    --output-folder stain_normalized \
    --split train
