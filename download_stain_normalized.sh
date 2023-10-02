# NOTE: The bucket must allow public access before data can be downloaded
# Create a directory for the data
mkdir -p data/stain

# Training data
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_train_x.h5?alt=media -P data/stain
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_train_y.h5?alt=media -P data/stain

# Validation data
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_valid_x.h5?alt=media -P data/stain
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_valid_y.h5?alt=media -P data/stain

# Test data
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_test_x.h5?alt=media -P data/stain
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_test_y.h5?alt=media -P data/stain

# Test data bundle (original, norm, E, H)
wget https://storage.googleapis.com/download/storage/v1/b/pcam-stain/o/camelyonpatch_level_2_split_test_all.h5?alt=media -P data/stain
