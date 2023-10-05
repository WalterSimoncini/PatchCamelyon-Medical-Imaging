# Create the data folder if it does not exist
mkdir -p data

# Download and unpack the training data and targets
curl -o data/camelyonpatch_level_2_split_train_x.h5.gz https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz?download=1
gdown -o  data/camelyonpatch_level_2_split_train_y.h5.gz https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz?download=1

gzip -d data/camelyonpatch_level_2_split_train_x.h5.gz 
gzip -d data/camelyonpatch_level_2_split_train_y.h5.gz 

# Download and unpack the validation data and targets
curl -o data/camelyonpatch_level_2_split_valid_x.h5.gz https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz?download=1
curl -o data/camelyonpatch_level_2_split_valid_y.h5.gz https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz?download=1

gzip -d data/camelyonpatch_level_2_split_valid_x.h5.gz
gzip -d data/camelyonpatch_level_2_split_valid_y.h5.gz

# Download and unpack the test data and targets
curl -o data/camelyonpatch_level_2_split_test_x.h5.gz https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz?download=1
curl -o data/camelyonpatch_level_2_split_test_y.h5.gz https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz?download=1

gzip -d data/camelyonpatch_level_2_split_test_x.h5.gz
gzip -d data/camelyonpatch_level_2_split_test_y.h5.gz
