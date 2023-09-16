# Create the data folder if it does not exist
mkdir -p data

# Download and unpack the training data and targets
gdown "https://drive.google.com/uc?export=download&id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2" -O data/camelyonpatch_level_2_split_train_x.h5.gz
gdown "https://drive.google.com/uc?export=download&id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG" -O data/camelyonpatch_level_2_split_train_y.h5.gz

gzip -d data/camelyonpatch_level_2_split_train_x.h5.gz
gzip -d data/camelyonpatch_level_2_split_train_y.h5.gz

# Download and unpack the validation data and targets
gdown "https://drive.google.com/uc?export=download&id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3" -O data/camelyonpatch_level_2_split_valid_x.h5.gz
gdown "https://drive.google.com/uc?export=download&id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO" -O data/camelyonpatch_level_2_split_valid_y.h5.gz

gzip -d data/camelyonpatch_level_2_split_valid_x.h5.gz
gzip -d data/camelyonpatch_level_2_split_valid_y.h5.gz

# Download and unpack the test data and targets
gdown "https://drive.google.com/uc?export=download&id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_" -O data/camelyonpatch_level_2_split_test_x.h5.gz
gdown "https://drive.google.com/uc?export=download&id=17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP" -O data/camelyonpatch_level_2_split_test_y.h5.gz

gzip -d data/camelyonpatch_level_2_split_test_x.h5.gz
gzip -d data/camelyonpatch_level_2_split_test_y.h5.gz
