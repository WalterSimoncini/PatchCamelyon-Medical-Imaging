# Create the data folder if it does not exist
mkdir -p data

# Download and unpack the training data
gdown "https://drive.google.com/uc?export=download&id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2" -O data/camelyonpatch_level_2_split_train_x.h5.gz
gzip -d data/camelyonpatch_level_2_split_train_x.h5.gz

# Download and unpack the targets
gdown "https://drive.google.com/uc?export=download&id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG" -O data/camelyonpatch_level_2_split_train_y.h5.gz
gzip -d data/camelyonpatch_level_2_split_train_y.h5.gz
