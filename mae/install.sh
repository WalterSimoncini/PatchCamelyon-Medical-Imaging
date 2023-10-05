# This script creates an environment to pretrain a
# masked autoencoder using mae_pretraining.py
conda create -n mae python=3.10

source activate mae

pip install torch torchvision datasets
# This script requires the from-source installation of transformers
pip install git+https://github.com/huggingface/transformers
# Install accelerate
# pip install transformers[torch]
pip install accelerate
