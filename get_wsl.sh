# Download a WSL image for camelyon17
mkdir -p data

aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON17/images/patient_192_node_3.tif data