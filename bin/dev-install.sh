# 🚨 USE THIS INSTALLATION ONLY FOR DEVELOPMENT ENVIRONMENTS!!!
# This script installs the required packages for the project in a new conda environment without requiring specific hardware.

# be sure to stop the script if any command fails
set -e

# create a new conda environment
conda create -n uomi-ai python=3.10 -y

# activate the environment
conda activate uomi-ai

# install the required packages
conda install pytorch
conda install torchvision
pip install transformers
pip install flask
pip install 'accelerate>=0.26.0'
pip install optimum
export BUILD_CUDA_EXT=0
pip install auto-gptq
