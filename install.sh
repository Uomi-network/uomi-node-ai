# be sure to stop the script if any command fails
set -e

# create a new conda environment
conda create -n uomi-ai python=3.10 -y

# make conda available in the shell
source ~/miniconda3/etc/profile.d/conda.sh

# activate the environment
conda activate uomi-ai

# install the required packages
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc=12.1 cuda-toolkit=12.1
pip install -U "optimum>=1.20.0"
# git clone https://github.com/AutoGPTQ/AutoGPTQ.git
# cd AutoGPTQ
# rm -rf build/*
# python3 setup.py install
pip install auto-gptq --no-build-isolation
pip install transformers
conda install numpy=1.24
pip install flash-attn --no-build-isolation
pip install flask
