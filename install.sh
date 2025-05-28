#!/bin/bash
# Check if environment exists before removing
if conda info --envs | grep -q "uomi-ai"; then
    echo "Removing existing uomi-ai environment..."
    conda env remove -n uomi-ai
else
    echo "Environment uomi-ai does not exist, skipping removal..."
fi
conda create -n uomi-ai python=3.10 -y
conda activate uomi-ai
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc=12.1 cuda-toolkit=12.1
conda install numpy=1.24
pip install flask
pip install psutil
pip install websocket-client requests
pip install transformers
pip install 'accelerate>=0.26.0'
pip install autoawq
pip install 'triton==3.2.0'
pip install diffusers
