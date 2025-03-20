FROM continuumio/miniconda3:latest

# Set environment variables for better GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set up conda to not require interactive confirmations
ENV CONDA_DEFAULT_YES=true

# Create a new directory for the application
WORKDIR /app

# Install build dependencies and C compiler
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git

# Create a new conda environment and install packages
RUN conda create -n uomi-ai python=3.10 -y && \
    echo "source activate uomi-ai" > ~/.bashrc && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate uomi-ai && \
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia && \
    conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc=12.1 cuda-toolkit=12.1 && \
    conda install numpy=1.24 && \
    pip install flask && \
    pip install psutil && \
    pip install transformers && \
    pip install 'accelerate>=0.26.0' && \
    pip install autoawq && \
    pip install 'triton==3.2.0' && \
    pip install diffusers

# Configure the container to always use the conda environment
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate uomi-ai && exec \"$@\"", "--"]

# Default command (modify as needed for your application)
CMD ["python", "uomi-ai.py"]