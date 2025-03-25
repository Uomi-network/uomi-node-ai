import os
import platform
import psutil
import subprocess
import re
import torch
from lib.config import MODELS_FOLDER

# System class is used to setup and check the system where the AI is executed.
class System:

    # Check if the system requirements are met.
    # A system is valid if the following files exist:
    # - OS is Linux
    # - CPU RAM has at least 48GB of RAM
    # - There are 2 or more GPUs available with 24GB+ of VRAM
    # - Disk has 100GB+ space total
    def check_system_requirements(self):
        try:
            # Check OS version
            os_info = platform.platform().lower()
            print(f"OS: {os_info}")
            if 'linux' not in os_info:
                print("🚨 Invalid OS")
                return False

            # Check RAM (convert to GB)
            total_ram = psutil.virtual_memory().total / (1024 ** 3)
            print(f"Total RAM: {total_ram}")
            if total_ram < 48:
                print("🚨 Insufficient RAM")
                return False

            # Check GPU count and VRAM
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
                gpus = nvidia_smi.decode('utf-8').strip().split('\n')
                
                # Count GPUs with 24GB+ VRAM
                valid_gpus = sum(1 for gpu in gpus if float(gpu) >= 24000)  # VRAM in MiB
                print(f"Valid GPUs: {valid_gpus}")
                if valid_gpus < 1:
                    print("🚨 Insufficient GPUs")
                    return False
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("🚨 Failed to detect GPUs")
                return False

            # Check disk space (convert to GB)
            disk = psutil.disk_usage('/')
            total_disk = disk.total / (1024 ** 3)
            print(f"Total disk: {total_disk}")
            if total_disk < 100:
                print("🚨 Insufficient disk space")
                return False

            return True
        except Exception as e:
            print(f"🚨 Error checking system requirements: {str(e)}")
            return False

    # Check if CUDA is available.
    # This function returns True if CUDA is available, False otherwise.
    def check_cuda_availability(self):
        try:
            return torch.cuda.is_available()
        except Exception as e:
            print(f"🚨 Error checking CUDA availability: {str(e)}")
            return False

    # Setup environment variables for the AI.
    # This function sets the following environment variables:
    # - CUDA_DEVICE_ORDER=PCI_BUS_ID
    # - CUDA_VISIBLE_DEVICES=0,1
    # - CUBLAS_WORKSPACE_CONFIG=:4096:8
    def setup_environment_variables(self):
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["TRANSFORMERS_CACHE"] = MODELS_FOLDER
        os.environ["HF_HOME"] = MODELS_FOLDER
