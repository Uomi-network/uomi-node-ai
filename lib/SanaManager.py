import os
import torch
import time
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass
from diffusers import SanaPipeline

@dataclass
class SanaModelConfig:
    model_name: str  # HuggingFace model name/path
    model_kwargs: Dict[str, Any]  # Additional kwargs for model loading
    tokenizer_kwargs: Dict[str, Any]  # Additional kwargs for tokenizer loading
    inference_kwargs: Dict[str, Any]  # Additional kwargs for inference

class SanaModelManager:
    def __init__(self, models_config: Dict[str, SanaModelConfig]):
        """
        Initialize SanaModelManager with multiple transformer models loaded on CPU.
        
        Args:
            models_config: Dictionary mapping model names to their configurations
        """
        self.models_config = models_config
        self.cpu_models = {}
        self.current_gpu_model = None
        self.current_gpu_model_name = None
        self.seed = 42  # Seed for deterministic behavior
        
        # Load all models and tokenizers on CPU
        for model_name, config in models_config.items():
            print(f"Loading model {model_name} on CPU")
            model = SanaPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
            )

            self.cpu_models[model_name] = model
    
    def switch_model(self, model_name: str):
        """
        Switch the model on GPU to a different one.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            The model that is now on GPU
            
        Raises:
            KeyError: If model_name is not found in configured models
        """
        time_start = time.time()
        if model_name not in self.cpu_models:
            raise KeyError(f"Model {model_name} not found in configured models")
            
        # Clear existing GPU model if present
        self.clear_model()
        
        # Move new model to GPU
        self.current_gpu_model = self.cpu_models[model_name].to("cuda", non_blocking=True)
        self.current_gpu_model.vae.to("cuda", dtype=torch.bfloat16, non_blocking=True)
        self.current_gpu_model.text_encoder.to("cuda", dtype=torch.bfloat16, non_blocking=True)
        self.current_gpu_model_name = model_name

        torch.cuda.synchronize()  # Synchronize CUDA operations

        print(f"Time taken to switch model: {time.time() - time_start:.2f}s")
        return self.current_gpu_model

    def clear_model(self):
        """
        Remove the current model from GPU if one exists.
        """
        time_start = time.time()
        if self.current_gpu_model is not None:
            # Move model back to CPU
            self.cpu_models[self.current_gpu_model_name] = self.current_gpu_model.to("cpu", non_blocking=True)
            self.cpu_models[self.current_gpu_model_name].vae.to("cpu", non_blocking=True)
            self.cpu_models[self.current_gpu_model_name].text_encoder.to("cpu", non_blocking=True)

            # Clear GPU model references
            self.current_gpu_model = None
            self.current_gpu_model_name = None
            
            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Ensure all CUDA operations are finished
            torch.cuda.synchronize()

        print(f"Time taken to clear model: {time.time() - time_start:.2f}s")

    def run_inference(self, prompt):
        """
        Run inference on the current GPU model.
        
        Args:
            prompt: Input prompt for the model
        
        Returns:
            The model's output
        """
        if self.current_gpu_model is None:
            print("No model loaded on GPU")
            return None
        start_time = time.time()

        image = self.current_gpu_model(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=25,
            generator=torch.Generator(device="cuda").manual_seed(self.seed),
        )[0]

        print(f"Time taken for inference: {time.time() - start_time:.2f}s")
        return image[0]

    def get_current_model(self):
        """
        Get the currently active GPU model.
        
        Returns:
            The current GPU model or None if no model is loaded
        """
        return self.current_gpu_model
