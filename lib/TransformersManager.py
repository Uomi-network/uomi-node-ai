import os
import torch
import time
import json
from typing import Dict, Optional, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class TransformersModelConfig:
    model_name: str  # HuggingFace model name/path
    deterministic: bool  # Whether the model is deterministic
    location: str  # Location of the model (cpu, disk)
    model_kwargs: Dict[str, Any]  # Additional kwargs for model loading
    tokenizer_kwargs: Dict[str, Any]  # Additional kwargs for tokenizer loading
    inference_kwargs: Dict[str, Any]  # Additional kwargs for inference

class TransformersModelManager:
    def __init__(self, models_config: Dict[str, TransformersModelConfig]):
        """
        Initialize TransformersModelManager with multiple transformer models loaded on CPU.
        
        Args:
            models_config: Dictionary mapping model names to their configurations
        """
        self.models_config = models_config
        self.cpu_models = {}
        self.tokenizers = {}
        self.current_gpu_model = None
        self.current_gpu_model_name = None
        self.seed = 42  # Seed for deterministic behavior
        
        # Load all models and tokenizers on CPU
        for model_name, config in models_config.items():
            # Load tokenizer
            print(f"Loading tokenizer for model {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                **config.tokenizer_kwargs
            )
            self.tokenizers[model_name] = tokenizer
            
            # Load model on CPU
            if config.location == "cpu":
                print(f"Loading model {model_name} on CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    device_map="cpu",
                    torch_dtype=torch.float16,  # Use float16 for memory efficiency
                    cache_dir='/app/models',
                    **config.model_kwargs
                )
                # Pin memory for faster GPU transfer
                for param in model.parameters():
                    param.data = param.data.pin_memory()
                for buffer in model.buffers():
                    buffer.data = buffer.data.pin_memory()

                self.cpu_models[model_name] = model
            
            # Load model from disk to gpu, then clear it
            elif config.location == "disk":
                print(f"Loading model {model_name} from disk to GPU")
                self.current_gpu_model_name = model_name
                self.current_gpu_model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    device_map="cuda",
                    torch_dtype=torch.float16,  # Use float16 for memory efficiency
                    cache_dir='/app/models',
                    **config.model_kwargs
                )
                self.run_inference([{ "role": "system", "content": "Hello, world!" }])
                self.clear_model()

            # Error if location is not cpu or disk
            else:
                raise ValueError(f"Invalid location {config.location} for model {model_name}")


    
    def switch_model(self, model_name: str):
        """
        Switch the model on GPU to a different one.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            The model that is now on GPU
        """
        time_start = time.time()

        # Load model configuration
        if model_name not in self.models_config:
            raise KeyError(f"Model {model_name} not found in configured models")
        model_config = self.models_config[model_name]

        # Clear existing GPU model if present
        self.clear_model()
        
        # Move new model to GPU from CPU
        if model_config.location == "cpu":
            self.current_gpu_model = self.cpu_models[model_name].to("cuda", non_blocking=True)
            self.current_gpu_model_name = model_name

            torch.cuda.synchronize()  # Synchronize CUDA operations
        
        # Load model from disk to GPU
        elif model_config.location == "disk":
            self.current_gpu_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                device_map="cuda",
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                cache_dir='/app/models',
                **model_config.model_kwargs
            )
            self.current_gpu_model_name = model_name
        
        # Error if location is not cpu or disk
        else:
            raise ValueError(f"Invalid location {model_config.location} for model {model_name}")

        print(f"Time taken to switch model: {time.time() - time_start:.2f}s")
        return self.current_gpu_model

    def clear_model(self):
        """
        Remove the current model from GPU if one exists.
        """
        time_start = time.time()
        if self.current_gpu_model is not None:
            if self.models_config[self.current_gpu_model_name].location == "cpu":
                # Move model back to CPU
                self.cpu_models[self.current_gpu_model_name] = self.current_gpu_model.to("cpu", non_blocking=True)

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

        # Switch between deterministic and non-deterministic behavior
        if self.models_config[self.current_gpu_model_name].deterministic:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)
        
        # Perform inference using the model
        tokenizer = self.tokenizers[self.current_gpu_model_name]
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.current_gpu_model.generate(
            **inputs,
            **self.models_config[self.current_gpu_model_name].inference_kwargs,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"Time taken for inference: {time.time() - start_time:.2f}s")
        return output

    def get_current_model(self):
        """
        Get the currently active GPU model.
        
        Returns:
            The current GPU model or None if no model is loaded
        """
        return self.current_gpu_model
    
    def get_tokenizer(self, model_name: str):
        """
        Get the tokenizer for a specific model.
        
        Args:
            model_name: Name of the model whose tokenizer to retrieve
            
        Returns:
            The tokenizer for the specified model
            
        Raises:
            KeyError: If model_name is not found in configured models
        """
        if model_name not in self.tokenizers:
            raise KeyError(f"Tokenizer for model {model_name} not found")
        return self.tokenizers[model_name]
