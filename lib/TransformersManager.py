import torch
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_TOKENS = 200
TEMPERATURE = 0.6

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

        # OLD VERSION: Old version of inference, without run token per token
        # inputs = tokenizer(text, return_tensors="pt").to("cuda")
        # outputs = self.current_gpu_model.generate(
        #     **inputs,
        #     **self.models_config[self.current_gpu_model_name].inference_kwargs,
        #     pad_token_id=tokenizer.eos_token_id,
        #     use_cache=True
        # )
        # output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # NEW VERSION: Inference with token per token execution in batch mode with batch size forced to 1
        inputs = tokenizer.encode(text, return_tensors="pt").to("cuda")
        batch_size = 1
        all_input_ids = []
        all_input_ids.append(inputs)
        # Initialize tracking variables for each sequence in the batch
        all_output_tokens = [[] for _ in range(batch_size)]
        first_new_token_ids = [None for _ in range(batch_size)]
        active_batch_indices = list(range(batch_size))
        # Create tracking for prompt lengths
        prompt_lengths = [ids.shape[1] for ids in all_input_ids]
        max_prompt_length = max(prompt_lengths)
        # Pre-allocate tensors with enough space for full sequence (prompt + all output tokens)
        full_sequence_length = max_prompt_length + MAX_TOKENS + 10
        # Create padded input tensors with attention masks
        batched_input_ids = torch.zeros((batch_size, full_sequence_length), dtype=torch.long, device="cuda")
        batched_attention_masks = torch.zeros((batch_size, full_sequence_length), dtype=torch.long, device="cuda")
        # Fill in the prompt portions
        for i, input_ids in enumerate(all_input_ids):
            seq_len = input_ids.shape[1]
            batched_input_ids[i, :seq_len] = input_ids.squeeze(0)
            batched_attention_masks[i, :seq_len] = 1
        # Track current token position for each sequence (starts at end of prompt)
        current_positions = prompt_lengths.copy()
        # Get the ID for the </s> token
        eos_token_id = tokenizer.eos_token_id
        # Track which sequences have completed (generated </s>)
        completed_sequences = [False] * batch_size
        # Process tokens step by step
        for step in range(MAX_TOKENS):
            print(f"Step {step + 1}/{MAX_TOKENS}")
            if not active_batch_indices:
                break  # All inferences have completed
            # Get active batch
            active_input_ids = batched_input_ids[active_batch_indices]
            active_attention_masks = batched_attention_masks[active_batch_indices]
            # Forward pass on the active batch
            outputs = self.current_gpu_model(input_ids=active_input_ids, attention_mask=active_attention_masks)
            # Process each active inference
            new_active_batch_indices = []
            for batch_pos, original_idx in enumerate(active_batch_indices):
                # Get the position of the last token in this sequence
                last_pos = current_positions[original_idx] - 1
                next_token_logits = outputs.logits[batch_pos, last_pos, :].unsqueeze(0)
                # Apply temperature (if not 1.0)
                if TEMPERATURE != 1.0:
                    next_token_logits = next_token_logits / TEMPERATURE
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                # Get top-k tokens by probability
                top_probs, top_indices = probs.topk(5, dim=-1)
                # Sample from the top-k tokens
                next_token_id = top_indices.select(-1, torch.multinomial(top_probs, num_samples=1).item()).unsqueeze(0)
                selected_token_id = next_token_id.item()
                all_output_tokens[original_idx].append(selected_token_id)
                # Record the first token if this is the first step
                if step == 0:
                    first_new_token_ids[original_idx] = selected_token_id
                # Add the selected token to the input sequence for next iteration
                pos = current_positions[original_idx]
                batched_input_ids[original_idx, pos] = selected_token_id
                batched_attention_masks[original_idx, pos] = 1
                current_positions[original_idx] += 1
                # Check if the generated token is the </s> token
                if selected_token_id == eos_token_id:
                    completed_sequences[original_idx] = True
                # Only keep sequences that haven't generated </s> in the active batch
                if not completed_sequences[original_idx]:
                    new_active_batch_indices.append(original_idx)
            # Update active batch indices
            active_batch_indices = new_active_batch_indices

        # Build final output
        full_sequence = batched_input_ids[0, :current_positions[0]].tolist()
        output = {
            "string": tokenizer.decode(full_sequence, skip_special_tokens=False),
            "tokens": all_output_tokens[0],
        }

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
