import torch
import time
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.config import MODELS_FOLDER, TRANSFORMERS_INFERENCE_MAX_TOKENS, TRANSFORMERS_INFERENCE_TEMPERATURE, USE_KV_CACHE
from transformers import LogitsProcessor
import torch
import time
import copy

from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    MinPLogitsWarper
)

class Sampling:
    def __init__(self, seed: int, device: str = "cpu"):
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, -1)
        # Avoid GPU<->CPU sync done by torch multinomial
        # See: https://github.com/pytorch/pytorch/blob/925a3788ec5c06db62ca732a0e9425a26a00916f/aten/src/ATen/native/Distributions.cpp#L631-L637
        q = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return probs.div_(q).argmax()

@dataclass
class TransformersModelConfig:
    model_name: str  # HuggingFace model name/path
    deterministic: bool  # Whether the model is deterministic
    location: str  # Location of the model (cpu, disk)
    model_kwargs: Dict[str, Any]  # Additional kwargs for model loading
    tokenizer_kwargs: Dict[str, Any]  # Additional kwargs for tokenizer loading

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
        self.device = 'cuda'
        self.streams = {}  # CUDA streams for async operations

        # Create CUDA stream for async operations
        if torch.cuda.is_available():
            self.main_stream = torch.cuda.Stream()
            
        self.warpers = [
            TemperatureLogitsWarper(TRANSFORMERS_INFERENCE_TEMPERATURE),
            TopKLogitsWarper(top_k=5)
        ]
        
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
                    cache_dir=MODELS_FOLDER,
                    **config.model_kwargs
                )
                
                # Create dedicated CUDA stream for this model
                if torch.cuda.is_available():
                    self.streams[model_name] = torch.cuda.Stream()
                
                # Store model in CPU models
                self.cpu_models[model_name] = model
                
                # Pin memory for faster transfers (only for parameters that support it)
                try:
                    for param in model.parameters():
                        if param.data.is_floating_point():  # Only pin floating point tensors
                            param.data = param.data.pin_memory()
                    for buffer in model.buffers():
                        if buffer.data.is_floating_point():  # Only pin floating point tensors
                            buffer.data = buffer.data.pin_memory()
                except Exception as e:
                    print(f"Warning: Could not pin memory for model {model_name}: {e}")

                # Verify the CPU model is successfully loaded
                self.switch_model(model_name)
                self.clear_model()
            
            # Load model from disk (will be loaded directly to GPU when needed)
            elif config.location == "disk":
                # For disk models, we don't load them until they're needed
                pass
            else:
                raise ValueError(f"Invalid location {config.location} for model {model_name}")

    def switch_model(self, model_name: str):
        """
        Switch the model on GPU to a different one using efficient CPU-GPU memory management.
        
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

        # If this model is already active on GPU, no need to switch
        if self.current_gpu_model_name == model_name and self.current_gpu_model is not None:
            return self.current_gpu_model

        # Clear existing GPU model if present
        self.clear_model()
        
        # Move new model to GPU from CPU
        if model_config.location == "cpu":
            # Use dedicated CUDA stream for this model if available
            stream = self.streams.get(model_name, None)
            
            with torch.cuda.stream(stream) if stream else torch.cuda.stream(self.main_stream):
                try:
                    # Create a new model instance on GPU by copying from CPU
                    # We use model.to() with non_blocking=True for asynchronous transfer
                    print(f"Moving model {model_name} to GPU...")
                    
                    # We need to first clone state_dict to avoid modifying the CPU model
                    self.current_gpu_model = copy.deepcopy(self.cpu_models[model_name])
                    self.current_gpu_model = self.current_gpu_model.to(
                        device="cuda", 
                        non_blocking=True
                    )
                    self.current_gpu_model.eval()  # Set to evaluation mode
                    self.current_gpu_model_name = model_name
                    
                    # Wait for the transfer to complete
                    if stream:
                        stream.synchronize()
                    else:
                        self.main_stream.synchronize()
                        
                except Exception as e:
                    print(f"Error moving model {model_name} to GPU: {e}")
                    # Fall back to standard synchronous transfer if async fails
                    self.current_gpu_model = self.cpu_models[model_name].to("cuda")
                    self.current_gpu_model_name = model_name
                    torch.cuda.synchronize()
        
        # Load model from disk to GPU
        elif model_config.location == "disk":
            print(f"Loading model {model_name} from disk to GPU")
            self.current_gpu_model = AutoModelForCausalLM.from_pretrained(
                model_config.model_name,
                device_map="cuda",
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                cache_dir=MODELS_FOLDER,
                **model_config.model_kwargs
            )
            self.current_gpu_model_name = model_name
        
        # Error if location is not cpu or disk
        else:
            raise ValueError(f"Invalid location {model_config.location} for model {model_name}")

        # Run warmup inference to initialize CUDA kernels (optional)
        with torch.no_grad():
            try:
                # Create a small dummy input tensor
                dummy_input = torch.ones((1, 10), dtype=torch.long, device="cuda")
                # Run a forward pass to initialize CUDA kernels
                _ = self.current_gpu_model(dummy_input)
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Warning: Warmup inference failed: {e}")

        print(f"Time taken to switch model: {time.time() - time_start:.2f}s")
        return self.current_gpu_model

    def clear_model(self):
        """
        Remove the current model from GPU and move it back to CPU efficiently.
        This preserves the model in CPU memory without reloading it from disk.
        """
        time_start = time.time()
        if self.current_gpu_model is not None:
            model_name = self.current_gpu_model_name
            
            if self.models_config[model_name].location == "cpu":
                try:
                    # Use dedicated stream for this model if available
                    stream = self.streams.get(model_name, None)
                    
                    with torch.cuda.stream(stream) if stream else torch.cuda.stream(self.main_stream):
                        print(f"Moving model {model_name} from GPU back to CPU...")
                        
                        # Move model back to CPU with non_blocking if possible
                        cpu_model = self.current_gpu_model.to(
                            device="cpu", 
                            non_blocking=True
                        )
                        
                        # We perform a deep copy to ensure the CPU model is fully independent
                        # This prevents CUDA errors during subsequent transfers
                        self.cpu_models[model_name] = cpu_model
                        
                        # Wait for transfer to complete
                        if stream:
                            stream.synchronize()
                        else:
                            self.main_stream.synchronize()
                            
                except Exception as e:
                    print(f"Error moving model {model_name} back to CPU: {e}")
                    # Fall back to synchronous transfer if async fails
                    self.cpu_models[model_name] = self.current_gpu_model.to("cpu")
                    torch.cuda.synchronize()
            
            # Release GPU memory explicitly
            self.current_gpu_model = None
            self.current_gpu_model_name = None
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"Time taken to clear model: {time.time() - time_start:.2f}s")

    def run_batch_executions(self, prompts, on_prompt_finished):
        """
        Run inference on the current GPU model on a batch of prompts using the generate() method.
        
        Args:
            prompts: Input prompts for the model
            on_prompt_finished: Callback function to call when a prompt is finished
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
        
        # Convert prompts to texts
        tokenizer = self.tokenizers[self.current_gpu_model_name]
        
        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Convert prompts to chat templates
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Ensure tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Tokenize all inputs
        batch_input_ids = tokenizer(texts, padding=True, return_tensors="pt").input_ids.to(self.device)
        
        # Setup generation parameters
        generation_config = {
            "max_new_tokens": TRANSFORMERS_INFERENCE_MAX_TOKENS,
            "temperature": TRANSFORMERS_INFERENCE_TEMPERATURE,
            "do_sample": not self.models_config[self.current_gpu_model_name].deterministic,
            "use_cache": USE_KV_CACHE,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        
        # Apply any warper-specific configurations
        for warper in self.warpers:
            # Adjust generation config based on warper type
            if hasattr(warper, "top_k") and warper.top_k is not None:
                generation_config["top_k"] = warper.top_k
            if hasattr(warper, "top_p") and warper.top_p is not None:
                generation_config["top_p"] = warper.top_p
            # Add other warper configurations as needed
        
        # Run generation
        outputs = self.current_gpu_model.generate(
            batch_input_ids,
            **generation_config
        )        
        # Process results
        generated_sequences = outputs.sequences
        scores = outputs.scores
        
        # Process each output sequence
        for i, (input_ids, generated_sequence) in enumerate(zip(batch_input_ids, generated_sequences)):
            # Get the generated text (only the new tokens, not the prompt)
            print(f"{generated_sequences=}")
            prompt_length = len(input_ids)
            generated_tokens = generated_sequence[prompt_length:]
            
            # Convert back to text
            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract token probabilities and prepare proof
            all_output_tokens = []
            
            for token_idx, token_id in enumerate(generated_tokens):
                # Get the score for this position
                if token_idx < len(scores):
                    token_scores = scores[token_idx][i]
                    # Get probabilities
                    token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
                    # Get top tokens
                    top_probs, top_indices = token_probs.topk(5)
                    
                    # Find this token's probability and rank
                    token_prob = None
                    token_rank = -1
                    
                    for rank, idx in enumerate(top_indices):
                        if idx.item() == token_id.item():
                            token_prob = top_probs[rank].item()
                            token_rank = rank
                            break
                    
                    if token_prob is None and token_id.item() < len(token_probs):
                        token_prob = token_probs[token_id.item()].item()
                else:
                    # For tokens beyond the available scores
                    token_prob = 0.0
                    token_rank = -1
                
                # Store token information
                token_info = {
                    "id": token_id.item(),
                    "prob": token_prob,
                    "index": token_rank
                }
                all_output_tokens.append(token_info)
            
            output = {
                "response": response,
                "proof": {
                    "tokens": all_output_tokens,
                    "full_sequence_length": len(generated_sequence)
                }
            }
            
            # Call the callback with the result
            on_prompt_finished(i, output)
        
        # Return execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        
        return execution_time

    def run_batch_checks(self, prompts, proofs, on_prompt_finished):
        """
        Verify that each token in the generated sequence is among the top 10 predicted tokens.

        Args:
            prompts: List of input prompts (e.g., chat messages).
            proofs: List of proofs, each containing generated token IDs (e.g., [{"id": token_id}, ...]).
            on_prompt_finished: Callback function to call with verification results.
        """
        if self.current_gpu_model is None:
            print("No model loaded on GPU")
            return None
        start_time = time.time()

        # Deterministic setup (as per your original code)
        deterministic_config = self.models_config[self.current_gpu_model_name].deterministic
        torch.manual_seed(self.seed)
        if deterministic_config:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)

        # Access tokenizer
        tokenizer = self.tokenizers[self.current_gpu_model_name]
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token_id

        # Prepare full input sequences (prompt + proof)
        full_input_ids = []
        full_attention_masks = []
        prompt_lengths = []
        for prompt, proof in zip(prompts, proofs):
            # Tokenize prompt with chat template
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]
            # Extract proof token IDs
            proof_ids = torch.tensor([t["id"] for t in proof["tokens"]], dtype=torch.long)
            # Concatenate prompt and proof
            full_ids = torch.cat([prompt_ids, proof_ids], dim=0)
            full_input_ids.append(full_ids)
            # Create attention mask (1 for all real tokens)
            attention_mask = torch.ones_like(full_ids)
            full_attention_masks.append(attention_mask)
            prompt_lengths.append(len(prompt_ids))

        # Pad sequences to the same length for batch processing
        max_len = max(len(ids) for ids in full_input_ids)
        padded_input_ids = torch.stack([
            torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id, dtype=ids.dtype)])
            for ids in full_input_ids
        ]).to(self.current_gpu_model.device)
        padded_attention_masks = torch.stack([
            torch.cat([mask, torch.zeros(max_len - len(mask), dtype=mask.dtype)])
            for mask in full_attention_masks
        ]).to(self.current_gpu_model.device)

        # Perform a single forward pass to get all logits
        with torch.no_grad():
            outputs = self.current_gpu_model(
                input_ids=padded_input_ids,
                attention_mask=padded_attention_masks,
                use_cache=False  # Single pass, cache not needed
            )
            logits = outputs.logits  # Shape: [batch_size, max_len, vocab_size]

        # Verify each generated token
        for batch_idx, (prompt_len, proof) in enumerate(zip(prompt_lengths, proofs)):
            generated_len = len(proof["tokens"])
            valid = True
            # Check each token in the generated sequence
            for i in range(generated_len):
                # Position j predicts the token at j+1
                j = prompt_len - 1 + i
                if j >= max_len - 1:
                    break  # Beyond sequence length due to padding
                current_logits = logits[batch_idx, j, :]  # Logits for next token
                top_tokens = torch.topk(current_logits, 10).indices  # Top 10 token IDs
                next_token = full_input_ids[batch_idx][j + 1].item()  # Actual next token
                if next_token not in top_tokens:
                    valid = False
                    break

            # Prepare response based on verification
            if valid:
                # Decode the verified generated sequence
                generated_ids = full_input_ids[batch_idx][prompt_len:prompt_len + generated_len]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                on_prompt_finished(batch_idx, {"response": response, "proof": None})
            else:
                on_prompt_finished(batch_idx, {"response": "", "proof": None})

        print(f"Batch processed in {time.time() - start_time:.2f}s")

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

    def _offload_model_to_cpu(self, model_name):
        """
        Helper method to safely offload a model from GPU to CPU.
        
        Args:
            model_name: Name of the model to offload
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.cpu_models:
            return False
            
        try:
            # Get the stream for this model if it exists
            stream = self.streams.get(model_name, self.main_stream)
            
            with torch.cuda.stream(stream):
                # If this model is currently on GPU, move it to CPU
                if self.current_gpu_model is not None and self.current_gpu_model_name == model_name:
                    # Create a new CPU copy of the model
                    print(f"Offloading model {model_name} from GPU to CPU")
                    
                    # Copy state_dict to avoid modifying tensors during transfer
                    cpu_state_dict = {k: v.detach().cpu() for k, v in self.current_gpu_model.state_dict().items()}
                    
                    # Load state dict into CPU model
                    self.cpu_models[model_name].load_state_dict(cpu_state_dict)
                    
                    # Clear CUDA memory for this model
                    del self.current_gpu_model
                    self.current_gpu_model = None
                    self.current_gpu_model_name = None
                    
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    stream.synchronize()
                    
            return True
        except Exception as e:
            print(f"Error offloading model {model_name} to CPU: {e}")
            return False
            
    def _load_model_to_gpu(self, model_name):
        """
        Helper method to safely load a model from CPU to GPU.
        
        Args:
            model_name: Name of the model to load to GPU
            
        Returns:
            The loaded GPU model or None if loading failed
        """
        if model_name not in self.cpu_models:
            return None
            
        try:
            # Get the stream for this model if it exists
            stream = self.streams.get(model_name, self.main_stream)
            
            with torch.cuda.stream(stream):
                # Make sure CUDA memory is clear before loading
                if self.current_gpu_model is not None:
                    self.clear_model()
                
                print(f"Loading model {model_name} to GPU")
                
                # Create a new CUDA model
                gpu_model = type(self.cpu_models[model_name])(**self.cpu_models[model_name].config.to_dict())
                
                # Copy state_dict to GPU
                with torch.inference_mode():
                    # Create a GPU state dict
                    gpu_state_dict = {
                        k: v.to(device="cuda", non_blocking=True) 
                        for k, v in self.cpu_models[model_name].state_dict().items()
                    }
                    
                    # Load state dict into GPU model
                    gpu_model = gpu_model.to("cuda")
                    gpu_model.load_state_dict(gpu_state_dict)
                
                # Set model to eval mode
                gpu_model.eval()
                
                # Update current GPU model
                self.current_gpu_model = gpu_model
                self.current_gpu_model_name = model_name
                
                # Synchronize stream to ensure all operations are complete
                stream.synchronize()
                
                return gpu_model
                
        except Exception as e:
            print(f"Error loading model {model_name} to GPU: {e}")
            return None

TRANSFORMERS_MODEL_CONFIG = {
    'casperhansen/mistral-small-24b-instruct-2501-awq': TransformersModelConfig(
        model_name='casperhansen/mistral-small-24b-instruct-2501-awq',
        deterministic=True,
        location="cpu",
        model_kwargs={
            "use_cache": True
        },
        tokenizer_kwargs={}
    ),
    'Qwen/QwQ-32B-AWQ': TransformersModelConfig(
        model_name='Qwen/QwQ-32B-AWQ',
        deterministic=False,
        location="cpu",
        model_kwargs={
            "use_cache": True
        },
        tokenizer_kwargs={}
    ),
    'SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B': TransformersModelConfig(
        model_name='SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B',
        deterministic=False,
        location="disk",
        model_kwargs={
            "use_cache": True
        },
        tokenizer_kwargs={}
    ),
}