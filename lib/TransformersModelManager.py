import torch
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.config import MODELS_FOLDER, TRANSFORMERS_INFERENCE_MAX_TOKENS, TRANSFORMERS_INFERENCE_TEMPERATURE

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
                    cache_dir=MODELS_FOLDER,
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
                    cache_dir=MODELS_FOLDER,
                    **config.model_kwargs
                )
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
                cache_dir=MODELS_FOLDER,
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

    def run_batch_executions(self, prompts, on_prompt_finished):
        """
        Run inference on the current GPU model on a batch of prompts.
        
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
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Convert texts to tensors
        all_input_ids = []
        for text in texts:
            input_ids = tokenizer.encode(
                text,
                return_tensors="pt"
            ).to("cuda")
            all_input_ids.append(input_ids)
        
        # Initialize tracking variables for each sequence in the batch
        batch_size = len(prompts)
        all_output_tokens = [[] for _ in range(batch_size)]
        first_new_token_ids = [None for _ in range(batch_size)]
        active_batch_indices = list(range(batch_size))

        # Create tracking for prompt lengths
        prompt_lengths = [ids.shape[1] for ids in all_input_ids]

        # Pre-allocate tensors with enough space for full sequence (prompt + all output tokens)
        full_sequence_length = max(prompt_lengths) + TRANSFORMERS_INFERENCE_MAX_TOKENS + 10

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
        for step in range(TRANSFORMERS_INFERENCE_MAX_TOKENS):
            print(f"Step execution {step + 1}/{TRANSFORMERS_INFERENCE_MAX_TOKENS}")
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
                if TRANSFORMERS_INFERENCE_TEMPERATURE != 1.0:
                    next_token_logits = next_token_logits / TRANSFORMERS_INFERENCE_TEMPERATURE
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                # Filter out tokens with probability <= 10^-4
                min_p_threshold = 1e-4
                valid_probs_mask = probs > min_p_threshold
                filtered_probs = probs.clone()
                filtered_probs[~valid_probs_mask] = 0.0
                # Renormalize probabilities after filtering
                if filtered_probs.sum() > 0:
                    filtered_probs = filtered_probs / filtered_probs.sum()
                # Get top-k tokens by probability
                top_probs, top_indices = filtered_probs.topk(5, dim=-1)
                top_tokens = {}
                for i, idx in enumerate(top_indices[0]):
                    prob = filtered_probs[0, idx].item()
                    top_tokens[idx.item()] = {
                        "prob": prob,
                        "index": i
                    }
                # Sample from the top-k tokens
                next_token_id = top_indices.select(-1, torch.multinomial(top_probs, num_samples=1).item()).unsqueeze(0)
                selected_token_id = next_token_id.item()
                # Record the first token if this is the first step
                if step == 0:
                    first_new_token_ids[original_idx] = selected_token_id
                # Add the selected token to the input sequence for next iteration
                pos = current_positions[original_idx]
                batched_input_ids[original_idx, pos] = selected_token_id
                batched_attention_masks[original_idx, pos] = 1
                current_positions[original_idx] += 1
                
                # Store the token on all_output_tokens
                all_output_tokens[original_idx].append({
                    "id": selected_token_id,
                    "prob": top_tokens[selected_token_id]["prob"],
                    "index": top_tokens[selected_token_id]["index"],
                })
                # Check if the generated token is the </s> token
                if selected_token_id == eos_token_id:
                    completed_sequences[original_idx] = True
                # Only keep sequences that haven't generated </s> in the active batch
                if completed_sequences[original_idx]:
                    prompt_text = tokenizer.decode(all_input_ids[original_idx].squeeze(0).tolist(), skip_special_tokens=True)
                    response = tokenizer.decode(batched_input_ids[original_idx, :current_positions[original_idx]].tolist(), skip_special_tokens=True)  
                    response = response[len(prompt_text):]
                    
                    output = {
                        "response": response,
                        "proof": {
                            "tokens": all_output_tokens[original_idx],
                            "full_sequence_length": full_sequence_length
                        }
                    }
                    on_prompt_finished(original_idx, output)
                else:
                    new_active_batch_indices.append(original_idx)
            # Update active batch indices
            active_batch_indices = new_active_batch_indices

    def run_batch_checks(self, prompts, proofs, on_prompt_finished):
        """
        Run inference on the current GPU model on a batch of prompts with proofs being sure proofs are valid.
        
        Args:
            prompts: Input prompts for the model
            proofs: Proofs to check while executing the prompts
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
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Convert texts to tensors
        all_input_ids = []
        for text in texts:
            input_ids = tokenizer.encode(
                text,
                return_tensors="pt"
            ).to("cuda")
            all_input_ids.append(input_ids)
        
        # Initialize tracking variables for each sequence in the batch
        batch_size = len(prompts)
        all_output_tokens = [[] for _ in range(batch_size)]
        first_new_token_ids = [None for _ in range(batch_size)]
        active_batch_indices = list(range(batch_size))

        # Create tracking for prompt lengths
        prompt_lengths = [ids.shape[1] for ids in all_input_ids]

        # Pre-allocate tensors with enough space for full sequence (take from the longest proof instead of max tokens)
        full_sequence_length = max(proof["full_sequence_length"] for proof in proofs)

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
        for step in range(TRANSFORMERS_INFERENCE_MAX_TOKENS):
            print(f"Step check {step + 1}/{TRANSFORMERS_INFERENCE_MAX_TOKENS}")
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
                proof = proofs[original_idx]
                current_token_id = proof["tokens"][step]["id"]
                # Get the position of the last token in this sequence
                last_pos = current_positions[original_idx] - 1
                next_token_logits = outputs.logits[batch_pos, last_pos, :].unsqueeze(0)
                # Apply temperature (if not 1.0)
                if TRANSFORMERS_INFERENCE_TEMPERATURE != 1.0:
                    next_token_logits = next_token_logits / TRANSFORMERS_INFERENCE_TEMPERATURE
                # Convert to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                # Get top-k tokens by probability
                top_probs, top_indices = probs.topk(10, dim=-1)
                # Be sure current token is in top-k
                if current_token_id not in top_indices:
                    completed_sequences[original_idx] = True
                    on_prompt_finished(original_idx, { "response": "", "proof": None })
                    continue

                # Sample forced to the current token
                selected_token_id = current_token_id
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
                if completed_sequences[original_idx]:
                    prompt_text = tokenizer.decode(all_input_ids[original_idx].squeeze(0).tolist(), skip_special_tokens=True)
                    response = tokenizer.decode(batched_input_ids[original_idx, :current_positions[original_idx]].tolist(), skip_special_tokens=True)  
                    response = response[len(prompt_text):]
                    
                    output = {
                        "response": response,
                        "proof": None
                    }
                    on_prompt_finished(original_idx, output)
                else:
                    new_active_batch_indices.append(original_idx)
            # Update active batch indices
            active_batch_indices = new_active_batch_indices

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

TRANSFORMERS_MODEL_CONFIG = {
    'casperhansen/mistral-small-24b-instruct-2501-awq': TransformersModelConfig(
        model_name='casperhansen/mistral-small-24b-instruct-2501-awq',
        deterministic=True,
        location="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={
            'do_sample': False,
            'num_beams': 1,
            'temperature': 1.0,
            'top_p': 1.0,
            'max_new_tokens': 8196,
        }
    ),
    'Qwen/QwQ-32B-AWQ': TransformersModelConfig(
        model_name='Qwen/QwQ-32B-AWQ',
        deterministic=False,
        location="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_new_tokens': 8196,
        }
    ),
    'SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B': TransformersModelConfig(
        model_name='SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B',
        deterministic=False,
        location="disk",
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_new_tokens': 8196,
        }
    ),
}