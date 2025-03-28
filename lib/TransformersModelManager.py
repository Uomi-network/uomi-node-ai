import torch
import time
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.config import MODELS_FOLDER, TRANSFORMERS_INFERENCE_MAX_TOKENS, TRANSFORMERS_INFERENCE_TEMPERATURE, USE_KV_CACHE

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
                # Pin memory for faster GPU transfer
                for param in model.parameters():
                    param.data.pin_memory()
                for buffer in model.buffers():
                    buffer.data.pin_memory()

                # Try a switch
                self.cpu_models[model_name] = model
                self.switch_model(model_name)
                self.clear_model()
            
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
            self.current_gpu_model = self.cpu_models[model_name].to("cuda")
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
                self.cpu_models[self.current_gpu_model_name] = self.current_gpu_model.to("cpu")

            # Clear GPU model references
            self.current_gpu_model = None
            self.current_gpu_model_name = None
            
            # Clear CUDA cache
            torch.cuda.empty_cache()

            # Ensure all CUDA operations are finished
            torch.cuda.synchronize()

        print(f"Time taken to clear model for TRANSFORMERS MODEL MANAGER: {time.time() - time_start:.2f}s")

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

        # Initialize past_key_values for KV caching
        past = None

        choice = Sampling(self.seed, 'cuda')
        
        for step in range(TRANSFORMERS_INFERENCE_MAX_TOKENS):
            print(f"Step execution {step + 1}/{TRANSFORMERS_INFERENCE_MAX_TOKENS}")
            if not active_batch_indices:
                break  # All inferences have completed
            
            if not USE_KV_CACHE:
                # NON-CACHING APPROACH: Process full sequences with dynamic batch sizing
                
                # Optimize sequence processing by only processing active sequences up to their current lengths
                active_sequences = [batched_input_ids[idx, :current_positions[idx]] for idx in active_batch_indices]
                
                # Find maximum sequence length in the current batch
                max_len = max(seq.size(0) for seq in active_sequences)
                
                # Create efficiently sized tensors for this step
                padded_sequences = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                active_attention_masks = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                
                # Fill in the padded tensors with actual sequence data
                for i, seq in enumerate(active_sequences):
                    seq_len = seq.size(0)
                    padded_sequences[i, :seq_len] = seq
                    active_attention_masks[i, :seq_len] = 1
                
                # Forward pass with optimally sized inputs
                outputs = self.current_gpu_model(
                    input_ids=padded_sequences,
                    attention_mask=active_attention_masks,
                    use_cache=False  # Explicitly disable KV caching
                )
                
                # Extract logits at the end of each sequence
                batch_indices = torch.arange(len(active_sequences))
                seq_lengths = torch.tensor([seq.size(0) - 1 for seq in active_sequences])
                next_token_logits = outputs.logits[batch_indices, seq_lengths]
                
            else:
                # KV CACHING APPROACH: Only process the new token each time
                if step == 0:
                    # First step: process full prompt
                    active_sequences = [batched_input_ids[idx, :current_positions[idx]] for idx in active_batch_indices]
                    
                    # Find maximum sequence length in the current batch
                    max_len = max(seq.size(0) for seq in active_sequences)
                    
                    # Create efficiently sized tensors for this step
                    padded_sequences = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                    active_attention_masks = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                    
                    # Fill in the padded tensors with actual sequence data
                    for i, seq in enumerate(active_sequences):
                        seq_len = seq.size(0)
                        padded_sequences[i, :seq_len] = seq
                        active_attention_masks[i, :seq_len] = 1
                    
                    # Forward pass on the active batch (first step)
                    outputs = self.current_gpu_model(
                        input_ids=padded_sequences, 
                        attention_mask=active_attention_masks,
                        use_cache=True  # Enable KV caching
                    )
                    
                    # Extract logits at the end of each sequence
                    batch_indices = torch.arange(len(active_sequences))
                    seq_lengths = torch.tensor([seq.size(0) - 1 for seq in active_sequences])
                    next_token_logits = outputs.logits[batch_indices, seq_lengths]
                    
                else:
                    # Subsequent steps: only process the last token
                    active_input_ids = torch.stack([
                        batched_input_ids[idx, current_positions[idx]-1:current_positions[idx]] 
                        for idx in active_batch_indices
                    ])
                    
                    # Create proper position IDs for the new tokens
                    position_ids = torch.tensor([
                        [current_positions[idx] - 1] for idx in active_batch_indices
                    ], device=active_input_ids.device)
                    
                    # We need full attention mask when using past_key_values
                    active_attention_masks = torch.stack([
                        batched_attention_masks[idx, :current_positions[idx]] 
                        for idx in active_batch_indices
                    ])
                    
                    # Forward pass with past key values
                    outputs = self.current_gpu_model(
                        input_ids=active_input_ids,
                        attention_mask=active_attention_masks,
                        position_ids=position_ids,
                        use_cache=True,
                        past_key_values=past
                    )
                    
                    # For KV caching in subsequent steps, logits are for the last token only
                    next_token_logits = outputs.logits[:, -1]
                
            # Apply warpers to all logits at once
            for warper in self.warpers:
                next_token_logits = warper(None, next_token_logits)
            
            # Sample next tokens for all sequences at once
            selected_token_ids = torch.tensor([choice(logits).item() for logits in next_token_logits])
            
            # Record first tokens if this is the first step
            if step == 0:
                for i, original_idx in enumerate(active_batch_indices):
                    first_new_token_ids[original_idx] = selected_token_ids[i].item()
            
            # Get top-k tokens and probabilities for logging/proof
            top_probs, top_indices = next_token_logits.topk(5, dim=-1)
            
            # Create a list to track which indices remain active after this step
            new_active_batch_indices = []
            new_past_indices = []
            
            # Update all sequences and check for completion in one pass
            for batch_pos, original_idx in enumerate(active_batch_indices):
                selected_token_id = selected_token_ids[batch_pos].item()
                
                # Add the selected token to the sequence
                pos = current_positions[original_idx]
                batched_input_ids[original_idx, pos] = selected_token_id
                batched_attention_masks[original_idx, pos] = 1
                current_positions[original_idx] += 1
                
                # Create top tokens dictionary for this sequence
                sequence_top_tokens = {
                    top_indices[batch_pos, i].item(): {
                        "prob": top_probs[batch_pos, i].item(),
                        "index": i
                    } for i in range(top_indices.size(1))
                }
                
                # Store token information
                token_info = {
                    "id": selected_token_id,
                    "prob": sequence_top_tokens.get(selected_token_id, {"prob": 0.0, "index": -1})["prob"],
                    "index": sequence_top_tokens.get(selected_token_id, {"prob": 0.0, "index": -1})["index"],
                }
                all_output_tokens[original_idx].append(token_info)
                
                # Check if the sequence is complete
                if selected_token_id == eos_token_id:
                    completed_sequences[original_idx] = True
                    
                # Process completed sequences
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
                    new_past_indices.append(batch_pos)
            
            # Update past_key_values for next iteration if using KV caching
            if USE_KV_CACHE and new_active_batch_indices and outputs.past_key_values is not None:
                try:
                    # For newer HuggingFace versions using Cache class
                    from transformers.generation.utils import Cache
                    if isinstance(outputs.past_key_values, Cache):
                        past = outputs.past_key_values.reorder_cache(new_past_indices)
                    else:
                        # For tuple-based format
                        past = tuple(
                            tuple(p_i[new_past_indices] for p_i in layer_past)
                            for layer_past in outputs.past_key_values
                        )
                except (ImportError, AttributeError):
                    # Fallback for older versions
                    past = tuple(
                        tuple(p_i[new_past_indices] for p_i in layer_past)
                        for layer_past in outputs.past_key_values
                    )
            elif not USE_KV_CACHE or not new_active_batch_indices:
                past = None
            
            # Update active batch indices
            active_batch_indices = new_active_batch_indices
            
            # Optional: Print token generation speed
            if step > 0 and step % 10 == 0:
                elapsed = time.time() - start_time
                tokens_generated = sum(current_positions[idx] - prompt_lengths[idx] for idx in range(batch_size) if not completed_sequences[idx])
                tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
                print(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tokens/sec)")
                
        # Return execution time
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")

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
        
        # Initialize past_key_values for KV caching
        past = None
        
        # Process tokens step by step
        for step in range(TRANSFORMERS_INFERENCE_MAX_TOKENS):
            print(f"Step check {step + 1}/{TRANSFORMERS_INFERENCE_MAX_TOKENS}")
            if not active_batch_indices:
                break  # All inferences have completed
            
            if not USE_KV_CACHE:
                # NON-CACHING APPROACH: Process full sequences with dynamic batch sizing
                
                # Optimize sequence processing by only processing active sequences up to their current lengths
                active_sequences = [batched_input_ids[idx, :current_positions[idx]] for idx in active_batch_indices]
                
                # Find maximum sequence length in the current batch
                max_len = max(seq.size(0) for seq in active_sequences)
                
                # Create efficiently sized tensors for this step
                padded_sequences = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                active_attention_masks = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                
                # Fill in the padded tensors with actual sequence data
                for i, seq in enumerate(active_sequences):
                    seq_len = seq.size(0)
                    padded_sequences[i, :seq_len] = seq
                    active_attention_masks[i, :seq_len] = 1
                
                # Forward pass with optimally sized inputs
                outputs = self.current_gpu_model(
                    input_ids=padded_sequences,
                    attention_mask=active_attention_masks,
                    use_cache=False  # Explicitly disable KV caching
                )
                
                # Extract logits at the end of each sequence
                batch_indices = torch.arange(len(active_sequences))
                seq_lengths = torch.tensor([seq.size(0) - 1 for seq in active_sequences])
                next_token_logits = outputs.logits[batch_indices, seq_lengths]
                
            else:
                # KV CACHING APPROACH: Only process the new token each time
                if step == 0:
                    # First step: process full prompt
                    active_sequences = [batched_input_ids[idx, :current_positions[idx]] for idx in active_batch_indices]
                    
                    # Find maximum sequence length in the current batch
                    max_len = max(seq.size(0) for seq in active_sequences)
                    
                    # Create efficiently sized tensors for this step
                    padded_sequences = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                    active_attention_masks = torch.zeros((len(active_sequences), max_len), dtype=torch.long, device=batched_input_ids.device)
                    
                    # Fill in the padded tensors with actual sequence data
                    for i, seq in enumerate(active_sequences):
                        seq_len = seq.size(0)
                        padded_sequences[i, :seq_len] = seq
                        active_attention_masks[i, :seq_len] = 1
                    
                    # Forward pass on the active batch (first step)
                    outputs = self.current_gpu_model(
                        input_ids=padded_sequences, 
                        attention_mask=active_attention_masks,
                        use_cache=True  # Enable KV caching
                    )
                    
                    # Extract logits at the end of each sequence
                    batch_indices = torch.arange(len(active_sequences))
                    seq_lengths = torch.tensor([seq.size(0) - 1 for seq in active_sequences])
                    next_token_logits = outputs.logits[batch_indices, seq_lengths]
                    
                else:
                    # Subsequent steps: only process the last token
                    active_input_ids = torch.stack([
                        batched_input_ids[idx, current_positions[idx]-1:current_positions[idx]] 
                        for idx in active_batch_indices
                    ])
                    
                    # Create proper position IDs for the new tokens
                    position_ids = torch.tensor([
                        [current_positions[idx] - 1] for idx in active_batch_indices
                    ], device=active_input_ids.device)
                    
                    # We need full attention mask when using past_key_values
                    active_attention_masks = torch.stack([
                        batched_attention_masks[idx, :current_positions[idx]] 
                        for idx in active_batch_indices
                    ])
                    
                    # Forward pass with past key values
                    outputs = self.current_gpu_model(
                        input_ids=active_input_ids,
                        attention_mask=active_attention_masks,
                        position_ids=position_ids,
                        use_cache=True,
                        past_key_values=past
                    )
                    
                    # For KV caching in subsequent steps, logits are for the last token only
                    next_token_logits = outputs.logits[:, -1]
            
            # Process each active inference
            new_active_batch_indices = []
            new_past_indices = []
            
            for batch_pos, original_idx in enumerate(active_batch_indices):
                proof = proofs[original_idx]
                current_token_id = proof["tokens"][step]["id"]
                
                # Get logits for this sequence
                logits = next_token_logits[batch_pos].unsqueeze(0)
                
                # Apply temperature (if not 1.0)
                if TRANSFORMERS_INFERENCE_TEMPERATURE != 1.0:
                    logits = logits / TRANSFORMERS_INFERENCE_TEMPERATURE
                    
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
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
                    new_past_indices.append(batch_pos)
                    
            # Update past_key_values for next iteration if using KV caching
            if USE_KV_CACHE and new_active_batch_indices and outputs.past_key_values is not None:
                try:
                    # For newer HuggingFace versions using Cache class
                    from transformers.generation.utils import Cache
                    if isinstance(outputs.past_key_values, Cache):
                        past = outputs.past_key_values.reorder_cache(new_past_indices)
                    else:
                        # For tuple-based format
                        past = tuple(
                            tuple(p_i[new_past_indices] for p_i in layer_past)
                            for layer_past in outputs.past_key_values
                        )
                except (ImportError, AttributeError):
                    # Fallback for older versions
                    past = tuple(
                        tuple(p_i[new_past_indices] for p_i in layer_past)
                        for layer_past in outputs.past_key_values
                    )
            elif not USE_KV_CACHE or not new_active_batch_indices:
                past = None
                
            # Update active batch indices
            active_batch_indices = new_active_batch_indices
            
            # Optional: Print token checking speed
            if step > 0 and step % 10 == 0:
                elapsed = time.time() - start_time
                tokens_checked = sum(current_positions[idx] - prompt_lengths[idx] for idx in range(batch_size) if not completed_sequences[idx])
                tokens_per_second = tokens_checked / elapsed if elapsed > 0 else 0
                print(f"Checked {tokens_checked} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tokens/sec)")

        # Return execution time
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
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