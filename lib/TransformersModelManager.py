import torch
import time
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lib.config import MODELS_FOLDER, TRANSFORMERS_INFERENCE_MAX_TOKENS, TRANSFORMERS_INFERENCE_TEMPERATURE, USE_KV_CACHE
from transformers import LogitsProcessor
import torch
import time

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
            if config.tokenizer_kwargs.get("chat_template") is not None:
                tokenizer.chat_template = config.tokenizer_kwargs["chat_template"]
            self.tokenizers[model_name] = tokenizer
            
            # Load model on CPU
            if config.location == "cpu":
                print(f"Loading model {model_name} on CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    # attn_implementation="flash_attention_2",
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
                    # attn_implementation="flash_attention_2",
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
                # attn_implementation="flash_attention_2",
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

    def run_batch_executions(self, prompts, enable_thinking_list, on_prompt_finished):
        """
        Run inference on the current GPU model on a batch of prompts using the generate() method.
        
        Args:
            prompts: Input prompts for the model
            enable_thinking_list: List of enable_thinking values for each prompt
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
        for prompt, enable_thinking in zip(prompts, enable_thinking_list):
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
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

    def run_batch_checks(self, prompts, enable_thinking_list, proofs, on_prompt_finished):
        """
        Verify that each token in the generated sequence is among the top 10 predicted tokens.

        Args:
            prompts: List of input prompts (e.g., chat messages).
            enable_thinking_list: List of enable_thinking values for each prompt
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
        for prompt, enable_thinking, proof in zip(prompts, enable_thinking_list, proofs):
            # Tokenize prompt with chat template
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
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

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

TRANSFORMERS_MODEL_CONFIG = {
    # 'casperhansen/mistral-small-24b-instruct-2501-awq': TransformersModelConfig(
    #     model_name='casperhansen/mistral-small-24b-instruct-2501-awq',
    #     deterministic=True,
    #     location="cpu",
    #     model_kwargs={
    #         "use_cache": True
    #     },
    #     tokenizer_kwargs={}
    # ),
    # 'Qwen/QwQ-32B-AWQ': TransformersModelConfig(
    #     model_name='Qwen/QwQ-32B-AWQ',
    #     deterministic=False,
    #     location="cpu",
    #     model_kwargs={
    #         "use_cache": True
    #     },
    #     tokenizer_kwargs={}
    # ),
    'SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B': TransformersModelConfig(
        model_name='SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B',
        deterministic=False,
        location="disk",
        model_kwargs={
            "use_cache": True
        },
        tokenizer_kwargs={}
    ),
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': TransformersModelConfig(
        model_name='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        deterministic=False,
        location="cpu",
        keep_in_memory=True,
        model_kwargs={
            "use_cache": True,
            # "quantization_config": quantization_config
            # "load_in_4bit": True,
        },
        tokenizer_kwargs={
            # "load_in_4bit": True,
            # "quantization_config": quantization_config
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{% set content = message['content'] %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{%- set ns.is_first = false -%}{%- set ns.is_last_user = true -%}{{'<｜User｜>' + content + '<｜Assistant｜>'}} {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}{%- if message['role'] == 'assistant' %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{% endif %}{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{%- endif %}{%- set ns.is_first = false %}{%- set ns.is_tool = false -%}{%- set ns.is_output_first = true %}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if content is none %}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{content + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none)%}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + content + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{{content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_last_user = false -%}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + content + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + content + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_last_user and not ns.is_tool %}{{'<｜Assistant｜>'}} \n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n {% endif %}"
        }
    ),
    }