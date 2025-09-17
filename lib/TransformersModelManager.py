import torch
import time
import os
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
from lib.continuous_batcher import ContinuousBatcher
from lib.fast_continuous_batcher import FastContinuousBatcher

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
    keep_in_memory: bool = False  # Whether to keep the model in memory after completion

class TransformersModelManager:
    def __init__(self, model_config: TransformersModelConfig):
        """Single-model manager (DeepSeek only) kept always on GPU (or CPU if CUDA unavailable)."""
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.warpers = [
            TemperatureLogitsWarper(TRANSFORMERS_INFERENCE_TEMPERATURE),
            TopKLogitsWarper(top_k=5)
        ]

        # Load tokenizer
        print(f"Loading tokenizer for model {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            **self.model_config.tokenizer_kwargs
        )
        # Ensure pad token and left padding for decoder-only models (eliminates right-padding warning)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        if self.model_config.tokenizer_kwargs.get("chat_template") is not None:
            self.tokenizer.chat_template = self.model_config.tokenizer_kwargs["chat_template"]

        # Load model
        print(f"Loading model {self.model_name} to device {self.device}")
        load_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        # Allow overriding attention implementation / device map / max memory via env without code change
        # ATTN_IMPL example: flash_attention_2 (if supported by installed transformers version)
        attn_impl = os.getenv("ATTN_IMPL")
        if attn_impl:
            # do not overwrite if user already passed something explicitly
            self.model_config.model_kwargs.setdefault("attn_implementation", attn_impl)

        # Parse MAX_MEMORY env: e.g. "0:20GiB,1:20GiB"
        max_memory_env = os.getenv("MAX_MEMORY")
        max_memory = None
        if max_memory_env:
            try:
                max_memory = {}
                for part in max_memory_env.split(','):
                    gid, cap = part.split(':', 1)
                    max_memory[int(gid.strip())] = cap.strip()
            except Exception as e:
                print(f"[model-load] Failed to parse MAX_MEMORY='{max_memory_env}': {e}")
                max_memory = None

        device_map_env = os.getenv("DEVICE_MAP", "auto")
        try:
            self.current_gpu_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                device_map=device_map_env,
                max_memory=max_memory,
                torch_dtype=load_dtype,
                cache_dir=MODELS_FOLDER,
                **self.model_config.model_kwargs
            )
        except Exception as e:
            print(f"[model-load] device_map='{device_map_env}' failed ({e}); retrying with 'auto'")
            self.current_gpu_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                device_map='auto',
                torch_dtype=load_dtype,
                cache_dir=MODELS_FOLDER,
                **self.model_config.model_kwargs
            )

        # Optional torch.compile acceleration
        if os.getenv("TORCH_COMPILE", "0") == "1":
            compile_mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
            try:
                import torch
                self.current_gpu_model = torch.compile(self.current_gpu_model, mode=compile_mode, fullgraph=False)
                print(f"[compile] Enabled torch.compile mode={compile_mode}")
            except Exception as e:
                print(f"[compile] Skipped torch.compile: {e}")

        # Report resolved device map for observability
        if hasattr(self.current_gpu_model, 'hf_device_map'):
            print(f"[model-load] hf_device_map={self.current_gpu_model.hf_device_map}")
        if self.device == 'cpu':
            print("CUDA not available, model loaded on CPU")

        # Continuous batcher disabled by default
        self.continuous_batcher: ContinuousBatcher | None = None
        self.fast_continuous_batcher: FastContinuousBatcher | None = None

    def enable_continuous(self, max_active: int | None = None, use_fast: bool = True):
        if use_fast:
            if self.fast_continuous_batcher is None:
                self.fast_continuous_batcher = FastContinuousBatcher(self.current_gpu_model, self.tokenizer, self.device, max_active=max_active or 5)
            return self.fast_continuous_batcher
        else:
            if self.continuous_batcher is None:
                self.continuous_batcher = ContinuousBatcher(self.current_gpu_model, self.tokenizer, self.device, max_active=max_active or 5)
            return self.continuous_batcher

    def submit_continuous(self, messages, enable_thinking, sampling_cfg, max_new_tokens, on_token, on_complete, is_check=False, forced_tokens=None):
        # Try fast batcher first, fallback to regular continuous batcher
        if self.fast_continuous_batcher is not None:
            return self.fast_continuous_batcher.submit(messages, enable_thinking, sampling_cfg, max_new_tokens, on_token, on_complete, is_check=is_check, forced_tokens=forced_tokens)
        elif self.continuous_batcher is not None:
            return self.continuous_batcher.submit(messages, enable_thinking, sampling_cfg, max_new_tokens, on_token, on_complete, is_check=is_check, forced_tokens=forced_tokens)
        else:
            raise RuntimeError("Continuous batching not enabled. Call enable_continuous() first.")

    def switch_model(self, model_name: str):
        if model_name != self.model_name:
            raise ValueError("Switching models not supported; single model manager.")
        return self.current_gpu_model

    def clear_model(self):
        return  # no-op to preserve interface

    def run_batch_executions(self, prompts, enable_thinking_list, on_prompt_finished):
        """
        Run inference on the current GPU model on a batch of prompts using the generate() method.
        
        Args:
            prompts: Input prompts for the model
            enable_thinking_list: List of enable_thinking values for each prompt
            on_prompt_finished: Callback function to call when a prompt is finished
        """
        # Always loaded
        start_time = time.time()

        if self.model_config.deterministic:
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
        
        tokenizer = self.tokenizer
        
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
            
        # Tokenize all inputs (left-padded) and keep attention mask to avoid warning
        tokenized = tokenizer(texts, padding=True, return_tensors="pt")
        batch_input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        # Setup generation parameters
        max_new_tokens = int(os.getenv("SMOKE_MAX_NEW_TOKENS", TRANSFORMERS_INFERENCE_MAX_TOKENS))
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": TRANSFORMERS_INFERENCE_TEMPERATURE,
            "do_sample": not self.model_config.deterministic,
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
            attention_mask=attention_mask,
            **generation_config
        )
        # Process results
        generated_sequences = outputs.sequences
        scores = outputs.scores
        
        # Process each output sequence
        for i, (input_ids, generated_sequence) in enumerate(zip(batch_input_ids, generated_sequences)):
            # Get the generated text (only the new tokens, not the prompt)
            if i == 0 and os.getenv("DEBUG_GENERATION"):
                print(f"generated_sequences shape={generated_sequences.shape}")
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
        # Model always loaded
        start_time = time.time()

        # Deterministic setup
        deterministic_config = self.model_config.deterministic
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
        tokenizer = self.tokenizer
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
        return self.current_gpu_model

    def get_tokenizer(self, _model_name: str | None = None):
        return self.tokenizer


DEEPSEEK_MODEL_CONFIG = TransformersModelConfig(
    model_name='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    deterministic=False,
    location='gpu',
    keep_in_memory=True,
    model_kwargs={
        'use_cache': True,
    },
    tokenizer_kwargs={
        'chat_template': "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}{%- for message in messages %}{%- if message['role'] == 'system' %}{%- if ns.is_first_sp %}{% set ns.system_prompt = ns.system_prompt + message['content'] %}{% set ns.is_first_sp = false %}{%- else %}{% set ns.system_prompt = ns.system_prompt + '\n\n' + message['content'] %}{%- endif %}{%- endif %}{%- endfor %}{{ bos_token }}{{ ns.system_prompt }}{%- for message in messages %}{% set content = message['content'] %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{%- set ns.is_first = false -%}{%- set ns.is_last_user = true -%}{{'<｜User｜>' + content + '<｜Assistant｜>'}} {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}{%- if message['role'] == 'assistant' %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{% endif %}{%- if message['role'] == 'assistant' and message['tool_calls'] is defined and message['tool_calls'] is not none %}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{%- endif %}{%- set ns.is_first = false %}{%- set ns.is_tool = false -%}{%- set ns.is_output_first = true %}{%- for tool in message['tool_calls'] %}{%- if not ns.is_first %}{%- if content is none %}{{'<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- else %}{{content + '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- set ns.is_first = true -%}{%- else %}{{'\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + '<｜tool▁call▁end｜>'}}{%- endif %}{%- endfor %}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- if message['role'] == 'assistant' and (message['tool_calls'] is not defined or message['tool_calls'] is none)%}{%- set ns.is_last_user = false -%}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + content + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{{content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_last_user = false -%}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + content + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\n<｜tool▁output▁begin｜>' + content + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_last_user and not ns.is_tool %}{{'<｜Assistant｜>'}} \n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n {% endif %}"
    }
)