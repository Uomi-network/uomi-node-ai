import math
import os
import time

import torch
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from lib.config import MODELS_FOLDER, TRANSFORMERS_INFERENCE_MAX_TOKENS, TRANSFORMERS_INFERENCE_TEMPERATURE, USE_KV_CACHE
from transformers import LogitsProcessor
from transformers import (
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    MinPLogitsWarper
)
from lib.continuous_batcher import ContinuousBatcher
from lib.fast_continuous_batcher import FastContinuousBatcher

# Note: We'll import accelerate lazily in the forced-device branch to avoid optional dependency warnings

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
    quantized_max_memory_multiplier: float = 1.82  # Planning headroom for quantized models (per GPU)

class TransformersModelManager:
    def __init__(self, model_config: TransformersModelConfig, force_device: str | None = None):
        """Single-model manager (DeepSeek only) kept always on GPU (or CPU if CUDA unavailable)."""
        # Clear GPU cache early to free any residual memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # When provided, force loading the entire model on a specific GPU device (e.g. 'cuda:0')
        self.force_device = force_device
        if self.force_device is not None and torch.cuda.is_available():
            # Align logical device to the forced target so downstream components use it
            self.device = self.force_device

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
        print(f"Loading model {self.model_name} to device {self.force_device or self.device}")
        load_dtype = torch.float16# if self.device == 'cuda' else torch.float32
        # Allow overriding attention implementation / device map / max memory via env without code change
        # ATTN_IMPL example: flash_attention_2 (if supported by installed transformers version)
        attn_impl = os.getenv("ATTN_IMPL")
        if attn_impl:
            # do not overwrite if user already passed something explicitly
            self.model_config.model_kwargs.setdefault("attn_implementation", attn_impl)

        # Parse MAX_MEMORY env: e.g. "0:20GiB,1:20GiB"
        max_memory_env = os.getenv("MAX_MEMORY")
        max_memory = None
        # Allow overriding the quantized multiplier (used when no explicit MAX_MEMORY is set)
        quantized_max_memory_multiplier = self.model_config.quantized_max_memory_multiplier
        multiplier_env = os.getenv("MAX_MEMORY_MULTIPLIER")
        if multiplier_env:
            try:
                # Keep sanity floor at 1.0 to avoid shrinking below physical VRAM
                quantized_max_memory_multiplier = max(1.0, float(multiplier_env))
            except ValueError:
                print(f"[model-load] Ignoring invalid MAX_MEMORY_MULTIPLIER='{multiplier_env}' (expected float)")
        if max_memory_env:
            try:
                max_memory = {}
                for part in max_memory_env.split(','):
                    gid, cap = part.split(':', 1)
                    max_memory[int(gid.strip())] = cap.strip()
            except Exception as e:
                print(f"[model-load] Failed to parse MAX_MEMORY='{max_memory_env}': {e}")
                max_memory = None

        # When force_device is set, place entire model on that GPU explicitly to avoid CPU or other GPU usage
        # Strategy: set allocator for fragmentation resilience, set current device, and load with device_map
        if self.force_device is not None and torch.cuda.is_available():
            print(f"[model-load] Forcing single-device placement on {self.force_device}")
            # Improve fragmentation resilience if not already set
            alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
            if not alloc_conf:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # Set current device so implicit allocations use the right GPU
            try:
                if ':' in self.force_device:
                    gid = int(self.force_device.split(':',1)[1])
                else:
                    gid = 0
                torch.cuda.set_device(gid)
            except Exception as e:
                print(f"[model-load] Warning: failed to set CUDA device context: {e}")
            # Empty cache to reduce fragmentation before loading
            torch.cuda.empty_cache()
            # Try zero-CPU load first via Accelerate; if not available, fail hard (no CPU fallback)
            try:
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch  # type: ignore
                from huggingface_hub import snapshot_download  # type: ignore
                # Download snapshot locally for accelerate (requires a local checkpoint path)
                local_dir = snapshot_download(self.model_config.model_name, cache_dir=MODELS_FOLDER)
                # Zero-CPU load: initialize empty model on meta and dispatch weights directly to GPU
                cfg = AutoConfig.from_pretrained(local_dir, cache_dir=MODELS_FOLDER)
                with init_empty_weights():
                    empty_model = AutoModelForCausalLM.from_config(cfg, dtype=load_dtype)
                with torch.cuda.device(gid):
                    self.current_gpu_model = load_checkpoint_and_dispatch(
                        empty_model,
                        checkpoint=local_dir,
                        device_map={"": self.force_device},
                        max_memory={gid: "18GiB"},  # Limit to 18GiB per GPU to leave more headroom
                        dtype=load_dtype,
                        no_split_module_classes=self.model_config.model_kwargs.get("no_split_module_classes")
                    )
                print("[model-load] Loaded via Accelerate with zero-CPU dispatch")
            except Exception as e:
                print(f"[model-load] Accelerate path failed ({e}); no CPU fallback allowed - failing startup")
                raise RuntimeError(f"Failed to load model on GPU {self.force_device} without CPU usage: {e}")
        else:
            # Auto-set max_memory for multi-GPU to ensure proper distribution without CPU offload
            if max_memory is None and torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    has_quantization = 'quantization_config' in self.model_config.model_kwargs
                    if has_quantization:
                        # Quantized models with hybrid architectures (e.g. Qwen3.5-35B-A3B) have
                        # ~57% actual GPU usage vs BF16 budget (mix of 4-bit + non-quantizable BF16).
                        # accelerate uses BF16 sizes for planning, so we need total budget > BF16 model
                        # size to prevent CPU dispatch, but must keep actual usage within physical VRAM.
                        #
                        # For RTX 4090 (24GiB): budget=37GiB → actual~21GiB < 24GiB (no OOM)
                        #                       total=74GiB > 70GB BF16 (no CPU dispatch)
                        # Formula: 1.55x actual VRAM (actual usage ≈ 0.57 × budget → 0.57×1.55 = 88%)
                        max_memory = {}
                        for i in range(num_gpus):
                            total_gb = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                            inflated_gib = math.ceil(total_gb * quantized_max_memory_multiplier)
                            max_memory[i] = f"{inflated_gib}GiB"
                        print(
                            "[model-load] Quantized model: using inflated max_memory="
                            f"{max_memory} ({quantized_max_memory_multiplier:.2f}x VRAM to prevent CPU dispatch while avoiding OOM)"
                        )
                    else:
                        # Non-quantized models: limit per-GPU to leave ~1GiB headroom for KV cache / OS
                        max_memory = {}
                        for i in range(num_gpus):
                            total_gb = torch.cuda.get_device_properties(i).total_memory // (1024 ** 3)
                            max_memory[i] = f"{total_gb - 1}GiB"
                        print(f"[model-load] Auto-detected {num_gpus} GPUs, setting max_memory={max_memory}")

            # Ensure allocator can grow instead of fragmenting when large blocks are requested mid-load
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
                if not alloc_conf or 'expandable_segments' not in alloc_conf:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            device_map_env = os.getenv("DEVICE_MAP", "auto")
            print(f"[model-load] Using device_map='{device_map_env}', max_memory={max_memory}")
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
                # Clean up any partial model state and GPU memory before retry
                try:
                    del self.current_gpu_model
                except AttributeError:
                    pass
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.current_gpu_model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_name,
                    device_map='auto',
                    torch_dtype=load_dtype,
                    cache_dir=MODELS_FOLDER,
                    **self.model_config.model_kwargs
                )
        
        # Only move model if device_map was NOT used (to preserve multi-GPU distribution)
        # and no forced device pinning was requested
        if (self.force_device is None) and (not hasattr(self.current_gpu_model, 'hf_device_map')):
            model_device = next(self.current_gpu_model.parameters()).device
            if str(model_device) == 'cpu' and self.device == 'cuda':
                print(f"[model-load] Moving model from CPU to {self.device}")
                try:
                    self.current_gpu_model = self.current_gpu_model.to(self.device)
                    print(f"[model-load] Successfully moved model to {self.device}")
                except Exception as e:
                    print(f"[model-load] Failed to move model to GPU: {e}")
        else:
            print(f"[model-load] Skipping .to() - model placed via device_map or forced device")

        # Optional torch.compile acceleration
        if os.getenv("TORCH_COMPILE", "0") == "1":
            compile_mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
            try:
                self.current_gpu_model = torch.compile(self.current_gpu_model, mode=compile_mode, fullgraph=False)
                print(f"[compile] Enabled torch.compile mode={compile_mode}")
            except Exception as e:
                print(f"[compile] Skipped torch.compile: {e}")

        # Report resolved device map for observability
        if hasattr(self.current_gpu_model, 'hf_device_map'):
            print(f"[model-load] hf_device_map={self.current_gpu_model.hf_device_map}")
        
        # Report actual model device placement (may show 'cpu' for dispatched models)
        try:
            model_device = next(self.current_gpu_model.parameters()).device
        except StopIteration:
            model_device = 'unknown'
        print(f"[model-load] Parameter device hint (may be 'cpu' for dispatched models): {model_device}")
        # Also report effective input device for generation
        effective_device = self._resolve_input_device()
        print(f"[model-load] Effective input device: {effective_device}")
        # Strict policy: if force_device is set, ensure we are truly on that CUDA device
        if self.force_device is not None:
            if not (isinstance(effective_device, str) and effective_device.startswith('cuda')):
                raise RuntimeError(f"Model effective device is '{effective_device}' but GPU '{self.force_device}' was requested; refusing CPU fallback")
            if str(effective_device) != str(self.force_device):
                print(f"⚠️  [model-load] Effective device '{effective_device}' does not match requested '{self.force_device}'")
        
        if self.device == 'cpu':
            print("CUDA not available, model loaded on CPU")
        elif str(model_device) == 'cpu' and self.device == 'cuda':
            print("⚠️  WARNING: Model ended up on CPU despite CUDA being available!")
            print("⚠️  This will cause 0% GPU utilization!")

        # Continuous batcher disabled by default
        self.continuous_batcher: ContinuousBatcher | None = None
        self.fast_continuous_batcher: FastContinuousBatcher | None = None

    def _resolve_input_device(self) -> str:
        """Determine the device inputs should be placed on for generate().
        If the model is dispatched with hf_device_map, prefer that mapping; otherwise fall back to force_device/self.device.
        """
        # Highest priority: explicit forced device
        if self.force_device is not None:
            return str(self.force_device)
        try:
            if hasattr(self.current_gpu_model, 'hf_device_map') and isinstance(self.current_gpu_model.hf_device_map, dict):
                dm = self.current_gpu_model.hf_device_map
                # Prefer any CUDA placement in the map
                cuda_devices = []
                for v in dm.values():
                    if isinstance(v, str) and v.startswith('cuda'):
                        cuda_devices.append(v)
                if cuda_devices:
                    # Pick the lowest-index CUDA device
                    try:
                        cuda_devices.sort(key=lambda s: int(s.split(':',1)[1]) if ':' in s else 0)
                    except Exception:
                        pass
                    return cuda_devices[0]
                # Fallback: if only CPU is found, acknowledge it
                for v in dm.values():
                    if isinstance(v, str):
                        return v
        except Exception:
            pass
        return str(self.device)

    def get_load(self) -> int:
        """Return a simple load metric for scheduling: pending + active in the active batcher."""
        if self.fast_continuous_batcher is not None:
            try:
                st = self.fast_continuous_batcher.status()
                return int(st.get('pending', 0)) + int(st.get('active', 0))
            except Exception:
                return 0
        if self.continuous_batcher is not None:
            try:
                st = self.continuous_batcher.status()
                return int(st.get('pending', 0)) + int(st.get('active', 0))
            except Exception:
                return 0
        return 0

    def enable_continuous(self, max_active: int | None = None, use_fast: bool = True):
        if use_fast:
            if self.fast_continuous_batcher is None:
                eff_dev = self._resolve_input_device()
                self.fast_continuous_batcher = FastContinuousBatcher(self.current_gpu_model, self.tokenizer, eff_dev, max_active=max_active or 5)
            return self.fast_continuous_batcher
        else:
            if self.continuous_batcher is None:
                eff_dev = self._resolve_input_device()
                self.continuous_batcher = ContinuousBatcher(self.current_gpu_model, self.tokenizer, eff_dev, max_active=max_active or 5)
            return self.continuous_batcher

    def submit_continuous(self, messages, enable_thinking, sampling_cfg, max_new_tokens, on_token, on_complete, is_check=False, forced_tokens=None):
        # If this is a verification (check) request, run the dedicated batch-check path
        # The fast continuous batcher currently ignores `is_check/forced_tokens`, so
        # dispatch checks to `run_batch_checks` which performs a single forward pass
        # and verifies the provided proof tokens.
        if is_check:
            import uuid
            # Build proof structure expected by run_batch_checks: list of {'tokens': [{'id': ...}, ...]}
            proof_obj = {'tokens': []}
            if forced_tokens is not None:
                for tid in forced_tokens:
                    proof_obj['tokens'].append({'id': int(tid)})

            # Wrap on_complete to match run_batch_checks' on_prompt_finished signature
            def _on_prompt_finished(idx, output):
                # run_batch_checks returns {'response': str, 'proof': None} for verified
                # Call original on_complete with a generated sid and the output
                sid = uuid.uuid4().hex
                resp = output.get('response', '')
                pf = output.get('proof', None)
                on_complete(sid, resp, pf)

            # run_batch_checks expects lists for prompts/enable_thinking/proofs
            prompts = [messages]
            enables = [enable_thinking]
            proofs = [proof_obj]
            # Execute verification synchronously (single-batch)
            self.run_batch_checks(prompts, enables, proofs, _on_prompt_finished)
            # Return a synthetic sid to maintain interface compatibility
            import uuid as _u
            return _u.uuid4().hex

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
        input_device = self._resolve_input_device()
        batch_input_ids = tokenized.input_ids.to(input_device)
        attention_mask = tokenized.attention_mask.to(input_device)
        
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

        print("Running batch checks...")
        print("Proofs:", proofs)
        print("Prompts:", prompts)

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
        input_device = self._resolve_input_device()
        padded_input_ids = torch.stack([
            torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id, dtype=ids.dtype)])
            for ids in full_input_ids
        ]).to(input_device)
        padded_attention_masks = torch.stack([
            torch.cat([mask, torch.zeros(max_len - len(mask), dtype=mask.dtype)])
            for mask in full_attention_masks
        ]).to(input_device)

        # Perform a single forward pass to get all logits
        with torch.no_grad():
            outputs = self.current_gpu_model(
                input_ids=padded_input_ids,
                attention_mask=padded_attention_masks,
                use_cache=False  # Single pass, cache not needed
            )
            logits = outputs.logits  # Shape: [batch_size, max_len, vocab_size]

        # Verify each generated token
        verbose = os.getenv('VERBOSE_VERIFY', '1') == '1'
        for batch_idx, (prompt_len, proof) in enumerate(zip(prompt_lengths, proofs)):
            generated_len = len(proof["tokens"])
            valid = True
            diagnostics = []
            # Check each token in the generated sequence
            for i in range(generated_len):
                # Position j predicts the token at j+1
                j = prompt_len - 1 + i
                if j >= max_len - 1:
                    # Out-of-range due to padding; break and mark invalid
                    valid = False
                    diagnostics.append({'pos': i, 'reason': 'padding_truncation'})
                    break
                current_logits = logits[batch_idx, j, :]  # Logits for next token
                topk = torch.topk(current_logits, 10)
                top_tokens = topk.indices
                top_probs = torch.nn.functional.softmax(topk.values, dim=-1)
                next_token = full_input_ids[batch_idx][j + 1].item()  # Actual next token
                in_topk = int(next_token in top_tokens)
                if not in_topk:
                    valid = False
                if verbose:
                    # Build readable diagnostics for this token
                    top_list = []
                    for rank, tid in enumerate(top_tokens.tolist()):
                        try:
                            txt = tokenizer.decode([tid], skip_special_tokens=True)
                        except Exception:
                            txt = ''
                        prob_val = float(top_probs[rank].item()) if rank < len(top_probs) else None
                        top_list.append({'rank': rank, 'id': int(tid), 'prob': prob_val, 'text': txt})
                    try:
                        next_text = tokenizer.decode([next_token], skip_special_tokens=True)
                    except Exception:
                        next_text = ''
                    diagnostics.append({'pos': i, 'next_token': int(next_token), 'next_text': next_text, 'in_topk': bool(in_topk), 'top': top_list})
                else:
                    if not in_topk:
                        diagnostics.append({'pos': i, 'next_token': int(next_token), 'in_topk': False})
                if not in_topk:
                    # stop at first mismatch
                    break

            # Prepare response based on verification
            if valid:
                # Decode the verified generated sequence
                generated_ids = full_input_ids[batch_idx][prompt_len:prompt_len + generated_len]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                out_proof = {
                    'tokens': proof['tokens'],
                    'full_sequence_length': prompt_len + generated_len,
                    'verified': True,
                }
                if diagnostics:
                    out_proof['diagnostics'] = diagnostics
                on_prompt_finished(batch_idx, {"response": response, "proof": out_proof})
            else:
                out_proof = {
                    'tokens': proof['tokens'],
                    'full_sequence_length': prompt_len + generated_len,
                    'verified': False,
                    'diagnostics': diagnostics
                }
                on_prompt_finished(batch_idx, {"response": "", "proof": out_proof})

        print(f"Batch processed in {time.time() - start_time:.2f}s")

    def get_current_model(self):
        return self.current_gpu_model

    def get_tokenizer(self, _model_name: str | None = None):
        return self.tokenizer


QWEN35_35B_A3B_MODEL_CONFIG = TransformersModelConfig(
    model_name='Qwen/Qwen3.5-35B-A3B',
    deterministic=False,
    location='gpu',
    keep_in_memory=True,
    quantized_max_memory_multiplier=1.82,
    model_kwargs={
        # INT8 quantization via bitsandbytes: ~36GB across 2x RTX 4090 (48GB total)
        # MoE: 35B total params but only 3B active per forward pass → very fast inference
        # Requires: pip install bitsandbytes accelerate
        # Released: February 24, 2026
        'quantization_config': BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",                    # Normal Float 4: better quality than standard int4
            bnb_4bit_use_double_quant=True,               # Nested quantization: extra ~0.4 bits saved
        ),
        'trust_remote_code': True,  # Load model code from HuggingFace repo (needed for new archs)
    },
    tokenizer_kwargs={},  # Qwen3.5 includes enable_thinking support in its default chat template
)

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