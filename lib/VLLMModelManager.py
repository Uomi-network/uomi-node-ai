"""
VLLMModelManager — drop-in replacement for TransformersModelManager for FP8 / large models.

vLLM handles:
  - tensor parallelism across GPUs natively (no staging-buffer OOM)
  - FP8 weight loading with hardware kernels on RTX 4090 (Ada Lovelace)
  - continuous batching internally

Public interface mirrors TransformersModelManager so runner.py needs only minimal changes.
"""

import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class VLLMModelConfig:
    model_name: str
    tensor_parallel_size: int = 2
    dtype: str = "auto"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.88
    # Extra kwargs forwarded to vllm.LLM()
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


QWEN35_35B_A3B_FP8_VLLM_CONFIG = VLLMModelConfig(
    model_name="Qwen/Qwen3.5-35B-A3B-FP8",
    tensor_parallel_size=2,
    dtype="auto",
    max_model_len=8192,
    gpu_memory_utilization=0.88,
)


class VLLMModelManager:
    """
    Wraps vllm.LLM to expose the same interface used by runner.py:
      - enable_continuous()
      - submit_continuous()
      - get_load()
      - get_tokenizer()
    """

    def __init__(self, model_config: VLLMModelConfig):
        from vllm import LLM  # type: ignore

        self.model_config = model_config
        self.model_name = model_config.model_name
        self.device = "cuda"
        self.force_device = None  # vLLM owns device placement via tensor_parallel_size

        print(f"[vllm] Loading {model_config.model_name} "
              f"(tensor_parallel_size={model_config.tensor_parallel_size}, "
              f"dtype={model_config.dtype}, "
              f"max_model_len={model_config.max_model_len}, "
              f"gpu_memory_utilization={model_config.gpu_memory_utilization})")

        self.llm = LLM(
            model=model_config.model_name,
            tensor_parallel_size=model_config.tensor_parallel_size,
            dtype=model_config.dtype,
            max_model_len=model_config.max_model_len,
            gpu_memory_utilization=model_config.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
            **model_config.extra_kwargs,
        )
        self._tokenizer = self.llm.get_tokenizer()
        print(f"[vllm] Model loaded successfully")

        self._active_count = 0
        self._lock = threading.Lock()
        # Thread pool for concurrent request handling.
        # vLLM serialises generate() calls internally so many workers is fine.
        max_workers = int(os.getenv("VLLM_THREAD_WORKERS", "32"))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    # ------------------------------------------------------------------
    # Interface expected by runner.py
    # ------------------------------------------------------------------

    def enable_continuous(self, max_active: Optional[int] = None, use_fast: bool = True):
        """No-op: vLLM handles concurrency and batching internally."""
        pass

    def get_load(self) -> int:
        with self._lock:
            return self._active_count

    def get_tokenizer(self, _model_name: Optional[str] = None):
        return self._tokenizer

    def submit_continuous(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        sampling_cfg: Dict,
        max_new_tokens: int,
        on_token: Callable,
        on_complete: Callable,
        is_check: bool = False,
        forced_tokens: Optional[List[int]] = None,
    ) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._active_count += 1
        self._executor.submit(
            self._run,
            sid, messages, enable_thinking, sampling_cfg,
            max_new_tokens, on_token, on_complete, is_check, forced_tokens,
        )
        return sid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_chat_template(self, messages, enable_thinking: bool) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Tokenizer doesn't support enable_thinking kwarg
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _run(
        self,
        sid: str,
        messages: List[Dict],
        enable_thinking: bool,
        sampling_cfg: Dict,
        max_new_tokens: int,
        on_token: Callable,
        on_complete: Callable,
        is_check: bool,
        forced_tokens: Optional[List[int]],
    ):
        from vllm import SamplingParams  # type: ignore

        try:
            prompt = self._apply_chat_template(messages, enable_thinking)

            temperature = float(sampling_cfg.get("temperature", 0.7))
            top_k = int(sampling_cfg.get("top_k", 5))

            if is_check and forced_tokens:
                # Verification: re-generate with same length and compare token ids
                n_tokens = len(forced_tokens)
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=n_tokens,
                    logprobs=0,
                )
            else:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_k=top_k,
                    max_tokens=max_new_tokens,
                )

            outputs = self.llm.generate(prompt, sampling_params)
            result = outputs[0].outputs[0]
            generated_text: str = result.text
            generated_ids: List[int] = list(result.token_ids)

            # Fire on_token once per generated token (mirrors streaming behaviour for callers
            # that log per-token — timing is approximate since vLLM returns all at once)
            for i, tid in enumerate(generated_ids):
                try:
                    tok_text = self._tokenizer.decode([tid], skip_special_tokens=True)
                    on_token(sid, tok_text, {"id": int(tid), "index": i})
                except Exception:
                    pass

            if is_check and forced_tokens:
                verified = generated_ids == list(forced_tokens)
                proof = {
                    "tokens": [{"id": int(t)} for t in generated_ids],
                    "verified": verified,
                }
                if not verified:
                    proof["error"] = "token_mismatch"
            else:
                proof = {
                    "tokens": [{"id": int(t)} for t in generated_ids],
                }

            on_complete(sid, generated_text, proof)

        except Exception as exc:
            print(f"[vllm] _run error (sid={sid[:6]}): {exc}")
            on_complete(sid, "", {"tokens": [], "error": str(exc)})
        finally:
            with self._lock:
                self._active_count -= 1
