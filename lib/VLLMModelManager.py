"""
VLLMModelManager — launches `vllm serve` as a subprocess and calls its
OpenAI-compatible REST API (/v1/chat/completions, non-streaming).

Why subprocess instead of vllm.LLM() directly:
  - Avoids multiprocessing spawn loop (vLLM forces spawn, which re-executes uomi-ai.py)
  - Avoids vLLM Python API bugs (e.g. qwen3_5.py multimodal_config=None crash)
  - `vllm serve` is the battle-tested path that already works

Token / proof contract (MUST be preserved for validator compatibility):
  - generate call  → proof = {"tokens": [{"id": int}, ...]}
  - check call     → proof = {"tokens": [...], "verified": bool}
                             "error": "token_mismatch" when verified=False
  - Token IDs are produced by tokenizer.encode(full_response_text) called ONCE on the
    complete response string — never chunk-by-chunk — so IDs are identical between
    generate and check runs on the same text.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class VLLMModelConfig:
    model_name: str
    tensor_parallel_size: int = 2
    dtype: str = "auto"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.96
    port: int = 8100          # REST port for vllm serve (not Flask's 8888)
    hf_overrides: Optional[Dict[str, Any]] = None
    extra_serve_args: List[str] = field(default_factory=list)


QWEN35_35B_A3B_FP8_VLLM_CONFIG = VLLMModelConfig(
    model_name="Qwen/Qwen3.5-35B-A3B-FP8",
    tensor_parallel_size=2,
    dtype="auto",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    port=8100,
    # HF config.json says Qwen3_5MoeForConditionalGeneration; vLLM class is Qwen3_5MoeForCausalLM
    hf_overrides={"architectures": ["Qwen3_5MoeForCausalLM"]},
)


class VLLMModelManager:
    """
    Starts `vllm serve <model>` in a child process, waits for /health to respond,
    then proxies generation through its OpenAI-compatible REST API.

    Public interface matches TransformersModelManager:
      enable_continuous(), submit_continuous(), get_load(), get_tokenizer()
    """

    def __init__(self, model_config: VLLMModelConfig):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.device = "cuda"
        self.force_device = None   # vLLM owns device placement

        self._base_url = f"http://localhost:{model_config.port}"
        self._proc: Optional[subprocess.Popen] = None
        self._active_count = 0
        self._lock = threading.Lock()

        max_workers = int(os.getenv("VLLM_THREAD_WORKERS", "32"))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Tokenizer loaded via transformers (CPU only — used for token ID encoding)
        self._tokenizer = self._load_tokenizer()
        self._start_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _find_vllm_bin(self) -> Optional[str]:
        candidate = os.path.join(os.path.dirname(sys.executable), "vllm")
        if os.path.isfile(candidate):
            return candidate
        return shutil.which("vllm")

    def _build_cmd(self) -> List[str]:
        cfg = self.model_config
        vllm_bin = self._find_vllm_bin()
        base = [vllm_bin, "serve"] if vllm_bin else [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        cmd = base + [
            cfg.model_name,
            "--tensor-parallel-size", str(cfg.tensor_parallel_size),
            "--dtype", cfg.dtype,
            "--max-model-len", str(cfg.max_model_len),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--port", str(cfg.port),
            "--trust-remote-code",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser", os.environ.get("VLLM_TOOL_CALL_PARSER", "hermes"),
        ]
        if cfg.hf_overrides:
            cmd += ["--hf-overrides", json.dumps(cfg.hf_overrides)]
        cmd += cfg.extra_serve_args
        return cmd

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(self.model_config.model_name, trust_remote_code=True)
            print(f"[vllm-serve] Tokenizer loaded for {self.model_config.model_name}")
            return tok
        except Exception as e:
            print(f"[vllm-serve] WARNING: could not load tokenizer: {e}")
            return None

    def _start_server(self):
        cmd = self._build_cmd()
        print(f"[vllm-serve] Starting: {' '.join(cmd)}", flush=True)
        # Don't capture stdout/stderr — let vllm write directly to the journal
        # so HuggingFace download progress bars are visible in `journalctl -f`.
        self._proc = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            preexec_fn=os.setsid,
        )
        self._wait_for_ready(timeout=600)

    def _wait_for_ready(self, timeout: int = 600):
        import urllib.request, urllib.error
        url = f"{self._base_url}/health"
        deadline = time.time() + timeout
        last_log = 0.0
        print(f"[vllm-serve] Waiting for {url} (timeout={timeout}s)…")
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self._proc.returncode} before becoming ready. "
                    "Check [vllm-server] log lines above."
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        print("[vllm-serve] Server ready ✓")
                        return
            except Exception:
                pass
            if time.time() - last_log > 15:
                print(f"[vllm-serve] Still waiting… ({int(time.time()-(deadline-timeout))}s elapsed)")
                last_log = time.time()
            time.sleep(2)
        raise RuntimeError(f"vllm serve did not become ready within {timeout}s")

    def shutdown(self):
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            print("[vllm-serve] Server stopped.")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Interface expected by runner.py
    # ------------------------------------------------------------------

    def enable_continuous(self, max_active: Optional[int] = None, use_fast: bool = True):
        """No-op: vLLM server handles batching internally."""
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
        tools: Optional[List[Dict]] = None,
    ) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._active_count += 1
        self._executor.submit(
            self._run,
            sid, messages, enable_thinking, sampling_cfg,
            max_new_tokens, on_token, on_complete, is_check, forced_tokens, tools,
        )
        return sid

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _topk_ids_from_logprobs(self, top_logprobs: list) -> set:
        """
        Convert OpenAI-format top_logprobs entries to a set of token IDs.

        Each entry is {"token": str, "logprob": float, "bytes": [int, ...]}.
        Two strategies are tried in order for each entry:
          1. tokenizer.encode(token_text) — works for most standard tokens.
          2. tokenizer.convert_tokens_to_ids(token_text) — works for raw vocab tokens
             that encode() would split differently.
        """
        if self._tokenizer is None:
            return set()
        top_ids: set = set()
        for lp in top_logprobs:
            tok_text = lp.get("token", "")
            # Strategy 1: encode the decoded token text directly.
            try:
                ids = self._tokenizer.encode(tok_text, add_special_tokens=False)
                if len(ids) == 1:
                    top_ids.add(ids[0])
            except Exception:
                pass
            # Strategy 2: direct vocab lookup (handles raw BPE / SentencePiece tokens).
            try:
                tok_id = self._tokenizer.convert_tokens_to_ids(tok_text)
                if isinstance(tok_id, int) and tok_id != self._tokenizer.unk_token_id:
                    top_ids.add(tok_id)
            except Exception:
                pass
        return top_ids

    def _encode_tokens(self, text: str) -> List[int]:
        """
        Encode *full* response text in one call.
        Called identically for both generate and check runs, so IDs are consistent.
        """
        if self._tokenizer is None:
            return []
        try:
            return list(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return []

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
        tools: Optional[List[Dict]] = None,
    ):
        import urllib.request, urllib.error

        try:
            temperature = float(sampling_cfg.get("temperature", 0.7))
            top_k = int(sampling_cfg.get("top_k", 5))
            n_tokens = len(forced_tokens) if (is_check and forced_tokens) else max_new_tokens
            # Clamp to max_model_len to avoid vLLM bad_request errors
            n_tokens = min(n_tokens, self.model_config.max_model_len)

            # Qwen3.5 thinking mode is controlled via chat_template_kwargs in extra_body.
            # The /think and /no_think soft-switches are NOT supported on Qwen3.5.
            #
            # vLLM strictly requires tool_calls in assistant messages to have an `id` field.
            # Inject a stable id if missing so callers don't need to track it.
            patched_messages = []
            for m in messages:
                m = dict(m)
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    fixed_calls = []
                    for i, tc in enumerate(m["tool_calls"]):
                        tc = dict(tc)
                        if "id" not in tc:
                            fn_name = tc.get("function", {}).get("name", "fn")
                            tc["id"] = f"call_{fn_name}_{i}"
                        fixed_calls.append(tc)
                    m["tool_calls"] = fixed_calls
                patched_messages.append(m)

            topk_verify = int(os.getenv("VLLM_TOPK_VERIFY", "10"))

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": patched_messages,
                "max_tokens": n_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
            # In check mode request logprobs so we can do top-K verification
            # instead of requiring exact token match (which fails with temperature > 0).
            if is_check and forced_tokens is not None:
                payload["logprobs"] = True
                payload["top_logprobs"] = topk_verify
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    raw = resp.read()
            except urllib.error.HTTPError as http_err:
                err_body = ""
                try:
                    err_body = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                print(f"[vllm-serve] HTTP {http_err.code} from vLLM: {err_body}")
                raise

            data = json.loads(raw)
            message = data["choices"][0]["message"]
            generated_text: str = message.get("content") or ""
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # Serialize tool_calls to JSON — encoded into token IDs for the proof,
                # consistently between generate and check runs.
                tool_calls_str = json.dumps(tool_calls, ensure_ascii=False)
                generated_text = (generated_text + "\n" + tool_calls_str).strip() if generated_text else tool_calls_str

            # Encode the FULL response text in one shot — never chunk-by-chunk.
            # This guarantees identical token IDs between generate and check runs.
            generated_ids = self._encode_tokens(generated_text)

            # Fire on_token for each token (used only for debug logging in runner.py)
            for i, tid in enumerate(generated_ids):
                try:
                    tok_text = self._tokenizer.decode([tid], skip_special_tokens=True) if self._tokenizer else ""
                    on_token(sid, tok_text, {"id": int(tid), "index": i})
                except Exception:
                    pass

            # Build proof
            if is_check and forced_tokens is not None:
                lp_content = (data["choices"][0].get("logprobs") or {}).get("content") or []
                forced_list = list(forced_tokens)

                if not lp_content or self._tokenizer is None:
                    # No logprobs available: fall back to exact match
                    verified = generated_ids == forced_list
                    print(f"[verify-log] exact-match fallback: verified={verified}")
                elif len(lp_content) != len(forced_list):
                    verified = False
                    print(f"[verify-log] length mismatch: got={len(lp_content)} expected={len(forced_list)}")
                else:
                    verified = True
                    for i, (forced_id, lp_pos) in enumerate(zip(forced_list, lp_content)):
                        top_ids = self._topk_ids_from_logprobs(lp_pos.get("top_logprobs") or [])
                        if int(forced_id) not in top_ids:
                            verified = False
                            print(f"[verify-log] position {i}: forced_id={forced_id} not in top-{topk_verify} ids={top_ids}")
                            break
                    if verified:
                        print(f"[verify-log] top-{topk_verify} verification passed ({len(forced_list)} tokens)")

                proof: Dict[str, Any] = {
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
            print(f"[vllm-serve] _run error (sid={sid[:6]}): {exc}")
            on_complete(sid, "", {"tokens": [], "error": str(exc)})
        finally:
            with self._lock:
                self._active_count -= 1
