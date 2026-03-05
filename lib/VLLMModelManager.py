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
import urllib.error
import urllib.request
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

    def _topk_ids_from_logprobs_dict(self, top_lp_dict: Dict[str, float]) -> set:
        """
        Convert a completions-API top_logprobs dict {token_text: logprob} to token IDs.
        Tries encode() first (handles decoded text), then convert_tokens_to_ids()
        (handles raw vocab tokens like 'Ġhello' or byte-level tokens).
        """
        if self._tokenizer is None:
            return set()
        top_ids: set = set()
        for tok_text in top_lp_dict.keys():
            try:
                ids = self._tokenizer.encode(tok_text, add_special_tokens=False)
                if len(ids) == 1:
                    top_ids.add(ids[0])
            except Exception:
                pass
            try:
                tid = self._tokenizer.convert_tokens_to_ids(tok_text)
                if isinstance(tid, int):
                    top_ids.add(tid)
            except Exception:
                pass
        return top_ids

    def _verify_topk(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        forced_tokens: List[int],
        topk_verify: int,
    ) -> Optional[bool]:
        """
        Verify that each forced token was within the top-K candidates at its position,
        using /v1/completions with echo=True.

        This is the only correct approach for top-K verification with temperature > 0:
        the logprob at position i is computed given the CORRECT prefix
        (forced_tokens[0..i-1]), not a freely-generated diverged prefix.

        Returns True (pass), False (fail), or None (cannot verify — caller decides).
        """
        if self._tokenizer is None:
            print("[verify-log] _verify_topk: no tokenizer, cannot verify")
            return None
        try:
            # Build token ID list: apply_chat_template + forced response tokens.
            # Passing token IDs directly to /v1/completions avoids any text
            # boundary re-tokenization artefacts.
            try:
                prompt_ids: List[int] = list(self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ))
            except TypeError:
                prompt_ids = list(self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                ))

            full_ids: List[int] = prompt_ids + list(forced_tokens)
            n_prompt = len(prompt_ids)
            n_resp = len(forced_tokens)

            print(f"[verify-log] completions echo: n_prompt={n_prompt} n_resp={n_resp} topk={topk_verify}")

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "prompt": full_ids,      # token IDs → no tokenization boundary issues
                "max_tokens": 1,         # 1 extra token so vLLM returns echo logprobs
                "logprobs": topk_verify,
                "echo": True,
                "temperature": 0,
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base_url}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                comp_data = json.loads(resp.read())

            lp_data = comp_data["choices"][0].get("logprobs") or {}
            # completions logprobs: {"tokens": [...], "top_logprobs": [{tok: logp}, ...]}
            all_top_lp: List[Dict] = lp_data.get("top_logprobs") or []
            resp_lp = all_top_lp[n_prompt: n_prompt + n_resp]

            print(f"[verify-log] logprobs returned: total={len(all_top_lp)} response_slice={len(resp_lp)}")

            if len(resp_lp) < n_resp:
                print(f"[verify-log] FAIL: logprobs too short ({len(resp_lp)} < {n_resp})")
                return False

            for i, (forced_id, top_lp_dict) in enumerate(zip(forced_tokens, resp_lp)):
                top_ids = self._topk_ids_from_logprobs_dict(top_lp_dict or {})
                if int(forced_id) not in top_ids:
                    forced_text = ""
                    try:
                        forced_text = self._tokenizer.decode([int(forced_id)], skip_special_tokens=True)
                    except Exception:
                        pass
                    top_sample = list(top_lp_dict.keys())[:5] if top_lp_dict else []
                    print(f"[verify-log] FAIL pos {i}: forced_id={forced_id} ('{forced_text}') "
                          f"not in top-{topk_verify}. Top candidates: {top_sample}")
                    return False

            print(f"[verify-log] PASS: all {n_resp} tokens verified in top-{topk_verify}")
            return True

        except Exception as exc:
            print(f"[verify-log] _verify_topk error: {exc}")
            return None

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
        try:
            # ------------------------------------------------------------------
            # CHECK PATH: top-K verification via /v1/completions echo=True.
            # No re-generation: evaluate logprobs of the forced tokens with the
            # correct prefix context at every position.
            # ------------------------------------------------------------------
            if is_check and forced_tokens is not None:
                topk_verify = int(os.getenv("VLLM_TOPK_VERIFY", "10"))
                verified_opt = self._verify_topk(messages, enable_thinking, list(forced_tokens), topk_verify)
                verified = bool(verified_opt) if verified_opt is not None else False
                if verified_opt is None:
                    print("[verify-log] top-K unavailable (no tokenizer or API error), marking failed")

                response_text = self._tokenizer.decode(list(forced_tokens), skip_special_tokens=True) if self._tokenizer else ""
                generated_ids = list(forced_tokens)

                for i, tid in enumerate(generated_ids):
                    try:
                        tok_text = self._tokenizer.decode([tid], skip_special_tokens=True) if self._tokenizer else ""
                        on_token(sid, tok_text, {"id": int(tid), "index": i})
                    except Exception:
                        pass

                proof_obj: Dict[str, Any] = {"tokens": [{"id": int(t)} for t in generated_ids], "verified": verified}
                if not verified:
                    proof_obj["error"] = "token_mismatch"
                on_complete(sid, response_text, proof_obj)
                return

            # ------------------------------------------------------------------
            # GENERATE PATH
            # ------------------------------------------------------------------
            temperature = float(sampling_cfg.get("temperature", 0.7))
            top_k = int(sampling_cfg.get("top_k", 5))
            n_tokens = min(max_new_tokens, self.model_config.max_model_len)

            # vLLM strictly requires tool_calls in assistant messages to have an `id` field.
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

            payload: Dict[str, Any] = {
                "model": self.model_name,
                "messages": patched_messages,
                "max_tokens": n_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
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
                tool_calls_str = json.dumps(tool_calls, ensure_ascii=False)
                generated_text = (generated_text + "\n" + tool_calls_str).strip() if generated_text else tool_calls_str

            # Encode the FULL response text in one shot — never chunk-by-chunk.
            generated_ids = self._encode_tokens(generated_text)

            # Fire on_token for each token (used only for debug logging in runner.py)
            for i, tid in enumerate(generated_ids):
                try:
                    tok_text = self._tokenizer.decode([tid], skip_special_tokens=True) if self._tokenizer else ""
                    on_token(sid, tok_text, {"id": int(tid), "index": i})
                except Exception:
                    pass

            proof: Dict[str, Any] = {
                "tokens": [{"id": int(t)} for t in generated_ids],
            }

            on_complete(sid, generated_text, proof)

        except Exception as exc:
            print(f"[vllm-serve] _run error (sid={sid[:6]}): {exc}")
            on_complete(sid, "", {"tokens": [], "error": str(exc)})
        finally:
            with self._lock:
                self._active_count -= 1
