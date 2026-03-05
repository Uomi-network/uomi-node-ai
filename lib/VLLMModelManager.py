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

import hashlib
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
    # To enable FP8 KV cache (doubles effective batch size, requires vLLM >= 0.4.3):
    #   extra_serve_args=["--kv-cache-dtype", "fp8"],
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
        proof_prompt_hash: Optional[str] = None,
    ) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._active_count += 1
        self._executor.submit(
            self._run,
            sid, messages, enable_thinking, sampling_cfg,
            max_new_tokens, on_token, on_complete, is_check, forced_tokens, tools, proof_prompt_hash,
        )
        return sid

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _get_stop_token_ids(self) -> List[int]:
        """Return EOS and common chat-template end tokens from the tokenizer."""
        if self._tokenizer is None:
            return []
        stop_ids: List[int] = []
        if self._tokenizer.eos_token_id is not None:
            stop_ids.append(int(self._tokenizer.eos_token_id))
        for special in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            tid = self._tokenizer.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid not in stop_ids:
                stop_ids.append(tid)
        return stop_ids

    def _compute_prompt_hash(self, messages: List[Dict], enable_thinking: bool) -> str:
        """SHA-256 (32 hex chars) of the chat-template token ID sequence."""
        if self._tokenizer is None:
            return ""
        try:
            try:
                ids = list(self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ))
            except TypeError:
                ids = list(self._tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                ))
            # Encode each token ID as 4 LE bytes for a deterministic byte sequence.
            raw = b"".join(id_.to_bytes(4, "little") for id_ in ids)
            return hashlib.sha256(raw).hexdigest()[:32]
        except Exception as exc:
            print(f"[verify-log] _compute_prompt_hash error: {exc}")
            return ""

    def _verify_topk(
        self,
        messages: List[Dict],
        enable_thinking: bool,
        forced_tokens: List[int],
        topk_verify: int,
        proof_prompt_hash: Optional[str] = None,
    ) -> Optional[bool]:
        """
        Verify each forced token is within the top-K candidates at its position.

        Three layered checks:
          1. Top-K per-token:  actual_logprob >= K-th best logprob at each position
          2. EOS completeness: last token must be an EOS/stop token (rejects truncated
             or padded-with-extra-tokens proofs)
          3. Mean logprob:     average logprob must exceed VLLM_MIN_MEAN_LOGPROB
             (rejects sequences that happen to be in top-K but are incoherent
             because they come from a completely different prompt context)

        Uses /v1/completions with echo=True and token IDs as prompt.
        Comparison is purely logprob-based — no token text-to-ID conversion needed.

        Returns True (pass), False (fail), or None (cannot verify — caller decides).
        """
        if self._tokenizer is None:
            print("[verify-log] _verify_topk: no tokenizer")
            return None

        if not forced_tokens:
            print("[verify-log] FAIL: empty forced_tokens")
            return False

        # --- Check 0: prompt hash — proof must commit to the exact prompt ---
        if proof_prompt_hash:
            actual_hash = self._compute_prompt_hash(messages, enable_thinking)
            if actual_hash and actual_hash != proof_prompt_hash:
                print(f"[verify-log] FAIL: prompt_hash mismatch "
                      f"expected={proof_prompt_hash} got={actual_hash}")
                return False

        # --- Check 2: EOS completeness ---
        stop_ids = self._get_stop_token_ids()
        if stop_ids and forced_tokens[-1] not in stop_ids:
            last_text = ""
            try:
                last_text = self._tokenizer.decode([int(forced_tokens[-1])], skip_special_tokens=False)
            except Exception:
                pass
            print(f"[verify-log] FAIL: last token id={forced_tokens[-1]} ('{last_text}') "
                  f"is not a stop token (stop_ids={stop_ids[:5]})")
            return False

        min_mean_logprob = float(os.getenv("VLLM_MIN_MEAN_LOGPROB", "-3.0"))

        try:
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
                "prompt": full_ids,      # token IDs — no text roundtrip
                "max_tokens": 1,         # 1 extra token so vLLM echoes the full sequence
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
            # token_logprobs[i] = log P(token_i | token_0..i-1)
            # top_logprobs[i]   = {token_text: logprob} for top-K at position i
            token_logprobs: List = lp_data.get("token_logprobs") or []
            all_top_lp: List[Dict] = lp_data.get("top_logprobs") or []

            resp_actual_lp = token_logprobs[n_prompt: n_prompt + n_resp]
            resp_top_lp    = all_top_lp[n_prompt: n_prompt + n_resp]

            print(f"[verify-log] logprobs: total={len(token_logprobs)} "
                  f"resp_actual={len(resp_actual_lp)} resp_top={len(resp_top_lp)}")

            if len(resp_actual_lp) < n_resp or len(resp_top_lp) < n_resp:
                print(f"[verify-log] FAIL: logprobs slice too short ({len(resp_actual_lp)} < {n_resp})")
                return False

            # --- Check 1: top-K per-token ---
            valid_lps: List[float] = []
            for i, (forced_id, actual_lp, top_lp_dict) in enumerate(
                    zip(forced_tokens, resp_actual_lp, resp_top_lp)):
                if actual_lp is None:
                    # First token in a sequence has no preceding context logprob
                    continue
                if not top_lp_dict:
                    print(f"[verify-log] FAIL pos {i}: empty top_logprobs entry")
                    return False
                # K-th best logprob = minimum value in the top-K dict.
                # Token passes iff its actual logprob >= K-th best logprob.
                kth_logprob = min(top_lp_dict.values())
                if actual_lp < kth_logprob:
                    forced_text = ""
                    try:
                        forced_text = self._tokenizer.decode([int(forced_id)], skip_special_tokens=True)
                    except Exception:
                        pass
                    top_sample = list(top_lp_dict.keys())[:5]
                    print(f"[verify-log] FAIL pos {i}: id={forced_id} ('{forced_text}') "
                          f"logprob={actual_lp:.4f} < kth={kth_logprob:.4f} "
                          f"top_tokens={top_sample}")
                    return False
                valid_lps.append(actual_lp)

            # --- Check 3: mean logprob threshold ---
            if valid_lps:
                mean_lp = sum(valid_lps) / len(valid_lps)
                print(f"[verify-log] mean logprob={mean_lp:.4f} threshold={min_mean_logprob}")
                if mean_lp < min_mean_logprob:
                    print(f"[verify-log] FAIL: mean logprob {mean_lp:.4f} < threshold {min_mean_logprob}")
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
        proof_prompt_hash: Optional[str] = None,
    ):
        try:
            # ------------------------------------------------------------------
            # CHECK PATH: top-K verification via /v1/completions echo=True.
            # No re-generation: evaluate logprobs of the forced tokens with the
            # correct prefix context at every position.
            # ------------------------------------------------------------------
            if is_check and forced_tokens is not None:
                topk_verify = int(os.getenv("VLLM_TOPK_VERIFY", "5"))
                verified_opt = self._verify_topk(messages, enable_thinking, list(forced_tokens), topk_verify, proof_prompt_hash)
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

            # Append EOS/stop token so the proof encodes sequence completeness.
            # Without this, truncated proofs would be indistinguishable from full ones.
            stop_token_ids = self._get_stop_token_ids()
            if stop_token_ids:
                generated_ids = generated_ids + [stop_token_ids[0]]

            # Fire on_token for each token (used only for debug logging in runner.py)
            for i, tid in enumerate(generated_ids):
                try:
                    tok_text = self._tokenizer.decode([tid], skip_special_tokens=True) if self._tokenizer else ""
                    on_token(sid, tok_text, {"id": int(tid), "index": i})
                except Exception:
                    pass

            proof: Dict[str, Any] = {
                "tokens": [{"id": int(t)} for t in generated_ids],
                "prompt_hash": self._compute_prompt_hash(messages, enable_thinking),
            }

            on_complete(sid, generated_text, proof)

        except Exception as exc:
            print(f"[vllm-serve] _run error (sid={sid[:6]}): {exc}")
            on_complete(sid, "", {"tokens": [], "error": str(exc)})
        finally:
            with self._lock:
                self._active_count -= 1
