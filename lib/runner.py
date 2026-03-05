import time
import threading
import uuid
import os
from lib.config import BATCH_WAIT_SEC, BATCH_MAX_SIZE, TRANSFORMERS_INFERENCE_MAX_TOKENS
from lib.executors import ChatExecutor, ImageExecutor
from lib.TestModelManager import TEST_MODEL_CONFIG, TestModelManager
from lib.TransformersModelManager import QWEN35_35B_A3B_MODEL_CONFIG, QWEN35_35B_A3B_FP8_MODEL_CONFIG, TransformersModelManager
from lib.VLLMModelManager import QWEN35_35B_A3B_FP8_VLLM_CONFIG, VLLMModelManager

# Active model config:
#   QWEN35_35B_A3B_FP8_MODEL_CONFIG  → FP8 weights (37.5 GB), fits 2x RTX 4090 natively, RECOMMENDED
#   QWEN35_35B_A3B_MODEL_CONFIG      → BF16 + BnB int8 (needs bitsandbytes with matching CUDA binary)
def _select_active_model_config():
    # Default: FP8 — native 8-bit weights, no bitsandbytes dependency, RTX 4090 has HW FP8 support.
    # The BnB path requires a bitsandbytes build whose CUDA binary matches the runtime exactly;
    # this is fragile across CUDA 12.x versions and dev builds.  FP8 avoids this entirely.
    variant = os.getenv("QWEN_MODEL_VARIANT", "fp8").strip().lower()
    if variant in {"4bit", "bnb", "nf4", "8bit", "int8", "qwen3.5-35b-a3b"}:
        return QWEN35_35B_A3B_MODEL_CONFIG
    return QWEN35_35B_A3B_FP8_MODEL_CONFIG


ACTIVE_MODEL_CONFIG = _select_active_model_config()
print(f"[runner] ACTIVE_MODEL_CONFIG={ACTIVE_MODEL_CONFIG.model_name} (QWEN_MODEL_VARIANT={os.getenv('QWEN_MODEL_VARIANT', 'fp8')})")
import torch
# from lib.SanaModelManager import SANA_MODEL_CONFIG, SanaModelManager

class RunnerQueue:
    def __init__(self):
        print('Initialize RunnerQueue')
        self.queue = {}
        self.lock = threading.Lock()

    def add_request(self, request):
        with self.lock:
            request_uuid = uuid.uuid4()
            self.queue[request_uuid] = {
                "status": "pending",
                "timestamp_pending": time.time(),
                "timestamp_running": None,
                "timestamp_finished": None,
                "uuid": request_uuid,
                "request": request,
                "output": None,
                "batch": None
            }
            return request_uuid
        
    def get_request(self, request_uuid):
        with self.lock:
            return self.queue.get(request_uuid)  # Use .get() to avoid KeyError

    def remove_request(self, request_uuid):
        with self.lock:
            if request_uuid in self.queue:
                del self.queue[request_uuid]
        
    def get_requests(self):
        with self.lock:
            # Return a copy to avoid external modification
            return dict(self.queue)
        
class RunnerExecutor:
    def __init__(self, queue, test_mode=False):
        print('Initialize RunnerExecutor')
        self.kill = False
        self.queue = queue
        self.active_model_name = ACTIVE_MODEL_CONFIG.model_name
        # Micro-batching accumulation window in milliseconds (set 0 to disable)
        self.microbatch_window_ms = 30
        self.test_model_manager = TestModelManager(TEST_MODEL_CONFIG)
        if test_mode:
            self.transformers_model_managers = []
            self.sana_model_manager = None
        else:
            # Create one independent replica of the model per available GPU and balance between them
            self.transformers_model_managers = []
            use_fast = os.getenv("FAST_CONTINUOUS_BATCHER", "1") == "1"
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                # Prefer GPUs with most free memory to minimize initial OOM risk
                try:
                    mem_stats = []
                    for gid in range(gpu_count):
                        stats = torch.cuda.memory_stats(gid)
                        # use total_reserved - allocated as free within PyTorch; not exact but indicative
                        reserved = int(stats.get('reserved_bytes.all.current', 0))
                        allocated = int(stats.get('allocated_bytes.all.current', 0))
                        free_est = max(0, reserved - allocated)
                        mem_stats.append((gid, free_est))
                    target_gpus = [gid for gid,_ in sorted(mem_stats, key=lambda x: x[1], reverse=True)]
                except Exception:
                    target_gpus = list(range(gpu_count))
                # Optional override to limit number of replicas
                max_replicas = int(os.getenv("MAX_GPU_REPLICAS", "0") or "0")
                if max_replicas > 0:
                    target_gpus = target_gpus[:max_replicas]
                if ACTIVE_MODEL_CONFIG is QWEN35_35B_A3B_FP8_MODEL_CONFIG:
                    # FP8 path: use vLLM which handles tensor parallelism and FP8 natively.
                    # Falls back to TransformersModelManager if vLLM is not installed.
                    print(f"🔧 Spawning vLLM instance of {QWEN35_35B_A3B_FP8_VLLM_CONFIG.model_name} "
                          f"(tensor_parallel_size={QWEN35_35B_A3B_FP8_VLLM_CONFIG.tensor_parallel_size})")
                    last_error = None
                    try:
                        tm = VLLMModelManager(QWEN35_35B_A3B_FP8_VLLM_CONFIG)
                        tm.enable_continuous(max_active=BATCH_MAX_SIZE, use_fast=use_fast)
                        self.transformers_model_managers.append(tm)
                        self.active_model_name = QWEN35_35B_A3B_FP8_VLLM_CONFIG.model_name
                    except Exception as e:
                        last_error = e
                        print(f"❌ vLLM load failed: {e}")
                        # Fallback to transformers if explicitly allowed
                        if os.getenv("QWEN_FALLBACK_ON_LOAD_FAILURE", "0") == "1":
                            fallback_configs = [QWEN35_35B_A3B_FP8_MODEL_CONFIG, QWEN35_35B_A3B_MODEL_CONFIG]
                            for cfg in fallback_configs:
                                print(f"⚠️  Trying transformers fallback: {cfg.model_name}")
                                try:
                                    tm = TransformersModelManager(cfg)
                                    tm.enable_continuous(max_active=BATCH_MAX_SIZE, use_fast=use_fast)
                                    self.transformers_model_managers.append(tm)
                                    self.active_model_name = cfg.model_name
                                    last_error = None
                                    break
                                except Exception as fe:
                                    last_error = fe
                                    print(f"❌ Transformers fallback failed ({cfg.model_name}): {fe}")
                    if not self.transformers_model_managers and last_error is not None:
                        print(f"❌ All model load attempts failed. Last error: {last_error}")
                elif ACTIVE_MODEL_CONFIG is QWEN35_35B_A3B_MODEL_CONFIG:
                    # BnB 8-bit path: single instance distributed across all GPUs (~35GB across 2x 4090).
                    self.active_model_name = ACTIVE_MODEL_CONFIG.model_name
                    print(f"🔧 Spawning single BnB 8-bit instance across {len(target_gpus)} GPU(s)")
                    try:
                        tm = TransformersModelManager(ACTIVE_MODEL_CONFIG)
                        tm.enable_continuous(max_active=BATCH_MAX_SIZE, use_fast=use_fast)
                        self.transformers_model_managers.append(tm)
                    except Exception as e:
                        print(f"❌ Failed to spawn BnB multi-GPU instance: {e}")
                else:
                    # Single-GPU model: one replica per GPU
                    for gid in target_gpus:
                        dev = f"cuda:{gid}"
                        print(f"🔧 Spawning model replica on {dev}")
                        try:
                            tm = TransformersModelManager(ACTIVE_MODEL_CONFIG, force_device=dev)
                            tm.enable_continuous(max_active=BATCH_MAX_SIZE, use_fast=use_fast)
                            self.transformers_model_managers.append(tm)
                        except Exception as e:
                            print(f"❌ Failed to spawn replica on {dev}: {e}")
                            continue
                if not self.transformers_model_managers:
                    raise RuntimeError("Failed to spawn any GPU replicas; check GPU memory and environment")
                print(f"🚀 {'Fast' if use_fast else 'Legacy'} continuous batcher enabled on {len(self.transformers_model_managers)} GPU(s)")
            else:
                # Explicitly avoid CPU fallback per requirements; raise if no CUDA
                raise RuntimeError("CUDA is required; CPU inference is not allowed by configuration")
            # self.sana_model_manager = SanaModelManager(SANA_MODEL_CONFIG)
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.start)
        self.thread.start()

    def stop(self):
        print('Stop RunnerExecutor')
        self.kill = True
        self.thread.join()
        

    def start(self):
        print('Start RunnerExecutor')
        chat_executor = ChatExecutor()
        image_executor = ImageExecutor()
        while not self.kill:
            # Poll pending requests and immediately dispatch individually to appropriate backend
            dispatched = False
            with self.lock:
                pending = [req for req in self.queue.get_requests().values() if req["status"] == "pending"]
            if not pending:
                time.sleep(BATCH_WAIT_SEC)
                continue
            for req in sorted(pending, key=lambda r: r["timestamp_pending"]):
                model = req["request"].get("model")
                request_id = req['request'].get('request_id', 'unknown')
                if model not in TEST_MODEL_CONFIG and model != self.active_model_name:
                    model = self.active_model_name
                    req["request"]["model"] = model
                is_check = "proof" in req["request"]
                # Mark running
                with self.lock:
                    if req["status"] != "pending":
                        continue
                    req["status"] = "running"
                    req["timestamp_running"] = time.time()
                try:
                    if model in TEST_MODEL_CONFIG:
                        # Use legacy batch path but one-at-a-time
                        def on_finished(_idx, output, rq=req):
                            with self.lock:
                                rq["status"] = "finished"
                                rq["timestamp_finished"] = time.time()
                                rq["output"] = output
                        if is_check:
                            ChatExecutor().check([req["request"]["input"]],[req["request"]["proof"]], self.test_model_manager, on_finished)
                        else:
                            ChatExecutor().execute([req["request"]["input"]], self.test_model_manager, on_finished)
                    elif model == self.active_model_name and self.transformers_model_managers:
                        # Continuous submission
                        print(f"🟢 Dispatching transformers request {req['uuid']} {request_id}")
                        input_json = req["request"]["input"]
                        import json
                        payload = json.loads(input_json)
                        messages = payload["messages"]
                        tools = payload.get("tools") or None
                        enable_thinking = payload.get("enable_thinking", req["request"].get("enable_thinking", True))
                        # Allow optional per-request sampling / max tokens overrides
                        sampling_cfg = payload.get("sampling", {"temperature":0.7, "top_k":5})
                        # Determine max_new_tokens with safe cap.
                        # Default is intentionally conservative to avoid minute-long generations
                        # when callers omit max_new_tokens.
                        req_max_new = req["request"].get("max_new_tokens") or payload.get("max_new_tokens")
                        try:
                            max_new_tokens = int(req_max_new) if req_max_new is not None else int(os.getenv("MAX_NEW_TOKENS", "256"))
                        except Exception:
                            max_new_tokens = 256
                        max_new_tokens = max(1, min(max_new_tokens, TRANSFORMERS_INFERENCE_MAX_TOKENS))  # hard cap to protect CPU/GPU
                        print(f"[request] request_id={request_id} max_new_tokens={max_new_tokens} enable_thinking={enable_thinking}")
                        if is_check:
                            # unzip proof
                            from lib.zipper import unzip_string
                            try:
                                proof_obj = json.loads(unzip_string(req["request"]["proof"]))
                            except Exception as e:
                                print(f"[verify-log] failed to unzip/parse proof: {e} request_id = {request_id}")
                                proof_obj = {"tokens": []}
                            # Server-side diagnostic: print received proof token ids and decoded tokens
                            try:
                                tokenizer = None
                                if self.transformers_model_managers:
                                    tokenizer = self.transformers_model_managers[0].get_tokenizer(None)
                                token_ids = [t.get('id') for t in proof_obj.get('tokens', [])]
                                decoded_tokens = []
                                if tokenizer is not None and token_ids:
                                    for tid in token_ids:
                                        try:
                                            decoded_tokens.append(tokenizer.decode([int(tid)], skip_special_tokens=True))
                                        except Exception:
                                            decoded_tokens.append('')
                                print(f"[verify-log] received proof token_ids={token_ids} for request_id = {request_id}")
                                print(f"[verify-log] received proof decoded_tokens={decoded_tokens} for request_id = {request_id}")
                                # Also print the actual prompt text the server will verify against
                                if tokenizer is not None:
                                    try:
                                        prompt_preview = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
                                        print(f"[verify-log] prompt_preview='{prompt_preview[:400]} ...' for request_id = {request_id}")
                                    except Exception as _:
                                        pass
                            except Exception as e:
                                print(f"[verify-log] error while logging proof diagnostics: {e} for request_id = {request_id}")
                            forced_ids = [t["id"] for t in proof_obj["tokens"]]
                            proof_prompt_hash = proof_obj.get("prompt_hash", "")
                            # In check mode, limit generation exactly to proof length
                            max_new_tokens = len(forced_ids)
                        else:
                            forced_ids = None
                            proof_prompt_hash = None
                        def on_token(sid, txt, meta, rq=req):
                            if os.getenv('CONTINUOUS_DEBUG','0') == '1':
                                print(f"[stream] req={rq['uuid']} sid={sid[:6]} token={meta.get('id')} txt='{txt}' for request_id = {request_id}")
                        def on_complete(sid, response, proof, rq=req):
                            from lib.zipper import zip_string
                            import json as _j
                            wrapped_proof = ""
                            result_flag = True
                            result_error = None
                            try:
                                if proof is not None:
                                    # Ensure proof contains verification flag when available
                                    if isinstance(proof, dict) and 'verified' in proof:
                                        result_flag = bool(proof.get('verified'))
                                    if isinstance(proof, dict) and proof.get('error'):
                                        result_flag = False
                                        result_error = str(proof.get('error'))
                                    wrapped_proof = zip_string(_j.dumps(proof))
                                else:
                                    wrapped_proof = ""
                            except Exception:
                                # On any error while handling proof, fall back to True
                                wrapped_proof = zip_string(_j.dumps(proof)) if proof is not None else ""
                            with self.lock:
                                rq["status"] = "finished"
                                rq["timestamp_finished"] = time.time()
                                # Provide an error field when verification failed so callers
                                # can safely reference output['error'] without KeyError
                                if result_flag:
                                    rq["output"] = {"result": True, "response": response, "proof": wrapped_proof}
                                else:
                                    rq["output"] = {"result": False, "response": response, "proof": wrapped_proof, "error": result_error or "verification_failed"}
                            if os.getenv('CONTINUOUS_DEBUG','0') == '1':
                                print(f"[complete] req={rq['uuid']} sid={sid[:6]} tokens={len(proof['tokens']) if proof else 0}")
                        # Pick the least loaded replica and submit
                        target_tm = self._pick_transformers_manager()
                        target_tm.submit_continuous(messages, enable_thinking, sampling_cfg, max_new_tokens, on_token, on_complete, is_check=is_check, forced_tokens=forced_ids, tools=tools, proof_prompt_hash=proof_prompt_hash if is_check else None)
                    # elif model in SANA_MODEL_CONFIG and self.sana_model_manager is not None:
                    #     def on_finished(_idx, output, rq=req):
                    #         with self.lock:
                    #             rq["status"] = "finished"
                    #             rq["timestamp_finished"] = time.time()
                    #             rq["output"] = output
                    #     ImageExecutor().execute([req["request"]["input"]], self.sana_model_manager, on_finished)
                    else:
                        with self.lock:
                            req["status"] = "finished"
                            req["timestamp_finished"] = time.time()
                            req["output"] = {"result": False, "error": "Model not valid"}
                    dispatched = True
                except Exception as e:
                    with self.lock:
                        req["status"] = "finished"
                        req["timestamp_finished"] = time.time()
                        req["output"] = {"result": False, "error": f"Processing error: {e}"}
            if not dispatched:
                time.sleep(BATCH_WAIT_SEC)

    def _pick_transformers_manager(self):
        """Pick the least-loaded transformers model replica for dispatch."""
        if not self.transformers_model_managers:
            raise RuntimeError("Transformers managers not initialized")
        loads = [(tm, tm.get_load()) for tm in self.transformers_model_managers]
        loads.sort(key=lambda x: x[1])
        chosen = loads[0][0]
        # Always log the chosen replica for debugging load balancing
        try:
            idx = self.transformers_model_managers.index(chosen)
            device = chosen.force_device or chosen.device
            print(f"[scheduler] chosen replica index={idx} device={device} load={loads[0][1]}")
        except Exception:
            print(f"[scheduler] chosen replica load={loads[0][1]}")
        return chosen
