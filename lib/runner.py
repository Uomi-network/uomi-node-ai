import time
import threading
import uuid
import os
from lib.config import BATCH_WAIT_SEC, BATCH_MAX_SIZE, TRANSFORMERS_INFERENCE_MAX_TOKENS
from lib.executors import ChatExecutor, ImageExecutor
from lib.TestModelManager import TEST_MODEL_CONFIG, TestModelManager
from lib.TransformersModelManager import DEEPSEEK_MODEL_CONFIG, TransformersModelManager
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
        # Micro-batching accumulation window in milliseconds (set 0 to disable)
        self.microbatch_window_ms = 30
        self.test_model_manager = TestModelManager(TEST_MODEL_CONFIG)
        if test_mode:
            self.transformers_model_manager = None
            self.sana_model_manager = None
        else:
            # Single DeepSeek transformers model always resident (enable continuous batching)
            self.transformers_model_manager = TransformersModelManager(DEEPSEEK_MODEL_CONFIG)
            self.transformers_model_manager.enable_continuous(max_active=BATCH_MAX_SIZE)
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
                if model not in TEST_MODEL_CONFIG and model != DEEPSEEK_MODEL_CONFIG.model_name:
                    model = DEEPSEEK_MODEL_CONFIG.model_name
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
                    elif model == DEEPSEEK_MODEL_CONFIG.model_name and self.transformers_model_manager is not None:
                        # Continuous submission
                        print(f"🟢 Dispatching transformers request {req['uuid']}")
                        input_json = req["request"]["input"]
                        import json
                        payload = json.loads(input_json)
                        messages = payload["messages"]
                        enable_thinking = payload.get("enable_thinking", True)
                        # Allow optional per-request sampling / max tokens overrides
                        sampling_cfg = payload.get("sampling", {"temperature":0.7, "top_k":5})
                        # Determine max_new_tokens with safe cap (env var MAX_NEW_TOKENS, default 128)
                        req_max_new = req["request"].get("max_new_tokens") or payload.get("max_new_tokens")
                        try:
                            max_new_tokens = int(req_max_new) if req_max_new is not None else int(os.getenv("MAX_NEW_TOKENS", f"{TRANSFORMERS_INFERENCE_MAX_TOKENS}"))
                        except Exception:
                            max_new_tokens = TRANSFORMERS_INFERENCE_MAX_TOKENS
                        max_new_tokens = max(1, min(max_new_tokens, 4096))  # hard cap to protect CPU/GPU
                        if is_check:
                            # unzip proof
                            from lib.zipper import unzip_string
                            proof_obj = json.loads(unzip_string(req["request"]["proof"]))
                            forced_ids = [t["id"] for t in proof_obj["tokens"]]
                            # In check mode, limit generation exactly to proof length
                            max_new_tokens = len(forced_ids)
                        else:
                            forced_ids = None
                        def on_token(sid, txt, meta, rq=req):
                            if os.getenv('CONTINUOUS_DEBUG','0') == '1':
                                print(f"[stream] req={rq['uuid']} sid={sid[:6]} token={meta.get('id')} txt='{txt}'")
                        def on_complete(sid, response, proof, rq=req):
                            from lib.zipper import zip_string
                            import json as _j
                            wrapped_proof = zip_string(_j.dumps(proof)) if proof is not None else ""
                            with self.lock:
                                rq["status"] = "finished"
                                rq["timestamp_finished"] = time.time()
                                rq["output"] = {"result": True, "response": response, "proof": wrapped_proof}
                            if os.getenv('CONTINUOUS_DEBUG','0') == '1':
                                print(f"[complete] req={rq['uuid']} sid={sid[:6]} tokens={len(proof['tokens']) if proof else 0}")
                        self.transformers_model_manager.submit_continuous(messages, enable_thinking, sampling_cfg, max_new_tokens, on_token, on_complete, is_check=is_check, forced_tokens=forced_ids)
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

