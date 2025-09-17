import time
import threading
import uuid
from lib.config import BATCH_WAIT_SEC, BATCH_MAX_SIZE
from lib.executors import ChatExecutor, ImageExecutor
from lib.TestModelManager import TEST_MODEL_CONFIG, TestModelManager
from lib.TransformersModelManager import DEEPSEEK_MODEL_CONFIG, TransformersModelManager
from lib.SanaModelManager import SANA_MODEL_CONFIG, SanaModelManager

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
            # Single DeepSeek transformers model always resident
            self.transformers_model_manager = TransformersModelManager(DEEPSEEK_MODEL_CONFIG)
            self.sana_model_manager = SanaModelManager(SANA_MODEL_CONFIG)
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
            # Step 1: Get batch to process (minimal lock time)
            batch = []
            model = None
            is_check = None
            
            with self.lock:
                pending_requests = [request for request in self.queue.get_requests().values() if request["status"] == "pending"]
                if len(pending_requests) == 0:
                    pass
                else:
                    pending_requests = sorted(pending_requests, key=lambda x: x["timestamp_pending"])
                    first = pending_requests[0]
                    model = first["request"]["model"]
                    if model not in TEST_MODEL_CONFIG and model not in SANA_MODEL_CONFIG and model != DEEPSEEK_MODEL_CONFIG.model_name:
                        model = DEEPSEEK_MODEL_CONFIG.model_name
                        first["request"]["model"] = model
                    is_check = "proof" in first["request"]
                    batch = [r for r in pending_requests if r["request"]["model"] == model and ("proof" in r["request"]) == is_check][:BATCH_MAX_SIZE]

            # Micro-batch accumulation (outside lock)
            if batch and len(batch) < BATCH_MAX_SIZE and self.microbatch_window_ms > 0:
                deadline = time.time() + self.microbatch_window_ms / 1000.0
                selected_ids = {r["uuid"] for r in batch}
                while time.time() < deadline and len(batch) < BATCH_MAX_SIZE:
                    time.sleep(0.005)
                    with self.lock:
                        more_pending = [req for req in self.queue.get_requests().values() if req["status"] == "pending"]
                        if not more_pending:
                            continue
                        more_pending = sorted(more_pending, key=lambda x: x["timestamp_pending"])
                        for req in more_pending:
                            if req["uuid"] in selected_ids:
                                continue
                            # Normalize model if legacy
                            req_model = req["request"].get("model")
                            if req_model not in TEST_MODEL_CONFIG and req_model not in SANA_MODEL_CONFIG and req_model != DEEPSEEK_MODEL_CONFIG.model_name:
                                req["request"]["model"] = DEEPSEEK_MODEL_CONFIG.model_name
                                req_model = DEEPSEEK_MODEL_CONFIG.model_name
                            if req_model == model and ("proof" in req["request"]) == is_check:
                                batch.append(req)
                                selected_ids.add(req["uuid"])
                                if len(batch) >= BATCH_MAX_SIZE:
                                    break

            # Mark batch as running after micro-accumulation
            if batch:
                with self.lock:
                    batch_ids = [req["uuid"] for req in batch]
                    for req in batch:
                        req["status"] = "running"
                        req["timestamp_running"] = time.time()
                        req["batch"] = batch_ids
            
            # Step 2: Handle no work case (outside lock)
            if not batch:
                time.sleep(BATCH_WAIT_SEC)  # Sleep outside the lock
                continue
                
            # Step 3: Process batch (outside any locks to prevent deadlock)
            try:
                print('🤖 Running batch with ' + str(len(batch)) + ' requests')
                self.test_model_manager.clear_model()
                # Do not clear transformers model (always resident)
                self.sana_model_manager.clear_model() if self.sana_model_manager is not None else None
                
                if model in TEST_MODEL_CONFIG:
                    self.test_model_manager.switch_model(model)
                    def on_output_finished(index, output):
                        # Use minimal lock for status update only
                        if index < len(batch):
                            with self.lock:
                                batch[index]["status"] = "finished"
                                batch[index]["timestamp_finished"] = time.time()
                                batch[index]["output"] = output
                    if is_check:
                        chat_executor.check([request["request"]["input"] for request in batch], [request["request"]["proof"] for request in batch], self.test_model_manager, on_output_finished)
                    else:
                        chat_executor.execute([request["request"]["input"] for request in batch], self.test_model_manager, on_output_finished)
                    self.test_model_manager.clear_model()
                elif model == DEEPSEEK_MODEL_CONFIG.model_name and self.transformers_model_manager is not None:
                    # Single transformers model already loaded
                    def on_output_finished(index, output):
                        # Use minimal lock for status update only
                        if index < len(batch):
                            with self.lock:
                                batch[index]["status"] = "finished"
                                batch[index]["timestamp_finished"] = time.time()
                                batch[index]["output"] = output
                    if is_check:
                        chat_executor.check([request["request"]["input"] for request in batch], [request["request"]["proof"] for request in batch], self.transformers_model_manager, on_output_finished)
                    else:
                        chat_executor.execute([request["request"]["input"] for request in batch], self.transformers_model_manager, on_output_finished)
                    # Always keep in memory now
                elif model in SANA_MODEL_CONFIG and self.sana_model_manager is not None:
                    self.sana_model_manager.switch_model(model)
                    def on_output_finished(index, output):
                        # Use minimal lock for status update only
                        if index < len(batch):
                            with self.lock:
                                batch[index]["status"] = "finished"
                                batch[index]["timestamp_finished"] = time.time()
                                batch[index]["output"] = output
                    if is_check:
                        # Handle unsupported check case
                        with self.lock:
                            for i, request in enumerate(batch):
                                request["status"] = "finished"
                                request["timestamp_finished"] = time.time()
                                request["output"] = {"result": False, "error": "Check not supported for the requested model"}
                    else:
                        image_executor.execute([request["request"]["input"] for request in batch], self.sana_model_manager, on_output_finished)
                    self.sana_model_manager.clear_model()
                else:
                    # Handle invalid model case
                    with self.lock:
                        for i, request in enumerate(batch):
                            request["status"] = "finished"
                            request["timestamp_finished"] = time.time()
                            request["output"] = {"result": False, "error": "Model not valid"}
            except Exception as e:
                # Handle exceptions and mark batch as failed
                print(f"Error processing batch: {e}")
                with self.lock:
                    for request in batch:
                        request["status"] = "finished"
                        request["timestamp_finished"] = time.time()
                        request["output"] = {"result": False, "error": f"Processing error: {str(e)}"}

