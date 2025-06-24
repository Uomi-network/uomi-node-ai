import time
import threading
import uuid
from lib.config import BATCH_WAIT_SEC, BATCH_MAX_SIZE
from lib.executors import ChatExecutor, ImageExecutor
from lib.TestModelManager import TEST_MODEL_CONFIG, TestModelManager
from lib.TransformersModelManager import TRANSFORMERS_MODEL_CONFIG, TransformersModelManager
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
            return self.queue[request_uuid]

    def remove_request(self, request_uuid):
        with self.lock:
            del self.queue[request_uuid]
        
    def get_requests(self):
        with self.lock:
            return self.queue
        
class RunnerExecutor:
    def __init__(self, queue, test_mode=False):
        print('Initialize RunnerExecutor')
        self.kill = False
        self.queue = queue
        self.test_model_manager = TestModelManager(TEST_MODEL_CONFIG)
        if test_mode:
            self.transformers_model_manager = None
            self.sana_model_manager = None
        else:
            self.transformers_model_manager = TransformersModelManager(TRANSFORMERS_MODEL_CONFIG)
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
            with self.lock:
                pending_requests = [request for request in self.queue.get_requests().values() if request["status"] == "pending"]
                if len(pending_requests) == 0:
                    time.sleep(BATCH_WAIT_SEC)
                    continue
                # Sort pending_requests by timestamp_pending
                pending_requests = sorted(pending_requests, key=lambda x: x["timestamp_pending"])
                # Take the model and is_check of the first request
                model = pending_requests[0]["request"]["model"]

                if model not in TEST_MODEL_CONFIG and model not in TRANSFORMERS_MODEL_CONFIG and model not in SANA_MODEL_CONFIG:
                    # fallback to deepseek an agent sent something from an older version of the pallet. Could happen if not all validators updated yet.
                    model = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

                is_check = True if "proof" in pending_requests[0]["request"] else False
                # Generate the batch by taking the first BATCH_MAX_SIZE requests with the same model and is_check
                batch = [request for request in pending_requests if request["request"]["model"] == model and ("proof" in request["request"]) == is_check][:BATCH_MAX_SIZE]
                for i, request in enumerate(batch):
                    request["status"] = "running"
                    request["timestamp_running"] = time.time()
                    request["batch"] = [request["uuid"] for request in batch]
                # Run the batch
                print('🤖 Running batch with ' + str(len(batch)) + ' requests')
                self.test_model_manager.clear_model()
                self.transformers_model_manager.clear_model() if self.transformers_model_manager is not None else None
                self.sana_model_manager.clear_model() if self.sana_model_manager is not None else None
                if model in TEST_MODEL_CONFIG:
                    self.test_model_manager.switch_model(model)
                    def on_output_finished(index, output):
                        batch[index]["status"] = "finished"
                        batch[index]["timestamp_finished"] = time.time()
                        batch[index]["output"] = output
                    if is_check:
                        chat_executor.check([request["request"]["input"] for request in batch], [request["request"]["proof"] for request in batch], self.test_model_manager, on_output_finished)
                    else:
                        chat_executor.execute([request["request"]["input"] for request in batch], self.test_model_manager, on_output_finished)
                    self.test_model_manager.clear_model()
                elif model in TRANSFORMERS_MODEL_CONFIG and self.transformers_model_manager is not None:
                    self.transformers_model_manager.switch_model(model)
                    def on_output_finished(index, output):
                        batch[index]["status"] = "finished"
                        batch[index]["timestamp_finished"] = time.time()
                        batch[index]["output"] = output
                    if is_check:
                        chat_executor.check([request["request"]["input"] for request in batch], [request["request"]["proof"] for request in batch], self.transformers_model_manager, on_output_finished)
                    else:
                        chat_executor.execute([request["request"]["input"] for request in batch], self.transformers_model_manager, on_output_finished)
                    keep_in_memory = TRANSFORMERS_MODEL_CONFIG[model].keep_in_memory if model in TRANSFORMERS_MODEL_CONFIG else False
                    if not keep_in_memory:
                        self.transformers_model_manager.clear_model()
                elif model in SANA_MODEL_CONFIG and self.sana_model_manager is not None:
                    self.sana_model_manager.switch_model(model)
                    def on_output_finished(index, output):
                        batch[index]["status"] = "finished"
                        batch[index]["timestamp_finished"] = time.time()
                        batch[index]["output"] = output
                    if is_check:
                        for i, request in enumerate(batch):
                            request["status"] = "finished"
                            request["timestamp_finished"] = time.time()
                            request["output"] = {"result": False, "error": "Check not supported for the requested model"}
                    else:
                        image_executor.execute([request["request"]["input"] for request in batch], self.sana_model_manager, on_output_finished)
                    self.sana_model_manager.clear_model()
                else:
                    for i, request in enumerate(batch):
                        request["status"] = "finished"
                        request["timestamp_finished"] = time.time()
                        request["output"] = {"result": False, "error": "Model not valid"}

