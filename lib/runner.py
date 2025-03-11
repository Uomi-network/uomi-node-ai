import time
import threading
import uuid
from lib.config import BATCH_WAIT_SEC, BATCH_MAX_SIZE
from lib.executors import ChatExecutor
from lib.TransformersModelManager import TRANSFORMERS_MODEL_CONFIG

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
                "output": None
            }
            return request_uuid
        
    def get_request(self, request_uuid):
        with self.lock:
            return self.queue[request_uuid]
        
    def get_requests(self):
        with self.lock:
            return self.queue
        
class RunnerExecutor:
    def __init__(self, queue, transformers_model_manager):
        print('Initialize RunnerExecutor')
        self.queue = queue
        self.transformers_model_manager = transformers_model_manager
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        print('Start RunnerExecutor')
        chat_executor = ChatExecutor()
        while True:
            with self.lock:
                pending_requests = [request for request in self.queue.get_requests().values() if request["status"] == "pending"]
                if len(pending_requests) == 0:
                    time.sleep(BATCH_WAIT_SEC)
                    continue
                # Sort pending_requests by timestamp_pending
                pending_requests = sorted(pending_requests, key=lambda x: x["timestamp_pending"])
                # Take the model of the first request
                model = pending_requests[0]["request"]["model"]
                # Generate the batch by taking the first BATCH_MAX_SIZE requests with the same model
                batch = [request for request in pending_requests if request["request"]["model"] == model][:BATCH_MAX_SIZE]
                for i, request in enumerate(batch):
                    request["status"] = "running"
                    request["timestamp_running"] = time.time()
                # Run the batch
                print('🤖 Running batch with ' + str(len(batch)) + ' requests')
                if model in TRANSFORMERS_MODEL_CONFIG:
                    self.transformers_model_manager.switch_model(model)
                    def on_output_finished(index, output):
                        batch[index]["status"] = "finished"
                        batch[index]["timestamp_finished"] = time.time()
                        batch[index]["output"] = output
                    outputs = chat_executor.execute([request["request"]["input"] for request in batch], self.transformers_model_manager, on_output_finished)
                    self.transformers_model_manager.clear_model()
                else:
                    for i, request in enumerate(batch):
                        request["status"] = "finished"
                        request["timestamp_finished"] = time.time()
                        request["output"] = {"result": False, "error": "Model not valid"}



                # # TEMP: Run requests in batch
                # for request in batch:
                #     request["status"] = "running"
                #     request["timestamp_running"] = time.time()
                #     request["status"] = "finished"
                #     request["timestamp_finished"] = time.time()
                #     request["output"] = {"result": True}
