import torch
import time
import torch
import torch.nn.functional as F
from typing import Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.config import MODELS_FOLDER, TRANSFORMERS_INFERENCE_MAX_TOKENS, TRANSFORMERS_INFERENCE_TEMPERATURE

@dataclass
class TestModelConfig:
    model_name: str  # HuggingFace model name/path

# NOTE: This class is a fake implementation of a generic ModelManager class used for tests.
class TestModelManager:
    def __init__(self, models_config: Dict[str, TestModelConfig]):
        self.models_config = models_config
        self.current_gpu_model = None
        self.current_gpu_model_name = None

    def switch_model(self, model_name: str):
        time_start = time.time()

        # Load model configuration
        if model_name not in self.models_config:
            raise KeyError(f"Model {model_name} not found in configured models")
        model_config = self.models_config[model_name]

        # Clear existing GPU model if present
        self.clear_model()
        
        self.current_gpu_model = True
        self.current_gpu_model_name = model_name

        print(f"Time taken to switch model: {time.time() - time_start:.2f}s")
        return self.current_gpu_model

    def clear_model(self):
        time_start = time.time()
        if self.current_gpu_model is not None:
            # Clear GPU model references
            self.current_gpu_model = None
            self.current_gpu_model_name = None

        print(f"Time taken to clear model for TEST MODEL MANAGER: {time.time() - time_start:.2f}s")

    def run_batch_executions(self, prompts, on_prompt_finished):
        if self.current_gpu_model is None:
            print("No model loaded on GPU")
            return None
        
        for i, prompt in enumerate(prompts):
            time.sleep(1)
            on_prompt_finished(i, { "response": "Test response", "proof": {
                "tokens": [{
                    "id": 0,
                    "prob": 1.0,
                    "index": 0,
                }],
                "full_sequence_length": 1
            } })

    def run_batch_checks(self, prompts, proofs, on_prompt_finished):
        if self.current_gpu_model is None:
            print("No model loaded on GPU")
            return None
        start_time = time.time()

        for i, prompt in enumerate(prompts):
            time.sleep(1)
            on_prompt_finished(i, { "response": "Test response", "proof": None })

    def get_current_model(self):
        return self.current_gpu_model
    
    def get_tokenizer(self, model_name: str):
        return True

TEST_MODEL_CONFIG = {
    "test/1": TestModelConfig(
        model_name="test/1",
    ),
    "test/2": TestModelConfig(
        model_name="test/2",
    ),
}