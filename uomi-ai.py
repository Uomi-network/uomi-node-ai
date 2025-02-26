import time
import os
import sys
import traceback
from flask import Flask, request, jsonify

from lib.System import System
from lib.TransformersManager import TransformersModelManager, TransformersModelConfig
from lib.SanaManager import SanaModelManager, SanaModelConfig
from lib.Runners import ChatRunner, ImageRunner

print(' ')
print('|' * 50)
print("🤖 Uomi Node AI")
print('|' * 50)
print(' ')

# Constants

UOMI_ENGINE_PALLET_VERSION = 3
print("UOMI_ENGINE_PALLET_VERSION:", UOMI_ENGINE_PALLET_VERSION)

USE_CACHE = False
print("USE_CACHE:", USE_CACHE)
print(' ')

# Define model configurations
TRANSFORMERS_MODEL_CONFIG = {
    'casperhansen/mistral-small-24b-instruct-2501-awq': TransformersModelConfig(
        model_name='casperhansen/mistral-small-24b-instruct-2501-awq',
        deterministic=True,
        location="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={
            'do_sample': False,
            'num_beams': 1,
            'temperature': 1.0,
            'top_p': 1.0,
            'max_new_tokens': 8196,
        }
    ),
    'casperhansen/deepseek-r1-distill-qwen-14b-awq': TransformersModelConfig(
        model_name='casperhansen/deepseek-r1-distill-qwen-14b-awq',
        deterministic=False,
        location="cpu",
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_new_tokens': 8196,
        }
    ),
    'SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B': TransformersModelConfig(
        model_name='SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B',
        deterministic=False,
        location="disk",
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'max_new_tokens': 8196,
        }
    ),
}
print("TRANSFORMERS_MODEL_CONFIG:", TRANSFORMERS_MODEL_CONFIG)
print(' ')
SANA_MODEL_CONFIG = {
    'Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers': SanaModelConfig(
        model_name='Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers',
        model_kwargs={},
        tokenizer_kwargs={},
        inference_kwargs={}
    ),
}
print("SANA_MODEL_CONFIG:", SANA_MODEL_CONFIG)
print(' ')

# Setup system
system = System()
sys.exit(0) if not system.check_system_requirements() else 1
sys.exit(0) if not system.check_cuda_availability() else 1
system.setup_environment_variables()
print("✅ System requirements met")
print(' ')

# Setup TransformersModelManager
transformers_model_manager = TransformersModelManager(TRANSFORMERS_MODEL_CONFIG)
print("✅ TransformersModelManager loaded")
print(' ')

# Setup SanaModelManager
sana_model_manager = SanaModelManager(SANA_MODEL_CONFIG)
print("✅ SanaModelManager loaded")
print(' ')

# Setup the Flask app
app = Flask(__name__)
request_running = False # Flag to indicate if a request is currently running or not
request_cache = {}

@app.route('/status', methods=['GET'])
def status_json():
    return jsonify({
        "UOMI_ENGINE_PALLET_VERSION": UOMI_ENGINE_PALLET_VERSION,
        "details": {
            "system_valid": system.check_system_requirements(),
            "cuda_available": system.check_cuda_availability(),
            "request_running": request_running
        }
    })

@app.route('/run', methods=['POST'])
def run_json():   
    global request_running, request_cache 

    print("Received request...")
    data = request.get_json()

    if USE_CACHE:
        body_hash = hash(str(data))
        if body_hash in request_cache:
            request_cached = request_cache[body_hash]
            if time.time() - request_cached["timestamp"] < 60 * 60: # Cache for 1 hour
                return jsonify(request_cached["output"])
            else:
                del request_cache[body_hash]

    # Validate parameters
    # be sure model name parameter is present
    if "model" not in data:
        return jsonify({"error": "model parameter is required"}), 400
    # be sure input parameter is present
    if "input" not in data:
        return jsonify({"error": "input parameter is required"}), 400
    # be sure input parameter is a string
    if not isinstance(data["input"], str):
        return jsonify({"error": "input parameter must be a string"}), 400

    # Wait for the previous request to finish
    if request_running:
        time_start = time.time()
        while request_running:
            time.sleep(1)
    request_running = True
    
    try:
        # Generate output
        output = None
        if data["model"] in TRANSFORMERS_MODEL_CONFIG:
            transformers_model_manager.switch_model(data["model"])
            runner = ChatRunner()
            output = runner.run(data["input"], transformers_model_manager)
            transformers_model_manager.clear_model()
        elif data["model"] in SANA_MODEL_CONFIG:
            sana_model_manager.switch_model(data["model"])
            runner = ImageRunner()
            output = runner.run(data["input"], sana_model_manager)
            sana_model_manager.clear_model()
        else:
            request_running = False
            return jsonify({"error": "model not found"}), 400

        # Check if output is valid
        if output == None or not output["result"]:
            request_running = False
            return jsonify({"error": output["error"]}), 400
        
        # Store output in cache
        if USE_CACHE:
            request_cache[body_hash] = {
                "output": output,
                "timestamp": time.time()
            }
            for key in list(request_cache): # clean up cache
                if time.time() - request_cache[key]["timestamp"] > 60 * 60:
                    del request_cache[key]

        # Return output
        request_running = False
        return jsonify(output)
    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        transformers_model_manager.clear_model()
        sana_model_manager.clear_model()
        request_running = False
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(host='0.0.0.0', port=8888, debug=False)
