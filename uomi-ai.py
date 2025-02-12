import time
import os
import sys
from flask import Flask, request, jsonify

from lib.System import System

from model_managers.FastDualModelGPUManager import FastDualModelGPUManager
from model_managers.SanaManager import SanaManager

from model_runners.ChatRunner import ChatRunner
from model_runners.ImageRunner import ImageRunner

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

MODELS = [
  { "name": "casperhansen/mistral-small-24b-instruct-2501-awq", "deterministic": True, "manager": "FastDualModelGPUManager", "runner": "ChatRunner" }, # UOMI Model ID 1
  # { "name": "casperhansen/deepseek-r1-distill-qwen-14b-awq", "deterministic": False, "runner": "ChatRunner" }, # UOMI Model ID 2
  # { "name": "SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B", "deterministic": False, "runner": "ChatRunner" } # UOMI Model ID 3
  { "name": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "deterministic": False, "manager": "SanaManager", "runner": "ImageRunner" }, # UOMI Model ID 4
]
print("MODELS:", MODELS)

# Setup system
system = System()
sys.exit(0) if not system.check_system_requirements() else 1
sys.exit(0) if not system.check_cuda_availability() else 1
system.setup_environment_variables()
print("✅ System requirements met")
print(' ')

# Load model managers
fast_dual_model_gpu_models = [model for model in MODELS if model["manager"] == "FastDualModelGPUManager"]
sana_models = [model for model in MODELS if model["manager"] == "SanaManager"]
fast_dual_model_gpu_manager = FastDualModelGPUManager(fast_dual_model_gpu_models) if fast_dual_model_gpu_models else None
sana_manager = SanaManager(sana_models) if sana_models else None
print("✅ Models loaded")
print(' ')

# Load model runners
chat_runner = ChatRunner()
image_runner = ImageRunner()
print("✅ Runners loaded")

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

  # Find model config
  model_config = MODELS[[model["name"] for model in MODELS].index(data["model"])]
  if not model_config:
    return jsonify({"error": "model not found"}), 400

  # Wait for the previous request to finish
  if request_running:
    time_start = time.time()
    while request_running:
      time.sleep(1)
  request_running = True
  
  try:
    # Run the model to get the response
    model_runner = None
    if model_config["runner"] == "ChatRunner":
      model_runner = chat_runner
    elif model_config["runner"] == "ImageRunner":
      model_runner = image_runner
    else:
      return jsonify({"error": "model runner not found"}), 400

    # Find the model manager
    model_manager = None
    if model_config["manager"] == "FastDualModelGPUManager":
      model_manager = fast_dual_model_gpu_manager
    elif model_config["manager"] == "SanaManager":
      model_manager = sana_manager
    else:
      return jsonify({"error": "model manager not found"}), 400
    
    # Switch to the model on the model manager
    model_manager.switch_model(model_config["name"])

    # Run the model
    output = model_runner.run(data["input"], model_manager)

    # Validate the output and return an error if needed
    if not output["result"]:
      model_manager.clear_model()
      request_running = False
      return jsonify({"error": output["error"]}), 400

    # Cache the output
    if USE_CACHE:
      request_cache[body_hash] = {
        "output": output,
        "timestamp": time.time()
      }
      for key in list(request_cache): # clean up cache
        if time.time() - request_cache[key]["timestamp"] > 60 * 60:
          del request_cache[key]

    # Return the output
    model_manager.clear_model()
    request_running = False
    return jsonify(output)
  except Exception as e:
    print("Error:", str(e))
    fast_dual_model_gpu_manager.clear_model()
    sana_manager.clear_model()
    request_running = False
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  print("🚀 Starting Flask app...")
  app.run(host='0.0.0.0', port=8888, debug=False)
  
