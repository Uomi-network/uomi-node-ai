import time
import os
import sys
from flask import Flask, request, jsonify
import lib.System
import lib.FastDualModelGPUManager
import lib.mistral_small_24b_instruct_2501_awq
import lib.deepseek_r1_distill_qwen_14b_awq
import lib.dobby_mini_unhinged_llama_31_8b

print(' ')
print('|' * 50)
print("🤖 Uomi Node AI")
print('|' * 50)
print(' ')

# Constants

UOMI_ENGINE_PALLET_VERSION = 3
print("UOMI_ENGINE_PALLET_VERSION:", UOMI_ENGINE_PALLET_VERSION)

MODELS = [
  # { "name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", "deterministic": True }, # UOMI Model ID 1 - Removed used on UOMI_ENGINE_PALLET_VERSION <= 2
  { "name": "casperhansen/mistral-small-24b-instruct-2501-awq", "deterministic": True, "runner": lib.mistral_small_24b_instruct_2501_awq.MistralSmall24bInstruct2501Awq() }, # UOMI Model ID 1
  { "name": "casperhansen/deepseek-r1-distill-qwen-14b-awq", "deterministic": False, "runner": lib.deepseek_r1_distill_qwen_14b_awq.DeepseekR1DistillQwen14bAwq() }, # UOMI Model ID 2
  # { "name": "SentientAGI/Dobby-Mini-Unhinged-Llama-3.1-8B", "deterministic": False, "runner": lib.dobby_mini_unhinged_llama_31_8b.DobbyMiniUnhingedLlama318b() } # UOMI Model ID 3
]
print("MODELS:", MODELS)

USE_CACHE = False
print("USE_CACHE:", USE_CACHE)
print(' ')

# Setup system
system = lib.System.System()
sys.exit(0) if not system.check_system_requirements() else 1
sys.exit(0) if not system.check_cuda_availability() else 1
system.setup_environment_variables()
print("✅ System requirements met")
print(' ')

# Load models
model_manager = lib.FastDualModelGPUManager.FastDualModelGPUManager(MODELS)
model_manager.switch_model(MODELS[0]["name"])
print("✅ Models loaded")
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
  # be sure model parameter is correct
  if data["model"] not in [model["name"] for model in MODELS]:
    return jsonify({"error": "model parameter is incorrect"}), 400
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
    # Load the model config
    model_config = MODELS[[model["name"] for model in MODELS].index(data["model"])]

    # Switch to the requested model
    model_manager.switch_model(model_config["name"])

    # Run the model to get the response
    output = model_config["runner"].run(data["input"], model_manager)

    # Validate the output and return an error if needed
    if not output["result"]:
      model_manager.switch_model(MODELS[0]["name"])
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
    model_manager.switch_model(MODELS[0]["name"])
    request_running = False
    return jsonify(output)
  except Exception as e:
    print("Error:", str(e))
    model_manager.switch_model(MODELS[0]["name"])
    request_running = False
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  print("🚀 Starting Flask app...")
  app.run(host='0.0.0.0', port=8888, debug=False)
  
