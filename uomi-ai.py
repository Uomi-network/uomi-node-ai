from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import numpy as np
import time
import os
import json
import psutil
import datetime
import gc

# INITIALIZER
##############################################################################################

UOMI_ENGINE_PALLET_VERSION = 2

models_available = [
  "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
]

request_history = []
max_history_size = 100  
service_start_time = datetime.datetime.now()

# Detect if system is valid
print("🕦 Checking system...")
system_valid = False
if os.path.exists("/proc/cpuinfo") and os.path.exists("/proc/meminfo") and os.path.exists("/proc/uptime"):
  cpuinfo = open("/proc/cpuinfo").read()
  meminfo = open("/proc/meminfo").read()
  uptime = open("/proc/uptime").read()
  if cpuinfo and meminfo and uptime:
    system_valid = True
if not system_valid:
  print("🚨 System is not valid. Setting only development environment...\n")
else:
  print("✅ System is valid!\n")

# Setup environment variables
print("🕦 Setting up environment variables...")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
print("✅ environment variables set!\n")

# Be sure cuda is available
print("🕦 Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
if not cuda_available:
  print("🚨 CUDA is not available.\n")
else:
  print("✅ CUDA is available!\n")

# Make model deterministic
print("🕦 Setting up model deterministic...")
if system_valid:
  random.seed(0) # Sets the seed for Python's built-in random module
  np.random.seed(0) # Sets the seed for NumPy's random number generator
  torch.manual_seed(0) # Sets the seed for PyTorch's CPU random number generator
  torch.cuda.manual_seed(0) # Sets the seed for the current GPU device
  torch.cuda.manual_seed_all(0) # Sets the seed for all available GPU devices
  torch.use_deterministic_algorithms(True, warn_only=True) # Ensures that only deterministic algorithms are used
  torch.backends.cuda.matmul.allow_tf32 = False # Disables TensorFloat32 (TF32) on matmul ops
  torch.backends.cudnn.allow_tf32 = False # Disables TF32 on cuDNN
  torch.backends.cudnn.benchmark = False # Disables the cuDNN auto-tuner
  torch.backends.cudnn.deterministic = True # Forces cuDNN to use deterministic algorithms
  torch.backends.cudnn.enabled = False # Disables cuDNN entirely
  print("✅ model deterministic set!\n")
else:
  print("🤖 skipping deterministic setup because system is not valid.\n")

# Load the model and tokenizer
print("🕦 Loading model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4...")
model_qwen = None
if system_valid:
  start_time = time.time()
  model_qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
    device_map="auto"
  )
  model_qwen.eval()
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
  model_qwen.resize_token_embeddings(len(tokenizer))
  end_time = time.time()
  time_taken = end_time - start_time
  print(f"✅ Model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 loaded in {time_taken:.2f} seconds!\n")
else:
  print("🤖 skipping model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 loading because system is not valid.\n")

# Setup the Flask app
print("🕦 Setting up Flask app...")
app = Flask(__name__)
request_running = False # Flag to indicate if a request is currently running or not
print("✅ Flask app set up!\n")

# STATUS API
##############################################################################################

@app.route('/status', methods=['GET'])
def status_json():
  cuda_available = torch.cuda.is_available()

  return jsonify({
    "UOMI_ENGINE_PALLET_VERSION": UOMI_ENGINE_PALLET_VERSION,
    "details": {
      "system_valid": system_valid,
      "request_running": request_running,
      "cuda_available": cuda_available,
      "models_available": models_available,
    }
  })

# MONITORING API
##############################################################################################

@app.route('/monitoring', methods=['GET'])
def monitoring_json():
    # Statistiche di sistema
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    
    # Statistiche CUDA
    cuda_stats = {}
    if cuda_available:
        try:
            cuda_stats = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "devices": []
            }
            
            for i in range(torch.cuda.device_count()):
                device_stats = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": torch.cuda.memory_allocated(i) / (1024**3),  # GB
                    "memory_reserved": torch.cuda.memory_reserved(i) / (1024**3),    # GB
                    "max_memory_allocated": torch.cuda.max_memory_allocated(i) / (1024**3),  # GB
                }
                cuda_stats["devices"].append(device_stats)
        except Exception as e:
            cuda_stats["error"] = str(e)
    
    # Statistiche del modello
    model_stats = {}
    if model_qwen is not None:
        try:
            model_stats = {
                "model_name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
                "device": str(next(model_qwen.parameters()).device),
                "param_count": sum(p.numel() for p in model_qwen.parameters()) / 1_000_000,  # milioni
            }
        except Exception as e:
            model_stats["error"] = str(e)
    
    # Statistiche del servizio
    uptime = datetime.datetime.now() - service_start_time
    uptime_seconds = uptime.total_seconds()
    
    # Statistiche delle richieste
    request_stats = {
        "total_requests": len(request_history),
        "average_request_time": sum(req["time_taken"] for req in request_history) / len(request_history) if request_history else 0,
        "average_tokens_per_second": sum(req["tokens_per_second"] for req in request_history) / len(request_history) if request_history else 0,
        "total_tokens_generated": sum(req["total_tokens_generated"] for req in request_history),
        "recent_requests": request_history[-10:] if request_history else []  # Ultime 10 richieste
    }
    
    # Esegui la garbage collection per garantire che le statistiche siano accurate
    collected = gc.collect()
    
    return jsonify({
        "timestamp": datetime.datetime.now().isoformat(),
        "uptime": {
            "days": uptime.days,
            "hours": uptime.seconds // 3600,
            "minutes": (uptime.seconds % 3600) // 60,
            "seconds": uptime.seconds % 60,
            "total_seconds": uptime_seconds
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_gb": memory_info.total / (1024**3),
                "available_gb": memory_info.available / (1024**3),
                "used_gb": memory_info.used / (1024**3),
                "percent": memory_info.percent
            },
            "disk": {
                "total_gb": disk_info.total / (1024**3),
                "used_gb": disk_info.used / (1024**3),
                "free_gb": disk_info.free / (1024**3),
                "percent": disk_info.percent
            }
        },
        "cuda": cuda_stats,
        "model": model_stats,
        "requests": request_stats,
        "gc_collected": collected
    })

# RUN API
##############################################################################################

@app.route('/run', methods=['POST'])
def run_json():
  global request_running, request_history
  try:
    print("Received request...")
    data = request.get_json()

    # Validate parameters
    # be sure model name parameter is present
    if "model" not in data:
      return jsonify({"error": "model parameter is required"}), 400
    # be sure model parameter is correct
    if data["model"] not in models_available:
      return jsonify({"error": "model parameter is incorrect"}), 400
    # be sure input parameter is present
    if "input" not in data:
      return jsonify({"error": "input parameter is required"}), 400
    # be sure input parameter is a string
    if not isinstance(data["input"], str):
      return jsonify({"error": "input parameter must be a string"}), 400

    # Initialize response variables
    response = None
    time_taken = None
    total_tokens_generated = None
    tokens_per_second = None

    # Model specific code
    # -------------------------------------------------------------------------- Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4
    if data["model"] == "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4":
      # fallback if system is not valid
      if not system_valid:
        response = "System is not valid. Setting only development environment..."
        return jsonify({"response": response, "time_taken": 0, "total_tokens_generated": 0, "tokens_per_second": 0})

      # try to parse input as a json object
      input_data = None
      try:
        input_data = json.loads(data["input"])
      except:
        return jsonify({"error": "input parameter must be a valid json string"}), 400
      
      # be sure messages input is a list
      if not isinstance(input_data["messages"], list):
        return jsonify({"error": "messages parameter must be a list"}), 400
      # be sure messages are objects with role and content keys, be sure content is a string and role is a string with values "system" or "user" or "assistant"
      for message in input_data["messages"]:
        if not isinstance(message, dict):
          return jsonify({"error": "each message must be an object"}), 400
        if "role" not in message:
          return jsonify({"error": "each message must have a role key"}), 400
        if "content" not in message:
          return jsonify({"error": "each message must have a content key"}), 400
        if not isinstance(message["role"], str):
          return jsonify({"error": "each message role must be a string"}), 400
        if not isinstance(message["content"], str):
          return jsonify({"error": "each message content must be a string"}), 400
        if message["role"] not in ["system", "user", "assistant"]:
          return jsonify({"error": "each message role must be 'system', 'user', or 'assistant'"}), 400

      # If another request is running wait for it to finish
      if request_running:
        time_start = time.time()
        while request_running and time.time() - time_start < 60:
          time.sleep(1)
      request_running = True
      
      # Tokenize the messages
      text = tokenizer.apply_chat_template(
        input_data["messages"],
        tokenize=False,
        add_generation_prompt=True,
      )
      model_inputs = tokenizer([text], return_tensors="pt").to(model_qwen.device)

      # Print the number of tokens in the input
      num_tokens = len(model_inputs['input_ids'][0])
      print(f"Number of input tokens: {num_tokens}")

      # Generate the response
      start_time = time.time()
      generated_ids = model_qwen.generate(
        **model_inputs,
        max_new_tokens=8192,
        do_sample=False,       # Disable sampling
        num_beams=1,           # Use greedy decoding, no beam search randomness
        temperature=1,         # No randomness in distribution scaling
        top_k=None,            # Disable top-k sampling
        top_p=None             # Disable nucleus (top-p) sampling
      )
      end_time = time.time()
      time_taken = end_time - start_time
      print(f"Time to generate response: {time_taken:.2f} seconds")
      total_tokens_generated = sum(len(ids) for ids in generated_ids)
      print(f"Total tokens generated: {total_tokens_generated}")
      tokens_per_second = total_tokens_generated / time_taken if time_taken > 0 else 0
      print(f"Average tokens per second: {tokens_per_second}")

      # Post-process the generated IDs
      generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
      ]

      response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
      request_running = False

    # -------------------------------------------------------------------------- Unsupported model
    else:
      return jsonify({"error": "model parameter is not supported yet"}), 400
    
    # Return the response
    if response is not None:
            request_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": data["model"],
                "time_taken": time_taken,
                "total_tokens_generated": total_tokens_generated,
                "tokens_per_second": tokens_per_second
            }
            request_history.append(request_info)
            
            # Limita la dimensione della cronologia
            if len(request_history) > max_history_size:
                request_history = request_history[-max_history_size:]
        
    return jsonify({"response": response, "time_taken": time_taken, "total_tokens_generated": total_tokens_generated, "tokens_per_second": tokens_per_second})
  except Exception as e:
    print("Error:", str(e))
    request_running = False
    return jsonify({"error": str(e)}), 500

# STARTUP
##############################################################################################

if __name__ == '__main__':
  print("🚀 Starting Flask app...")
  app.run(host='0.0.0.0', port=8888, debug=False)
