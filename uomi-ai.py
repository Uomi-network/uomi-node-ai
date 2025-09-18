import time
import os
import sys
import datetime
import gc
import psutil
import torch
import atexit
import json
from flask import Flask, request, jsonify
from lib.config import UOMI_ENGINE_PALLET_VERSION, CACHE_ENABLED
from lib.runner import RunnerQueue, RunnerExecutor
from lib.system import System
from lib.zipper import unzip_string, zip_string
from lib.monitoring import MonitoringService
from lib.TGIClient import TGIClient

print(' ')
print('|' * 50)
print("🧠 Uomi Node AI")
print('|' * 50)
print(' ')

system = System()
sys.exit(0) if not system.check_system_requirements() else 1
sys.exit(0) if not system.check_cuda_availability() else 1
system.setup_environment_variables()
print('🚀 System setup completed!')
print('\n')

# Initialize TGI client first
tgi_client = TGIClient()
tgi_available = tgi_client.health_check()
use_tgi = tgi_available and os.getenv('USE_TGI', '1') == '1'

if use_tgi:
    print('🚀 TGI client initialized and connected!')
    tgi_info = tgi_client.info()
    if tgi_info:
        print(f'🔧 TGI model info: {tgi_info}')
    print('📝 Using TGI - skipping local model initialization')
    runner_queue = None
    runner_executor = None
else:
    print('⚠️ TGI not available, initializing local models')
    runner_queue = RunnerQueue()
    runner_executor = RunnerExecutor(runner_queue)
    print('🚀 Runner setup completed!')
print('\n')

app = Flask(__name__)
app_cache = {}

# Global variables for monitoring
service_start_time = datetime.datetime.now()
request_history = []
cuda_available = system.check_cuda_availability()

# TGI configuration
use_tgi = tgi_available and os.getenv('USE_TGI', '1') == '1'

# Initialize monitoring service
monitoring_service = MonitoringService(app)

# Setup cleanup on exit
def cleanup_services():
    monitoring_service.stop()

atexit.register(cleanup_services)

@app.route('/status', methods=['GET'])
def status_json():
    tgi_status = {
        "available": tgi_available,
        "enabled": os.getenv('USE_TGI', '1') == '1',
        "base_url": tgi_client.base_url if tgi_client else None
    }
    
    if tgi_available:
        try:
            tgi_status["info"] = tgi_client.info()
        except:
            tgi_status["info"] = "Error getting TGI info"
    
    return jsonify({
        "UOMI_ENGINE_PALLET_VERSION": UOMI_ENGINE_PALLET_VERSION,
        "details": {
            "system_valid": system.check_system_requirements(),
            "cuda_available": system.check_cuda_availability(),
            "tgi": tgi_status
        }
    })

@app.route('/run', methods=['POST'])
def run_json():
    global app_cache
    time_start = time.time()

    # print('💬 Received request...')
    data = request.get_json()

    # Check if the response can be returned from cache
    body_hash = None
    if CACHE_ENABLED:
        # print('💬 Checking response on cache...')
        body_hash = hash(str(data))
        if body_hash in app_cache:
            response_cached = app_cache[body_hash]
            if time.time() - response_cached["timestamp"] < 60 * 60: # Cache for 1 hour
                print('✅ Returning cached response in ' + str(time.time() - time_start) + ' seconds')
                return jsonify(response_cached["output"])
            else:
                del app_cache[body_hash]

    # Validate parameters
    print('💬 Validating parameters...')
    # be sure model name parameter is present
    if "model" not in data:
        print('❌ Model parameter is required')
        return jsonify({"error": "model parameter is required"}), 400
    # be sure input parameter is present
    if "input" not in data:
        print('❌ Input parameter is required')
        return jsonify({"error": "input parameter is required"}), 400
    
    # Handle both string and object formats for input
    if isinstance(data["input"], str):
        # Current format: input is a JSON string
        try:
            input_data = json.loads(data["input"])
        except:
            print('❌ Input parameter must be a valid JSON string or object')
            return jsonify({"error": "input parameter must be a valid JSON string or object"}), 400
    elif isinstance(data["input"], dict):
        # New format: input is already an object
        input_data = data["input"]
    else:
        print('❌ Input parameter must be a string or object')
        return jsonify({"error": "input parameter must be a string or object"}), 400
    
    # validate enable_thinking parameter (can be in top level or inside input)
    enable_thinking = data.get("enable_thinking", input_data.get("enable_thinking", False))
    if not isinstance(enable_thinking, bool):
        print('❌ enable_thinking parameter must be a boolean')
        return jsonify({"error": "enable_thinking parameter must be a boolean"}), 400
    
    # Ensure enable_thinking is included in the input data
    input_data["enable_thinking"] = enable_thinking
    
    if use_tgi:
        # print('💬 Processing with TGI...')
        # Direct processing with TGI (no queue needed)
        try:
            # Extract messages from input_data
            messages = input_data.get("messages", [])
            if not messages:
                print('❌ Messages array is required for TGI')
                return jsonify({"error": "messages array is required for TGI"}), 400
            
            # Prepare TGI parameters
            tgi_parameters = {
                "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "512")),
                "temperature": input_data.get("temperature", 0.7),
                "top_k": input_data.get("top_k", 5),
                "top_p": input_data.get("top_p", 0.9),
                "repetition_penalty": input_data.get("repetition_penalty", 1.1),
                "do_sample": True
            }
            
            # Allow override from data
            if "parameters" in data:
                tgi_parameters.update(data["parameters"])
            
            # Handle proof checking mode
            forced_tokens = None
            is_check = "proof" in data
            if is_check:
                # print('💬 TGI proof checking mode...')
                try:
                    proof_obj = json.loads(unzip_string(data["proof"]))
                    forced_tokens = [t["id"] for t in proof_obj["tokens"]]
                    tgi_parameters["max_new_tokens"] = len(forced_tokens)
                except Exception as e:
                    print(f'❌ Error parsing proof: {e}')
                    return jsonify({"error": f"Invalid proof format: {e}"}), 400
            
            # Call TGI
            # If we are in check mode, pass the proof and optional verification_mode
            verification_mode = data.get('verification_mode')
            proof_obj = None
            if is_check:
                # proof_obj is the decompressed proof
                proof_obj = proof_obj if 'proof_obj' in locals() and proof_obj is not None else proof_obj
                try:
                    proof_obj = json.loads(unzip_string(data['proof']))
                except Exception:
                    # already validated earlier, ignore here
                    pass

            result = tgi_client.generate_stream(
                messages=messages,
                parameters=tgi_parameters,
                enable_thinking=enable_thinking,
                forced_tokens=forced_tokens,
                proof=proof_obj,
                verification_mode=verification_mode
            )
            
            if not result["result"]:
                err = result.get("error", "TGI processing failed")
                verification_error = result.get('verification_error')
                print(f'❌ TGI error: {err}')
                if verification_error:
                    print(f"   Verification error details: {verification_error}")
                    return jsonify({"error": err, "verification_error": verification_error}), 500
                return jsonify({"error": err}), 500
            
            # Format output similar to the original system
            if result["proof"]:
                # Create zipped proof like the original system
                proof_compressed = zip_string(json.dumps(result["proof"]))
                output = {
                    "result": True,
                    "response": result["response"],
                    "proof": proof_compressed
                }
            else:
                output = {
                    "result": True,
                    "response": result["response"],
                    "proof": ""
                }
            
        except Exception as e:
            print(f'❌ TGI processing error: {e}')
            return jsonify({"error": f"TGI processing error: {e}"}), 500
    
    else:
        # print('💬 Processing with local models...')
        # Convert back to string format for internal processing
        data["input"] = json.dumps(input_data)
        
        # Add request to queue
        # print('💬 Adding request to queue...')
        request_uuid = runner_queue.add_request(data)
        # print('💬 Request added to queue with UUID ' + str(request_uuid))

        # Wait for the request to be processed (with timeout)
        # print('💬 Waiting for request to be processed...')
        deadline = time.time() + float(os.getenv('REQUEST_TIMEOUT_SECONDS', '3600'))
        last_log = 0
        while True:
            request_data = runner_queue.get_request(request_uuid)
            if request_data is None:
                print('❌ Request disappeared from queue unexpectedly')
                return jsonify({"error": "Internal queue error"}), 500
            if request_data["status"] == "finished":
                # print('💬 Request finished!')
                output = request_data["output"]
                runner_queue.remove_request(request_uuid)
                break
            if time.time() > deadline:
                print('❌ Request timed out')
                return jsonify({"error": "Request timed out"}), 504
            if time.time() - last_log > 5:
                # print(f"⏳ Still waiting... status={request_data['status']} id={request_uuid}")
                last_log = time.time()
            time.sleep(0.1)

    # Check if output is valid
    # print('💬 Checking output...')
    if output == None or not output["result"]:
        print('❌ Invalid output')
        return jsonify({"error": output["error"] if output != None else "Invalid output"}), 400
    
    # Store output in cache
    if CACHE_ENABLED:
        # print('💬 Storing response in cache...')
        app_cache[body_hash] = {
            "output": output,
            "timestamp": time.time()
        }
        for key in list(app_cache): # clean up cache
            if time.time() - app_cache[key]["timestamp"] > 60 * 60:
                del app_cache[key]

    # Track request statistics for monitoring
    request_time = time.time() - time_start
    timestamp = datetime.datetime.now().isoformat()
    
    # Handle input text extraction differently for TGI vs local models
    if use_tgi:
        # For TGI, the input is the messages array
        input_text = json.dumps(input_data.get("messages", []))
        model_name = f"tgi-{data.get('model', 'unknown')}"
    else:
        # For local models, use the original input format
        input_text = data.get("input", "")
        model_name = data.get("model", "unknown")
    
    # Create a unique ID by concatenating timestamp and a hash of the input
    request_id = f"{timestamp}_{hash(input_text) & 0xffffffff}"
    
    request_record = {
        "id": request_id,
        "timestamp": timestamp,
        "time_taken": request_time,
        "model": model_name,
        "input_length": len(input_text),
        "tokens_per_second": 0,
        "total_tokens_generated": 0,
        "success": True,
        "input": input_text,
        "output": output.get("response", ""),
        "processing_method": "tgi" if use_tgi else "local"
    }
    
    # Try to extract token information from output if available
    if output and "proof" in output and output["proof"]:
        try:
            if use_tgi:
                # For TGI, we might have already parsed proof data
                if isinstance(output.get("proof"), str):
                    output_proof = json.loads(unzip_string(output["proof"]))
                else:
                    output_proof = output.get("proof", {})
            else:
                # For local models, decompress the proof
                output_proof = json.loads(unzip_string(output["proof"]))
            
            tokens = output_proof.get("tokens", [])
            # Calculate token statistics
            if tokens:
                request_record["tokens_per_second"] = len(tokens) / request_time if request_time > 0 else 0
                request_record["total_tokens_generated"] = len(tokens)
            else:
                request_record["tokens_per_second"] = 0
                request_record["total_tokens_generated"] = 0
        except Exception as e:
            print(f'❌ Error extracting token information: {e}')
            request_record["tokens_per_second"] = 0
            request_record["total_tokens_generated"] = 0
    request_history.append(request_record)
    
    # Keep only the last 1000 requests in history (up from 100)
    # This allows for more comprehensive monitoring data
    if len(request_history) > 1000:
        request_history.pop(0)

    # Return response
    # print('✅ Returning response in ' + str(time.time() - time_start) + ' seconds')
    return jsonify(output)

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
    
    
    # Statistiche del servizio
    uptime = datetime.datetime.now() - service_start_time
    uptime_seconds = uptime.total_seconds()
    
    # Statistiche delle richieste
    request_stats = {
        "total_requests": len(request_history),
        "average_request_time": sum(req["time_taken"] for req in request_history) / len(request_history) if request_history else 0,
        "average_tokens_per_second": sum(req["tokens_per_second"] for req in request_history) / len(request_history) if request_history else 0,
        "total_tokens_generated": sum(req["total_tokens_generated"] for req in request_history),
        "recent_requests": request_history[-10:] if request_history else [],  # Last 10 requests for backwards compatibility
        "all_requests": request_history  # All requests in the history
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
        "requests": request_stats,
        "gc_collected": collected
    })

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    
    # Start monitoring service
    monitoring_service.start()
    
    try:
        app.run(host='0.0.0.0', port=8888, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    finally:
        cleanup_services()