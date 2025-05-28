import time
import sys
import datetime
import gc
import psutil
import torch
import atexit
from flask import Flask, request, jsonify
from lib.config import UOMI_ENGINE_PALLET_VERSION, CACHE_ENABLED
from lib.runner import RunnerQueue, RunnerExecutor
from lib.system import System
from lib.monitoring import MonitoringService

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

# Initialize monitoring service
monitoring_service = MonitoringService(app)

# Setup cleanup on exit
def cleanup_services():
    monitoring_service.stop()

atexit.register(cleanup_services)

@app.route('/status', methods=['GET'])
def status_json():
    return jsonify({
        "UOMI_ENGINE_PALLET_VERSION": UOMI_ENGINE_PALLET_VERSION,
        "details": {
            "system_valid": system.check_system_requirements(),
            "cuda_available": system.check_cuda_availability()
        }
    })

@app.route('/run', methods=['POST'])
def run_json():
    global app_cache
    time_start = time.time()

    print('💬 Received request...')
    data = request.get_json()

    # Check if the response can be returned from cache
    if CACHE_ENABLED:
        print('💬 Checking response on cache...')
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
    # be sure input parameter is a string
    if not isinstance(data["input"], str):
        print('❌ Input parameter must be a string')
        return jsonify({"error": "input parameter must be a string"}), 400
    
    # Add request to queue
    print('💬 Adding request to queue...')
    request_uuid = runner_queue.add_request(data)
    print('💬 Request added to queue with UUID ' + str(request_uuid))

    # Wait for the request to be processed
    print('💬 Waiting for request to be processed...')
    while True:
        request_data = runner_queue.get_request(request_uuid)
        if request_data["status"] == "finished":
            print('💬 Request finished!')
            output = request_data["output"]
            runner_queue.remove_request(request_uuid)
            break
        time.sleep(0.1)

    # Check if output is valid
    print('💬 Checking output...')
    if output == None or not output["result"]:
        print('❌ Invalid output')
        return jsonify({"error": output["error"] if output != None else "Invalid output"}), 400
    
    # Store output in cache
    if CACHE_ENABLED:
        print('💬 Storing response in cache...')
        app_cache[body_hash] = {
            "output": output,
            "timestamp": time.time()
        }
        for key in list(app_cache): # clean up cache
            if time.time() - app_cache[key]["timestamp"] > 60 * 60:
                del app_cache[key]

    # Track request statistics for monitoring
    request_time = time.time() - time_start
    request_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "time_taken": request_time,
        "model": data.get("model", "unknown"),
        "input_length": len(data.get("input", "")),
        "tokens_per_second": 0,  # This would need to be calculated based on actual token generation
        "total_tokens_generated": 0,  # This would need to be extracted from the output
        "success": True
    }
    
    # Try to extract token information from output if available
    if output and "result" in output:
        # You might need to adjust this based on your actual output structure
        if "tokens" in output:
            request_record["total_tokens_generated"] = output.get("tokens", 0)
            if request_time > 0:
                request_record["tokens_per_second"] = output.get("tokens", 0) / request_time
    
    request_history.append(request_record)
    
    # Keep only the last 100 requests in history
    if len(request_history) > 100:
        request_history.pop(0)

    # Return response
    print('✅ Returning response in ' + str(time.time() - time_start) + ' seconds')
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