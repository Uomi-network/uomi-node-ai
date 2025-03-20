import time
import sys
from flask import Flask, request, jsonify
from lib.config import UOMI_ENGINE_PALLET_VERSION, CACHE_ENABLED
from lib.runner import RunnerQueue, RunnerExecutor
from lib.system_tmp import System

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

    # Return response
    print('✅ Returning response in ' + str(time.time() - time_start) + ' seconds')
    return jsonify(output)

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(host='0.0.0.0', port=8888, debug=False)