import os

# UOMI Engine version
UOMI_ENGINE_PALLET_VERSION = 5
# Enable/disable usage of caching on the server
CACHE_ENABLED = False
# Max number of requests executed in batch mode
BATCH_MAX_SIZE = 5
# Time to wait before starting a new batch
BATCH_WAIT_SEC = 0.1
# Folder for caching models
MODELS_FOLDER = "./models"
# Inference settings for transformers models
TRANSFORMERS_INFERENCE_MAX_TOKENS = 1024
TRANSFORMERS_INFERENCE_TEMPERATURE = 0.6
# KV Cache
USE_KV_CACHE = True
# Monitoring websocket configuration
MONITORING_WEBSOCKET_URL = os.getenv('MONITORING_WEBSOCKET_URL', "ws://telemetry-ai.uomi.ai:3001")  # Set to websocket URL to enable monitoring
MONITORING_INTERVAL_SECONDS = int(os.getenv('MONITORING_INTERVAL_SECONDS', 10))  # Interval in seconds to send monitoring data