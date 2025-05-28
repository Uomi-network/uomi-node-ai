# UOMI Node AI

UOMI Node AI is a crucial component for running a validator node on the UOMI blockchain. This service manages the AI model required for blockchain validation operations.

## 🔍 Overview

This repository contains the necessary components to set up and run the AI service required for UOMI blockchain validation.

## 🚀 Features

- Automated installation of dependencies via install script
- Systemd service integration for reliable operation
- CUDA-optimized AI model execution
- Deterministic model outputs for consistent validation
- RESTful API endpoint for model interactions
- Real-time logging and monitoring capabilities

## 📋 Requirements

- CUDA-capable GPU(s)
- Ubuntu/Debian-based system
- Conda package manager
- Systemd (for service management)
- Minimum 64GB RAM recommended
- CUDA Toolkit 11.x or higher

## Nvidia Driver Installation

```bash
sudo apt-get purge nvidia-*
sudo apt-get update
sudo apt-get autoremove
sudo apt install libnvidia-common-530
sudo apt install nvidia-driver-530
# Reboot
nvidia-smi
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/uomi-node-ai
cd uomi-node-ai
```

2. Run the installation script:
```bash
chmod +x install.sh
./install.sh
```

3. Configure the systemd service:
```bash
sudo cp uomi-ai.service /etc/systemd/system/
sudo nano /etc/systemd/system/uomi-ai.service  # Edit paths as needed
```

4. Enable and start the service:
```bash
sudo systemctl enable uomi-ai
sudo systemctl start uomi-ai
```

## 📊 Monitoring

UOMI Node AI includes built-in monitoring capabilities that can send real-time performance data to a WebSocket endpoint.

### Configuration

To enable monitoring, set the following environment variables:

```bash
# Required: WebSocket URL to send monitoring data to
export MONITORING_WEBSOCKET_URL="ws://your-monitoring-server.com:8080/monitoring"

# Optional: Monitoring interval in seconds (default: 10)
export MONITORING_INTERVAL_SECONDS=15
```

### Monitoring Data

The monitoring service sends the following data every interval:

- **System metrics**: CPU usage, memory usage, disk usage
- **CUDA metrics**: GPU memory allocation, device information
- **Request statistics**: Total requests, average response time, tokens per second
- **Service uptime**: Days, hours, minutes, seconds since startup
- **Garbage collection**: Memory cleanup statistics

### Example Monitoring Data

```json
{
  "type": "monitoring",
  "timestamp": "2025-05-28T10:30:00.123456",
  "data": {
    "uptime": {
      "total_seconds": 3600,
      "days": 0,
      "hours": 1,
      "minutes": 0,
      "seconds": 0
    },
    "system": {
      "cpu_percent": 45.2,
      "memory": {
        "total_gb": 64.0,
        "used_gb": 32.1,
        "percent": 50.2
      }
    },
    "cuda": {
      "device_count": 2,
      "devices": [
        {
          "name": "NVIDIA RTX 4090",
          "memory_allocated": 12.5
        }
      ]
    },
    "requests": {
      "total_requests": 150,
      "average_request_time": 2.34,
      "average_tokens_per_second": 45.6
    }
  }
}
```

### Testing Monitoring

Use the included test WebSocket server:

```bash
# Install websockets dependency for testing
pip install websockets

# Start test server
./test_monitoring.sh

# In another terminal, start uomi-ai with monitoring
export MONITORING_WEBSOCKET_URL="ws://localhost:8080"
python uomi-ai.py
```

## 🔧 API Usage

The service exposes an HTTP endpoint at `http://localhost:8888/run` accepting POST requests with the following JSON structure:

```json
{
  "model": "casperhansen/mistral-small-24b-instruct-2501-awq",
  "input": {
    "messages": [
      {
        "role": "system",
        "content": "System message here"
      },
      {
        "role": "user",
        "content": "User input here"
      }
    ]
  }
}
```

## ⚙️ Configuration

The service is configured for optimal performance with:
- Deterministic model execution
- CUDA optimization settings
- Automatic GPU device selection
- Fixed random seeds for reproducibility

## 🔒 Security Notes

- The service runs on port 8888 by default
- Implement appropriate firewall rules if exposing the service

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🧑‍💻 Testing

To run tests, execute the following command:
```bash
python -m unittest discover -s tests -p "*_test.py"
```

## ⚠️ Troubleshooting

If you encounter issues:

1. Check CUDA availability:
```bash
nvidia-smi
```

2. Verify conda environment:
```bash
conda env list
```

3. Check service logs for errors:
```bash
journalctl -xe -u uomi-ai
```

## Useful links

[UOMI website](https://uomi.ai)

[Docs](https://docs.uomi.ai)
