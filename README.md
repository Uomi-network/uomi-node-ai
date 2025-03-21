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

View service logs in real-time:
```bash
journalctl -f -u uomi-ai
```

Check service status:
```bash
systemctl status uomi-ai
```

## 🔧 API Usage

The service exposes an HTTP endpoint at `http://localhost:8888/run` accepting POST requests with the following JSON structure:

```json
{
  "model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
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
