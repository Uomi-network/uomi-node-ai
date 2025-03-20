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

-----

Refactor this code to be sure ignoring tokens with min_p <= 10^-4

for step in range(TRANSFORMERS_INFERENCE_MAX_TOKENS):
    print(f"Step execution {step + 1}/{TRANSFORMERS_INFERENCE_MAX_TOKENS}")
    if not active_batch_indices:
        break  # All inferences have completed
    # Get active batch
    active_input_ids = batched_input_ids[active_batch_indices]
    active_attention_masks = batched_attention_masks[active_batch_indices]
    # Forward pass on the active batch
    outputs = self.current_gpu_model(input_ids=active_input_ids, attention_mask=active_attention_masks)
    # Process each active inference
    new_active_batch_indices = []
    for batch_pos, original_idx in enumerate(active_batch_indices):
        # Get the position of the last token in this sequence
        last_pos = current_positions[original_idx] - 1
        next_token_logits = outputs.logits[batch_pos, last_pos, :].unsqueeze(0)
        # Apply temperature (if not 1.0)
        if TRANSFORMERS_INFERENCE_TEMPERATURE != 1.0:
            next_token_logits = next_token_logits / TRANSFORMERS_INFERENCE_TEMPERATURE
        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        # Get top-k tokens by probability
        top_probs, top_indices = probs.topk(5, dim=-1)
        top_tokens = {}
        for i, idx in enumerate(top_indices[0]):
            prob = probs[0, idx].item()
            top_tokens[idx.item()] = {
                "prob": prob,
                "index": i
            }
        # Sample from the top-k tokens
        next_token_id = top_indices.select(-1, torch.multinomial(top_probs, num_samples=1).item()).unsqueeze(0)
        selected_token_id = next_token_id.item()
        # Record the first token if this is the first step
        if step == 0:
            first_new_token_ids[original_idx] = selected_token_id
        # Add the selected token to the input sequence for next iteration
        pos = current_positions[original_idx]
        batched_input_ids[original_idx, pos] = selected_token_id
        batched_attention_masks[original_idx, pos] = 1
        current_positions[original_idx] += 1
        
        # Store the token on all_output_tokens
        all_output_tokens[original_idx].append({
            "id": selected_token_id,
            "prob": top_tokens[selected_token_id]["prob"],
            "index": top_tokens[selected_token_id]["index"],
        })
        # Check if the generated token is the </s> token
        if selected_token_id == eos_token_id:
            completed_sequences[original_idx] = True
        # Only keep sequences that haven't generated </s> in the active batch
        if completed_sequences[original_idx]:
            prompt_text = tokenizer.decode(all_input_ids[original_idx].squeeze(0).tolist(), skip_special_tokens=True)
            response = tokenizer.decode(batched_input_ids[original_idx, :current_positions[original_idx]].tolist(), skip_special_tokens=True)  
            response = response[len(prompt_text):]
            
            output = {
                "response": response,
                "proof": {
                    "tokens": all_output_tokens[original_idx],
                    "full_sequence_length": full_sequence_length
                }
            }
            on_prompt_finished(original_idx, output)
        else:
            new_active_batch_indices.append(original_idx)
    # Update active batch indices
    active_batch_indices = new_active_batch_indices