import sys
import os

# Add the parent directory to the Python path so we can import from lib/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.OptimizedModelManager import OptimizedModelManager

# Initialize model manager with a predefined model
manager = OptimizedModelManager(model_name="tiiuae/falcon-rw-1b")

# Define test prompts to validate inference
test_prompts = [
    "What is the capital of France?",
    "Who wrote the play Romeo and Juliet?",
    "What is 5 multiplied by 7?",
    "Explain photosynthesis in simple terms.",
    "Why do we see rainbows?"
]

# Running inference for each prompt
print("\n🧪 Running Inference Tests...\n")
for prompt in test_prompts:
    print(f"🧠 Prompt: {prompt}")
    response = manager.run_inference(prompt)
    print(f"🗣️ Response: {response}\n{'-' * 60}\n")
