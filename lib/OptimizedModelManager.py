from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class OptimizedModelManager:
    def __init__(self, model_name: str = "tiiuae/falcon-rw-1b"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def run_inference(self, prompt: str) -> str:
        # Tokenize the prompt text
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Generate the model's response
        outputs = self.model.generate(
            inputs['input_ids'],
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id  # Handling pad token
        )

        # Decode the generated response and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
