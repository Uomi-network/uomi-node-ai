import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from accelerate.hooks import remove_hook_from_submodules

class FastDualModelGPUManager:
  def __init__(self, models_config):
    self.models_config = models_config
    self.models = {}
    self.tokenizers = {}
    self.current_model = None
    self.gpu_devices = [0, 1]
    self.seed = 1312

    # Load models into CPU memory (pinned for fast GPU transfer).
    for model_config in models_config:
      name = model_config["name"]
      print(f"Loading model {name}...")

      model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="cpu",  # Keep entire model on CPU.
        torch_dtype=torch.float16,
      )
      
      self._pin_memory(model)  # Pin memory for faster GPU transfer.
      self.models[name] = model
      self.tokenizers[name] = AutoTokenizer.from_pretrained(name)

  # Switch to a new model by loading its weights into GPU memory.
  def switch_model(self, model_name):
    print(f"[Switch] Switching to model '{model_name}'...")

    # Skip if the model is already loaded.
    if model_name == self.current_model:
      print(f"[Switch] Model '{model_name}' is already loaded")
      return True
    start_time = time.time()

    # Clear GPU memory for the current model.
    if self.current_model:
      old_model = self.models[self.current_model]
      self._clear_gpu_memory(old_model)

    # Dispatch the new model to GPU with an optimal device map.
    new_model = self.models[model_name]
    device_map = self._create_device_map(new_model)
    dispatch_model(new_model, device_map=device_map, main_device="cuda:0")
    self.current_model = model_name

    switch_time = time.time() - start_time
    print(f"[Switch] Switching to model '{model_name}' took {switch_time:.2f} s")
    return True

  # Run an inference on the current model.
  def run_inference(self, messages):
    # Set deterministic or non-deterministic behavior.
    model_config = None
    for config in self.models_config:
      if config["name"] == self.current_model:
        model_config = config
        break
    if model_config["deterministic"]:
      self._set_deterministic()
      default_kwargs = {
        'do_sample': False,
        'num_beams': 1,
        'temperature': 1.0,
        'top_p': 1.0,
        'max_new_tokens': 8196,
      }
    else:
      self._set_nondeterministic()
      default_kwargs = {
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'max_new_tokens': 8196,
      }
    
    # Get the model and tokenizer for the current model.
    model = self.models[self.current_model]
    tokenizer = self.tokenizers[self.current_model]
    text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Run the inference and measure the time taken.
    print(f"[Inference] Running inference on '{self.current_model}'...")
    start_time = time.time()
    outputs = model.generate(
      **inputs,
      **default_kwargs,
      pad_token_id=tokenizer.eos_token_id,
      use_cache=True
      # use_cache=not deterministic # it seems it's not a problem for determinism. use_cache true increases a lot the speed 
    )
    gen_time = time.time() - start_time
    print(f"[Inference] Inference on '{self.current_model}' took {gen_time:.2f} s")

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


  # Set all seeds and backend settings for deterministic behavior.
  def _set_deterministic(self):
    torch.manual_seed(self.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(self.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

  # Set all seeds and backend settings for deterministic behavior.
  def _set_nondeterministic(self):
    torch.manual_seed(self.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

  # Pin all model parameters and buffers for faster CPU-to-GPU transfers.
  def _pin_memory(self, model):
    for param in model.parameters():
      param.data = param.data.pin_memory()
    for buffer in model.buffers():
      buffer.data = buffer.data.pin_memory()

  # Move the model back to CPU and clear GPU memory.
  def _clear_gpu_memory(self, model):
    dispatch_model(model, device_map={"": "cpu"})
    remove_hook_from_submodules(model)
    torch.cuda.empty_cache()

  # Generate an optimal device map for GPU offloading.
  def _create_device_map(self, model):
    max_memory = get_balanced_memory(
      model,
      max_memory={k: "22GB" for k in self.gpu_devices},
      no_split_module_classes=model._no_split_modules
    )
    return infer_auto_device_map(
      model,
      max_memory=max_memory,
      no_split_module_classes=model._no_split_modules
    )