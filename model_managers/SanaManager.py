import time
import torch
from diffusers import SanaPipeline
from lib.ModelManager import ModelManager

class SanaManager(ModelManager):
  def __init__(self, models_config):
    self.models_config = models_config
    self.models = {}
    self.current_model = None
    self.default_model = models_config[0]["name"]
    self.seed = 1312

    # Load models into CPU memory (pinned for fast GPU transfer).
    for model_config in models_config:
      name = model_config["name"]
      print(f"Loading model {name}...")

      pipe = SanaPipeline.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
      )
      pipe.to("cpu")
      pipe.vae.to("cpu")
      pipe.text_encoder.to("cpu")
      self.models[name] = pipe

  def switch_model(self, model_name):
    print(f"[Switch] Switching to model '{model_name}'...")

    # Skip if the model is already loaded.
    if model_name == self.current_model:
      print(f"[Switch] Model '{model_name}' is already loaded")
      return True
    start_time = time.time()

    # TODO: Da gestire come riportare il modello attivo su cpu

    model = self.models[model_name]
    model.to("cuda")
    model.vae.to("cuda", dtype=torch.bfloat16)
    model.text_encoder.to("cuda", dtype=torch.bfloat16)
    self.current_model = model_name

    switch_time = time.time() - start_time
    print(f"[Switch] Switching to model '{model_name}' took {switch_time:.2f} s")
    return True


  def clear_model(self):
    if not self.current_model:
      print("[Clear] No model loaded")
      return False

    print(f"[Clear] Clearing model '{self.current_model}'...")
    model = self.models[self.current_model]
    model.to("cpu")
    model.vae.to("cpu")
    model.text_encoder.to("cpu")
    self.current_model = None
    return True

  def run_inference(self, prompt):
    model = self.models[self.current_model]
    image = model(
      prompt=prompt,
      height=1024,
      width=1024,
      guidance_scale=3.5,
      num_inference_steps=25,
      generator=torch.Generator(device="cuda").manual_seed(self.seed),
    )[0]

    return image[0]