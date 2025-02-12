class ModelManager:
  def __init__(self, models_config):
    raise NotImplementedError("Method not implemented")

  def switch_model(self, model_name):
    raise NotImplementedError("Method not implemented")

  def clear_model(self):
    raise NotImplementedError("Method not implemented")

  def run_inference(self, data):
    raise NotImplementedError("Method not implemented")