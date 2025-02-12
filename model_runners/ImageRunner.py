import json
from lib.ModelRunner import ModelRunner

class ImageRunner(ModelRunner):
  def run(self, input, model_manager):
    # be sure input is a string
    if not isinstance(input, str):
      return self._generate_error("input parameter must be a string")

    response = model_manager.run_inference(input)
    # response.save("ImageRunner_last.jpg")
    response_as_bytes = response.tobytes()
    response_as_hex_string = response_as_bytes.hex()
    return self._generate_output(response_as_hex_string)
