class ModelRunner:
  def run(self, input):
    raise NotImplementedError("run method not implemented")

  def _generate_error(self, message):
    return {
      "result": False,
      "error": message
    }

  def _generate_output(self, response):
    return {
      "result": True,
      "response": response
    }