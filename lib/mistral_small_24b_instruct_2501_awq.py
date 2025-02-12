import json
import lib.ModelRunner

class MistralSmall24bInstruct2501Awq(lib.ModelRunner.ModelRunner):
  def run(self, input, model_manager):
    print("Running MistralSmall24bInstruct2501Awq model...", input)
    # try to parse input as a json object
    input_data = None
    try:
      input_data = json.loads(input)
    except:
      return self._generate_error("input parameter must be a valid json string")

    # be sure messages input is a list
    if not isinstance(input_data["messages"], list):
      return self._generate_error("messages parameter must be a list")
    # be sure messages are objects with role and content keys, be sure content is a string and role is a string with values "system" or "user" or "assistant"
    for message in input_data["messages"]:
      if not isinstance(message, dict):
        return self._generate_error("each message must be an object")
      if "role" not in message:
        return self._generate_error("each message must have a role key")
      if "content" not in message:
        return self._generate_error("each message must have a content key")
      if not isinstance(message["role"], str):
        return self._generate_error("each message role must be a string")
      if not isinstance(message["content"], str):
        return self._generate_error("each message content must be a string")
      if message["role"] not in ["system", "user", "assistant"]:
        return self._generate_error("each message role must be 'system', 'user', or 'assistant'")

    response = model_manager.run_inference(input_data["messages"])
    return self._generate_output(response)
