import json

class AbstractRunner:
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
  
class ChatRunner(AbstractRunner):
    def run(self, input, model_manager):
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
        with open("ChatRunner_last.json", "w") as f:
            json.dump(response, f)
        return self._generate_output(response)

class ImageRunner(AbstractRunner):
    def run(self, input, model_manager):
        # be sure input is a string
        if not isinstance(input, str):
            return self._generate_error("input parameter must be a string")

        response = model_manager.run_inference(input)
        response.save("ImageRunner_last.jpg")
        response_as_bytes = response.tobytes()
        response_as_hex_string = response_as_bytes.hex()
        return self._generate_output(response_as_hex_string)
