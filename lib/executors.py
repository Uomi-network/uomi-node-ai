import json

class AbstractExecutor:
    def execute(self, inputs, model_manager, on_input_finished):
        raise NotImplementedError("execute method not implemented")

    def _generate_error(self, message):
        return {
            "result": False,
            "error": message
        }

    def _generate_output(self, response, tokens=None):
        return {
            "result": True,
            "response": response,
            "tokens": tokens
        }

class ChatExecutor(AbstractExecutor):
    def execute(self, inputs, model_manager, on_input_finished):
        valid_inputs = []

        # Validate inputs and exclude invalid
        for i, input in enumerate(inputs):
            # parse input as json
            input_data = None
            try:
                input_data = json.loads(input)
            except:
                on_input_finished(i, self._generate_error("input parameter must be a valid json string"))
                continue
            # be sure messages input is a list
            if not isinstance(input_data["messages"], list):
                on_input_finished(i, self._generate_error("messages parameter must be a list"))
                continue
            # be sure messages are objects with role and content keys, be sure content is a string and role is a string with values "system" or "user" or "assistant"
            messages_valid = True
            for message in input_data["messages"]:
                if not isinstance(message, dict):
                    on_input_finished(i, self._generate_error("each message must be an object"))
                    messages_valid = False
                    break
                if "role" not in message:
                    on_input_finished(i, self._generate_error("each message must have a role key"))
                    messages_valid = False
                    break
                if "content" not in message:
                    on_input_finished(i, self._generate_error("each message must have a content key"))
                    messages_valid = False
                    break
                if not isinstance(message["role"], str):
                    on_input_finished(i, self._generate_error("each message role must be a string"))
                    messages_valid = False
                    break
                if not isinstance(message["content"], str):
                    on_input_finished(i, self._generate_error("each message content must be a string"))
                    messages_valid = False
                    break
                if message["role"] not in ["system", "user", "assistant"]:
                    on_input_finished(i, self._generate_error("each message role must be 'system', 'user', or 'assistant'"))
                    messages_valid = False
                    break
            if not messages_valid:
                continue
            valid_inputs.append({
                "messages": input_data["messages"],
                "index": i
            })

        # Run valid inputs
        def on_valid_input_finished(index, output):
            on_input_finished(valid_inputs[index]["index"], self._generate_output(output['string'], output['tokens']))
        model_manager.run_batch_executions([input["messages"] for input in valid_inputs], on_valid_input_finished)

