import json
from lib.zipper import zip_string, unzip_string

class AbstractExecutor:
    def execute(self, inputs, model_manager, on_input_finished):
        raise NotImplementedError("execute method not implemented")

    def check(self, inputs, proofs, model_manager, on_input_finished):
        raise NotImplementedError("check method not implemented")

    def _generate_error(self, message):
        return {
            "result": False,
            "error": message
        }

    def _generate_output(self, response, proof=None):
        return {
            "result": True,
            "response": response,
            "proof": proof
        }

class ChatExecutor(AbstractExecutor):
    def _validate_input(self, input):
        # Parse input as json
        input_data = None
        try:
            input_data = json.loads(input)
        except:
            return self._generate_error("input parameter must be a valid json string")
        # Be sure messages input is a list
        if not isinstance(input_data["messages"], list):
            return self._generate_error("messages parameter must be a list")
        # Be sure messages are objects with role and content keys, be sure content is a string and role is a string with values "system" or "user" or "assistant"
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
        return None

    def _validate_proof(self, proof):
        # Be sure proof is a string
        if not isinstance(proof, str):
            return self._generate_error("proof parameter must be a string")
        # Unzip proof
        proof_string = None
        try:
            proof_string = unzip_string(proof)
        except:
            return self._generate_error("proof parameter must be a valid base64 string")
        # Parse proof as json
        proof_data = None
        try:
            proof_data = json.loads(proof_string)
        except:
            return self._generate_error("proof parameter must be a valid json string")
        # Be sure proof has a tokens key
        if "tokens" not in proof_data:
            return self._generate_error("proof parameter must have a tokens key")
        # Be sure tokens is a list
        if not isinstance(proof_data["tokens"], list):
            return self._generate_error("proof tokens must be a list")
        # Be sure tokens are objects with id(integer), prob(float), and index(integer) keys
        for token in proof_data["tokens"]:
            if not isinstance(token, dict):
                return self._generate_error("each token must be an object")
            if "id" not in token:
                return self._generate_error("each token must have an id key")
            if "prob" not in token:
                return self._generate_error("each token must have a prob key")
            if "index" not in token:
                return self._generate_error("each token must have an index key")
            if not isinstance(token["id"], int):
                return self._generate_error("each token id must be an integer")
            if not isinstance(token["prob"], float):
                return self._generate_error("each token prob must be a float")
            if not isinstance(token["index"], int):
                return self._generate_error("each token index must be an integer")
        # Be sure proof has a full_sequence_length key
        if "full_sequence_length" not in proof_data:
            return self._generate_error("proof parameter must have a full_sequence_length key")
        # Be sure full_sequence_length is an integer
        if not isinstance(proof_data["full_sequence_length"], int):
            return self._generate_error("proof full_sequence_length must be an integer")
        return None

    def execute(self, inputs, model_manager, on_input_finished):
        valid_inputs = []

        # Validate inputs and exclude invalid
        for i, input in enumerate(inputs):
            error = self._validate_input(input)
            if error is None:
                valid_inputs.append({
                    "messages": json.loads(input)["messages"],
                    "index": i
                })
            else:
                on_input_finished(i, error)
        if len(valid_inputs) == 0:
            return

        # Run valid inputs
        def on_valid_input_finished(index, output):
            on_input_finished(valid_inputs[index]["index"], self._generate_output(output['response'], zip_string(json.dumps(output['proof'])) if output['proof'] is not None else None))
        model_manager.run_batch_executions([input["messages"] for input in valid_inputs], on_valid_input_finished)

    def check(self, inputs, proofs, model_manager, on_input_finished):
        valid_inputs_with_tokens = []

        # Validate inputs and proofs and exclude invalid
        for i, input in enumerate(inputs):
            error_input = self._validate_input(input)
            error_proof = self._validate_proof(proofs[i])
            if error_input is None and error_proof is None:
                valid_inputs_with_tokens.append({
                    "messages": json.loads(input)["messages"],
                    "proof": json.loads(unzip_string(proofs[i])),
                    "index": i
                })
            else:
                on_input_finished(i, error_input if error_input is not None else error_proof)
        if len(valid_inputs_with_tokens) == 0:
            return

        # Run valid inputs
        def on_valid_input_finished(index, output):
            on_input_finished(valid_inputs_with_tokens[index]["index"], self._generate_output(output['response'], zip_string(json.dumps(output['proof'])) if output['proof'] is not None else None))
        model_manager.run_batch_checks([input["messages"] for input in valid_inputs_with_tokens], [input["proof"] for input in valid_inputs_with_tokens], on_valid_input_finished)