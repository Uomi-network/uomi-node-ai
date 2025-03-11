import unittest
from lib.runner import RunnerQueue, RunnerExecutor

class TestRunner(unittest.TestCase):
    def test_runner_queue(self):
        runner_queue = RunnerQueue()

        request_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request_uuid = runner_queue.add_request(request_content)
        self.assertTrue(request_uuid is not None)

        request_data = runner_queue.get_request(request_uuid)
        self.assertEqual(request_data["uuid"], request_uuid)
        self.assertEqual(request_data["status"], "pending")
        self.assertEqual(request_data["request"], request_content)
        self.assertTrue(request_data["timestamp_pending"] is not None)
        self.assertTrue(request_data["timestamp_running"] is None)
        self.assertTrue(request_data["timestamp_finished"] is None)
        self.assertTrue(request_data["output"] is None)
        self.assertTrue(request_data["batch"] is None)

        requests_data = runner_queue.get_requests()
        self.assertTrue(isinstance(requests_data, dict))
        self.assertEqual(len(requests_data), 1)
        self.assertEqual(requests_data[request_uuid], request_data)

        runner_queue.remove_request(request_uuid)
        requests_data = runner_queue.get_requests()
        self.assertTrue(isinstance(requests_data, dict))
        self.assertEqual(len(requests_data), 0)

    def test_runner_executor_with_invalid_model(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request_content = {"model": "invalid", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request_uuid = runner_queue.add_request(request_content)
        while runner_queue.get_request(request_uuid)["status"] != "finished":
            pass

        request_data = runner_queue.get_request(request_uuid)
        self.assertEqual(request_data["status"], "finished")
        self.assertFalse(request_data["output"]["result"])
        self.assertTrue("error" in request_data["output"])
        self.assertTrue(len(request_data["batch"]) == 1)

        runner_executor.stop()

    def test_runner_executor_with_invalid_input(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request_content = {"model": "test/1", "input": "invalid"}
        request_uuid = runner_queue.add_request(request_content)
        while runner_queue.get_request(request_uuid)["status"] != "finished":
            pass

        request_data = runner_queue.get_request(request_uuid)
        self.assertEqual(request_data["status"], "finished")
        self.assertFalse(request_data["output"]["result"])
        self.assertTrue("error" in request_data["output"])
        self.assertTrue(len(request_data["batch"]) == 1)

        runner_executor.stop()

    def test_runner_executor_with_invalid_proof(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}", "proof": "invalid"}
        request_uuid = runner_queue.add_request(request_content)
        while runner_queue.get_request(request_uuid)["status"] != "finished":
            pass

        request_data = runner_queue.get_request(request_uuid)
        self.assertEqual(request_data["status"], "finished")
        self.assertFalse(request_data["output"]["result"])
        self.assertTrue("error" in request_data["output"])
        self.assertTrue(len(request_data["batch"]) == 1)

        runner_executor.stop()

    def test_runner_executor_valid_execution(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request_uuid = runner_queue.add_request(request_content)
        while runner_queue.get_request(request_uuid)["status"] != "finished":
            pass

        request_data = runner_queue.get_request(request_uuid)
        self.assertEqual(request_data["status"], "finished")
        self.assertTrue(request_data["output"]["result"])
        self.assertTrue("response" in request_data["output"])
        self.assertTrue("proof" in request_data["output"])
        self.assertTrue(isinstance(request_data["output"]["response"], str))
        self.assertTrue(isinstance(request_data["output"]["proof"], str))
        self.assertTrue(len(request_data["batch"]) == 1)

        runner_executor.stop()

    def test_runner_executor_valid_check(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}", "proof": "eJyrVirJz07NK1ayUoiuVspMAdIGOgpKBUX5SUCmoR6Ik5mXkloBkqiNBfLSSnNy4otTC0tT85JT43NS89JLMkBKawFSVBc7"}
        request_uuid = runner_queue.add_request(request_content)
        while runner_queue.get_request(request_uuid)["status"] != "finished":
            pass

        request_data = runner_queue.get_request(request_uuid)
        self.assertEqual(request_data["status"], "finished")
        self.assertTrue(request_data["output"]["result"])
        self.assertTrue("response" in request_data["output"])
        self.assertTrue("proof" in request_data["output"])
        self.assertTrue(isinstance(request_data["output"]["response"], str))
        self.assertTrue(request_data["output"]["proof"] is None)
        self.assertTrue(len(request_data["batch"]) == 1)

        runner_executor.stop()

    def test_runner_executor_batched(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request1_uuid = runner_queue.add_request(request_content)
        request2_uuid = runner_queue.add_request(request_content)
        while runner_queue.get_request(request1_uuid)["status"] != "finished" or runner_queue.get_request(request2_uuid)["status"] != "finished":
            pass

        request1_data = runner_queue.get_request(request1_uuid)
        self.assertEqual(request1_data["status"], "finished")
        self.assertTrue(request1_data["output"]["result"])
        self.assertTrue("response" in request1_data["output"])
        self.assertTrue("proof" in request1_data["output"])
        self.assertTrue(isinstance(request1_data["output"]["response"], str))
        self.assertTrue(isinstance(request1_data["output"]["proof"], str))
        self.assertTrue(len(request1_data["batch"]) == 2)

        request2_data = runner_queue.get_request(request2_uuid)
        self.assertEqual(request2_data["status"], "finished")
        self.assertTrue(request2_data["output"]["result"])
        self.assertTrue("response" in request2_data["output"])
        self.assertTrue("proof" in request2_data["output"])
        self.assertTrue(isinstance(request2_data["output"]["response"], str))
        self.assertTrue(isinstance(request2_data["output"]["proof"], str))
        self.assertTrue(len(request2_data["batch"]) == 2)

        self.assertEqual(request1_data["batch"], request2_data["batch"])
        
        runner_executor.stop()

    def test_runner_executor_not_batched_different_models(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request1_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request2_content = {"model": "test/2", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request1_uuid = runner_queue.add_request(request1_content)
        request2_uuid = runner_queue.add_request(request2_content)
        while runner_queue.get_request(request1_uuid)["status"] != "finished" or runner_queue.get_request(request2_uuid)["status"] != "finished":
            pass

        request1_data = runner_queue.get_request(request1_uuid)
        self.assertEqual(request1_data["status"], "finished")
        self.assertTrue(request1_data["output"]["result"])
        self.assertTrue("response" in request1_data["output"])
        self.assertTrue("proof" in request1_data["output"])
        self.assertTrue(isinstance(request1_data["output"]["response"], str))
        self.assertTrue(isinstance(request1_data["output"]["proof"], str))
        self.assertTrue(len(request1_data["batch"]) == 1)

        request2_data = runner_queue.get_request(request2_uuid)
        self.assertEqual(request2_data["status"], "finished")
        self.assertTrue(request2_data["output"]["result"])
        self.assertTrue("response" in request2_data["output"])
        self.assertTrue("proof" in request2_data["output"])
        self.assertTrue(isinstance(request2_data["output"]["response"], str))
        self.assertTrue(isinstance(request2_data["output"]["proof"], str))
        self.assertTrue(len(request2_data["batch"]) == 1)

        runner_executor.stop()

    def test_runner_executor_not_batched_different_run(self):
        runner_queue = RunnerQueue()
        runner_executor = RunnerExecutor(runner_queue, True)

        request1_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}"}
        request2_content = {"model": "test/1", "input": "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a chat and you have to help user\"}, {\"role\":\"user\",\"content\":\"Whats the name of Berlusconi? Answer only with the name.\"}]}", "proof": "eJyrVirJz07NK1ayUoiuVspMAdIGOgpKBUX5SUCmoR6Ik5mXkloBkqiNBfLSSnNy4otTC0tT85JT43NS89JLMkBKawFSVBc7"}
        request1_uuid = runner_queue.add_request(request1_content)
        request2_uuid = runner_queue.add_request(request2_content)
        while runner_queue.get_request(request1_uuid)["status"] != "finished" or runner_queue.get_request(request2_uuid)["status"] != "finished":
            pass

        request1_data = runner_queue.get_request(request1_uuid)
        self.assertEqual(request1_data["status"], "finished")
        self.assertTrue(request1_data["output"]["result"])
        self.assertTrue("response" in request1_data["output"])
        self.assertTrue("proof" in request1_data["output"])
        self.assertTrue(isinstance(request1_data["output"]["response"], str))
        self.assertTrue(isinstance(request1_data["output"]["proof"], str))
        self.assertTrue(len(request1_data["batch"]) == 1)

        request2_data = runner_queue.get_request(request2_uuid)
        self.assertEqual(request2_data["status"], "finished")
        self.assertTrue(request2_data["output"]["result"])
        self.assertTrue("response" in request2_data["output"])
        self.assertTrue("proof" in request2_data["output"])
        self.assertTrue(isinstance(request2_data["output"]["response"], str))
        self.assertTrue(request2_data["output"]["proof"] is None)
        self.assertTrue(len(request2_data["batch"]) == 1)

        runner_executor.stop()

if __name__ == '__main__':
    unittest.main()