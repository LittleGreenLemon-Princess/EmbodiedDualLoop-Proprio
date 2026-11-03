import json
import unittest

from orchestrator.openrouter_agent_host import parallel_tool_rejection_messages


class ParallelToolRejectionTests(unittest.TestCase):
    def test_rejects_every_parallel_call_without_running_any(self):
        calls = [{"id": "call-a"}, {"id": "call-b"}]

        messages = parallel_tool_rejection_messages(calls)

        self.assertEqual([message["tool_call_id"] for message in messages],
                         ["call-a", "call-b"])
        self.assertTrue(all(message["role"] == "tool" for message in messages))
        payloads = [json.loads(message["content"]) for message in messages]
        self.assertTrue(all("No tool in the parallel response was executed" in payload["error"]
                            for payload in payloads))
        self.assertTrue(all("do not replay actions" in payload["error"]
                            for payload in payloads))

    def test_refuses_to_build_rejection_for_a_single_call(self):
        with self.assertRaises(ValueError):
            parallel_tool_rejection_messages([{"id": "call-a"}])


if __name__ == "__main__":
    unittest.main()
