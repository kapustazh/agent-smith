import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.sandbox_connector import SandboxClient


class ConnectorTests(unittest.TestCase):
    def client(self, body: str) -> SandboxClient:
        spawn = subprocess.Popen
        script = "import sys, time\n" + body
        with patch("src.sandbox_connector.subprocess.Popen", side_effect=(
            lambda *args, **kwargs: spawn(
                [sys.executable, "-u", "-c", script], **kwargs,
            )
        )):
            client = SandboxClient()
        self.addCleanup(client.close)
        return client

    def test_fragmented_response_waits_for_complete_json(self) -> None:
        payload = json.dumps({"success": True, "tool_name": "start_task",
                              "sandbox_output": "готово 😀"},
                             ensure_ascii=False).encode()
        client = self.client(
            "for line in sys.stdin:\n"
            " if line.strip() != '<<<END_COMMAND>>>': continue\n"
            f" data = {payload!r}\n"
            " for byte in data:\n"
            "  sys.stdout.buffer.write(bytes([byte])); sys.stdout.flush()\n"
            "  time.sleep(.001)\n"
            " print()\n",
        )
        result = client.run_tool("print(start_task())", max_sec=2)
        self.assertTrue(result.success, result)
        self.assertEqual(result.sandbox_output, "готово 😀")

    def test_timeout_is_explicit_and_closes_pending_request(self) -> None:
        client = self.client("sys.stdin.readline(); time.sleep(30)\n")
        result = client.run_tool("start_task()", max_sec=.05)
        self.assertFalse(result.success)
        self.assertEqual(result.status, "timeout")
        self.assertIn("timed out", str(result.sandbox_output))
        self.assertNotIn("Invalid JSON", str(result.sandbox_output))
        self.assertIsNotNone(client.process.poll())

    def test_eof_reports_process_failure(self) -> None:
        client = self.client("sys.stdin.readline(); sys.exit(7)\n")
        result = client.run_tool("start_task()", max_sec=1)
        self.assertFalse(result.success)
        self.assertIn("closed", str(result.sandbox_output))

    def test_two_responses_in_one_read_remain_separate(self) -> None:
        result = {"success": True, "tool_name": "python",
                  "sandbox_output": "one"}
        second = {**result, "sandbox_output": "two"}
        client = self.client(
            "for line in sys.stdin:\n"
            " if line.strip() != '<<<END_COMMAND>>>': continue\n"
            f" print({json.dumps(result)!r})\n"
            f" print({json.dumps(second)!r})\n"
            " sys.stdin.readline()\n"
            " break\n",
        )
        first = client.run_tool("pass", max_sec=1)
        self.assertTrue(first.success)
        self.assertEqual(first.sandbox_output, "one")
        second_result = client.run_tool("pass", max_sec=1)
        self.assertEqual(second_result.sandbox_output, "two")

    def test_delayed_response_is_not_parsed_as_empty_json(self) -> None:
        payload = json.dumps({"success": True, "tool_name": "start_task",
                              "sandbox_output": "started"})
        client = self.client(
            "for line in sys.stdin:\n"
            " if line.strip() != '<<<END_COMMAND>>>': continue\n"
            " time.sleep(.1)\n"
            f" print({payload!r})\n",
        )
        self.assertTrue(client.run_tool("start_task()", max_sec=1).success)

    def test_default_wait_allows_docker_preparation(self) -> None:
        payload = json.dumps({"success": True, "tool_name": "run_tests",
                              "sandbox_output": "evaluation complete"})
        client = self.client(
            "for line in sys.stdin:\n"
            " if line.strip() != '<<<END_COMMAND>>>': continue\n"
            f" print({payload!r})\n",
        )
        with patch("src.sandbox_connector.monotonic",
                   side_effect=[0, *[15] * 10]):
            result = client.run_tool("print(run_tests())")
        self.assertTrue(result.success, result)
        self.assertIsNone(client.process.poll())
