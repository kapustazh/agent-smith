import asyncio
import json
import os
import shlex
import tempfile
import unittest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import TextContent
from mcp_tools_swebench import create_server
from src.config_model import SandboxConfig, SWEBenchTaskInput
from src.swebench_docker import DockerSession


IMAGE = os.environ.get("SWE_DOCKER_TEST_IMAGE", "")
EVAL = ("cd /testbed\n"
        "python -c 'from calc import add; assert add(2, 3) == 5'\n"
        "echo evaluator-only > evaluation-artifact.txt\n")


def task_input() -> SWEBenchTaskInput:
    return SWEBenchTaskInput(
        instance_id="fixture-1", problem_statement="Fix add.",
        docker_image=IMAGE, eval_script=EVAL,
    )


@unittest.skipUnless(IMAGE, "Run make test-docker for Docker checks.")
class DockerTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        self.config = SandboxConfig(max_memory_mb=64,
                                    max_execution_time_seconds=2)
        self.session = DockerSession(task_input(), self.config)
        self.addAsyncCleanup(self.session.close)
        await self.session.start()

    async def test_limits_and_host_isolation(self) -> None:
        settings = json.loads(await self.session._control(
            "inspect", "--format", "{{json .HostConfig}}",
            self.session.container,
        ))
        self.assertEqual(settings["NetworkMode"], "none")
        self.assertEqual(settings["Memory"], 64 * 1024 * 1024)
        self.assertEqual(settings["MemorySwap"], settings["Memory"])
        self.assertEqual(settings["PidsLimit"], 128)
        self.assertEqual(len(settings["Binds"] or settings["Mounts"]), 1)
        with tempfile.NamedTemporaryFile() as secret:
            result = json.loads(await self.session.run_command(
                "test -f " + shlex.quote(secret.name),
            ))
            self.assertNotEqual(result["exit_code"], 0)
        with self.assertRaises(ToolError):
            await self.session.run_command("true", "../")

    async def test_script_logs_and_patch_stay_independent(self) -> None:
        failed = json.loads(await self.session.run_tests())
        self.assertEqual(failed["exit_code"], 0)
        self.assertIn("AssertionError", failed["stderr"])
        self.assertNotIn("success", failed)
        artifact = self.session.root / "evaluation-artifact.txt"
        self.assertFalse(artifact.exists())
        self.session.repository.edit_file("calc.py", "a - b", "a + b")
        passed = json.loads(await self.session.run_tests())
        self.assertEqual(passed["exit_code"], 0)
        self.assertNotIn("AssertionError", passed["stderr"])
        await self.session.run_command(
            "printf 'VALUE = 1\\n' > extra.py; chmod +x calc.py",
        )
        patch = await self.session.get_patch()
        self.assertIn("+    return a + b", patch)
        self.assertIn("diff --git a/extra.py b/extra.py", patch)
        self.assertNotIn("evaluation-artifact", patch)
        self.assertNotIn("old mode", patch)
        staged = json.loads(await self.session.run_command(
            "git -c safe.directory=/testbed diff --cached",
        ))
        self.assertEqual(staged["stdout"], "")
        patch_command = "git apply --check - <<'PATCH'\n" + patch + "PATCH\n"
        # Reverse the edit before checking that the generated patch applies.
        await self.session.run_command(
            "git -c safe.directory=/testbed checkout -- calc.py; "
            "unlink extra.py",
        )
        applied = json.loads(await self.session.run_command(patch_command))
        self.assertEqual(applied["exit_code"], 0, applied)

    async def test_timeout_stops_children_and_next_call_recovers(self) -> None:
        result = json.loads(await self.session.run_command(
            "echo started; sleep 30 & wait",
        ))
        self.assertTrue(result["timed_out"])
        self.assertIn("started", result["stdout"])
        running = await self.session._control(
            "inspect", "--format", "{{.State.Running}}",
            self.session.container,
        )
        self.assertEqual(running, "false")
        recovered = json.loads(await self.session.run_command(
            "echo recovered",
        ))
        self.assertEqual(recovered["stdout"], "recovered\n")

    async def test_cancellation_kills_remote_execution(self) -> None:
        pending = asyncio.create_task(self.session.run_command(
            "touch started; sleep 30 & wait",
        ))
        try:
            for _ in range(100):
                if (self.session.root / "started").exists():
                    break
                await asyncio.sleep(.02)
            self.assertTrue((self.session.root / "started").exists())
        finally:
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assertEqual(await self.session._control(
            "inspect", "--format", "{{.State.Running}}",
            self.session.container,
        ), "false")

    async def test_memory_output_limits_and_oversized_patch(self) -> None:
        killed = json.loads(await self.session.run_command(
            "python -c 'value = bytearray(256 * 1024 * 1024)'",
        ))
        self.assertEqual(killed["exit_code"], 137, killed)
        self.config.max_output_chars = 100
        output = json.loads(await self.session.run_command(
            "python -c \"print('😀' * 10000)\"",
        ))
        self.assertTrue(output["truncated"])
        self.assertLessEqual(len(output["stdout"] + output["stderr"]), 100)
        self.session.repository.edit_file("calc.py", "a - b", "a + b")
        with self.assertRaisesRegex(ToolError, "Patch unavailable"):
            await self.session.get_patch()


@unittest.skipUnless(IMAGE, "Run make test-docker for Docker checks.")
class DockerMCPTests(unittest.IsolatedAsyncioTestCase):

    async def check_client(self, client: Client) -> None:
        async with client:
            tools = await client.list_tools()
            self.assertEqual(len(tools), 9)
            self.assertTrue(all(tool.description for tool in tools))
            response = await client.call_tool("read_file", {
                "filepath": "/testbed/calc.py", "start_line": 1, "end_line": 2,
            })
            content = response.content[0]
            assert isinstance(content, TextContent)
            self.assertIn("return a - b", content.text)
            response = await client.call_tool("run_command", {
                "command": "printf out; printf err >&2; exit 7",
            })
            content = response.content[0]
            assert isinstance(content, TextContent)
            result = json.loads(content.text)
            self.assertEqual(result["stdout"], "out")
            self.assertEqual(result["stderr"], "err")
            self.assertEqual(result["exit_code"], 7)

    async def test_in_process_lifecycle(self) -> None:
        await self.check_client(Client(create_server(task=task_input())))
