import asyncio
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import tempfile
import unittest

from anyio import move_on_after
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from mcp.types import TextContent

from mcp_tools_swebench import create_server
from src.config_model import SandboxConfig, SWEBenchTaskInput
from src.swebench_docker import DockerSession
from src.docker_process import capture


ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("SWE_DOCKER_TEST_IMAGE", "")


def task_input() -> SWEBenchTaskInput:
    return SWEBenchTaskInput(
        instance_id="fixture-1", problem_statement="Fix add.",
        docker_image=IMAGE, eval_script=(
            "cd /testbed\n"
            "python -c 'from calc import add; assert add(2, 3) == 5'\n"
            "echo evaluator-only > evaluation-artifact.txt\n"),
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

    async def test_evaluation_preserves_installed_dependencies(self) -> None:
        await self.session.run_command(
            "printf 'VALUE = 7\\n' > "
            "/usr/local/lib/python3.10/site-packages/smith_dependency.py",
        )
        self.session.task.eval_script = (
            "python -c 'import smith_dependency; "
            "assert smith_dependency.VALUE == 7'\n"
            "echo evaluation-only > /testbed/evaluation-output\n"
        )
        result = json.loads(await self.session.run_tests())
        self.assertEqual(result["exit_code"], 0, result)
        self.assertFalse((self.session.root / "evaluation-output").exists())
        self.assertEqual(self.session.images, set())

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

    async def test_mcp_cancel_scope_cleans_commands_and_evaluation(
        self,
    ) -> None:
        with move_on_after(0.3) as command_scope:
            await self.session.run_command("sleep 30 & wait")
        self.assertTrue(command_scope.cancel_called)
        self.assertEqual(await self.session._control(
            "inspect", "--format", "{{.State.Running}}",
            self.session.container,
        ), "false")
        self.session.task.eval_script = "sleep 30 & wait"
        with move_on_after(0.3) as test_scope:
            await self.session.run_tests()
        self.assertTrue(test_scope.cancel_called)
        self.assertEqual(self.session.images, set())
        self.assertEqual(set(self.session.containers),
                         {self.session.container})

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

    async def test_stdio_lifecycle(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as task:
            task.write(task_input().model_dump_json())
            task.flush()
            await self.check_client(Client(StdioTransport(
                command=sys.executable,
                args=[str(ROOT / "mcp_tools_swebench.py"),
                      "--task-file", task.name],
                cwd=str(ROOT),
            )))

    async def test_http_sigterm_cleans_up_container(self) -> None:
        before = await capture([
            "docker", "ps", "-aq", "--filter", "name=smith-swe-",
        ], 5, 4096)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as task:
            task.write(task_input().model_dump_json())
            task.flush()
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(ROOT / "mcp_tools_swebench.py"),
                "--task-file", task.name, "--transport", "http", "--port",
                str(port), stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                for _ in range(200):
                    try:
                        _, writer = await asyncio.open_connection(
                            "127.0.0.1", port,
                        )
                    except OSError:
                        await asyncio.sleep(.05)
                    else:
                        writer.close()
                        await writer.wait_closed()
                        break
                else:
                    self.fail("Docker HTTP server did not start")
                await self.check_client(Client(f"http://127.0.0.1:{port}/mcp"))
            finally:
                process.terminate()
                await asyncio.wait_for(process.wait(), 10)
        after = await capture([
            "docker", "ps", "-aq", "--filter", "name=smith-swe-",
        ], 5, 4096)
        self.assertEqual(after.stdout, before.stdout)

    async def test_startup_failure_cleans_up(self) -> None:
        broken_task = task_input()
        broken_task.docker_image = "python:3.10-slim"
        session = DockerSession(broken_task, SandboxConfig())
        try:
            with self.assertRaises(ToolError):
                await session.start()
        finally:
            await session.close()
        self.assertFalse(session.root.exists())
        self.assertEqual(session.containers, {})
