"""Contract, real Deno isolation, lifecycle, and MCP transport checks."""

import asyncio
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mcp.types import TextContent
from pydantic import ValidationError

import mcp_tools_mbpp as mbpp
from src.config_model import SandboxConfig
import src.mbpp_executor as executor
from src.mbpp_executor import MBPPExecutionRequest, MBPPExecutionResult


ROOT = Path(__file__).resolve().parents[1]


def decode_result(text: str) -> dict[str, Any]:
    return MBPPExecutionResult.model_validate_json(text).model_dump()


class ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_inputs_do_not_launch_runtime(self) -> None:
        with patch.object(mbpp, "execute_mbpp", new_callable=AsyncMock) as run:
            for code, tests in [("", ["pass"]), ("pass", []),
                                ("pass", [" "])]:
                result = decode_result(await mbpp.run_tests(code, tests))
                self.assertFalse(result["success"])
            run.assert_not_called()

    async def test_sandbox_owns_configuration(self) -> None:
        config = SandboxConfig(max_execution_time_seconds=2,
                               max_memory_mb=256, max_output_bytes=73)
        with (
            patch.object(mbpp, "sandbox_config", config),
            patch.object(mbpp, "execute_mbpp", new_callable=AsyncMock) as run,
        ):
            run.return_value = MBPPExecutionResult(success=True, output="OK")
            await mbpp.run_tests("pass", ["pass"])
        request = run.call_args.args[0]
        self.assertEqual(request.config.model_dump(), config.model_dump())
        self.assertEqual(request.code, "pass")
        self.assertEqual(request.version, 1)

    def test_invalid_limits_and_result_types_are_rejected(self) -> None:
        for field in ("max_execution_time_seconds", "max_memory_mb",
                      "max_output_bytes"):
            for value in (0, -1, True, "30"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValidationError):
                        SandboxConfig.model_validate({field: value})
        with self.assertRaises(ValidationError):
            MBPPExecutionResult.model_validate(
                {"success": "true", "output": ""})
        with self.assertRaises(ValidationError):
            MBPPExecutionRequest.model_validate({
                "version": 2, "code": "pass", "test_list": ["pass"],
                "config": {},
            })

    async def test_missing_runtime_is_reported(self) -> None:
        with patch.object(executor.shutil, "which", return_value=None):
            result = await executor.execute_mbpp(MBPPExecutionRequest(
                code="pass", test_list=["pass"], config=SandboxConfig(),
            ))
        self.assertFalse(result.success)
        self.assertIn("Deno", result.output)


class RuntimeCase(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("deno") is None:
            raise RuntimeError("Deno must be on PATH (use uv run make test).")
        if not (ROOT / "node_modules/pyodide/pyodide.mjs").is_file():
            raise RuntimeError("Install offline runtime assets: npm ci")

    async def asyncSetUp(self) -> None:
        self.config = SandboxConfig(max_execution_time_seconds=2)
        config = patch.object(mbpp, "sandbox_config", self.config)
        config.start()
        self.addCleanup(config.stop)

    async def run_code(
        self, code: str, tests: list[str] | None = None,
    ) -> dict[str, Any]:
        return decode_result(await mbpp.run_tests(
            code, tests if tests is not None else ["assert True"],
        ))


class RuntimeTests(RuntimeCase):
    async def test_passing_imports_and_failed_assertion(self) -> None:
        code = "import math\ndef solve(): return math.factorial(5)"
        for answer in (0, 120):
            result = await self.run_code(code, [f"assert solve() == {answer}"])
            self.assertEqual(result["success"], answer == 120, result)
            if answer == 0:
                self.assertIn("AssertionError", result["output"])

    async def test_submodule_allowlist_and_custom_import_policy(self) -> None:
        result = await self.run_code(
            "import collections.abc\n"
            "assert isinstance([], collections.abc.Sequence)",
        )
        self.assertTrue(result["success"], result)
        self.config.authorized_imports = ["math"]
        result = await self.run_code("import re")
        self.assertFalse(result["success"], result)
        self.assertIn("Import denied: re", result["output"])

    async def test_syntax_runtime_and_early_exit_are_failures(self) -> None:
        for code, error in [("def broken(", "SyntaxError"),
                            ("1 / 0", "ZeroDivisionError"),
                            ("raise SystemExit(0)", "SystemExit")]:
            result = await self.run_code(code)
            self.assertFalse(result["success"], result)
            self.assertIn(error, result["output"])

    async def test_timeout_preserves_partial_output(self) -> None:
        self.config.max_execution_time_seconds = 1
        result = await self.run_code("print('started')\nwhile True: pass")
        self.assertFalse(result["success"], result)
        self.assertIn("started", result["output"])
        self.assertIn("timed out after 1s", result["output"])

    async def test_whole_process_memory_limit_includes_startup(self) -> None:
        self.config.max_memory_mb = 32
        result = await self.run_code("pass")
        self.assertFalse(result["success"], result)
        self.assertIn("Memory limit exceeded (32 MiB)", result["output"])

    async def test_candidate_memory_exhaustion_is_a_failure(self) -> None:
        result = await self.run_code("x = bytearray(600 * 1024 * 1024)")
        self.assertFalse(result["success"], result)
        self.assertIn("Memory", result["output"])

    async def test_monitor_failure_cannot_leave_unmonitored_execution(
        self,
    ) -> None:
        create = asyncio.create_subprocess_exec

        async def fail_ps(*args: Any, **kwargs: Any) -> Any:
            if args[0] == "ps":
                raise OSError("ps unavailable")
            return await create(*args, **kwargs)

        with patch.object(executor.asyncio, "create_subprocess_exec", fail_ps):
            result = await self.run_code("while True: pass")
        self.assertFalse(result["success"], result)
        self.assertIn("Memory monitoring failed", result["output"])

    async def test_output_limit_is_bytes_and_diagnostics_fit(self) -> None:
        self.config.max_output_bytes = 80
        result = await self.run_code("print('я' * 10000)")
        self.assertTrue(result["success"], result)
        self.assertLessEqual(len(result["output"].encode()), 80)
        self.assertIn("truncated", result["output"])

    async def test_imports_restricted_builtins_and_paths(self) -> None:
        for code, error in [
            ("import os", "Import denied"),
            ("import js", "Import denied"),
            ("eval('1 + 1')", "NameError"),
            ("open('/etc/passwd')", "File access denied"),
            ("open('/tmp/agent/../outside', 'w')", "File access denied"),
        ]:
            result = await self.run_code(code)
            self.assertFalse(result["success"], result)
            self.assertIn(error, result["output"])

    async def test_files_are_virtual_and_fresh_for_each_call(self) -> None:
        self.config.allowed_directories = ["/scratch"]
        result = await self.run_code(
            "with open('/scratch/example', 'w') as f: f.write('hello')",
            ["assert open('/scratch/example').read() == 'hello'"],
        )
        self.assertTrue(result["success"], result)
        result = await self.run_code("open('/scratch/example')")
        self.assertFalse(result["success"], result)
        self.assertIn("FileNotFoundError", result["output"])

    async def test_deno_blocks_host_access_even_with_js_import_allowed(
        self,
    ) -> None:
        # Exercise the outer permission boundary, independent of import guards.
        self.config.authorized_imports.append("js")
        for operation in [
            "js.Deno.readTextFileSync('/etc/passwd')",
            "js.Deno.env.get('HOME')",
            "js.Deno.Command.new('/bin/sh').spawn()",
            "js.WebSocket.new('ws://1.1.1.1:80')",
        ]:
            result = await self.run_code("import js\n" + operation)
            self.assertFalse(result["success"], result)
            self.assertIn("NotCapable", result["output"])

    async def test_concurrent_calls_are_independent(self) -> None:
        results = await asyncio.gather(*[
            self.run_code(
                f"open('/tmp/agent/value', 'w').write('{value}')",
                [f"assert open('/tmp/agent/value').read() == '{value}'"],
            ) for value in ("one", "two")
        ])
        self.assertTrue(all(result["success"] for result in results), results)

    async def test_cancellation_reaps_the_runtime(self) -> None:
        create = asyncio.create_subprocess_exec
        children: list[asyncio.subprocess.Process] = []

        async def record(*args: Any, **kwargs: Any) -> Any:
            child = await create(*args, **kwargs)
            if Path(args[0]).name == "deno":
                children.append(child)
            return child

        with patch.object(executor.asyncio, "create_subprocess_exec", record):
            task = asyncio.create_task(self.run_code("while True: pass"))
            try:
                while not children:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.5)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertIsNotNone(children[0].returncode)


class TransportTests(RuntimeCase):
    async def check_client(self, client: Client[Any]) -> None:
        async with client:
            tools = await client.list_tools()
            self.assertEqual([tool.name for tool in tools], ["run_tests"])
            self.assertEqual(set(tools[0].input_schema["properties"]),
                             {"code", "test_list"})
            self.assertEqual(len(await client.list_resources()), 1)
            self.assertEqual(len(await client.list_prompts()), 1)
            for expected in [False, True]:
                response = await client.call_tool("run_tests", {
                    "code": "def solve(): return 42",
                    "test_list": [
                        f"assert solve() == {42 if expected else 0}"],
                })
                self.assertFalse(response.is_error)
                content = response.content[0]
                assert isinstance(content, TextContent)
                result = decode_result(content.text)
                self.assertEqual(result["success"], expected, result)

    async def test_stdio(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config:
            config.write(SandboxConfig(max_output_bytes=80).model_dump_json())
            config.flush()
            transport = StdioTransport(
                command=sys.executable,
                args=[str(ROOT / "mcp_tools_mbpp.py"),
                      "--sandbox-config", config.name], cwd=str(ROOT),
            )
            await self.check_client(Client(transport))
            async with Client(transport) as client:
                response = await client.call_tool("run_tests", {
                    "code": "print('x' * 1000)", "test_list": ["assert True"],
                })
                content = response.content[0]
                assert isinstance(content, TextContent)
                result = decode_result(content.text)
                self.assertTrue(result["success"], result)
                self.assertLessEqual(len(result["output"].encode()), 80)
                self.assertIn("truncated", result["output"])

    async def test_streamable_http(self) -> None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / "mcp_tools_mbpp.py"),
            "--transport", "http", "--port", str(port),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        try:
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                try:
                    _, writer = await asyncio.open_connection(
                        "127.0.0.1", port,
                    )
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    if process.returncode is not None:
                        self.fail("HTTP server exited before becoming ready")
                    if asyncio.get_running_loop().time() > deadline:
                        self.fail("HTTP server did not become ready")
                    await asyncio.sleep(0.05)
            await self.check_client(Client(f"http://127.0.0.1:{port}/mcp"))
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.communicate(), 10)


if __name__ == "__main__":
    unittest.main()
