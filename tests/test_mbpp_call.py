"""Task imports, actionable errors, and the subject's public MCP contract."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch

from fastmcp import Client
from mcp.types import TextContent
from pydantic import ValidationError
from typer.testing import CliRunner

import mcp_tools_mbpp as mbpp
from src.config_model import SandboxConfig
import src.mbpp_executor as executor
from src.mbpp_executor import MBPPExecutionRequest


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = """def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return res
"""
ASSERTION = ("assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) "
             "== set((4, 5))")


class ErrorTests(unittest.IsolatedAsyncioTestCase):
    def test_invalid_config_is_a_cli_error_without_a_traceback(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config:
            config.write('{"max_execution_time_seconds": 0}')
            config.flush()
            response = CliRunner().invoke(
                mbpp.app, ["--sandbox-config", config.name],
            )
        self.assertEqual(response.exit_code, 2)
        self.assertIn("Cannot load sandbox configuration", response.output)
        self.assertNotIn("Traceback", response.output)

    async def test_setup_error_recovery(self) -> None:
        request = MBPPExecutionRequest(
            code=SOLUTION, test_list=[ASSERTION], config=SandboxConfig(),
        )
        for missing in ("all", "pyodide.asm.wasm"):
            with (
                patch.object(executor.shutil, "which", return_value="deno"),
                patch.object(Path, "is_file", autospec=True,
                             side_effect=lambda path: missing != "all" and
                             path.name != missing),
                patch.object(executor.asyncio, "create_subprocess_exec",
                             new_callable=AsyncMock) as create,
            ):
                result = await executor.execute_mbpp(request)
            self.assertFalse(result.success)
            self.assertIn("setup error", result.output)
            self.assertIn("No code or tests were executed", result.output)
            self.assertIn("npm ci --ignore-scripts", result.output)
            self.assertIn(str(ROOT), result.output)
            self.assertIn("pyodide.asm.wasm", result.output)
            create.assert_not_called()
        with patch.object(executor.shutil, "which", return_value=None):
            result = await executor.execute_mbpp(request)
        self.assertIn("Deno was not found on PATH", result.output)
        self.assertIn("make install", result.output)
        self.assertIn("No code or tests were executed", result.output)

    def test_request_validates_imports_and_empty_inputs(self) -> None:
        for code, tests, imports in [
            (" ", [ASSERTION], []), (SOLUTION, [" "], []),
            (SOLUTION, [ASSERTION], [""]),
        ]:
            with self.assertRaises(ValidationError):
                MBPPExecutionRequest(code=code, test_list=tests,
                                     test_imports=imports,
                                     config=SandboxConfig())

    async def test_optional_imports_are_forwarded_to_executor(self) -> None:
        with patch.object(mbpp, "execute_mbpp", new_callable=AsyncMock) as run:
            run.return_value = executor.MBPPExecutionResult(
                success=True, output="OK",
            )
            await mbpp.MBPPTools().run_tests(
                "def solve(): return math.pi", ["assert solve() > 3"],
                ["import math"])
        self.assertEqual(run.call_args.args[0].test_imports, ["import math"])

    async def test_invalid_request_feedback_and_tiny_budgets(self) -> None:
        for limit in (8192, 1, 17):
            tools = mbpp.MBPPTools(SandboxConfig(max_output_bytes=limit))
            result = json.loads(await tools.run_tests(SOLUTION, [], [""]))
            self.assertFalse(result["success"])
            self.assertLessEqual(len(result["output"].encode()), limit)
            if limit == 8192:
                self.assertIn("Invalid MBPP request", result["output"])
                self.assertIn("No code or tests were executed",
                              result["output"])
                self.assertIn("test_list", result["output"])
        with patch.object(executor.shutil, "which", return_value=None):
            result = await executor.execute_mbpp(MBPPExecutionRequest(
                code=SOLUTION, test_list=[ASSERTION],
                config=SandboxConfig(max_output_bytes=1),
            ))
        self.assertFalse(result.success)
        self.assertLessEqual(len(result.output.encode()), 1)


class CallTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tools = mbpp.MBPPTools()

    async def call(
        self, code: str, tests: list[str], imports: list[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(
            await self.tools.run_tests(code, tests, imports),
        )
        self.assertEqual(set(result), {"success", "output"})
        return result

    async def test_user_example_and_failed_assertion(self) -> None:
        result = await self.call(SOLUTION, [ASSERTION])
        self.assertTrue(result["success"], result)
        result = await self.call(SOLUTION, ["assert similar_elements((), ())"])
        self.assertFalse(result["success"], result)
        self.assertIn("AssertionError", result["output"])
        self.assertIn("<test 1>", result["output"])

    async def test_imports_run_before_solution_and_tests(self) -> None:
        result = await self.call(
            "def root(x): return math.sqrt(x)",
            ["assert root(9) == math.sqrt(9)"], ["import math"],
        )
        self.assertTrue(result["success"], result)
        result = await self.call(
            "from __future__ import annotations\n"
            "def root(x: float) -> float: return math.sqrt(x)",
            ["assert root(9) == 3"], ["import math"],
        )
        self.assertTrue(result["success"], result)

    async def test_import_failures_identify_the_import_stage(self) -> None:
        for imports, error in [(["import os"], "ImportError"),
                               (["from math import ("], "SyntaxError")]:
            result = await self.call("raise RuntimeError('solution ran')",
                                     ["assert True"], imports)
            self.assertFalse(result["success"], result)
            self.assertIn("<test imports>", result["output"])
            self.assertIn(error, result["output"])
            self.assertNotIn("solution ran", result["output"])

    async def test_malformed_pasted_code_reports_syntax_error(self) -> None:
        result = await self.call(
            "def solve():\n\u00a0 return 1", ["assert solve() == 1"],
        )
        self.assertFalse(result["success"], result)
        self.assertIn("SyntaxError", result["output"])
        self.assertIn("<solution>", result["output"])

    async def test_syntax_error_points_to_supplied_assertion(self) -> None:
        broken = ASSERTION.replace("==", "=")
        result = await self.call(SOLUTION, [broken])
        self.assertFalse(result["success"])
        self.assertIn("Error while executing <test 1>", result["output"])
        self.assertIn('File "<test 1>", line 1', result["output"])
        self.assertIn(broken, result["output"])
        self.assertIn("^", result["output"])
        self.assertIn("SyntaxError", result["output"])
        self.assertNotIn("<exec>", result["output"])

    async def test_runtime_error_keeps_solution_lines_and_chained_causes(
        self,
    ) -> None:
        result = await self.call(
            "def solve():\n    try:\n        1 / 0\n"
            "    except ZeroDivisionError as error:\n"
            "        raise ValueError('bad input') from error",
            ["assert solve() == 1"],
        )
        self.assertFalse(result["success"])
        self.assertIn('File "<solution>", line 3', result["output"])
        self.assertIn("1 / 0", result["output"])
        self.assertIn('File "<test 1>", line 1', result["output"])
        self.assertIn("assert solve() == 1", result["output"])
        self.assertIn("ZeroDivisionError", result["output"])
        self.assertIn("ValueError: bad input", result["output"])
        self.assertNotIn("<exec>", result["output"])

    async def test_timeout_and_truncation_are_both_reported(self) -> None:
        with patch.object(self.tools, "config", SandboxConfig(
            max_execution_time_seconds=1, max_output_bytes=256,
        )):
            result = await self.call(
                "print('x' * 10000)\nwhile True: pass", ["assert True"],
            )
        self.assertFalse(result["success"], result)
        self.assertIn("timed out", result["output"])
        self.assertIn("partial", result["output"])
        self.assertIn("truncated", result["output"])
        self.assertLessEqual(len(result["output"].encode()), 256)

    async def test_memory_measurement_errors_stop_execution(self) -> None:
        create = asyncio.create_subprocess_exec
        for command in ("/usr/bin/false", "/usr/bin/true"):
            async def failed_ps(*args: Any, **kwargs: Any) -> Any:
                if args[0] == "ps":
                    return await create(command, **kwargs)
                return await create(*args, **kwargs)

            with patch.object(executor.asyncio,
                              "create_subprocess_exec", failed_ps):
                result = await self.call("while True: pass", ["assert True"])
            self.assertFalse(result["success"], result)
            self.assertIn("Memory monitoring failed", result["output"])
            self.assertIn("ps failed", result["output"])

    async def test_real_mcp_optional_import_argument(self) -> None:
        async with Client(mbpp.mcp) as client:
            response = await client.call_tool("run_tests", {
                "code": "def root(): return math.sqrt(9)",
                "test_list": ["assert root() == 3"],
                "test_imports": ["import math"],
            })
            self.assertFalse(response.is_error)
            content = response.content[0]
            assert isinstance(content, TextContent)
            self.assertTrue(json.loads(content.text)["success"])

    async def test_fastmcp_cli_example(self) -> None:
        process = await asyncio.create_subprocess_exec(
            str(Path(sys.executable).with_name("fastmcp")), "call",
            str(ROOT / "mcp_tools_mbpp.py"), "run_tests",
            "code=" + SOLUTION, "test_list=" + json.dumps([ASSERTION]),
            "--json", stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=ROOT,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 30)
        self.assertEqual(process.returncode, 0, stderr.decode())
        envelope = json.loads(stdout)
        result = json.loads(envelope["structured_content"]["result"])
        self.assertEqual(result, {"success": True,
                                  "output": "All 1 tests passed."})


if __name__ == "__main__":
    unittest.main()
