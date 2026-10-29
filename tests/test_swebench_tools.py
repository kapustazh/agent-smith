import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from mcp.types import TextContent
from mcp_tools_swebench import create_server
from src.config_model import SandboxConfig
from src.swebench_tools import RepositoryTools

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ("def solve(value):\n"
          "    return value + 1\n"
          "\n"
          "class Worker:\n"
          "    async def solve(self):\n"
          "        return 7\n"
          "\n"
          "answer = solve(1)\n")


class RepositoryCase(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "module.py").write_text(SOURCE)
        (self.repo / "consumer.py").write_text(
            "from module import solve as calculate\nanswer = calculate(2)\n",
        )
        self.tools = RepositoryTools(self.repo, SandboxConfig())

    def test_read_inclusive_ranges_and_absolute_paths(self) -> None:
        self.assertEqual(self.tools.read_file("module.py", 2, 3),
                         "2:     return value + 1\n3: ")
        self.assertEqual(self.tools.read_file(str(self.repo / "module.py"), 8),
                         "8: answer = solve(1)")
        self.assertIn("No lines", self.tools.read_file("module.py", 99))
        for start, end in ((0, 2), (3, 2)):
            with self.assertRaises(ToolError):
                self.tools.read_file("module.py", start, end)

    def test_errors_also_respect_output_limit(self) -> None:
        self.tools.config.max_output_chars = 40
        with self.assertRaises(ToolError) as caught:
            self.tools.read_file("x" * 500)
        self.assertLessEqual(len(str(caught.exception)), 40)

    def test_container_paths_map_to_private_workspace(self) -> None:
        self.tools.visible_root = Path("/testbed")
        self.assertEqual(self.tools.read_file("/testbed/module.py", 1, 1),
                         "1: def solve(value):")
        outputs = (self.tools.list_files(), self.tools.search_code("solve"),
                   self.tools.find_references("solve", "module.py", 1))
        for output in outputs:
            self.assertIn("/testbed/", output)
            self.assertNotIn(str(self.repo), output)

    def test_exact_edit_preserves_newlines_and_permissions(self) -> None:
        path = self.repo / "windows.py"
        path.write_bytes(b"def f():\r\n    return 1\r\n")
        path.chmod(0o755)
        self.tools.edit_file("windows.py", "return 1", "return 10")
        self.assertEqual(path.read_bytes(), b"def f():\r\n    return 10\r\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o755)

    def test_failed_and_ambiguous_edits_leave_original_unchanged(self) -> None:
        for old in ("", "not present", "solve"):
            with self.subTest(old=old), self.assertRaises(ToolError):
                self.tools.edit_file("module.py", old, "broken")
            self.assertEqual((self.repo / "module.py").read_text(), SOURCE)
        with patch("src.swebench_tools.os.replace", side_effect=OSError):
            with self.assertRaises(ToolError):
                self.tools.edit_file("module.py", "return value + 1", "pass")
        self.assertEqual((self.repo / "module.py").read_text(), SOURCE)
        self.assertFalse(list(self.repo.glob(".agent-edit-*")))

    def test_concurrent_edits_have_one_winner(self) -> None:
        def edit(value: int) -> bool:
            try:
                self.tools.edit_file("module.py", "value + 1", str(value))
                return True
            except ToolError:
                return False
        with ThreadPoolExecutor() as pool:
            results = list(pool.map(edit, (42, 43)))
        self.assertEqual(sum(results), 1)

    def test_overlap_and_invalid_utf8_edits_preserve_original(self) -> None:
        path = self.repo / "text.txt"
        path.write_text("banana")
        for old, new in (("ana", "x"), ("banana", "\ud800")):
            with self.assertRaises(ToolError):
                self.tools.edit_file("text.txt", old, new)
            self.assertEqual(path.read_text(), "banana")

    def test_edit_reports_saved_python_syntax_error(self) -> None:
        message = self.tools.edit_file("module.py", "return value + 1", "if:")
        self.assertIn("File saved", message)
        self.assertIn("syntax is invalid", message)
        self.assertIn("line 2", message)
        self.assertIn("    if:", (self.repo / "module.py").read_text())

    def test_listing_skips_generated_directories(self) -> None:
        for name in ("pkg", ".git", ".venv", "node_modules", "__pycache__"):
            (self.repo / name).mkdir()
            (self.repo / name / "extra.py").write_text("pass\n")
        files = self.tools.list_files(".", "*.py").splitlines()
        self.assertEqual(files, [str(self.repo / name) for name in
                                 ("consumer.py", "module.py", "pkg/extra.py")])
        self.assertEqual(self.tools.list_files("pkg", "extra.py"),
                         str(self.repo / "pkg/extra.py"))
        self.assertEqual(self.tools.list_files(".", "pkg/*.py"),
                         str(self.repo / "pkg/extra.py"))
        self.assertEqual(self.tools.list_files(".", "absent*"),
                         "No matching files.")

    def test_search_regex_globs_and_numbered_output(self) -> None:
        (self.repo / "binary").write_bytes(b"\x00solve\xff\n")
        self.assertEqual(self.tools.search_code(r"^def\s+solve", "*.py"),
                         f"{self.repo}/module.py:1 def solve(value):")
        self.assertEqual(self.tools.search_code("absent"), "No matches.")
        with self.assertRaisesRegex(ToolError, "Invalid regular expression"):
            self.tools.search_code("[")

    def test_expensive_regex_times_out(self) -> None:
        (self.repo / "long.txt").write_text("a" * 200000 + "!")
        self.tools.config.max_execution_time_seconds = 1
        with self.assertRaisesRegex(ToolError, "timed out"):
            self.tools.search_code("(a+)+$", "long.txt")

    def test_definitions_include_async(self) -> None:
        (self.repo / "broken.py").write_text("def broken(:")
        (self.repo / "comment.txt").write_text("def solve(): pass")
        output = self.tools.search_function_or_class_definition_in_code(
            "solve")
        self.assertIn(f"{self.repo}/module.py:1 def solve(value):", output)
        self.assertIn(f"{self.repo}/module.py:5     async def solve", output)
        self.assertIn("Skipped 1", output)
        self.assertNotIn("comment.txt", output)
        self.assertIn("module.py:4 class Worker:",
                      self.tools.search_function_or_class_definition_in_code(
                          "Worker"))

    def test_paths_are_confined(self) -> None:
        outside = self.root / "secret.py"
        outside.write_text("secret = 42\n")
        for name in ("../secret.py", str(outside), ".git/config", ""):
            with self.subTest(name=name):
                with self.assertRaises(ToolError):
                    self.tools.read_file(name)
                with self.assertRaises(ToolError):
                    self.tools.edit_file(name, "secret", "public")
        with self.assertRaises(ToolError):
            self.tools.list_files("..")
        self.assertEqual(outside.read_text(), "secret = 42\n")

    def test_links_and_special_files_are_rejected(self) -> None:
        outside = self.root / "secret.py"
        outside.write_text("secret = 42\n")
        (self.repo / "link.py").symlink_to(outside)
        (self.repo / "linked_dir").symlink_to(
            self.root, target_is_directory=True,
        )
        os.link(outside, self.repo / "hard.py")
        os.mkfifo(self.repo / "pipe")
        for path in ("link.py", "linked_dir/secret.py", "hard.py", "pipe"):
            with self.subTest(path=path), self.assertRaises(ToolError):
                self.tools.read_file(path)
            with self.assertRaises(ToolError):
                self.tools.edit_file(path, "secret", "public")
        self.assertNotIn("link.py", self.tools.list_files())
        self.assertNotIn("hard.py", self.tools.list_files())
        self.assertEqual(self.tools.search_code("secret"), "No matches.")
        self.assertEqual(outside.read_text(), "secret = 42\n")

    def test_missing_testbed_has_no_cwd_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            tools = RepositoryTools(None, SandboxConfig())
            with self.assertRaisesRegex(ToolError, "TESTBED_PATH"):
                tools.list_files()
        with patch.dict(os.environ, {"TESTBED_PATH": str(self.repo)}):
            self.assertIn("module.py", tools.list_files())

    def test_unicode_output_and_truncation(self) -> None:
        (self.repo / "unicode.txt").write_text("😀" * 10 + "\n")
        self.tools.config.max_output_chars = 13
        self.assertEqual(self.tools.read_file("unicode.txt"), "1: " + "😀" * 10)
        self.tools.config.max_output_chars = 80
        (self.repo / "unicode.txt").write_text("😀" * 1000)
        for output in (self.tools.read_file("unicode.txt"),
                       self.tools.search_code("😀", "unicode.txt"),
                       self.tools.list_files()):
            self.assertLessEqual(len(output), 80)
            self.assertIn("Output truncated", output)

    def test_references_follow_aliases(self) -> None:
        output = self.tools.find_references("solve", "module.py", 1)
        self.assertIn("module.py:1 def solve(value):", output)
        self.assertIn("module.py:8 answer = solve(1)", output)
        self.assertIn("consumer.py:2 answer = calculate(2)", output)
        self.assertNotIn("module.py:5", output)
        (self.repo / "module.py").write_text(SOURCE + "solve(3)\n")
        self.assertIn("module.py:9 solve(3)",
                      self.tools.find_references("solve", "module.py", 1))

    def test_references_reject_missing_or_ambiguous_symbols(self) -> None:
        for name, line in (("solve", 0), ("absent", 1), ("solve", 99)):
            with self.assertRaises(ToolError):
                self.tools.find_references(name, "module.py", line)
        (self.repo / "duplicate.py").write_text("f(f())\n# f\ntext = 'f'\n")
        for line in (1, 2, 3):
            with self.assertRaises(ToolError):
                self.tools.find_references("f", "duplicate.py", line)

    def test_analysis_never_runs_repository_code(self) -> None:
        marker = self.root / "executed"
        (self.repo / "consumer.py").write_text(
            "from module import solve as calculate\n" +
            f"open({str(marker)!r}, 'w').write('unsafe')\n" +
            "answer = calculate(2)\n",
        )
        output = self.tools.find_references("solve", "module.py", 1)
        self.assertIn("consumer.py:3 answer = calculate(2)", output)
        self.assertFalse(marker.exists())


class MCPTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / "module.py").write_text(SOURCE)

    async def check_client(self, client: Client) -> None:
        async with client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            self.assertEqual(set(tools), {
                "read_file", "edit_file", "list_files", "search_code",
                "search_function_or_class_definition_in_code",
                "find_references",
            })
            schema = tools["find_references"].input_schema
            self.assertEqual(set(schema["required"]),
                             {"name", "filepath", "line"})
            for name, arguments, expected in (
                ("list_files", {"directory": ".", "pattern": "*.py"},
                 str(self.root / "module.py")),
                ("read_file", {"filepath": "module.py", "start_line": 1,
                               "end_line": 1}, "1: def solve(value):"),
                ("edit_file", {"filepath": "module.py", "old_str": "value + 1",
                               "new_str": "value + 2"}, "Replaced one"),
                ("search_code", {"pattern": "value \\+ 2"}, "module.py:2"),
                ("search_function_or_class_definition_in_code",
                 {"name": "Worker"}, "module.py:4 class Worker:"),
                ("find_references", {"name": "solve", "filepath": "module.py",
                                     "line": 1}, "module.py:8 answer"),
            ):
                response = await client.call_tool(name, arguments)
                self.assertFalse(response.is_error)
                content = response.content[0]
                assert isinstance(content, TextContent)
                self.assertIn(expected, content.text)
            response = await client.call_tool(
                "read_file", {"filepath": "../secret"}, raise_on_error=False,
            )
            self.assertTrue(response.is_error)

    async def test_in_process_tool_contract(self) -> None:
        await self.check_client(Client(create_server(self.root)))

    async def test_stdio_with_testbed_environment(self) -> None:
        await self.check_client(Client(StdioTransport(
            command=sys.executable, args=[str(ROOT / "mcp_tools_swebench.py")],
            cwd=str(ROOT), env={"TESTBED_PATH": str(self.root)},
        )))

    async def test_http_transport(self) -> None:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "mcp_tools_swebench.py"),
             "--transport", "http", "--port", str(port)], cwd=ROOT,
            env={**os.environ, "TESTBED_PATH": str(self.root)},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(200):
                if process.poll() is not None:
                    self.fail("HTTP server exited during startup")
                try:
                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1", port,
                    )
                except OSError:
                    await asyncio.sleep(0.05)
                else:
                    writer.close()
                    await writer.wait_closed()
                    break
            else:
                self.fail("HTTP server did not start")
            await self.check_client(Client(f"http://127.0.0.1:{port}/mcp"))
        finally:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
