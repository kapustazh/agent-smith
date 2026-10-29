from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastmcp.exceptions import ToolError
from src.config_model import SandboxConfig
from src.swebench_tools import RepositoryTools

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
