from pathlib import Path
import tempfile
import unittest
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
