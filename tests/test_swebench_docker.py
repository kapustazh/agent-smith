import json
import os
import shlex
import tempfile
import unittest
from fastmcp.exceptions import ToolError
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
