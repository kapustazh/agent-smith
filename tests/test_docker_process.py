import asyncio
import sys
import unittest
from typing import Any
from unittest.mock import patch

from src.docker_process import capture, start_process


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def delayed_spawn(
        self, *args: Any, **kwargs: Any,
    ) -> asyncio.subprocess.Process:
        self.child = await self.original_spawn(*args, **kwargs)
        self.spawned.set()
        await asyncio.sleep(0.05)
        return self.child

    async def test_cancel_during_spawn_reaps_the_unclaimed_process(
        self,
    ) -> None:
        self.original_spawn = asyncio.create_subprocess_exec
        self.spawned = asyncio.Event()
        with patch.object(
            asyncio, "create_subprocess_exec", self.delayed_spawn,
        ):
            pending = asyncio.create_task(start_process(
                sys.executable, "-c", "import time; time.sleep(10)",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            ))
            try:
                await asyncio.wait_for(self.spawned.wait(), 2)
            finally:
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
        self.assertIsNotNone(self.child.returncode)

    async def test_exit_code_and_separate_streams(self) -> None:
        result = await capture([
            sys.executable, "-c",
            "import sys; print('out'); print('err', file=sys.stderr); exit(7)",
        ], 2, 100)
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.stdout, "out\n")
        self.assertEqual(result.stderr, "err\n")

    async def test_unicode_spanning_pipe_chunks(self) -> None:
        result = await capture([
            sys.executable, "-c", "print('😀' * 3000, end='')",
        ], 2, 3000)
        self.assertEqual(result.stdout, "😀" * 3000)
        self.assertFalse(result.truncated)

    async def test_flooding_both_streams_is_bounded(self) -> None:
        result = await capture([
            sys.executable, "-c",
            "import sys; print('x'*200000); print('y'*200000,file=sys.stderr)",
        ], 2, 100)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.truncated)
        self.assertLessEqual(len(result.stdout + result.stderr), 100)
        self.assertIn("Output truncated", result.stderr)

    async def test_timeout_preserves_partial_output(self) -> None:
        result = await capture([
            sys.executable, "-u", "-c",
            "import time; print('started'); time.sleep(30)",
        ], .2, 100)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertIn("started", result.stdout)
        self.assertIn("timed out", result.stderr)

    async def test_cancellation_finishes_without_pipe_deadlock(self) -> None:
        task = asyncio.create_task(capture([
            sys.executable, "-u", "-c",
            "while True: print('x' * 4096)",
        ], 30, 100))
        await asyncio.sleep(.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
