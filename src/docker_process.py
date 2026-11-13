import asyncio
import codecs
import os
import signal
from typing import Any

from anyio import CancelScope
from pydantic import BaseModel


class ProcessResult(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    truncated: bool = False


def kill_process_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def start_process(
    *arguments: str, **options: Any,
) -> asyncio.subprocess.Process:
    options["start_new_session"] = True
    creation = asyncio.create_task(asyncio.create_subprocess_exec(
        *arguments, **options))
    try:
        with CancelScope(shield=True):
            process = await asyncio.shield(creation)
        return process
    except asyncio.CancelledError:
        with CancelScope(shield=True):
            process = await creation
            kill_process_group(process)
            await process.communicate()
        raise


class ProcessCapture:
    def __init__(self, process: asyncio.subprocess.Process,
                 limit: int) -> None:
        self.process = process
        self.limit = limit
        self.remaining = limit
        self.outputs: list[list[str]] = [[], []]
        self.result = ProcessResult()

    async def read(self, stream: asyncio.StreamReader, index: int) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            chunk = await stream.read(4096)
            text = decoder.decode(chunk, final=not chunk)
            accepted = text[:self.remaining]
            if accepted:
                self.outputs[index].append(accepted)
            self.remaining -= len(accepted)
            self.result.truncated |= len(text) > len(accepted)
            if not chunk:
                break

    async def write(self, data: bytes) -> None:
        writer = self.process.stdin
        assert writer is not None
        try:
            writer.write(data)
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            writer.close()

    async def run(self, timeout: float, data: bytes) -> ProcessResult:
        process = self.process
        assert process.stdout is not None and process.stderr is not None
        tasks = [asyncio.create_task(self.read(process.stdout, 0)),
                 asyncio.create_task(self.read(process.stderr, 1)),
                 asyncio.create_task(self.write(data))]
        work = asyncio.gather(process.wait(), *tasks)
        try:
            await asyncio.wait_for(asyncio.shield(work), timeout)
            self.result.exit_code = process.returncode
        except TimeoutError:
            self.result.timed_out = True
        finally:
            with CancelScope(shield=True):
                kill_process_group(process)
                if process.returncode is None:
                    await process.wait()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.gather(work, return_exceptions=True)
        return self.finish()

    def finish(self) -> ProcessResult:
        result = self.result
        result.stdout, result.stderr = (
            "".join(parts) for parts in self.outputs)
        notices = []
        if result.timed_out:
            notices.append("Execution timed out.")
        if result.truncated:
            notices.append("Output truncated.")
        if result.exit_code == 137:
            notices.append("Exit 137; check the memory limit.")
        if notices:
            notice = "\n[" + " ".join(notices) + "]"
            if (len(result.stdout + result.stderr) >
                    max(0, self.limit - len(notice)) and not result.truncated):
                result.truncated = True
                notices.append("Output truncated.")
                notice = "\n[" + " ".join(notices) + "]"
            room = max(0, self.limit - len(notice))
            result.stdout = result.stdout[:room]
            result.stderr = (result.stderr[:room - len(result.stdout)] +
                             notice[:self.limit])
        return result


async def capture(
    arguments: list[str], timeout: float, limit: int, data: bytes = b"",
    cwd: str | None = None,
) -> ProcessResult:
    options: dict[str, Any] = {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    if cwd is not None:
        options["cwd"] = cwd
    process = await start_process(*arguments, **options)
    return await ProcessCapture(process, limit).run(timeout, data)
