import asyncio
import json
from pathlib import Path
import os
import shutil
import signal
import sys
import tempfile
from typing import Literal
from uuid import uuid4

from anyio import CancelScope
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config_model import SandboxConfig
from src.docker_process import capture, start_process


class MBPPExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    code: str = Field(min_length=1)
    test_list: list[str] = Field(min_length=1)
    test_imports: list[str] = Field(default_factory=list)
    config: SandboxConfig

    @model_validator(mode="after")
    def require_nonblank_input(self) -> "MBPPExecutionRequest":
        if not self.code.strip() or any(
            not text.strip() for text in self.test_list + self.test_imports
        ):
            raise ValueError(
                "Code, tests and import entries must be non-empty.")
        return self


class MBPPExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    success: bool
    output: str


def execution_result(
    success: bool, output: str, config: SandboxConfig, diagnostic: str = "",
) -> MBPPExecutionResult:
    limit = config.max_output_bytes
    data = output.encode("utf-8", "replace")
    if len(data) > limit and not diagnostic:
        diagnostic = "Output truncated."
    if diagnostic:
        suffix = ("[" + diagnostic + "]").encode("utf-8", "replace")
        room = max(0, limit - len(suffix) - 1)
        output = data[:room].decode("utf-8", "ignore").rstrip()
        data = ((output.encode() + b"\n") if output else b"") + suffix
    return MBPPExecutionResult(
        success=success, output=data[:limit].decode("utf-8", "ignore").strip(),
    )


async def execute_mbpp(request: MBPPExecutionRequest) -> MBPPExecutionResult:
    deno = shutil.which("deno")
    if deno is None:
        return execution_result(
            False, "MBPP setup error: Deno was not found on PATH. "
            "No code or tests were executed. Install the locked dependencies "
            "with `make install` and launch this server with "
            "`uv run python mcp_tools_mbpp.py`, or install Deno on PATH.",
            request.config,
        )
    root = Path(__file__).resolve().parents[1]
    assets = root / "node_modules/pyodide"
    missing = [name for name in (
        "pyodide.mjs", "pyodide.asm.js", "pyodide.asm.wasm",
        "pyodide.asm.data", "pyodide_py.tar", "repodata.json",
    ) if not (assets / name).is_file()]
    if missing:
        return execution_result(
            False, "MBPP setup error: the local Pyodide installation is "
            "missing or incomplete. Missing files: " + ", ".join(missing) +
            ". No code or tests were executed. From the project directory "
            f"({root}), run `make install` (installs Pyodide using "
            "`npm ci --ignore-scripts`), then retry the call.",
            request.config,
        )
    if sys.platform not in ("linux", "darwin"):
        return execution_result(
            False, "MBPP setup error: memory monitoring requires Linux or "
            "macOS. No code or tests were executed. Run the server on a "
            "supported operating system.",
            request.config,
        )

    return await MBPPExecutor(request, root, deno).run()


class MBPPExecutor:
    def __init__(
        self, request: MBPPExecutionRequest, root: Path, deno: str,
    ) -> None:
        self.request = request
        self.config = request.config
        self.root = root
        self.deno = deno
        self.token = "__MBPP_" + uuid4().hex + "__"
        self.marker = ("\n" + self.token + "\n").encode()
        self.prefix = bytearray()
        self.completed = False
        self.truncated = False
        self.failure = ""
        self.process: asyncio.subprocess.Process | None = None
        self.readers: list[asyncio.Task[None]] = []
        self.monitor: asyncio.Task[None] | None = None

    def retain(self, data: bytes) -> None:
        room = self.config.max_output_bytes - len(self.prefix)
        self.prefix.extend(data[:max(0, room)])
        self.truncated |= len(data) > room

    async def read_output(
        self, stream: asyncio.StreamReader, framed: bool = False,
    ) -> None:
        pending = b""
        while chunk := await stream.read(4096):
            if not framed:
                self.retain(chunk)
                continue
            pending += chunk
            if self.marker in pending:
                self.completed = True
                pending = pending.replace(self.marker, b"")
            room = max(0, len(pending) - len(self.marker) + 1)
            self.retain(pending[:room])
            pending = pending[room:]
        self.retain(pending)

    def kill(self) -> None:
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    async def monitor_memory(self) -> None:
        assert self.process is not None
        try:
            while self.process.returncode is None:
                result = await capture([
                    "ps", "-o", "rss=", "-p", str(self.process.pid),
                ], 2, 64)
                if result.exit_code or not result.stdout.strip():
                    if self.process.returncode is not None:
                        return
                    raise OSError(
                        "ps failed or returned no memory measurement")
                if int(result.stdout) > self.config.max_memory_mb * 1024:
                    self.failure = ("Memory limit exceeded "
                                    f"({self.config.max_memory_mb} MiB).")
                    self.kill()
                    return
                await asyncio.sleep(0.02)
        except Exception as error:
            self.failure = f"Memory monitoring failed: {error}"
            self.kill()

    async def start(self, directory: str) -> None:
        memory = self.config.max_memory_mb
        self.process = await start_process(
            self.deno, "run", "--no-prompt", "--no-config", "--no-lock",
            "--cached-only", "--no-remote", "--node-modules-dir=manual",
            "--allow-read=" + str(self.root / "node_modules") + "," +
            str(self.root / "runtime"),
            "--allow-env=WS_NO_BUFFER_UTIL,WS_NO_UTF_8_VALIDATE",
            "--deny-net", "--deny-run", "--deny-write", "--deny-ffi",
            "--deny-sys", "--deny-import",
            f"--v8-flags=--wasm-max-mem-pages={memory * 16},"
            f"--max-old-space-size={memory}",
            str(self.root / "runtime/mbpp.mjs"),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=directory,
            start_new_session=True,
            env={"HOME": directory, "PATH": os.defpath,
                 "DENO_NO_UPDATE_CHECK": "1", "NO_COLOR": "1"},
        )
        assert self.process.stderr is not None
        self.readers.append(asyncio.create_task(
            self.read_output(self.process.stderr)))
        self.monitor = asyncio.create_task(self.monitor_memory())

    async def initialize(self) -> None:
        process = self.process
        assert process is not None
        assert process.stdin is not None and process.stdout is not None
        payload = self.request.model_dump()
        payload["token"] = self.token
        process.stdin.write(json.dumps(payload).encode())
        await process.stdin.drain()
        process.stdin.close()
        ready = await process.stdout.readline()
        if ready != (self.token + "\n").encode():
            self.retain(ready)
            self.failure = self.failure or "Runtime failed before execution."
        self.readers.append(asyncio.create_task(
            self.read_output(process.stdout, framed=True)))

    async def close(self) -> None:
        with CancelScope(shield=True):
            self.kill()
            if self.process is not None:
                await self.process.wait()
            if self.monitor is not None:
                self.monitor.cancel()
                await asyncio.gather(self.monitor, return_exceptions=True)
            await asyncio.gather(*self.readers, return_exceptions=True)

    async def run(self) -> MBPPExecutionResult:
        try:
            with tempfile.TemporaryDirectory(prefix="mbpp-") as directory:
                try:
                    await self.start(directory)
                    await asyncio.wait_for(self.initialize(), 30)
                    assert self.process is not None
                    try:
                        await asyncio.wait_for(
                            self.process.wait(),
                            self.config.max_execution_time_seconds,
                        )
                    except asyncio.TimeoutError:
                        seconds = self.config.max_execution_time_seconds
                        self.failure = ("Execution timed out after "
                                        f"{seconds}s; "
                                        "output is partial.")
                finally:
                    await self.close()
        except asyncio.TimeoutError:
            self.failure = ("MBPP runtime initialization timed out after 30s. "
                            "No code or tests were executed.")
        except (OSError, ValueError) as error:
            self.failure = self.failure or f"Runtime execution failed: {error}"
        return self.finish()

    def finish(self) -> MBPPExecutionResult:
        output = bytes(self.prefix).decode("utf-8", "replace")
        success = (not self.failure and self.completed and
                   self.process is not None and self.process.returncode == 0)
        diagnostic = " ".join(text for text in (
            self.failure, "Output truncated." if self.truncated else "",
        ) if text)
        if not success and not diagnostic and not output.strip():
            status = self.process.returncode if self.process else None
            diagnostic = (f"MBPP runtime exited with status {status} before "
                          "confirming that all tests completed. "
                          "The result is a failure, not a passing test run.")
        if not output.strip() and not diagnostic:
            output = f"All {len(self.request.test_list)} tests passed."
        return execution_result(success, output, self.config, diagnostic)
