import subprocess
import os
import selectors
from itertools import count
from time import monotonic
import shlex

from src.config_model import SandboxIntermediateResult
from pydantic import ValidationError


class SandboxClient:
    def __init__(self, param: str = "") -> None:

        cmd = ["uv", "run", "sandbox"]

        if param:
            cmd.extend(shlex.split(param))

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Inherit stderr for diagnostics
            # In this mode i can't read more then 1 char without bloking
            # text=True,
            bufsize=0,
        )

        self._ids = count(1)
        self._response_buffer = b""

        if not self.process:
            return

        if self.process.stdout is None:
            raise RuntimeError("Failed to open stdout pipe for Sandbox")

        # make stdout not locking on OS level
        os.set_blocking(self.process.stdout.fileno(), False)

        # Register the process stdout stream to a selector
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.process.stdout, selectors.EVENT_READ)

    def write_std(self, text: str) -> None:
        if self.process.stdin is None or self.process.stdin.closed:
            raise RuntimeError("Sandbox session closed; start a new client.")
        if self.process.stdin:
            self.process.stdin.write(text.encode("utf-8"))
            self.process.stdin.flush()

    def read_std(self, max_sec: float = 0.5) -> str:

        # Check for data with a timeout (e.g., 0.5 seconds)
        # If timeout=0, it acts as a completely non-blocking check
        events = self.sel.select(timeout=max_sec)

        if not events:
            return ""

        if self.process.stdout is None:
            raise RuntimeError("Sandbox stdout is unavailable")

        chunks = []
        while True:
            try:
                chunk = self.process.stdout.read(4096)  # read as byte
                if not chunk:  # Check EOF
                    break
                chunks.append(chunk)
                if len(chunk) < 4096:
                    break
            except (BlockingIOError, TypeError):
                # no more data to read
                break

        # Decode in text
        return b"".join(chunks).decode("utf-8", errors="ignore")

    def request(self, param: str, max_sec: float = 1) -> str:
        if self.process.stdin is None:
            raise RuntimeError("Sandbox stdin is unavailable")
        if self.process.stdout is None:
            raise RuntimeError("Sandbox stdout is unavailable")

        self.write_std(f"{param.strip()}\n<<<END_COMMAND>>>\n")

        return self.read_std(max_sec=max_sec)

    def run_tool(
            self,
            tool_with_param: str,
            max_sec: float = 360) -> SandboxIntermediateResult:
        # Docker tools may do several bounded setup and cleanup operations.
        self.write_std(f"{tool_with_param.strip()}\n<<<END_COMMAND>>>\n")
        deadline = monotonic() + max_sec
        assert self.process.stdout is not None
        while b"\n" not in self._response_buffer:
            remaining = deadline - monotonic()
            if remaining <= 0 or not self.sel.select(timeout=remaining):
                self.close()
                return SandboxIntermediateResult(
                    success=False, tool_name="sandbox", status="timeout",
                    sandbox_output=(f"Sandbox response timed out after "
                                    f"{max_sec:g}s. Session closed; restart "
                                    "before retrying the command."),
                )
            chunk = self.process.stdout.read(4096)
            if not chunk:
                self.close()
                return SandboxIntermediateResult(
                    success=False, tool_name="sandbox", status="error",
                    sandbox_output="Sandbox stdout closed before a response.",
                )
            self._response_buffer += chunk
        raw, self._response_buffer = self._response_buffer.split(b"\n", 1)
        try:
            res_ = SandboxIntermediateResult.model_validate_json(raw)
        except ValidationError as ex:
            res_ = SandboxIntermediateResult(
                success=False,
                tool_name="sandbox", status="error",
                sandbox_output=f"Invalid sandbox response: {ex}")
        return res_

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.write(b"exit()\n<<<END_COMMAND>>>\n")
                self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            finally:
                self.process.stdin.close()

        self.process.terminate()  # Ask nicely to stop
        try:
            self.process.wait(timeout=5)  # Wait up to 5 seconds for it to stop
        except subprocess.TimeoutExpired:
            self.process.kill()  # Forcefully kill it if it takes too long
            self.process.wait()  # Clean up the dead process completely
        if self.process.stdout:
            self.process.stdout.close()
        self.sel.close()
