import subprocess
import os
import selectors
from itertools import count
from time import sleep
import shlex

from src.config_model import SandboxIntermediateResult


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
            max_sec: float = 2) -> SandboxIntermediateResult:
        json_str = self.request(tool_with_param, max_sec=max_sec)
        # print("json_str:", json_str)
        return SandboxIntermediateResult.model_validate_json(json_str)

    def close(self) -> None:
        if self.process.stdin:
            try:
                self.process.stdin.write(b"exit()\n")
                self.process.stdin.flush()
                sleep(1)
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except Exception:
                pass

        self.process.terminate()  # Ask nicely to stop
        try:
            self.process.wait(timeout=5)  # Wait up to 5 seconds for it to stop
        except subprocess.TimeoutExpired:
            self.process.kill()  # Forcefully kill it if it takes too long
            self.process.wait()  # Clean up the dead process completely
