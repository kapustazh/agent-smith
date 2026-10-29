import asyncio
from pathlib import Path
import tempfile
from uuid import uuid4
from fastmcp.exceptions import ToolError
from src.config_model import SandboxConfig, SWEBenchTaskInput
from src.docker_process import ProcessResult, capture
from src.swebench_tools import RepositoryTools


class DockerSession:

    def __init__(self, task: SWEBenchTaskInput, config: SandboxConfig) -> None:
        self.task = task
        self.config = config
        self.directory = tempfile.TemporaryDirectory(prefix="smith-swe-")
        self.root = Path(self.directory.name).resolve() / "repo"
        self.root.mkdir()
        self.containers: dict[str, Path | None] = {}
        self.container = ""
        self.base = ""
        self.lock = asyncio.Lock()
        self.repository = RepositoryTools(self.root, config, Path("/testbed"))

    async def _control(self, *arguments: str, check: bool = True) -> str:
        try:
            result = await capture(
                ["docker", *arguments], 30, 4096,
            )
        except FileNotFoundError:
            raise self.repository._error(
                "Docker is missing. Install and start Docker.")
        if check and (result.exit_code != 0 or result.timed_out):
            raise self.repository._error(
                (result.stderr or result.stdout or "Docker operation failed.")
                [:self.config.max_output_chars],
            )
        return result.stdout.strip()

    async def _create(self, workspace: Path | None) -> str:
        name = "smith-swe-" + uuid4().hex
        self.containers[name] = workspace
        arguments = [
            "create", "--name", name, "--network", "none",
            "--memory", f"{self.config.max_memory_mb}m",
            "--memory-swap", f"{self.config.max_memory_mb}m",
            "--pids-limit", "128", "--cap-drop", "ALL",
            "--cap-add", "DAC_OVERRIDE", "--security-opt",
            "no-new-privileges", "--entrypoint", "/bin/sh",
            "--workdir", "/testbed",
        ]
        if workspace is not None:
            arguments.extend([
                "--mount", f"type=bind,src={workspace},dst=/testbed",
            ])
        arguments.extend([
            "--", self.task.docker_image, "-c",
            "while :; do sleep 3600; done",
        ])
        try:
            await self._control(*arguments)
        except BaseException:
            await asyncio.shield(self._remove(name))
            raise
        return name

    async def start(self) -> None:
        if not self.task.docker_image.strip():
            raise self.repository._error(
                "Task docker_image must not be empty.")
        try:
            await self._control(
                "image", "inspect", "--", self.task.docker_image,
            )
        except ToolError as error:
            raise self.repository._error(
                f"Cannot use Docker image {self.task.docker_image}. "
                "Start Docker and pull the task image before launching. "
                f"{error}",
            ) from None
        seed = await self._create(None)
        try:
            await self._control("cp", f"{seed}:/testbed/.", str(self.root))
        finally:
            await self._remove(seed)
        self.container = await self._create(self.root)
        await self._control("start", self.container)
        self.base = await self._control(
            "exec", self.container, "git", "-c", "safe.directory=/testbed",
            "rev-parse", "--verify", "HEAD",
        )
        if len(self.base) not in (40, 64) or any(
            char not in "0123456789abcdef" for char in self.base
        ):
            raise self.repository._error(
                "Cannot determine the task's base Git commit.")

    async def _execute(
        self, container: str, arguments: list[str], data: bytes = b"",
        workdir: str = "/testbed",
    ) -> ProcessResult:
        await self._control("start", container)
        try:
            result = await capture(
                ["docker", "exec", "-i", "--workdir", workdir,
                 container, *arguments],
                self.config.max_execution_time_seconds,
                self.config.max_output_chars, data,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._control("kill", container, check=False))
            raise
        if result.timed_out:
            await self._control("kill", container, check=False)
        return result

    async def _remove(self, name: str) -> None:
        workspace = self.containers.get(name)
        if workspace is not None:
            await self._control("kill", name, check=False)
            await self._control("start", name, check=False)
            await self._control(
                "exec", name, "find", "/testbed", "-mindepth", "1", "-delete",
                check=False,
            )
        try:
            await self._control("rm", "--force", name)
        except ToolError as error:
            if "no such container" not in str(error).lower():
                raise
        self.containers.pop(name, None)

    async def close(self) -> None:
        for name in list(self.containers):
            await self._remove(name)
        self.directory.cleanup()
