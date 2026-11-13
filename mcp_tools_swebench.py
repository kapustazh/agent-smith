import asyncio
import shlex
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Callable, cast

import typer
from anyio import CancelScope
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from src.config_model import SandboxConfig
from src.docker_process import capture
from src.swebench_docker import DockerSession
from src.swebench_tools import RepositoryTools


class SWEBenchMCP:
    def __init__(self, root: Path | None, config: SandboxConfig) -> None:
        self.config = config
        self.fallback_root = root
        self.session: DockerSession | None = None
        self.lock = asyncio.Lock()
        self.repository = RepositoryTools(root, config)
        self.server = FastMCP(
            "Agent Smith - SWE-bench", lifespan=self.lifespan)
        for method, description in (
            (self.read_file, "Read numbered UTF-8 lines."),
            (self.edit_file, "Replace exactly one text match."),
            (self.list_files, "Recursively list matching files."),
            (self.search_code, "Find regex matches and locations."),
            (
                self.search_function_or_class_definition_in_code,
                "Find exact Python function and class definitions.",
            ),
            (
                self.find_references,
                "Find Python references and aliases at a 1-based source line.",
            ),
        ):
            self.server.tool(method, description=description)
        self.server.tool(
            self.run_command,
            description=(
                "Run a shell command and return output and exit code."),
        )
        self.server.tool(
            self.run_tests,
            description=(
                "Run eval_script with the prepared environment "
                "on a disposable copy.\n\nReturns script logs and exit_code, "
                "not a benchmark "
                "verdict."
            ),
        )
        self.server.tool(
            self.get_patch,
            description=(
                "Get a patch against the original commit, including new files."
                "\n\nOversized patches return an error instead of "
                "an incomplete patch."
            ),
        )
        self.server.tool(
            self.start_task,
            description=(
                "Replace the current container with docker_image "
                "and eval_script."
            ),
        )
        self.server.resource(
            "agent://swebench/task",
            description=("Task metadata and execution configuration status."),
        )(self.task_resource)
        self.server.prompt(
            description=("Instructions for solving the task with these tools.")
        )(self.swebench_solver)

    async def call_repository(
        self, method: str | Callable[..., str], *args: Any,
    ) -> str:
        async with self.lock:
            if isinstance(method, str):
                method = cast(
                    Callable[..., str], getattr(self.repository, method),
                )
            work = asyncio.create_task(asyncio.to_thread(method, *args))
            try:
                return await asyncio.shield(work)
            finally:
                with CancelScope(shield=True):
                    await work

    async def read_file(
        self,
        filepath: str,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> str:
        return await self.call_repository(
            "read_file", filepath, start_line, end_line,
        )

    async def edit_file(
        self,
        filepath: str,
        old_str: str,
        new_str: str,
    ) -> str:
        return await self.call_repository(
            "edit_file", filepath, old_str, new_str,
        )

    async def list_files(
        self,
        directory: str = ".",
        pattern: str = "*",
    ) -> str:
        return await self.call_repository("list_files", directory, pattern)

    async def search_code(self, pattern: str, file_pattern: str = "*") -> str:
        return await self.call_repository(
            "search_code", pattern, file_pattern,
        )

    async def search_function_or_class_definition_in_code(
        self,
        name: str,
    ) -> str:
        return await self.call_repository(
            "search_function_or_class_definition_in_code", name,
        )

    async def find_references(
        self,
        name: str,
        filepath: str,
        line: int,
    ) -> str:
        return await self.call_repository(
            "find_references", name, filepath, line,
        )

    async def _close_session(self) -> None:
        session = self.session
        if session is None:
            return
        self.session = None
        self.repository = RepositoryTools(self.fallback_root, self.config)
        try:
            with CancelScope(shield=True):
                await asyncio.shield(session.close())
        except Exception:
            self.session = session
            self.repository = session.repository
            raise

    async def _shutdown(self) -> None:
        async with self.lock:
            await self._close_session()

    @asynccontextmanager
    async def lifespan(self, server: FastMCP) -> AsyncIterator[dict[str, str]]:
        try:
            yield {}
        finally:
            with CancelScope(shield=True):
                await asyncio.shield(self._shutdown())

    def _host_cwd(self, workdir: str) -> str:
        if workdir == "/testbed" or workdir.startswith("/testbed/"):
            suffix = workdir.removeprefix("/testbed").lstrip("/")
            workdir = suffix or "."
        relative = self.repository._relative(workdir)
        with self.repository._directory(relative):
            return str(self.repository.root / relative)

    async def _host_command(self, command: str, workdir: str) -> str:
        if not command.strip() or "\x00" in command:
            raise self.repository._error(
                "Supply a non-empty shell command without null bytes.")
        result = await capture(
            ["/bin/bash", "-c", command],
            self.config.max_execution_time_seconds,
            self.config.max_output_chars,
            cwd=self._host_cwd(workdir),
        )
        return result.model_dump_json()

    async def _host_patch(self) -> str:
        root = self.repository.root
        quoted = shlex.quote(str(root))
        git = f"git -c core.fileMode=false -c safe.directory={quoted} "
        script = (
            "set -e\n"
            f"base=$({git}rev-parse --verify HEAD)\n"
            "index=$(mktemp /tmp/smith-index.XXXXXX)\n"
            "trap 'rm -f \"$index\" \"$index.lock\"' EXIT\n"
            "export GIT_INDEX_FILE=$index\n"
            f"{git}read-tree \"$base\"\n"
            f"{git}add -A -- .\n"
            f"{git}diff --cached --binary --no-ext-diff --no-textconv "
            "\"$base\" --\n"
        )
        result = await capture(
            ["/bin/bash", "-c", script],
            self.config.max_execution_time_seconds,
            self.config.max_output_chars,
            cwd=str(root),
        )
        if result.exit_code != 0 or result.timed_out or result.truncated:
            raise self.repository._error(
                "Patch unavailable; check Git errors and trusted output "
                "limits. " + result.stderr,
            )
        return result.stdout

    async def start_task(self, docker_image: str, eval_script: str) -> str:
        if not docker_image.strip() or "\x00" in docker_image:
            raise self.repository._error(
                "docker_image must be a non-empty image name.")
        if not eval_script.strip() or "\x00" in eval_script:
            raise self.repository._error(
                "eval_script must be a non-empty shell script.")
        async with self.lock:
            await self._close_session()
            session = DockerSession(docker_image, eval_script, self.config)
            try:
                await session.start()
            except BaseException:
                with CancelScope(shield=True):
                    await asyncio.shield(session.close())
                raise
            self.session = session
            self.repository = session.repository
        return f"Started {docker_image} at /testbed."

    async def run_command(
        self,
        command: str,
        workdir: str = "/testbed",
    ) -> str:
        async with self.lock:
            if self.session is not None:
                return await self.session.run_command(command, workdir)
            return await self._host_command(command, workdir)

    async def run_tests(self) -> str:
        async with self.lock:
            if self.session is None:
                raise self.repository._error(
                    "No task is started. Call start_task(docker_image, "
                    "eval_script) before run_tests."
                )
            return await self.session.run_tests()

    async def get_patch(self) -> str:
        async with self.lock:
            if self.session is not None:
                return await self.session.get_patch()
            return await self._host_patch()

    def task_resource(self) -> dict[str, str]:
        if self.session is None:
            try:
                repository = str(self.repository.root)
            except ToolError:
                repository = ""
            return {"execution": "idle", "repository": repository}
        return {
            "docker_image": self.session.docker_image,
            "repository": self.session.workdir,
            "execution": "docker",
        }

    def swebench_solver(self) -> str:
        return (
            "Inspect the repository at /testbed. Edit the source, "
            "run_tests(), read the logs even when exit_code is zero, "
            "and get_patch(). Pass that patch to final_answer()."
        )


def create_server(
    root: Path | None = None,
    config: SandboxConfig | None = None,
) -> FastMCP:
    return SWEBenchMCP(root, config or SandboxConfig()).server


mcp = create_server()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[
        Path | None,
        typer.Option(
            "--sandbox-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    if transport not in ("stdio", "http"):
        raise typer.BadParameter("Transport must be stdio or http.")
    policy = SandboxConfig()
    if config is not None:
        try:
            policy = SandboxConfig.model_validate_json(config.read_text())
        except (OSError, ValueError) as error:
            raise typer.BadParameter(
                f"Cannot load sandbox configuration: {error}",
            ) from None
    server = create_server(config=policy)

    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if transport == "http":
        server.run(transport="http", host=host, port=port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    app()
