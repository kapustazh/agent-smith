import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import signal
from typing import Annotated, Any, AsyncIterator, Callable

from anyio import CancelScope
from fastmcp import FastMCP
import typer

from src.config_model import SandboxConfig, SWEBenchTaskInput
from src.swebench_docker import DockerSession
from src.swebench_tools import RepositoryTools


class SWEBenchMCP:
    def __init__(
        self, root: Path | None, config: SandboxConfig,
        task: SWEBenchTaskInput | None,
    ) -> None:
        self.session = DockerSession(
            task, config) if task is not None else None
        self.lock = self.session.lock if self.session else asyncio.Lock()
        self.repository = (self.session.repository if self.session else
                           RepositoryTools(root, config))
        self.server = FastMCP("Agent Smith - SWE-bench",
                              lifespan=self.lifespan)
        for method, description in (
            (self.read_file, "Read numbered UTF-8 lines."),
            (self.edit_file, "Replace exactly one text match."),
            (self.list_files, "Recursively list matching files."),
            (self.search_code, "Find regex matches and locations."),
            (self.search_function_or_class_definition_in_code,
             "Find exact Python function and class definitions."),
            (self.find_references,
             "Find Python references and aliases at a 1-based source line."),
        ):
            self.server.tool(method, description=description)
        self.server.tool(self.run_command)
        self.server.tool(self.run_tests)
        self.server.tool(self.get_patch)
        self.server.resource("agent://swebench/task")(self.task_resource)
        self.server.prompt()(self.swebench_solver)

    async def call_repository(
        self, method: Callable[..., str], *args: Any,
    ) -> str:
        async with self.lock:
            work = asyncio.create_task(asyncio.to_thread(
                method, *args))
            try:
                return await asyncio.shield(work)
            finally:
                # Cancellation cannot release the lock while a thread edits.
                with CancelScope(shield=True):
                    await work

    async def read_file(
        self, filepath: str, start_line: int = 1, end_line: int | None = None,
    ) -> str:
        return await self.call_repository(
            self.repository.read_file, filepath, start_line, end_line)

    async def edit_file(
        self, filepath: str, old_str: str, new_str: str,
    ) -> str:
        return await self.call_repository(
            self.repository.edit_file, filepath, old_str, new_str)

    async def list_files(
        self, directory: str = ".", pattern: str = "*",
    ) -> str:
        return await self.call_repository(
            self.repository.list_files, directory, pattern)

    async def search_code(self, pattern: str, file_pattern: str = "*") -> str:
        return await self.call_repository(
            self.repository.search_code, pattern, file_pattern)

    async def search_function_or_class_definition_in_code(
        self, name: str,
    ) -> str:
        return await self.call_repository(
            self.repository.search_function_or_class_definition_in_code, name)

    async def find_references(
        self, name: str, filepath: str, line: int,
    ) -> str:
        return await self.call_repository(
            self.repository.find_references, name, filepath, line)

    @asynccontextmanager
    async def lifespan(self, server: FastMCP) -> AsyncIterator[dict[str, str]]:
        try:
            if self.session is not None:
                await self.session.start()
            yield {}
        finally:
            if self.session is not None:
                with CancelScope(shield=True):
                    await asyncio.shield(self.session.close())

    def execution(self) -> DockerSession:
        if self.session is None:
            raise self.repository._error(
                "Execution requires --task-file with docker_image and "
                "eval_script. TESTBED_PATH configures repository tools only.")
        return self.session

    async def run_command(
        self, command: str, workdir: str = "/testbed",
    ) -> str:
        """Run a container command and return output and exit code."""
        return await self.execution().run_command(command, workdir)

    async def run_tests(self) -> str:
        """Run eval_script with the prepared environment on a disposable copy.

        Returns script logs and exit_code, not a benchmark verdict.
        """
        return await self.execution().run_tests()

    async def get_patch(self) -> str:
        """Get a patch against the original commit, including new files.

        Oversized patches return an error instead of an incomplete patch.
        """
        return await self.execution().get_patch()

    def task_resource(self) -> dict[str, str]:
        """Task metadata and execution configuration status."""
        if self.session is None:
            return {"execution": "requires --task-file",
                    "repository": str(self.repository.root)}
        return {"instance_id": self.session.task.instance_id,
                "problem_statement": self.session.task.problem_statement,
                "repository": "/testbed", "execution": "docker"}

    def swebench_solver(self) -> str:
        """Instructions for solving the task with these tools."""
        return (
            "Inspect the task and repository. Edit the source, run_tests(), "
            "read the test logs even when exit_code is zero, and get_patch(). "
            "Pass that patch to the sandbox's final_answer(). Execution tools "
            "require the server to be started with --task-file."
        )


def create_server(
    root: Path | None = None, config: SandboxConfig | None = None,
    task: SWEBenchTaskInput | None = None,
) -> FastMCP:
    return SWEBenchMCP(root, config or SandboxConfig(), task).server


def terminate(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


mcp = create_server()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path | None, typer.Option(
        "--sandbox-config", exists=True, dir_okay=False, readable=True,
    )] = None,
    task_file: Annotated[Path | None, typer.Option(
        "--task-file", exists=True, dir_okay=False, readable=True,
    )] = None,
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
    task = None
    if task_file is not None:
        try:
            task = SWEBenchTaskInput.model_validate_json(task_file.read_text())
        except (OSError, ValueError) as error:
            raise typer.BadParameter(f"Cannot load SWE-bench task: {error}")
    server = create_server(config=policy, task=task)

    signal.signal(signal.SIGTERM, terminate)
    if transport == "http":
        server.run(transport="http", host=host, port=port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    app()
