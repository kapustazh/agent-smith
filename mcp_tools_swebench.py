import asyncio
from contextlib import asynccontextmanager
from functools import wraps
from pathlib import Path
import signal
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable

from fastmcp import FastMCP
import typer

from src.config_model import SandboxConfig, SWEBenchTaskInput
from src.swebench_docker import DockerSession
from src.swebench_tools import RepositoryTools


def create_server(
    root: Path | None = None, config: SandboxConfig | None = None,
    task: SWEBenchTaskInput | None = None,
) -> FastMCP:
    policy = config or SandboxConfig()
    session = DockerSession(task, policy) if task is not None else None
    tools = session.repository if session else RepositoryTools(root, policy)

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[dict[str, str]]:
        try:
            if session is not None:
                await session.start()
            yield {}
        finally:
            if session is not None:
                await asyncio.shield(session.close())

    def serialized(
        method: Callable[..., str],
    ) -> Callable[..., Awaitable[str]]:
        @wraps(method)
        async def call(*args: Any, **kwargs: Any) -> str:
            assert session is not None
            async with session.lock:
                return await asyncio.to_thread(method, *args, **kwargs)
        return call

    server = FastMCP("Agent Smith - SWE-bench", lifespan=lifespan)
    for method, description in (
        (tools.read_file, "Read UTF-8 lines using inclusive, 1-based ranges."),
        (tools.edit_file, "Replace exactly one text match."),
        (tools.list_files, "Recursively list files matching a glob pattern."),
        (tools.search_code, "Search text by regex; return paths and lines."),
        (tools.search_function_or_class_definition_in_code,
         "Find exact Python function and class definitions."),
        (tools.find_references,
         "Find Python references and aliases at a 1-based source line."),
    ):
        server.tool(serialized(method) if session else method,
                    description=description)
    if session is not None:
        server.tool(session.run_command, description=(
            "Run a shell command inside the task container. Returns stdout, "
            "stderr, exit_code, timed_out and truncated as JSON."))
        server.tool(session.run_tests, description=(
            "Run the task eval_script on a disposable repository copy. "
            "Returns script output and exit_code, not a benchmark verdict; "
            "read the test log even if the script exits zero."))
        server.tool(session.get_patch, description=(
            "Get a Git patch against the task's original commit, including "
            "new files. Oversized patches return an error."))
    return server


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

    def terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    if transport == "http":
        server.run(transport="http", host=host, port=port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    app()
