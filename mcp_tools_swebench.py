from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
import typer

from src.config_model import SandboxConfig
from src.swebench_tools import RepositoryTools


def create_server(
    root: Path | None = None, config: SandboxConfig | None = None,
) -> FastMCP:
    tools = RepositoryTools(root, config or SandboxConfig())
    server = FastMCP("Agent Smith - SWE-bench")
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
        server.tool(method, description=description)
    return server


mcp = create_server()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path | None, typer.Option(
        "--sandbox-config", exists=True, dir_okay=False, readable=True,
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
    server = create_server(config=policy)
    if transport == "http":
        server.run(transport="http", host=host, port=port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    app()
