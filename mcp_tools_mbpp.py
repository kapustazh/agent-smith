"""MBPP MCP tools backed by the configured Deno sandbox executor."""

from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
import typer

from src.config_model import SandboxConfig
from src.mbpp_executor import MBPPExecutionRequest, execute_mbpp


mcp = FastMCP("Agent Smith - MBPP")
app = typer.Typer(add_completion=False)
sandbox_config = SandboxConfig()


@mcp.tool()
async def run_tests(code: str, test_list: list[str]) -> str:
    """Test a candidate against supplied assertions in the Deno sandbox.

    Include required imports in code. Returns a JSON string containing
    success and output. The server uses the sandbox's trusted configuration;
    the caller cannot override execution limits. No hidden tests are accessed.
    """
    if not code.strip() or not test_list or any(
        not test.strip() for test in test_list
    ):
        return ('{"success": false, "output": '
                '"Code and at least one test must be non-empty."}')
    result = await execute_mbpp(MBPPExecutionRequest(
        code=code, test_list=test_list, config=sandbox_config,
    ))
    return result.model_dump_json()


@mcp.resource("agent://mbpp/task")
def task_resource() -> dict[str, str]:
    """Server metadata; the caller supplies task code and public tests."""
    return {"runtime": "deno/pyodide", "tool": "run_tests(code, test_list)"}


@mcp.prompt()
def mbpp_solver() -> str:
    return (
        "Solve the MBPP task. Include required imports in your Python code, "
        "then call run_tests(code=code, test_list=the_public_assertions). "
        "Use the real test feedback to improve the solution. "
        "When finished, call the sandbox's final_answer(solution_code)."
    )


@app.command()
def main(
    config: Annotated[Path | None, typer.Option(
        "--sandbox-config", exists=True, dir_okay=False, readable=True,
    )] = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    global sandbox_config
    if transport not in ("stdio", "http"):
        raise typer.BadParameter("Transport must be stdio or http.")
    if config is not None:
        sandbox_config = SandboxConfig.model_validate_json(config.read_text())
    if transport == "http":
        mcp.run(transport="http", host=host, port=port, show_banner=False)
    else:
        mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    app()
