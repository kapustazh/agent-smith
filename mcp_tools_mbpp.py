from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import ValidationError
import typer

from src.config_model import SandboxConfig
from src.mbpp_executor import (
    MBPPExecutionRequest, execute_mbpp, execution_result,
)


app = typer.Typer(add_completion=False)


class MBPPTools:
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    async def run_tests(
        self, code: str, test_list: list[str],
        test_imports: list[str] | None = None,
    ) -> str:
        try:
            request = MBPPExecutionRequest(
                code=code, test_list=test_list, config=self.config,
                test_imports=test_imports if test_imports is not None else [],
            )
        except ValidationError as error:
            details = "; ".join(
                ".".join(str(part)
                         for part in item["loc"]) + ": " + item["msg"]
                for item in error.errors(
                    include_input=False, include_url=False)
            )
            return execution_result(
                False, "Invalid MBPP request: " + details +
                " No code or tests were executed. Supply code and assertions; "
                "import entries must be non-empty.",
                self.config,
            ).model_dump_json()
        result = await execute_mbpp(request)
        return result.model_dump_json()


def task_resource() -> dict[str, str]:
    return {"runtime": "deno/pyodide",
            "tool": "run_tests(code, test_list, test_imports=None)"}


def mbpp_solver() -> str:
    return (
        "Solve the MBPP task. Call run_tests(code=solution_code, "
        "test_list=the_public_assertions, test_imports=task.test_imports). "
        "The server runs those imports before the solution and assertions. "
        "If a setup error says tests did not run, repair the server setup "
        "before retrying; do not treat it as assertion feedback. "
        "Use the real test feedback to improve the solution. "
        "When finished, call the sandbox's final_answer(solution_code)."
    )


def create_server(config: SandboxConfig | None = None) -> FastMCP:
    server = FastMCP("Agent Smith - MBPP")
    server.tool(MBPPTools(config).run_tests, description=(
        "Test a candidate against supplied assertions in the Deno sandbox."
        "\n\nOptional test_imports run before the solution and assertions.\n"
        "Returns JSON with success and output, including setup/test errors.\n"
        "Limits come from server configuration. No hidden tests are read."
    ))
    server.resource("agent://mbpp/task", description=(
        "Server metadata; the caller supplies task code and public tests."
    ))(task_resource)
    server.prompt()(mbpp_solver)
    return server


mcp = create_server()


@app.command()
def main(
    config: Annotated[Path | None, typer.Option(
        "--sandbox-config", exists=True, dir_okay=False, readable=True,
    )] = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    policy = SandboxConfig()
    if transport not in ("stdio", "http"):
        raise typer.BadParameter("Transport must be stdio or http.")
    if config is not None:
        try:
            policy = SandboxConfig.model_validate_json(
                config.read_text(),
            )
        except (OSError, ValidationError) as error:
            raise typer.BadParameter(
                f"Cannot load sandbox configuration: {error}",
                param_hint="--sandbox-config",
            ) from None
    server = create_server(policy)
    if transport == "http":
        server.run(transport="http", host=host, port=port, show_banner=False)
    else:
        server.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    app()
