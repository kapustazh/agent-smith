import typer
import re
import sys
import asyncio
import shlex
from contextlib import AsyncExitStack
from typing import Optional, Any
from pathlib import Path
from fastmcp import Client
from mcp.types import Tool
from fastmcp.client.transports import StdioTransport

# Init Typer app
app = typer.Typer(
    add_completion=False,
    help=("Agent Smith - SandBox (MCP Client)"
          " - Launch an interactive sandbox environment.")
)


def get_python_type(prop: Any) -> str:
    types = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict[str, Any]",
    }

    if "anyOf" in prop:
        options = prop["anyOf"]
        non_null = [p for p in options if p.get("type") != "null"]
        result = " | ".join(get_python_type(p) for p in non_null)
        return f"{result} | None" if len(non_null) < len(options) else result

    typ = prop.get("type")

    if typ == "array":
        item_type = get_python_type(prop.get("items", {}))
        return f"list[{item_type}]"

    # if typ == "object":
    #     properties = prop.get("properties", {})
    #     if properties:
    #         value_types = {get_python_type(v) for v in properties.values()}
    #         if len(value_types) == 1:
    #             return f"dict[str, {next(iter(value_types))}]"
    #     return "dict[str, Any]"

    return types.get(typ, "Any")


def schema_to_signature(tool: Tool) -> str:
    schema = tool.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    params = []
    for name, prop in properties.items():
        typ = get_python_type(prop)

        default = "" if name in required else " = None"
        params.append(f"{name}: {typ}{default}")

    return f"{tool.name}({', '.join(params)})"


def manual(tools: list[Tool] | None) -> str:
    """Generate the dynamic manual the LLM receives in the system prompt."""

    lines = [
        "Sandbox execution manual:",
        "- Write exactly one Python code block per step.",
        "- Tool calls are ordinary Python function calls.",
        "- Use final_answer(answer_string) when the task is solved.",
        "- Positional arguments are rejected for MCP tools;" +
        "use keyword arguments.\n",
        "Available tools:",
        "- final_answer(answer: str): terminate the agent loop and " +
        "return the solution."
    ]

    if tools:
        for tool in tools:
            lines.append(schema_to_signature(tool))

    return "\n".join(lines)


def run_read_loop() -> None:
    data = " "

    while (data != "exit" and data != "exit()" and data != ""):
        data = sys.stdin.read().strip()

        match = re.search(r"\bfinal_answer\s*\(", data)

        if match:
            start = match.end()
            depth = 1

            for i in range(start, len(data)):
                if data[i] == "(":
                    depth += 1
                elif data[i] == ")":
                    depth -= 1

                    if depth == 0:
                        answer = data[start:i]
                        print(f"\nSandbox: the final answer is:\n{answer}")
                        return
        # -- run the command
        print("-"*20, "Received command for sandbox:\n", data, "\n", "-"*20)


async def sandbox(
        template: Optional[Path] = None,
        mcp_stdio: Optional[str] = None,
        mcp_server: Optional[str] = None) -> None:
    async with AsyncExitStack() as stack:
        tools: Optional[list[Tool]] = None
        client: Optional[Client] = None

        if mcp_stdio:
            cmd = shlex.split(mcp_stdio)
            client = Client(
                StdioTransport(
                    command=cmd[0],
                    args=cmd[1:]
                )
            )
        elif mcp_server:
            # EXAMPLE: uv run sandbox --mcp-server https://mcp.webrobot.eu/mcp
            client = Client(mcp_server)

        if client is not None:
            active_client = await stack.enter_async_context(client)
            tools = await active_client.list_tools()

        typer.echo(manual(tools))

        run_read_loop()

        print("Sandbox: Work is complete,",
              "all connections terminate, all resources released. Bye!")


@app.command()
def main(
    template: Optional[Path] = typer.Argument(
        None,
        help="Path to the sandbox template JSON file.",
    ),
    mcp_stdio: Optional[str] = typer.Option(
        None,
        "--mcp-stdio",
        help="Command to run MCP tools via stdio (e.g., 'python mcp_tool.py')."
    ),
    mcp_server: Optional[str] = typer.Option(
        None,
        "--mcp-server",
        help="MCP server URL.",
    )
) -> None:
    """
    Sandbox CLI - for Agent Smith.

    Write command and press Ctrl+D
    """

    # if mcp_stdio and mcp_server:
    #     wrong format ???

    asyncio.run(sandbox(template, mcp_stdio, mcp_server))


if __name__ == "__main__":
    app()
