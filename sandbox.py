import typer
import sys
import asyncio
import shlex
import ast
import resource
import subprocess
import builtins
import os
import selectors
import time
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Optional, Any, Type, Callable
from pathlib import Path
from fastmcp import Client
from mcp.types import Tool
from fastmcp.client.transports import StdioTransport
from dataclasses import dataclass
from src.config_model import SandboxConfig, SandboxIntermediateResult


SANDBOX_TEMPLATE_PATH = "src/sandbox_template.json"
COMMAND_END_MARKER = "<<<END_COMMAND>>>"
EXIT_COMMANDS = {"exit", "exit()"}


@dataclass(frozen=True)
class ParsedCall:
    name: str
    positional_arguments: tuple[Any, ...]
    keyword_arguments: dict[str, Any]


class FinalAnswerSignal(Exception):
    def __init__(self, answer: str) -> None:
        self.answer = answer
        super().__init__("final_answer called")


def final_answer(answer: Any) -> None:
    raise FinalAnswerSignal(answer)


def make_mcp_wrapper(
    tool_name: str,
    invoke: Callable[[str, dict[str, Any]], Any]
) -> Callable[..., Any]:
    def wrapper(**kwargs: Any) -> Any:
        return invoke(tool_name, kwargs)

    wrapper.__name__ = tool_name
    return wrapper


# Init Typer app
app = typer.Typer(
    add_completion=False,
    help=("Agent Smith - SandBox (MCP Client)"
          " - Launch an interactive sandbox environment.")
)


def parse_tool_call(source: str) -> ParsedCall:
    try:
        tree = ast.parse(source.strip(), mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc.msg}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError("Expected exactly one function call")

    expression = tree.body[0].value

    if not isinstance(expression, ast.Call):
        raise ValueError("Expected a function call")

    if not isinstance(expression.func, ast.Name):
        raise ValueError("Only direct function calls are allowed")

    positional_args: list[Any] = []

    for arg in expression.args:
        try:
            positional_args.append(ast.literal_eval(arg))
        except (ValueError, TypeError, SyntaxError) as e:
            raise ValueError(
                "Positional arguments must be literal values"
            ) from e

    keyword_args: dict[str, Any] = {}

    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("Unpacking keyword arguments is not allowed")

        if keyword.arg in keyword_args:
            raise ValueError(f"Duplicate argument: {keyword.arg}")

        try:
            keyword_args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError) as e:
            raise ValueError(
                f"Argument '{keyword.arg}' must be a literal value"
            ) from e

    return ParsedCall(
        name=expression.func.id,
        positional_arguments=tuple(positional_args),
        keyword_arguments=keyword_args
    )


def validate_tool_call(
    call: ParsedCall,
    tool_registry: dict[str, Tool]
) -> Tool | None:
    if call.name == "final_answer":
        positional_args = call.positional_arguments
        keyword_args = call.keyword_arguments

        if not (len(positional_args) == 1 and not keyword_args
                or not positional_args and set(keyword_args) == {"answer"}):
            raise ValueError(
                "final_answer requires exactly one positional argument "
                "or one 'answer' keyword argument"
            )
        # if not isinstance(call.arguments["answer"], str):
        #     raise ValueError("final_answer.answer must be a string")
        return None

    if call.positional_arguments:
        raise ValueError(
            f"MCP tool '{call.name}' accepts keyword arguments only"
        )

    tool = tool_registry.get(call.name)
    if tool is None:
        raise ValueError(f"Unknown tool: {call.name}")

    schema = tool.input_schema or {}

    try:
        validator = Draft202012Validator(schema)
        validator.check_schema(schema)
        validator.validate(call.keyword_arguments)
    except SchemaError as exc:
        raise ValueError(
            f"Invalid schema for MCP tool '{call.name}'"
        ) from exc
    except ValidationError as exc:
        path = ".".join(str(item) for item in exc.absolute_path)
        location = f" at '{path}'" if path else ""

        raise ValueError(
            f"Invalid arguments for '{call.name}'{location}: {exc.message}"
        ) from exc

    return tool


async def dispatch_tool_call(
    call: ParsedCall,
    tool_registry: dict[str, Tool],
    client: Optional[Client]
) -> SandboxIntermediateResult:
    tool = validate_tool_call(call, tool_registry)

    if tool is None:
        answer = (
            call.positional_arguments[0]
            if call.positional_arguments
            else call.keyword_arguments["answer"]
        )
        return SandboxIntermediateResult(
            success=True,
            tool_name="final_answer",
            sandbox_output=answer
        )

    if not client:
        raise ValueError("There is no active MCP client")

    try:
        result = await client.call_tool(
            call.name,
            call.keyword_arguments
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return SandboxIntermediateResult(
            success=False,
            tool_name=call.name,
            sandbox_output=f"MCP tool failed: {exc}"
        )

    return SandboxIntermediateResult(
        success=True,
        tool_name=call.name,
        sandbox_output=str(result)
    )


def get_python_type(prop: Any) -> str:
    types = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict[str, Any]",
        "null": "None"
    }

    if "anyOf" in prop:
        options = prop["anyOf"]
        non_null = [p for p in options if p.get("type") != "null"]
        if not non_null:
            return "None"
        return " | ".join(get_python_type(p) for p in non_null)

    if "enum" in prop:
        return "Literal[" + ", ".join(repr(value) for value in prop["enum"]) + "]"

    typ = prop.get("type")

    if typ == "array":
        item_type = get_python_type(prop.get("items", {}))
        return f"list[{item_type}]"

    return types.get(typ, "Any")


def schema_to_signature(tool: Tool) -> str:
    schema = tool.input_schema or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    params: list[str] = []

    for name, prop in properties.items():
        annotation = get_python_type(prop)

        if name in required:
            params.append(f"{name}: {annotation}")
        else:
            params.append(f"{name}: {annotation} | None = None")

    return f"- {tool.name}({', '.join(params)}): {tool.description or ''})"


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
        # "- show_tools(): show available tool names",
        # "- print(text: str): print a result"
    ]

    for tool in tools or []:
        lines.append(schema_to_signature(tool))

    return "\n".join(lines)


def build_tool_registry(tools: list[Tool]) -> dict[str, Tool]:
    registry: dict[str, Tool] = {}

    for tool in tools:
        if tool.name in registry:
            raise ValueError(f"Duplicate MCP tool name: {tool.name}")
        registry[tool.name] = tool

    return registry


class Sandbox:
    def __init__(
        self,
        template_path: Optional[Path] = None,
        mcp_stdio: Optional[str] = None,
        mcp_server: Optional[str] = None
    ) -> None:
        self.template_path = template_path or Path(SANDBOX_TEMPLATE_PATH)
        self.mcp_stdio = mcp_stdio
        self.mcp_server = mcp_server

        self.config: Optional[SandboxConfig] = None
        self.client: Optional[Client] = None
        self.tool_registry: dict[str, Tool] = {}
        self.mcp_wrappers: dict[str, Callable[..., Any]] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "Sandbox":
        await self._stack.__aenter__()

        with open(self.template_path) as f:
            self.config = SandboxConfig.model_validate_json(f.read())

        if self.mcp_stdio:
            cmd = shlex.split(self.mcp_stdio)
            self.client = Client(
                StdioTransport(
                    command=cmd[0],
                    args=cmd[1:]
                )
            )
        elif self.mcp_server:
            # EXAMPLE: uv run sandbox --mcp-server https://mcp.webrobot.eu/mcp
            self.client = Client(self.mcp_server)

        if self.client is not None:
            active_client = await self._stack.enter_async_context(self.client)
            tools = await active_client.list_tools()
            self.tool_registry = build_tool_registry(tools)
            self.mcp_wrappers = {
                name: make_mcp_wrapper(name, self.invoke_mcp_tool)
                for name in self.tool_registry
            }

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_value, traceback)

    def _limits(self) -> None:
        assert self.config is not None

        resource.setrlimit(
            resource.RLIMIT_CPU,
            (
                self.config.max_execution_time_seconds,
                self.config.max_execution_time_seconds
            )
        )
        resource.setrlimit(
            resource.RLIMIT_AS,
            (
                self.config.max_memory_mb * 1024 * 1024,
                self.config.max_memory_mb * 1024 * 1024
            )
        )

    def _execution_directory(self) -> str:
        if self.config is None:
            raise RuntimeError("Sandbox configuration is not loaded")

        for directory in self.config.allowed_directories:
            path = Path(directory)

            if path.is_dir():
                return str(path)

        raise RuntimeError("No allowed execution directory exists")

    def _allowed_path(self, path: str | os.PathLike[str]) -> Path:
        if self.config is None:
            raise RuntimeError("Sandbox configuration is not loaded")

        candidate = Path(path).expanduser().resolve()

        allowed = any(
            candidate == directory or directory in candidate.parents
            for configured in self.config.allowed_directories
            if (directory := Path(configured).resolve()).is_dir()
        )

        if not allowed:
            raise PermissionError(f"Access denied: {candidate}")

        return candidate

    def _safe_open(
        self,
        file: str | os.PathLike[str],
        mode: str = "r",
        *args: Any,
        **kwargs: Any
    ) -> Any:
        path = self._allowed_path(file)

        if any(flag in mode for flag in ("w", "a", "x", "+")):
            if not any(path == Path(directory).resolve()
                       or Path(directory).resolve() in path.parents
                       for directory in self.config.allowed_directories):
                raise PermissionError(f"Write access denied: {path}")

        return builtins.open(path, mode, *args, **kwargs)

    def _read_process_output(
        self,
        process: subprocess.Popen[bytes]
    ) -> tuple[bytes, bytes, bool, bool]:
        assert self.config is not None

        selector = selectors.DefaultSelector()
        streams = {
            process.stdout: bytearray(),
            process.stderr: bytearray()
        }

        for stream in streams:
            assert stream is not None
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)

        deadline = time.monotonic() + self.config.max_execution_time_seconds
        output_limit = self.config.max_output_bytes
        total_bytes = 0
        truncated = False
        timed_out = False

        while selector.get_map():
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                timed_out = True
                process.kill()
                break

            for key, _ in selector.select(remaining):
                stream = key.fileobj
                chunk = stream.read(4096)

                if not chunk:
                    selector.unregister(stream)
                    continue

                available = output_limit - total_bytes

                if available > 0:
                    streams[stream].extend(chunk[:available])
                    total_bytes += min(len(chunk), available)

                if len(chunk) > available:
                    truncated = True

        if timed_out:
            process.kill()

        process.wait()

        for stream in list(selector.get_map().values()):
            selector.unregister(stream.fileobj)

        return (
            bytes(streams[process.stdout]),
            bytes(streams[process.stderr]),
            timed_out,
            truncated
        )

    def restricted_builtins(self) -> dict[str, Any]:
        allowed_names = {
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "filter",
            "float",
            "int",
            "len",
            "list",
            "map",
            "max",
            "min",
            "print",
            "range",
            "reversed",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "zip",
            "Exception",
            "ValueError",
            "TypeError",
            "RuntimeError"
        }

        restricted = {
            name: getattr(builtins, name)
            for name in allowed_names
        }

        restricted["open"] = self._safe_open

        return restricted

    def build_execution_namespace(self) -> dict[str, Any]:
        namespace: dict[str, Any] = {
            "__builtins__": self.restricted_builtins(),
            "final_answer": final_answer
        }
        namespace.update(self.mcp_wrappers)
        return namespace

    def execute_in_namespace(
        self,
        source: str
    ) -> SandboxIntermediateResult:
        namespace = self.build_execution_namespace()

        try:
            exec(compile(source, "<sandbox>", "exec"), namespace, namespace)
        except FinalAnswerSignal as signal:
            return SandboxIntermediateResult(
                success=True,
                tool_name="final_answer",
                sandbox_output=signal.answer
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                sandbox_output=str(exc)
            )

        return SandboxIntermediateResult(
            success=True,
            tool_name="python",
            sandbox_output=""
        )

    def show_manual(self) -> None:
        print(manual(list(self.tool_registry.values())))
        sys.stdout.flush()

    def show_tools(self) -> list[str]:
        tools = ["final_answer", "show_tools"]
        tools.extend([t for t in self.tool_registry])
        return tools

    def check_imports(self, names: list[str]) -> None:
        if not self.config:
            raise RuntimeError("Sandbox configuration is not loaded")

        unauth_imports: list[str] = []

        for name in names:
            allowed = any(
                name == rule
                or (
                    rule.endswith(".*") and name.startswith(rule[:-2] + ".")
                )
                for rule in self.config.authorized_imports
            )
            if not allowed:
                unauth_imports.append(name)
        if unauth_imports:
            imports = ", ".join(sorted(set(unauth_imports)))
            raise ValueError(f"Unauthorized import(s) detected: {imports}")

    def validate_python_source(self, source: str) -> None:
        tree = ast.parse(source, mode="exec")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                self.check_imports(names)
            elif isinstance(node, ast.ImportFrom):
                self.check_imports([node.module or ""])
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError("Global declarations are not allowed")

    async def invoke_mcp_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        call = ParsedCall(
            name=name,
            positional_arguments=(),
            keyword_arguments=arguments
        )
        tool = validate_tool_call(call, self.tool_registry)

        if tool is None:
            raise ValueError("final_answer is not an MCP tool")

        if self.client is None:
            raise RuntimeError("No active MCP client")

        return await self.client.call_tool(name, arguments)

    def run_python(self, source: str) -> SandboxIntermediateResult:
        if not self.config:
            raise RuntimeError("Sandbox configuration is not loaded")

        try:
            self.validate_python_source(source)
        except (SyntaxError, ValueError, RuntimeError) as e:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                sandbox_output=str(e)
            )

        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", source],
                cwd=self._execution_directory(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=self._limits
            )

            stdout, stderr, timed_out, truncated = self._read_process_output(
                process
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError as e:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                sandbox_output=f"Execution failed: {e}"
            )

        output = stdout + (b"\n" if stdout and stderr else b"") + stderr
        text = output.decode("utf-8", errors="replace")

        if timed_out:
            text = f"Execution timed out. Partial output:\n{text}"

        if truncated:
            text += "\nOutput truncated due to configured size limits."

        if process.returncode != 0 and not timed_out:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                sandbox_output=text
            )
        return SandboxIntermediateResult(
            success=not timed_out,
            tool_name="python",
            sandbox_output=text
        )

    async def read_command(self) -> str | None:
        lines: list[str] = []

        import codeop

        while True:
            prompt = "sandbox> " if not lines else "... "

            try:
                line = await asyncio.to_thread(input, prompt)
            except EOFError:
                return "\n".join(lines) if lines else None

            if not lines and line.strip() in EXIT_COMMANDS:
                return line.strip()

            lines.append(line)
            source = "\n".join(lines)

            try:
                compiled = codeop.compile_command(
                    source, "<sandbox>", "single"
                )
            except (SyntaxError, OverflowError, ValueError):
                return source

            if compiled is not None:
                return source

    async def run(self) -> None:
        self.show_manual()

        while True:
            data = await self.read_command()

            if data is None or data in EXIT_COMMANDS:
                # TODO: some quit function
                print("Sandbox: Work complete,",
                      "all connections terminated,",
                      "all resources released. Bye!")
                return

            if not data:
                continue

            if data in ("manual", "manual()"):
                self.show_manual()
                continue
            if data in ("show_tools", "show_tools()"):
                result = SandboxIntermediateResult(
                    success=True,
                    tool_name="show_tools",
                    sandbox_output=self.show_tools()
                )
                print(result.model_dump_json())
                sys.stdout.flush()
                continue

            call: Optional[ParsedCall] = None

            try:
                result = self.run_python(data)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                result = SandboxIntermediateResult(
                    success=False,
                    tool_name=call.name if call else "sandbox",
                    sandbox_output=str(e)
                )
            print(result.model_dump_json())
            sys.stdout.flush()


async def sandbox_main(
    template_path: Optional[Path] = None,
    mcp_stdio: Optional[str] = None,
    mcp_server: Optional[str] = None
) -> None:
    async with Sandbox(template_path, mcp_stdio, mcp_server) as sandbox:
        await sandbox.run()


@app.command()
def main(
    template_path: Optional[Path] = typer.Argument(
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

    if mcp_stdio and mcp_server:
        raise ValueError("Wrong format 'uv run --mcp-stdio ... --mcp-server "
                         "...'. It must contain either mcp-stdio or mcp-server"
                         ", but not both")

    asyncio.run(sandbox_main(template_path, mcp_stdio, mcp_server))


if __name__ == "__main__":
    app()
