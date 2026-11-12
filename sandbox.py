import typer
import sys
import asyncio
import shlex
import ast
# import resource
# import subprocess
import builtins
import os
import re
# import selectors
# import time
# import codeop
import signal
import json
import keyword
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


# class FinalAnswerSignal(Exception):
#     def __init__(self, answer: str) -> None:
#         self.answer = answer


# def final_answer(answer: Any) -> None:
#     raise FinalAnswerSignal(answer)


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

    for key_word in expression.keywords:
        if key_word.arg is None:
            raise ValueError("Unpacking keyword arguments is not allowed")

        if key_word.arg in keyword_args:
            raise ValueError(f"Duplicate argument: {key_word.arg}")

        try:
            keyword_args[key_word.arg] = ast.literal_eval(key_word.value)
        except (ValueError, TypeError, SyntaxError) as e:
            raise ValueError(
                f"Argument '{key_word.arg}' must be a literal value"
            ) from e

    return ParsedCall(
        name=expression.func.id,
        positional_arguments=tuple(positional_args),
        keyword_arguments=keyword_args
    )


def validate_tool_call(
    call: ParsedCall,
    tool_registry: dict[str, Tool]
) -> Optional[Tool]:
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
        return (
            "Literal[" + ", ".join(repr(value) for value in prop["enum"]) + "]"
        )

    typ = prop.get("type")

    if typ == "array":
        item_type = get_python_type(prop.get("items", {}))
        return f"list[{item_type}]"

    return types.get(typ, "Any")


def schema_to_signature(tool: Tool, python_name: Optional[str] = None) -> str:
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

    name = python_name or tool.name
    result = f"- {name}({', '.join(params)})"

    if name != tool.name:
        result += f" [MCP name: {tool.name}]"

    if tool.description:
        # result += f": {tool.description}"
        result += f"\n  Description: {tool.description}"

    # # Include the actual JSON Schema for exact constraints.
    # result += (
    #     "\n  Input schema: "
    #     + json.dumps(schema, ensure_ascii=False)
    # )

    return result

    # return f"- {tool.name}({', '.join(params)}): {tool.description or ''})"


def manual(
    tools: Optional[list[Tool]],
    aliases: Optional[dict[str, str]] = None,
    # resources: Optional[list[Any]] = None,
    # prompts: Optional[list[Any]] = None,
    # resource_templates: Optional[list[Any]] = None,
    # discovery_errors: Optional[dict[str, str]] = None
) -> str:
    """Generate the dynamic manual the LLM receives in the system prompt."""

    lines = [
        # #  # Propose start
        # "Sandbox execution manual",
        # "",
        # "Write Python code in one code block per agent step.",
        # "MCP tools are available as Python functions.",
        # "Use keyword arguments for MCP tool calls.",
        # "MCP results can be strings, dictionaries, or lists.",
        # "Use final_answer(answer_string) to finish.",
        # "",
        # "Built-in function:",
        # "- final_answer(answer: str)",
        # # "- validate_edited_python(source: str, filename: str)",
        # "",
        # "MCP helper functions:",
        # "- mcp_call_tool(name: str, **kwargs)",
        # "- mcp_list_resources()",
        # "- mcp_list_resource_templates()",
        # "- mcp_read_resource(uri: str)",
        # "- mcp_list_prompts()",
        # "- mcp_get_prompt(name: str, arguments: dict | None = None)",
        # "",
        # "Available MCP tools:"
        # #  # Propose end
        "Sandbox execution manual:",
        "- Write exactly one Python code block per step.",
        "- Tool calls are ordinary Python function calls.",
        "- Use final_answer(answer_string) when the task is solved.",
        "- Positional arguments are rejected for MCP tools;" +
        "use keyword arguments.\n",
        "Available tools:",
        "- final_answer(answer: str)" +
        "\n  Description: terminate the agent loop and return the solution."
        # "- show_tools(): show available tool names",
        # "- print(text: str): print a result"
    ]

    registry = {tool.name: tool for tool in (tools or [])}

    if aliases:
        for python_name, original_name in aliases.items():
            lines.append(
                schema_to_signature(
                    registry[original_name], python_name
                )
            )
    # else:
    #     lines.append("(No MCP tools available)")

    # lines.append("\nAvailable MCP resources:")

    # for resource in resources or []:
    #     lines.append(
    #         f"- {resource.uri}: "
    #         f"{resource.description or ''}"
    #     )

    # lines.append("\nAvailable resource templates:")

    # for template in resource_templates or []:
    #     lines.append(
    #         f"- {template.uri_template}: "
    #         f"{template.description or ''}"
    #     )

    # lines.append("\nAvailable MCP prompts:")

    # for prompt in prompts or []:
    #     args = [
    #         f"{arg.name}{' (required)' if arg.required else ''}"
    #         for arg in (prompt.arguments or [])
    #     ]
    #     lines.append(
    #         f"- {prompt.name}({', '.join(args)}): "
    #         f"{prompt.description or ''}"
    #     )

    # if discovery_errors:
    #     lines.append("\nMCP discovery warnings:")
    #     for capability, error in discovery_errors.items():
    #         lines.append(f"- {capability}: {error}")

    return "\n".join(lines)


def build_tool_registry(tools: list[Tool]) -> dict[str, Tool]:
    registry: dict[str, Tool] = {}

    for tool in tools:
        if tool.name in registry:
            raise ValueError(f"Duplicate MCP tool name: {tool.name}")
        registry[tool.name] = tool

    return registry


def build_mcp_aliases(
    registry: dict[str, Tool]
) -> dict[str, str]:
    """Map valid Python function names to original MCP tool names."""

    reserved = set(dir(builtins)) | {
        "final_answer",
        # "mcp_call_tool",
        # "mcp_read_resource",
        # "mcp_get_prompt",
        # "mcp_list_resources",
        # "mcp_list_resource_templates",
        # "mcp_list_prompts",
        "validate_edited_python"
    }

    aliases: dict[str, str] = {}

    for number, name in enumerate(sorted(registry), start=1):
        if (
            name.isidentifier()
            and not keyword.iskeyword(name)
            and not name.startswith("_")
            and name not in reserved
        ):
            alias = name
        else:
            alias = f"mcp_tool_{number}"

            while alias in reserved or alias in registry or alias in aliases:
                number += 1
                alias = f"mcp_tool_{number}"

        aliases[alias] = name
        reserved.add(alias)

    return aliases


def extract_llm_code(
    response: str
) -> tuple[Optional[str], Optional[str]]:
    """Extract one Python block, reporting any recovery."""

    blocks: list[tuple[str, str, bool]] = []
    in_block: bool = False
    language: str = ""
    current: list[str] = []

    for line in response.splitlines(keepends=True):
        text = line.strip()

        if not in_block:
            if text.startswith("```"):
                language = text[3:].strip().lower()
                current = []
                in_block = True
            continue

        if text == "```":
            blocks.append(
                (language, "".join(current), True)
            )
            in_block = False
            current = []
        else:
            current.append(line)

    if in_block:
        blocks.append(
            (language, "".join(current), False)
        )

    if not blocks:
        return None, (
            "No valid code block found. "
            "Provide exactly one ```python block."
        )

    if len(blocks) != 1:
        return None, (
            f"Found {len(blocks)} fenced blocks. "
            "Expected exactly one."
        )

    language, code, closed = blocks[0]

    if language not in ("python", "py", ""):
        return None, (
            f"Unsupported code language: {language}. "
            "Expected Python."
        )

    if not code.strip():
        return None, "Python code block is empty."

    recovery = []

    if not language:
        recovery.append(
            "Unlabeled code fence interpreted as Python."
        )

    if not closed:
        recovery.append(
            "Closing fence missing; all remaining text "
            "was interpreted as Python code."
        )

    return (
        code.rstrip("\r\n"),
        " ".join(recovery) if recovery else None
    )


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
        self.tool_aliases: dict[str, str] = {}
        # self.resources: list[Any] = []
        # self.resource_templates: list[Any] = []
        # self.prompts: list[Any] = []
        # self.discovery_errors: dict[str, str] = {}
        self._stack = AsyncExitStack()
        self._worker: Optional[asyncio.subprocess.Process] = None

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
            active_client = await self._stack.enter_async_context(
                self.client
            )

            # Discover tools dynamically.
            tools = await active_client.list_tools()
            self.tool_registry = build_tool_registry(tools)
            self.tool_aliases = build_mcp_aliases(self.tool_registry)

            # Discover optional MCP capabilities.
            capabilities = {
                "resources": "list_resources",
                "resource_templates": "list_resource_templates",
                "prompts": "list_prompts"
            }

            for attribute, method_name in capabilities.items():
                try:
                    method = getattr(active_client, method_name)
                    items = await method()
                    setattr(self, attribute, items)
                except Exception as exc:
                    self.discovery_errors[attribute] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    setattr(self, attribute, [])

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType]
    ) -> None:
        try:
            await self._stop_worker()
        finally:
            await self._stack.__aexit__(exc_type, exc_value, traceback)

    def _execution_directory(self) -> str:
        if self.config is None:
            raise RuntimeError("Sandbox configuration is not loaded")

        for directory in self.config.allowed_directories:
            path = Path(directory)

            if path.is_dir():
                return str(path)

        # raise RuntimeError("No allowed execution directory exists")
        path = Path("/tmp/agent")
        os.mkdir(path)
        return str(path)

    async def _start_worker(self) -> None:
        """Start the worker if it is not already running."""
        if self._worker is not None and self._worker.returncode is None:
            return

        if self.config is None:
            raise RuntimeError("Sandbox configuration is not loaded")

        worker_path = (
            Path(__file__).resolve().parent / "src" / "sandbox_worker.py"
        )

        self._worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-u",
            str(worker_path),
            cwd=self._execution_directory(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8"
            }
        )

        try:
            assert self._worker.stdin is not None
            assert self._worker.stdout is not None

            request = {
                "type": "init",
                "config": self.config.model_dump(),
                "tools": self.tool_aliases
            }

            self._worker.stdin.write(
                (json.dumps(request) + "\n").encode()
            )
            await self._worker.stdin.drain()

            response = await asyncio.wait_for(
                self._worker.stdout.readline(),
                timeout=5
            )

            if not response:
                raise RuntimeError("Worker exited during initialization")
            message = json.loads(response)
            if message.get("type") != "ready":
                raise RuntimeError(
                    f"Worker initialization failed: \
                    {message.get('error', message)}"
                )

        except BaseException:
            await self._stop_worker()
            raise

    async def _stop_worker(self) -> None:
        """Terminate the worker and any processes in its process group."""

        if self._worker is None:
            return

        process = self._worker
        self._worker = None

        if process.returncode is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()

        await process.wait()

        if process.stdin is not None:
            process.stdin.close()

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

    def show_manual(self) -> None:
        print(
            manual(
                tools=list(self.tool_registry.values()),
                aliases=self.tool_aliases,
                # resources=self.resources,
                # prompts=self.prompts,
                # resource_templates=self.resource_templates,
                # discovery_errors=self.discovery_errors
            )
        )
        sys.stdout.flush()

    def show_tools(self) -> list[str]:
        return [
            "final_answer",
            # "mcp_call_tool",
            # "mcp_read_resource",
            # "mcp_get_prompt",
            # "mcp_list_resources",
            # "mcp_list_resource_templates",
            # "mcp_list_prompts",
            # "validate_edited_python",
            *self.tool_aliases.keys()
        ]

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
        arguments: dict[str, Any]
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

    async def _handle_mcp_request(
        self,
        request: dict[str, Any]
    ) -> dict[str, Any]:
        """Run MCP operations in the trusted parent process."""

        request_id = request.get("id")

        try:
            if self.client is None or self.config is None:
                raise RuntimeError("No active MCP connection")

            operation = request.get("operation")

            def as_dict(obj: Any) -> Any:
                if hasattr(obj, "model_dump"):
                    return obj.model_dump(
                        mode="json", exclude_none=True
                    )
                return obj

            if operation == "call_tool":
                name = request["name"]
                arguments = request["arguments"]

                if not (isinstance(name, str) and isinstance(arguments, dict)):
                    raise ValueError("Invalid MCP tool request")

                tool = self.tool_registry.get(name)
                if tool is None:
                    raise ValueError(f"Unknown MCP tool: {name}")

                # Validate against the server's JSON Schema.
                schema = tool.input_schema
                Draft202012Validator.check_schema(schema)
                Draft202012Validator(schema).validate(arguments)

                # Run in the parent, NOT in the restricted worker.
                result = await self.client.call_tool_mcp(
                    name, arguments
                )

                if result.is_error:
                    errors = [
                        getattr(block, "text", str(block))
                        for block in result.content
                    ]
                    raise RuntimeError(
                        "MCP tool reported an error: "
                        + " ".join(errors)
                    )

                value = result.structured_content

                if value is None:
                    blocks = [as_dict(b) for b in result.content]
                    if (
                        len(blocks) == 1
                        and blocks[0].get("type") == "text"
                    ):
                        value = blocks[0]["text"]
                    else:
                        value = blocks

            elif operation == "list_resources":
                items = await self.client.list_resources()
                value = [as_dict(item) for item in items]

            elif operation == "list_resource_templates":
                items = await self.client.list_resource_templates()
                value = [as_dict(item) for item in items]

            elif operation == "read_resource":
                uri = request["uri"]
                if not isinstance(uri, str):
                    raise ValueError("Resource URI must be a string")

                items = await self.client.read_resource(uri)
                value = [as_dict(item) for item in items]

            elif operation == "list_prompts":
                items = await self.client.list_prompts()
                value = [as_dict(item) for item in items]

            elif operation == "get_prompt":
                name = request["name"]
                arguments = request.get("arguments") or {}

                if not isinstance(name, str):
                    raise ValueError("Prompt name must be a string")

                if not isinstance(arguments, dict) or not all(
                    isinstance(k, str) and isinstance(v, str)
                    for k, v in arguments.items()
                ):
                    raise ValueError(
                        "Prompt arguments must be a dict[str, str]"
                    )

                result = await self.client.get_prompt(
                    name, arguments
                )
                value = as_dict(result)

            else:
                raise ValueError(
                    f"Unsupported MCP operation: {operation}"
                )

            # Limit the data sent back to sandboxed Python.
            encoded = json.dumps(
                value, ensure_ascii=False, default=str
            ).encode("utf-8")

            limit = self.config.max_output_bytes
            truncated = len(encoded) > limit

            if truncated:
                preview = encoded[:max(0, limit - 100)].decode(
                    "utf-8", errors="ignore"
                )
                value = (
                    preview
                    + "\n[MCP output truncated: JSON preview only]"
                )

            return {
                "type": "mcp_response",
                "id": request_id,
                "ok": True,
                "value": value,
                "truncated": truncated
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
            )[:2000]

            return {
                "type": "mcp_response",
                "id": request_id,
                "ok": False,
                "error": error,
                "truncated": False
            }

    async def run_python(self, source: str) -> SandboxIntermediateResult:
        if self.config is None:
            raise RuntimeError("Sandbox configuration is not loaded")

        try:
            self.validate_python_source(source)
        except SyntaxError as e:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                sandbox_output=f"Syntax error: {e}",
                status="syntax_error"
            )
        except (ValueError, RuntimeError) as e:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                sandbox_output=str(e),
                status="error"
            )

        await self._start_worker()
        process = self._worker
        assert process is not None
        assert process.stdin is not None
        assert process.stdout is not None

        pieces: list[str] = []
        byte_count = 0
        char_count = 0
        truncated = False
        tool_truncated = False

        def append_output(value: str) -> None:
            nonlocal byte_count, char_count, truncated

            remaining_bytes = max(
                0, self.config.max_output_bytes - byte_count
            )
            remaining_chars = max(
                0, self.config.max_output_chars - char_count
            )

            accepted = (
                value.encode("utf-8")[:remaining_bytes]
                .decode("utf-8", errors="ignore")[:remaining_chars]
            )

            if accepted:
                pieces.append(accepted)
            byte_count += len(accepted.encode("utf-8"))
            char_count += len(accepted)

            if accepted != value:
                truncated = True

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.max_execution_time_seconds

        try:
            request = {"type": "run", "code": source}
            process.stdin.write(
                (json.dumps(request) + "\n").encode()
            )
            await asyncio.wait_for(
                process.stdin.drain(),
                timeout=max(0.001, deadline - loop.time()),
            )

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()

                raw = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=remaining
                )

                if not raw:
                    await self._stop_worker()
                    return SandboxIntermediateResult(
                        success=False,
                        tool_name="python",
                        status="error",
                        sandbox_output=(
                            "".join(pieces) +
                            "\nWorker exited unexpectedly."
                        ),
                        partial=True,
                        truncated=truncated
                    )

                message = json.loads(raw)
                kind = message.get("type")

                if kind == "output":
                    append_output(message.get("text", ""))
                    continue

                if kind == "mcp_request":
                    # MCP work executes outside the restricted worker.
                    # Pause the sandbox-code timeout during this operation.
                    pause_started = loop.time()

                    try:
                        response = await self._handle_mcp_request(message)

                        process.stdin.write(
                            (json.dumps(response, ensure_ascii=False) + "\n")
                            .encode("utf-8")
                        )

                        await asyncio.wait_for(
                            process.stdin.drain(),
                            timeout=5
                        )

                        if response.get("truncated"):
                            tool_truncated = True
                    finally:
                        deadline += loop.time() - pause_started

                    continue

                if kind == "control":
                    await self._stop_worker()
                    if message.get("name") == "KeyboardInterrupt":
                        raise KeyboardInterrupt()
                    raise SystemExit(message.get("code"))

                if kind == "result":
                    status = message["status"]

                    if status == "final_answer":
                        answer = message["answer"]
                        return SandboxIntermediateResult(
                            success=True,
                            tool_name="final_answer",
                            sandbox_output=answer,
                            status="final_answer",
                            final_answer=answer,
                            truncated=(truncated or tool_truncated)
                        )

                    append_output(message.get("error", ""))
                    output = "".join(pieces)

                    if truncated:
                        output += (
                            "\n[Sandbox: Output truncated due to size limits.]"
                        )

                    if tool_truncated:
                        output += (
                            "\n[Sandbox: MCP tool output was truncated "
                            "due to configured size limits.]"
                        )

                    was_truncated = truncated or tool_truncated

                    return SandboxIntermediateResult(
                        success=(status == "success"),
                        tool_name="python",
                        sandbox_output=output,
                        status=(
                            "truncated"
                            if was_truncated and status == "success"
                            else status
                        ),
                        truncated=was_truncated
                    )

                if kind == "memory_limit":
                    await self._stop_worker()

                raise RuntimeError(f"Unexpected worker message: {kind}")

        except asyncio.TimeoutError:
            await self._stop_worker()

            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                status="timeout",
                partial=True,
                truncated=truncated,
                sandbox_output=(
                    f"Execution timed out after "
                    f"{self.config.max_execution_time_seconds} seconds.\n"
                    f"Partial output:\n{''.join(pieces)}\n"
                    "Worker terminated. Namespace state was lost."
                    + (
                        "\n[Sandbox: Output truncated due to size limits.]"
                        if truncated else ""
                    )
                )
            )

        except asyncio.CancelledError:
            await self._stop_worker()
            raise

        except (BrokenPipeError, ConnectionResetError,
                json.JSONDecodeError, RuntimeError) as exc:
            await self._stop_worker()

            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                status="error",
                sandbox_output=f"Worker communication failed: {exc}",
            )

    async def execute_llm_response(
        self,
        response: str
    ) -> SandboxIntermediateResult:
        """Extract and execute code from an LLM response."""

        code, feedback = extract_llm_code(response)

        if code is None:
            return SandboxIntermediateResult(
                success=False,
                tool_name="python",
                status="no_code",
                sandbox_output=feedback or "No executable code."
            )

        # Execute using the existing restricted worker.
        result = await self.run_python(code)

        if feedback:
            output = result.sandbox_output

            if isinstance(output, list):
                output = "\n".join(output)

            result = result.model_copy(
                update={
                    "sandbox_output": (
                        f"[Code recovery] {feedback}\n{output}"
                    ),
                    "status": (
                        "recovered_code"
                        if result.status == "success"
                        else result.status
                    ),
                    "code_recovered": True,
                    "recovery_details": feedback
                }
            )

        return result

    # #  With end_marker only
    # async def read_command(self) -> tuple[Optional[str], bool]:
    #     """Read one command terminated by <<<END_COMMAND>>>.

    #     Returns (command, complete).
    #     """

    #     lines: list[str] = []
    #     terminal_mode = sys.stdin.isatty()

    #     while True:
    #         if terminal_mode:
    #             prompt = "sandbox> " if not lines else "... "
    #             print(prompt, end="")
    #             sys.stdout.flush()

    #         line = await asyncio.to_thread(sys.stdin.readline)

    #         if not line:
    #             # EOF
    #             if not lines:
    #                 return None, True
    #             return "".join(lines), False

    #         text = line.rstrip("\r\n")

    #         if not lines and text.strip() in EXIT_COMMANDS:
    #             return text.strip(), True

    #         if COMMAND_END_MARKER in text:
    #             before_marker = text.split(COMMAND_END_MARKER, 1)[0]
    #             lines.append(before_marker)

    #             command = "".join(lines).strip()

    #             # Convert literal \n sequences into actual newlines
    #             if "\\n" in command and "\n" not in command:
    #                 command = command.replace("\\n", "\n")

    #             return command, True

    #         lines.append(line)

    # without end_marker
    async def read_command(self) -> Optional[str]:
        lines: list[str] = []
        terminal_mode = sys.stdin.isatty()

        while True:
            if terminal_mode:
                prompt = "sandbox> " if not lines else "... "
                print(prompt, end="")
                sys.stdout.flush()

            line = await asyncio.to_thread(sys.stdin.readline)

            if not line:
                # EOF
                if not lines:
                    return None
                return "".join(lines)

            text = line.rstrip("\r\n")

            if not lines and text.strip() in EXIT_COMMANDS:
                return text.strip()

            if COMMAND_END_MARKER in text:
                before_marker = text.split(COMMAND_END_MARKER, 1)[0]
                lines.append(before_marker)

                command = "".join(lines).strip()

                # Convert literal \n sequences into actual newlines
                if "\\n" in command and "\n" not in command:
                    command = command.replace("\\n", "\n")

                return command

            lines.append(line)

    async def run(self) -> None:
        """Run the persistent interactive Sandbox.

        Accept raw Python or LLM responses through stdin.
        Each command ends with <<<END_COMMAND>>>.
        """

         #  With ctrl+D as end of a command
        terminal_mode = sys.stdin.isatty()
        if terminal_mode:
            self.show_manual()

        while True:
            data = await self.read_command()

            if data is None or data.strip() in EXIT_COMMANDS:
                # TODO: some quit function
                print(
                    "Sandbox: Work complete,",
                    "all connections terminated,",
                    "all resources released. Bye!"
                )
                return

            if not data.strip():
                continue

            try:
                if data.strip() in ("manual", "manual()"):
                    self.show_manual()
                    continue
                if data.strip() in ("show_tools", "show_tools()"):
                    result = SandboxIntermediateResult(
                        success=True,
                        tool_name="show_tools",
                        sandbox_output=self.show_tools(),
                        status="success"
                    )
                # Process LLM responses containing code fences
                elif "```" in data:
                    result = await self.execute_llm_response(data)
                # Execute raw Python code
                else:
                    result = await self.run_python(data)

            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                result = SandboxIntermediateResult(
                    success=False,
                    tool_name="sandbox",
                    sandbox_output=f"{type(e).__name__}: {e}",
                    status="error"
                )
            print(result.model_dump_json())
            sys.stdout.flush()

        # terminal_mode = sys.stdin.isatty()

        # if terminal_mode:
        #     self.show_manual()

        # while True:
        #     data, complete = await self.read_command()

        #     # Clean EOF.
        #     if data is None:
        #         # TODO: some quit function
        #         return

        #     # EOF before the expected delimiter.
        #     if not complete:
        #         # TODO: some quit function
        #         result = SandboxIntermediateResult(
        #             success=False,
        #             tool_name="sandbox",
        #             status="incomplete_input",
        #             sandbox_output=(
        #                 "Input ended before "
        #                 f"{COMMAND_END_MARKER}. "
        #                 "The partial command was not executed."
        #             )
        #         )
        #         print(result.model_dump_json())
        #         sys.stdout.flush()
        #         return

        #     if data.strip() in EXIT_COMMANDS:
        #         # TODO: some quit function
        #         return

        #     if not data.strip() and terminal_mode:
        #         continue

        #     try:
        #         if terminal_mode and data.strip() in (
        #             "manual", "manual()"
        #         ):
        #             result = SandboxIntermediateResult(
        #                 success=True,
        #                 tool_name="manual",
        #                 sandbox_output=self.get_manual()
        #             )

        #         elif terminal_mode and data.strip() in (
        #             "show_tools", "show_tools()"
        #         ):
        #             result = SandboxIntermediateResult(
        #                 success=True,
        #                 tool_name="show_tools",
        #                 sandbox_output=self.show_tools()
        #             )

        #         elif terminal_mode and "```" not in data:
        #             # Terminal REPL supports raw Python.
        #             result = await self.run_python(data)

        #         else:
        #             # Complete LLM response.
        #             # Used by the Agent and for fenced terminal input.
        #             result = await self.execute_llm_response(data)

        #     except (KeyboardInterrupt, SystemExit):
        #         raise
        #     except Exception as exc:
        #         result = SandboxIntermediateResult(
        #             success=False,
        #             tool_name="sandbox",
        #             status="error",
        #             sandbox_output=(
        #                 f"{type(exc).__name__}: {exc}"
        #             )
        #         )

        #     print(result.model_dump_json())
        #     sys.stdout.flush()


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
        help="Path to the sandbox template JSON file."
    ),
    mcp_stdio: Optional[str] = typer.Option(
        None,
        "--mcp-stdio",
        help="Command to run MCP tools via stdio (e.g., 'python mcp_tool.py')."
    ),
    mcp_server: Optional[str] = typer.Option(
        None,
        "--mcp-server",
        help="MCP server URL."
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
