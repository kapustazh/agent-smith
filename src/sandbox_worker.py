"""Worker process for sandbox"""

import contextlib
import json
import os
import sys
import traceback
import importlib.util
from pathlib import Path


_security_file = Path(__file__).resolve().with_name("sandbox_security.py")
_spec = importlib.util.spec_from_file_location(
    "_sandbox_security_impl", _security_file
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load security module: {_security_file}")
_security = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_security)
_rpc_counter = 0
enforce_security = _security.enforce_security
restricted_builtins = _security.restricted_builtins


class FinalAnswerSignal(BaseException):
    def __init__(self, answer: str):
        self.answer = answer


def final_answer(answer: str) -> None:
    if not isinstance(answer, str):
        raise TypeError("final_answer() requires a string")
    raise FinalAnswerSignal(answer)


def send(message: dict) -> None:
    """Send a JSON message to the parent process."""
    sys.__stdout__.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def mcp_rpc(operation: str, **arguments):
    """Synchronously request an MCP operation from the parent."""
    global _rpc_counter
    _rpc_counter += 1
    request_id = _rpc_counter

    send({
        "type": "mcp_request",
        "id": request_id,
        "operation": operation,
        **arguments
    })

    # Parent sends one response over the existing stdin pipe.
    raw = sys.stdin.readline()

    if not raw:
        raise RuntimeError("MCP parent disconnected")

    response = json.loads(raw)

    if (
        response.get("type") != "mcp_response"
        or response.get("id") != request_id
    ):
        raise RuntimeError(
            "MCP communication error: unexpected response"
        )

    if not response.get("ok"):
        raise RuntimeError(
            f"MCP request failed: {response.get('error')}"
        )

    return response.get("value")


def create_mcp_namespace(
    aliases: dict[str, str]
) -> dict:
    """Build Python-callable MCP wrappers inside the worker."""

    def mcp_call_tool(name: str, **kwargs):
        return mcp_rpc(
            "call_tool", name=name, arguments=kwargs
        )

    def mcp_read_resource(uri: str):
        return mcp_rpc("read_resource", uri=uri)

    def mcp_get_prompt(
        name: str,
        arguments: dict[str, str] | None = None
    ):
        return mcp_rpc(
            "get_prompt",
            name=name,
            arguments=arguments or {}
        )

    def mcp_list_resources():
        return mcp_rpc("list_resources")

    def mcp_list_resource_templates():
        return mcp_rpc("list_resource_templates")

    def mcp_list_prompts():
        return mcp_rpc("list_prompts")

    namespace = {
        "mcp_call_tool": mcp_call_tool,
        "mcp_read_resource": mcp_read_resource,
        "mcp_get_prompt": mcp_get_prompt,
        "mcp_list_resources": mcp_list_resources,
        "mcp_list_resource_templates": mcp_list_resource_templates,
        "mcp_list_prompts": mcp_list_prompts
    }

    def create_tool_wrapper(original_name: str):
        def wrapper(**kwargs):
            return mcp_call_tool(original_name, **kwargs)
        return wrapper

    for python_name, original_name in aliases.items():
        namespace[python_name] = create_tool_wrapper(
            original_name
        )

    return namespace


class OutputStream:
    """Forward sandbox output to the parent as JSON messages."""

    def __init__(self, name: str):
        self.name = name

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("Expected text")
        for index in range(0, len(text), 1024):
            send({"type": "output", "stream": self.name,
                  "text": text[index:index + 1024]})
        return len(text)

    # def flush(self) -> None:
    #     pass


def execute(code: str, namespace: dict) -> None:
    try:
        with (
            contextlib.redirect_stdout(OutputStream("stdout")),
            contextlib.redirect_stderr(OutputStream("stderr"))
        ):
            exec(compile(code, "<sandbox>", "exec"), namespace, namespace)

    except FinalAnswerSignal as exc:
        send(
            {"type": "result", "status": "final_answer", "answer": exc.answer}
        )
    except (KeyboardInterrupt, SystemExit) as exc:
        code = getattr(exc, "code", None)
        if not isinstance(code, (str, int, type(None))):
            code = str(code)
        send({"type": "control", "name": type(exc).__name__, "code": code})
    except MemoryError:
        send({"type": "result", "status": "memory_limit",
              "error": "Sandbox memory limit exceeded. Worker terminated."})
        os._exit(137)
    except SyntaxError:
        send({"type": "result", "status": "syntax_error",
              "error": traceback.format_exc()})
    except Exception:
        send({"type": "result", "status": "error",
              "error": traceback.format_exc()})
    else:
        send({"type": "result", "status": "success"})


def main() -> None:
    initialized = False
    namespace: dict = {}

    for raw_line in sys.stdin:
        try:
            message = json.loads(raw_line)
            kind = message.get("type")

            if kind == "init":
                if initialized:
                    raise RuntimeError("Worker was already initialized")
                config = message["config"]
                # Create safe builtins before filesystem access is restricted.
                safe = restricted_builtins(
                    config["authorized_imports"], config["allowed_directories"]
                )
                # Fail closed: never emit 'ready' unless ALL restrictions work.
                enforce_security(config)
                namespace = {
                    "__name__": "__sandbox__",
                    "__builtins__": safe,
                    "final_answer": final_answer
                }
                namespace.update(
                    create_mcp_namespace(message.get("tools", {}))
                )
                initialized = True
                send({"type": "ready"})

            elif kind == "run":
                if not initialized:
                    raise RuntimeError(
                        "Worker must be initialized before running code"
                    )
                execute(message["code"], namespace)

            elif kind == "shutdown":
                return

            else:
                raise ValueError(f"Unknown worker command: {kind}")

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            send({"type": "result", "status": "error",
                  "error": traceback.format_exc()})
            if not initialized:
                return  # Startup failed; sandbox remains locked.


if __name__ == "__main__":
    main()
