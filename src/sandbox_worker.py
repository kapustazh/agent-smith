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
