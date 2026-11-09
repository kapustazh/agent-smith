
import contextlib
import json
import sys
import traceback


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

    def __init__(self, stream: str):
        self.stream = stream

    def write(self, value: str) -> int:
        for pos in range(0, len(value), 1024):
            send({
                "type": "output",
                "stream": self.stream,
                "text": value[pos:pos + 1024],
            })
        return len(value)

    # def flush(self) -> None:
    #     pass


def execute(source: str, namespace: dict) -> None:
    """Execute Python code while preserving the worker namespace."""
    try:
        with (
            contextlib.redirect_stdout(OutputStream("stdout")),
            contextlib.redirect_stderr(OutputStream("stderr")),
        ):
            exec(
                compile(source, "<sandbox>", "exec"),
                namespace,
                namespace,
            )

    except FinalAnswerSignal as signal:
        send({
            "type": "result",
            "status": "final_answer",
            "answer": signal.answer,
        })

    except (KeyboardInterrupt, SystemExit) as exc:
        code = getattr(exc, "code", None)
        if not isinstance(code, (str, int, type(None))):
            code = str(code)

        send({
            "type": "control",
            "name": type(exc).__name__,
            "code": code,
        })

    except SyntaxError:
        send({
            "type": "result",
            "status": "syntax_error",
            "error": traceback.format_exc(),
        })

    except Exception:
        send({
            "type": "result",
            "status": "error",
            "error": traceback.format_exc(),
        })

    else:
        send({"type": "result", "status": "success"})


def main() -> None:
    namespace = {
        "__name__": "__sandbox__",
        "final_answer": final_answer,
    }

    for line in sys.stdin:
        try:
            message = json.loads(line)
            kind = message.get("type")

            if kind == "init":
                send({"type": "ready"})
            elif kind == "run":
                execute(message["code"], namespace)
            elif kind == "shutdown":
                break
            else:
                send({
                    "type": "result",
                    "status": "error",
                    "error": f"Unknown message: {kind}",
                })
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            send({
                "type": "result",
                "status": "error",
                "error": traceback.format_exc(),
            })


if __name__ == "__main__":
    main()
