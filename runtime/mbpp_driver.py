"""Trusted Python bootstrap executed inside Pyodide, never on host Python."""

import builtins
import importlib
import io
import json
import os
import traceback
import sys
from typing import Any

import js


def _run(request_json: str) -> bool:
    request = json.loads(request_json)
    config = request["config"]
    emit = js.__mbpp_emit

    class Output(io.TextIOBase):
        def write(self, text: str) -> int:
            emit(text)
            return len(text)

    sys.stdout = sys.stderr = Output()
    sys.stdin = io.StringIO("")
    patterns = config["authorized_imports"]
    directories = [os.path.abspath(path)
                   for path in config["allowed_directories"]]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    original_import = builtins.__import__
    original_open = builtins.open
    execute = builtins.exec
    compile_code = builtins.compile

    def allowed_import(name: str) -> bool:
        return any(name == item or
                   (item.endswith(".*") and
                    (name == item[:-2] or name.startswith(item[:-1])))
                   for item in patterns)

    def checked_import(
        name: str, globals: Any = None, locals: Any = None,
        fromlist: tuple[str, ...] = (), level: int = 0,
    ) -> Any:
        if level or not allowed_import(name):
            raise ImportError("Import denied: " + name)
        return original_import(name, globals, locals, fromlist, level)

    def checked_open(path: str, *args: Any, **kwargs: Any) -> Any:
        resolved = os.path.realpath(path)
        if not any(os.path.commonpath([resolved, directory]) == directory
                   for directory in directories):
            raise PermissionError("File access denied: " + str(path))
        return original_open(path, *args, **kwargs)

    # Load allowed stdlib modules before restricting the candidate's imports.
    try:
        for pattern in patterns:
            importlib.import_module(pattern.removesuffix(".*"))
    except (ImportError, ModuleNotFoundError):
        traceback.print_exc()
        return False

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "import" and not allowed_import(args[0]):
            raise ImportError("Import denied: " + args[0])
        if event in ("open", "os.listdir", "os.scandir", "os.chdir",
                     "os.mkdir", "os.remove", "os.rmdir", "os.chmod",
                     "os.rename", "os.link", "os.symlink"):
            paths = args[:2] if event in (
                "os.rename", "os.link", "os.symlink") else args[:1]
            for path in paths:
                if not isinstance(path, (str, bytes, os.PathLike)):
                    raise PermissionError("File descriptors are denied.")
                resolved = os.path.realpath(os.fsdecode(path))
                if not any(os.path.commonpath([resolved, directory]) ==
                           directory for directory in directories):
                    raise PermissionError("File access denied: " + str(path))

    sys.addaudithook(audit)

    safe_builtins = dict(vars(builtins))
    for name in ("breakpoint", "eval", "exec", "compile", "input"):
        safe_builtins.pop(name, None)
    safe_builtins["__import__"] = checked_import
    safe_builtins["open"] = checked_open
    namespace = {"__name__": "__main__", "__builtins__": safe_builtins}
    try:
        execute(compile_code(request["code"], "<solution>", "exec"), namespace)
        for index, test in enumerate(request["test_list"], 1):
            execute(compile_code(test, "<test %d>" % index, "exec"), namespace)
    except BaseException:
        traceback.print_exc()
        return False
    return True


_run(globals()["_request_json"])
