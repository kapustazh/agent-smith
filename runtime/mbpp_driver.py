"""Restricted MBPP Python; loaded only inside the Deno/Pyodide process."""

import ast
import builtins
from functools import partial
import importlib
import io
import json
import os
import string
import sys
import traceback
import types
import _string
from typing import Any


class Output(io.TextIOBase):
    def __init__(self, emit: Any) -> None:
        self.emit = emit

    def write(self, text: str) -> int:
        self.emit(text)
        return len(text)


class CandidateGate(ast.NodeTransformer):
    blocked = frozenset({
        "gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code",
        "tb_frame", "tb_next", "f_back", "f_globals", "f_locals", "f_builtins",
        "f_code", "format_field",
    })

    def visit(self, node: ast.AST) -> Any:
        for field in ("id", "arg", "name", "asname", "rest"):
            name = getattr(node, field, None)
            if isinstance(name, str) and (
                name.startswith(("__", "_sandbox_")) or name in self.blocked
            ):
                if not (isinstance(node, ast.FunctionDef) and field == "name"
                        and name in {
                            "__init__", "__repr__", "__str__", "__len__",
                            "__bool__", "__iter__", "__next__", "__getitem__",
                            "__setitem__", "__contains__", "__eq__", "__ne__",
                            "__lt__", "__le__", "__gt__", "__ge__", "__hash__",
                            "__add__", "__sub__", "__mul__", "__truediv__",
                            "__floordiv__", "__mod__", "__neg__", "__abs__",
                }):
                    raise PermissionError("Reserved name: " + name)
        if isinstance(node, ast.ClassDef) and node.keywords:
            raise PermissionError("Custom metaclasses are denied.")
        if isinstance(node, ast.MatchClass) and any(
            name.startswith("_") or name in self.blocked
            for name in node.kwd_attrs
        ):
            raise PermissionError("Private class patterns are denied.")
        return super().visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("_") or node.attr in self.blocked:
            raise PermissionError("Attribute access denied: " + node.attr)
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Call(
                func=ast.Name(id="_sandbox_attribute", ctx=ast.Load()),
                args=[node.value, ast.Constant(node.attr)], keywords=[],
            ), node)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        if node.level or any(part.startswith("_") for part in
                             (node.module or "").split(".")):
            if node.module != "__future__":
                raise ImportError("Import denied: " + str(node.module))
        for alias in node.names:
            if alias.name.startswith("_"):
                raise ImportError("Import denied: " + alias.name)
        return self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            if any(part.startswith("_") for part in alias.name.split(".")):
                raise ImportError("Import denied: " + alias.name)
        return self.generic_visit(node)


# Plain functions prevent deepcopy from copying the supervisor through self.
def candidate_import(
    name: str, globals: Any = None, locals: Any = None,
    fromlist: tuple[str, ...] = (), level: int = 0,
) -> Any:
    if name == "_strptime" and _runtime.allowed("datetime"):
        return _runtime.strptime
    if level or name not in _runtime.modules:
        raise ImportError("Import denied: " + name)
    module = _runtime.modules[name]
    for member in fromlist or ():
        if member != "*" and (member.startswith("_") or
                              not hasattr(module, member)):
            raise ImportError("Import denied: " + name + "." + member)
    return module if fromlist else _runtime.modules[name.split(".")[0]]


def candidate_attribute(value: Any, name: str) -> Any:
    if name.startswith("_") or name in CandidateGate.blocked:
        raise PermissionError("Attribute access denied: " + name)
    if name in ("format", "format_map"):
        if isinstance(value, str):
            return partial(candidate_format, name, value)
        if isinstance(value, type) and issubclass(value, str):
            return partial(candidate_format, name)
    return getattr(value, name)


def candidate_format(
    method: str, text: str, *args: Any, **kwargs: Any,
) -> str:
    mapping = args[0] if method == "format_map" else kwargs
    return CandidateFormatter().vformat(
        text, () if method == "format_map" else args, mapping,
    )


def candidate_type(value: Any) -> Any:
    result = type(value)
    return candidate_type if result is type else result


class CandidateFormatter(string.Formatter):
    def get_field(
        self, field_name: str, args: Any, kwargs: Any,
    ) -> tuple[Any, Any]:
        first, rest = _string.formatter_field_name_split(field_name)
        value = self.get_value(first, args, kwargs)
        for attribute, key in rest:
            value = (candidate_attribute(value, key) if attribute else
                     value[key])
        return value, first


class CandidateRuntime:
    def __init__(self, config_json: str, emit: Any) -> None:
        config = json.loads(config_json)
        self.patterns = tuple(config["authorized_imports"])
        self.directories = tuple(os.path.abspath(path)
                                 for path in config["allowed_directories"])
        self.sources: dict[str, str] = {}
        for directory in self.directories:
            os.makedirs(directory, exist_ok=True)
        sys.stdout = sys.stderr = Output(emit)
        sys.stdin = io.StringIO("")
        self.modules = self.load_modules()
        # CPython datetime imports this helper via the caller's builtins.
        self.strptime = importlib.import_module("_strptime")
        datetime = importlib.import_module("datetime")
        datetime.datetime.strptime("2000", "%Y")
        self.namespace = {"__name__": "mbpp_candidate",
                          "__builtins__": self.builtins(),
                          "_sandbox_attribute": candidate_attribute}
        sys.addaudithook(self.audit)

    def allowed(self, name: str) -> bool:
        return any(name == item or (item.endswith(".*") and
                   (name == item[:-2] or name.startswith(item[:-1])))
                   for item in self.patterns)

    def load_modules(self) -> dict[str, types.ModuleType]:
        exports = {
            "math": None, "cmath": None, "itertools": None, "bisect": None,
            "heapq": "heapify heappop heappush heappushpop heapreplace merge "
            "nlargest nsmallest",
            "collections": "ChainMap Counter OrderedDict UserDict UserList "
            "UserString defaultdict deque namedtuple",
            "collections.abc": None,
            "re": "__all__", "json": "__all__", "datetime": "__all__",
            "array": "array ArrayType typecodes",
            "copy": "copy deepcopy Error error",
            "random": "__all__", "operator": "__all__",
            "functools": "cache cached_property cmp_to_key lru_cache partial "
            "partialmethod reduce total_ordering",
            "string": "ascii_letters ascii_lowercase ascii_uppercase capwords "
            "digits hexdigits octdigits printable punctuation Template "
            "whitespace",
            "typing": "Any AnyStr Annotated BinaryIO Callable ClassVar "
            "Concatenate "
            "Final Generic IO Literal NewType NoReturn Optional ParamSpec "
            "Protocol SupportsAbs SupportsBytes SupportsComplex "
            "SupportsFloat SupportsIndex SupportsInt SupportsRound Text "
            "TextIO Tuple Type TypeVar Union cast "
            "runtime_checkable TYPE_CHECKING List Dict Set FrozenSet "
            "DefaultDict Deque Counter ChainMap OrderedDict Sequence "
            "MutableSequence Mapping MutableMapping Iterable Iterator "
            "Generator Collection Container Hashable Reversible Sized",
            "__future__": "annotations",
        }
        modules: dict[str, types.ModuleType] = {}
        for name, members in exports.items():
            if name != "__future__" and not self.allowed(name):
                continue
            original = importlib.import_module(name)
            module = types.ModuleType(name)
            names = (getattr(original, "__all__") if members == "__all__" else
                     members.split() if members is not None else
                     [key for key in vars(original)
                      if not key.startswith("_")])
            for key in names:
                if key.startswith("_") or key in (
                        "attrgetter", "methodcaller"):
                    continue
                value = getattr(original, key)
                if not isinstance(value, types.ModuleType):
                    setattr(module, key, value)
            setattr(module, "__all__", tuple(
                key for key in vars(module) if not key.startswith("_")))
            modules[name] = module
        for name, module in tuple(modules.items()):
            if "." in name and name.rpartition(".")[0] in modules:
                parent, _, child = name.rpartition(".")
                setattr(modules[parent], child, module)
        return modules

    def builtins(self) -> dict[str, Any]:
        safe = {name: getattr(builtins, name) for name in (
            "abs all any ascii bin bool bytearray bytes callable chr "
            "classmethod complex dict divmod enumerate filter float format "
            "frozenset hash hex id int isinstance issubclass iter len list "
            "map "
            "max min next object oct open ord pow print property range repr "
            "reversed round set slice sorted staticmethod str sum super "
            "tuple zip"
        ).split()}
        safe.update({name: value for name, value in vars(builtins).items()
                     if isinstance(value, type) and
                     issubclass(value, BaseException)})
        safe.update({"__import__": candidate_import, "type": candidate_type,
                     "__build_class__": builtins.__build_class__})
        return safe

    def audit(self, event: str, args: tuple[Any, ...]) -> None:
        if event not in {
            "open", "os.listdir", "os.scandir", "os.chdir", "os.mkdir",
            "os.remove", "os.rmdir", "os.chmod", "os.rename", "os.link",
            "os.symlink",
        }:
            return
        paths = args[:2] if event in (
            "os.rename", "os.link", "os.symlink") else args[:1]
        for path in paths:
            if not isinstance(path, (str, bytes, os.PathLike)):
                raise PermissionError("File descriptors are denied.")
            resolved = os.path.realpath(os.fsdecode(path))
            if not any(os.path.commonpath([resolved, directory]) == directory
                       for directory in self.directories):
                raise PermissionError("File access denied: " + str(path))

    def execute(self, source: str, filename: str) -> bool:
        self.sources[filename] = source
        try:
            tree = CandidateGate().visit(ast.parse(source, filename))
            builtins.exec(builtins.compile(
                ast.fix_missing_locations(tree), filename, "exec",
            ), self.namespace)
        except BaseException as error:
            self.report_error(error, filename)
            return False
        return True

    def report_error(self, error: BaseException, filename: str) -> None:
        print("Error while executing %s:" % filename, file=sys.stderr)
        report = traceback.TracebackException.from_exception(error)
        pending = [report]
        while pending:
            entry = pending.pop()
            frames = []
            for frame in entry.stack:
                if frame.filename == "<exec>":
                    continue
                lines = self.sources.get(frame.filename, "").splitlines()
                line = (lines[frame.lineno - 1] if frame.lineno and
                        frame.lineno <= len(lines) else frame.line)
                frames.append(traceback.FrameSummary(
                    frame.filename, frame.lineno, frame.name,
                    line=line, lookup_line=False,
                ))
            entry.stack = traceback.StackSummary.from_list(frames)
            pending.extend(related for related in (
                entry.__cause__, entry.__context__,
            ) if related is not None)
        print("".join(report.format()), end="", file=sys.stderr)


_runtime = CandidateRuntime(globals()["_config_json"], globals()["_emit"])
_runtime
