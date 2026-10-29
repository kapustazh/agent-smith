import ast
from contextlib import contextmanager
import fnmatch
import os
from pathlib import Path
import stat
from threading import Lock
from typing import Iterable, Iterator, TextIO
from uuid import uuid4
from fastmcp.exceptions import ToolError
from src.config_model import SandboxConfig

SKIP_DIRECTORIES = {".git", ".venv", "venv", "node_modules", "__pycache__"}


class RepositoryTools:

    def __init__(self, root: Path | None, config: SandboxConfig) -> None:
        self._root = root
        self.config = config
        self._edit_lock = Lock()

    def _error(self, message: str) -> ToolError:
        return ToolError(message[:self.config.max_output_chars])

    @property
    def root(self) -> Path:
        if self._root is None:
            value = os.environ.get("TESTBED_PATH", "")
            if not value.strip():
                raise self._error("Set TESTBED_PATH to the repository root.")
            self._root = Path(value).resolve()
        root = self._root.resolve()
        if not root.is_dir():
            raise self._error("TESTBED_PATH must name an existing directory.")
        return root

    def _relative(self, value: str) -> Path:
        if not value.strip() or "\x00" in value:
            raise self._error("Supply a non-empty repository path.")
        path = Path(value)
        if path.is_absolute():
            try:
                path = path.relative_to(self.root)
            except ValueError:
                raise self._error(
                    "Path must be inside TESTBED_PATH.") from None
        if ".." in path.parts or ".git" in path.parts:
            raise self._error(
                "Parent traversal and .git access are forbidden.")
        return path

    @contextmanager
    def _directory(self, path: Path) -> Iterator[int]:
        descriptor = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in path.parts:
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = child
            yield descriptor
        except TimeoutError:
            raise
        except OSError:
            raise self._error(
                "Cannot access directory; it must exist inside TESTBED_PATH "
                "and must not be a symlink.",
            ) from None
        finally:
            os.close(descriptor)

    @contextmanager
    def _file(self, path: Path, edit: bool = False) -> Iterator[TextIO]:
        with self._directory(path.parent) as directory:
            descriptor = None
            try:
                descriptor = os.open(
                    path.name, (os.O_RDWR if edit else os.O_RDONLY) |
                    os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory,
                )
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise self._error(
                        "Only regular files without hard links are allowed.",
                    )
                handle = os.fdopen(
                    descriptor, "r+" if edit else "r",
                    encoding="utf-8", newline="",
                )
                descriptor = None
                with handle:
                    yield handle
            except TimeoutError:
                raise
            except OSError:
                raise self._error(
                    f"Cannot access {path}; check that it exists, is "
                    "readable/writable and is not a symlink.",
                ) from None
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def _render(self, lines: Iterable[str], empty: str) -> str:
        limit = self.config.max_output_chars
        output = ""
        for line in lines:
            text = ("\n" if output else "") + line
            if len(output) + len(text) > limit:
                notice = "\n[Output truncated; narrow the query.]"
                return ((output + text)[:max(0, limit - len(notice))] +
                        notice)[:limit]
            output += text
        return output or empty[:limit]

    def read_file(
        self, filepath: str, start_line: int = 1, end_line: int | None = None,
    ) -> str:
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise self._error("Require 1 <= start_line <= end_line.")
        with self._file(self._relative(filepath)) as handle:
            def lines() -> Iterator[str]:
                for number, line in enumerate(handle, 1):
                    if end_line is not None and number > end_line:
                        break
                    if number >= start_line:
                        yield f"{number}: " + line.rstrip("\r\n")
            try:
                return self._render(lines(), "No lines in requested range.")
            except UnicodeError:
                raise self._error("File is not UTF-8 text.") from None

    def edit_file(self, filepath: str, old_str: str, new_str: str) -> str:
        if not old_str:
            raise self._error("old_str must not be empty.")
        try:
            new_str.encode("utf-8")
        except UnicodeError:
            raise self._error("new_str must be valid UTF-8 text.") from None
        with self._edit_lock, self._file(
            self._relative(filepath), edit=True,
        ) as handle:
            try:
                source = handle.read()
            except UnicodeError:
                raise self._error("File is not UTF-8 text.") from None
            first = source.find(old_str)
            if first < 0 or source.find(old_str, first + 1) >= 0:
                raise self._error(
                    "Expected one exact match; found none or multiple. "
                    "File unchanged; include more context.",
                )
            path = self._relative(filepath)
            updated = source.replace(old_str, new_str, 1)
            message = "Replaced one occurrence."
            if path.suffix in (".py", ".pyi"):
                try:
                    ast.parse(updated)
                except (SyntaxError, ValueError) as error:
                    message += (" File saved, but Python syntax is invalid: "
                                f"{error}")
            with self._directory(path.parent) as directory:
                temporary = ".agent-edit-" + uuid4().hex
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600, dir_fd=directory,
                )
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8",
                                   newline="") as replacement:
                        os.fchmod(replacement.fileno(), stat.S_IMODE(
                            os.fstat(handle.fileno()).st_mode,
                        ))
                        replacement.write(updated)
                    os.replace(temporary, path.name, src_dir_fd=directory,
                               dst_dir_fd=directory)
                finally:
                    try:
                        os.unlink(temporary, dir_fd=directory)
                    except FileNotFoundError:
                        pass
        return message[:self.config.max_output_chars]

    def _files(self, directory: str, pattern: str) -> Iterator[Path]:
        path = self._relative(directory)
        if not pattern or Path(pattern).is_absolute() or ".." in (
            Path(pattern).parts
        ):
            raise self._error("Supply a relative glob pattern, e.g. '*.py'.")
        with self._directory(path) as descriptor:
            for current, directories, files, current_fd in os.fwalk(
                ".", follow_symlinks=False, dir_fd=descriptor,
            ):
                directories[:] = sorted(
                    name for name in directories
                    if name not in SKIP_DIRECTORIES
                )
                for name in sorted(files):
                    relative = Path(current) / name
                    if not (fnmatch.fnmatchcase(name, pattern) or
                            relative.match(pattern)):
                        continue
                    info = os.stat(
                        name, dir_fd=current_fd, follow_symlinks=False,
                    )
                    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        yield path / relative

    def list_files(self, directory: str = ".", pattern: str = "*") -> str:
        return self._render(
            (str(self.root / path)
             for path in self._files(directory, pattern)),
            "No matching files.",
        )
