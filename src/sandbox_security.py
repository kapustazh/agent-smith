"""Linux x86-64 sandbox restrictions. Requires working Landlock + seccomp.

Only standard-library modules are used. Missing kernel facilities fail closed.
"""

import builtins
import ctypes
import errno
import importlib
import os
import platform
import resource
from pathlib import Path


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int),
        ("reserved", ctypes.c_uint32)
    ]


class _BpfInsn(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32)
    ]


class _BpfProgram(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("instructions", ctypes.POINTER(_BpfInsn))
    ]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long
_libc.prctl.restype = ctypes.c_int
_libc.prctl.argtypes = [
    ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
    ctypes.c_ulong, ctypes.c_ulong
]


def _syscall(number: int, *arguments: object) -> int:
    result = _libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def _no_new_privileges() -> None:
    if _libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _landlock(directories: list[str]) -> None:
    # Syscall numbers 444, 445, 446 are shared by Linux x86-64 and arm64;
    # this implementation is explicitly restricted to x86-64.
    try:
        abi = _syscall(
            444,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(1)
        )
    except OSError as exc:
        raise RuntimeError(
            "Landlock is unavailable in this Linux environment; "
            "secure execution is disabled (fail closed)."
        ) from exc

    # Handle ALL supported file rights (deny by default), but grant only
    # ordinary file operations within allowed directories.
    handled = (1 << 13) - 1  # v1: bits 0..12
    if abi >= 2:
        handled |= 1 << 13   # REFER
    if abi >= 3:
        handled |= 1 << 14   # TRUNCATE
    if abi >= 5:
        handled |= 1 << 15   # IOCTL_DEV
    if abi >= 9:
        handled |= 1 << 16   # RESOLVE_UNIX

    # Read files and directories, update/remove files, create regular files,
    # directories, and symlinks. No execution, device creation, sockets,
    # special files, or device ioctl access.
    granted = sum(1 << i for i in (1, 2, 3, 4, 5, 7, 8, 12))
    if abi >= 2:
        granted |= 1 << 13
    if abi >= 3:
        granted |= 1 << 14

    attr = _Ruleset(handled)
    ruleset_fd = _syscall(
        444, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint(0)
    )
    rules_added = 0
    try:
        for directory in directories:
            # A configured path must already be a directory inside the worker.
            if not os.path.isdir(directory):
                continue
            fd = os.open(directory, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathRule(granted, fd, 0)
                _syscall(445, ctypes.c_int(ruleset_fd), ctypes.c_int(1),
                         ctypes.byref(rule), ctypes.c_uint(0))
                rules_added += 1
            finally:
                os.close(fd)

        if not rules_added:
            raise RuntimeError(
                "None of the configured allowed directories exists"
            )

        _no_new_privileges()
        _syscall(446, ctypes.c_int(ruleset_fd), ctypes.c_uint(0))
    finally:
        os.close(ruleset_fd)


def _seccomp() -> None:
    # Linux x86-64 syscall numbers. Deny networking, process creation,
    # cross-process access, and common sandbox-bypass primitives.
    denied = {
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        288, 299, 307,   # socket, socketpair, network operations
        56, 57, 58, 59, 322, 435,  # clone, fork, exec, clone3
        62, 200, 234, 424,   # kill, tkill, tgkill, pidfd_send_signal
        101, 310, 311, 434,  # ptrace, process_vm_*, pidfd_open
        155, 161, 165, 166, 272, 308,  # chroot, mount, namespaces
        303, 304, 298, 321, 323,       # file handles, perf, bpf, userfaultfd
        425, 426, 427                  # io_uring
    }

    LOAD_ABS = 0x20
    JUMP_EQ = 0x15
    JUMP_SET = 0x45
    RETURN = 0x06
    KILL = 0x80000000
    ERROR = 0x00050000 | errno.EPERM
    ALLOW = 0x7FFF0000
    X86_64 = 0xC000003E

    instructions = [
        _BpfInsn(LOAD_ABS, 0, 0, 4),  # seccomp_data.arch
        _BpfInsn(JUMP_EQ, 1, 0, X86_64),
        _BpfInsn(RETURN, 0, 0, KILL),
        _BpfInsn(LOAD_ABS, 0, 0, 0),  # seccomp_data.nr
        # Reject x32 ABI syscalls, which otherwise bypass an x86-64 blocklist.
        _BpfInsn(JUMP_SET, 0, 1, 0x40000000),
        _BpfInsn(RETURN, 0, 0, KILL)
    ]
    for syscall_number in sorted(denied):
        instructions.extend([
            _BpfInsn(JUMP_EQ, 0, 1, syscall_number),
            _BpfInsn(RETURN, 0, 0, ERROR)
        ])
    instructions.append(_BpfInsn(RETURN, 0, 0, ALLOW))

    buffer = (_BpfInsn * len(instructions))(*instructions)
    program = _BpfProgram(len(instructions), buffer)
    _no_new_privileges()
    if _libc.prctl(22, 2, ctypes.addressof(program), 0, 0) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def permitted_import(name: str, rules: list[str]) -> bool:
    return any(
        name == rule or
        (rule.endswith(".*") and name.startswith(rule[:-2] + "."))
        for rule in rules
    )


def preload_modules(rules: list[str]) -> None:
    """Import authorized modules before the filesystem is sealed.

    Wildcard entries authorize submodules, but cannot themselves be imported.
    Submodules needed after sealing must be preloaded here.
    """
    for rule in rules:
        if not rule.endswith(".*"):
            importlib.import_module(rule)
    if permitted_import("collections.abc", rules):
        importlib.import_module("collections.abc")


def restricted_builtins(rules: list[str], directories: list[str]) -> dict:
    original_import = builtins.__import__
    original_open = builtins.open
    roots = [Path(p).resolve() for p in directories if Path(p).is_dir()]

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or not permitted_import(name, rules):
            raise ImportError(f"Import not authorized: {name}")
        return original_import(name, globals, locals, fromlist, level)

    def safe_open(file, mode="r", *args, **kwargs):
        if isinstance(file, int):
            raise PermissionError(
                "Opening arbitrary file descriptors is prohibited"
            )
        candidate = Path(file).resolve()
        if not any(
            candidate == root or root in candidate.parents for root in roots
        ):
            raise PermissionError(f"Filesystem access denied: {candidate}")
        return original_open(candidate, mode, *args, **kwargs)

    names = {
        "__build_class__", "abs", "all", "any", "bool", "bytes", "bytearray",
        "callable", "chr", "classmethod", "dict", "divmod", "enumerate",
        "Exception", "FileNotFoundError", "filter", "float", "getattr",
        "hasattr", "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "object", "ord", "PermissionError",
        "pow", "print", "property", "range", "repr", "reversed", "round",
        "set", "slice", "sorted", "staticmethod", "str", "sum", "super",
        "tuple", "type", "ValueError", "TypeError", "RuntimeError",
        "ImportError", "MemoryError", "ZeroDivisionError", "IndexError",
        "KeyError", "NotImplementedError", "zip", "True", "False", "None"
    }
    available = {name: getattr(builtins, name) for name in names}
    available["__import__"] = safe_import
    available["open"] = safe_open
    return available


def enforce_security(config: dict) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "This security implementation requires Linux x86-64"
        )
    directories = config["allowed_directories"]
    preload_modules(config["authorized_imports"])
    os.umask(0o077)
    # Hard limits cannot be raised by executed code.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    limit = config["max_memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    _landlock(directories)  # Throws if Landlock is unavailable: fail closed.
    _seccomp()             # Throws if seccomp is unavailable: fail closed.
