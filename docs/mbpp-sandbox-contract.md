# MBPP sandbox execution contract (version 1)

The agent sandbox and MBPP executor share `SandboxConfig` from
`src/config_model.py`. The MCP server delegates candidate execution to
`execute_mbpp(MBPPExecutionRequest)` in `src/mbpp_executor.py`. The result model is
`MBPPExecutionResult`. The executor owns runtime startup, restrictions, resource
monitoring and cleanup. MCP only validates tool inputs and returns the result.

## Configuration ownership

The sandbox launcher must start the MCP server with the same trusted JSON
configuration file it uses itself:

```bash
uv run python mcp_tools_mbpp.py --sandbox-config src/sandbox_template.json
```

The server loads the file once. Restart it to apply changes. Without a file, it
uses `SandboxConfig` defaults. The `run_tests` schema has only `code` and
`test_list`; the model cannot change policy through a tool call.

| Field | Meaning | Default |
| --- | --- | --- |
| `max_execution_time_seconds` | Candidate execution time after runtime ready | 30 |
| `max_memory_mb` | Whole Deno process RSS, including Python and startup | 512 MiB |
| `max_output_bytes` | Maximum UTF-8 output size, including diagnostics | 8192 |
| `authorized_imports` | Allowed Python modules; `name.*` permits submodules | Shared model allowlist |
| `allowed_directories` | Allowed directories in the isolated in-memory filesystem | `/testbed`, `/tmp/agent` |
| `max_output_chars` | Existing agent sandbox observation limit; separate from the MCP byte budget | Shared model default |

The three resource limits must be positive JSON integers. The template keeps its
existing `max_output_chars` setting. MBPP's byte budget does not reinterpret that
field or let an MCP action change the agent sandbox's own timeout.

## Request and response

The typed execution request serializes as:

```json
{
  "version": 1,
  "code": "def add(a, b): return a + b",
  "test_list": ["assert add(2, 3) == 5"],
  "config": {
    "authorized_imports": ["math", "math.*"],
    "allowed_directories": ["/tmp/agent"],
    "max_execution_time_seconds": 30,
    "max_memory_mb": 512,
    "max_output_chars": 5000,
    "max_output_bytes": 8192
  }
}
```

The result is always the same shape:

```json
{"success": true, "output": "All 1 tests passed."}
```

The public MCP returns that result as a JSON string. Empty code, empty test lists,
failed assertions, exceptions (including `SystemExit(0)`), incomplete execution,
startup errors, timeout and memory termination cannot report success. Cancelled
calls propagate cancellation after killing and reaping the runtime. Output is
truncated to the configured byte budget; diagnostics share that budget, so very
small limits may truncate the diagnostic itself.

For direct integration from trusted Python code:

```python
config = SandboxConfig.model_validate_json(config_path.read_text())
result = await execute_mbpp(MBPPExecutionRequest(
    code=code, test_list=test_list, config=config,
))
```

Do not copy limits into the MCP module or add per-call model-controlled limits.
The interactive sandbox's MCP client/manual/agent loop remains a separate
integration task; this change supplies the candidate executor it can reuse.

## Runtime boundary

Node/npm installs the pinned Pyodide 0.21.3 package and its integrity-locked assets.
This version supplies CPython 3.10.2, matching the subject's Python 3.10 target.
It is an older Pyodide release; upgrading it requires explicitly checking the
Python version and rerunning the isolation checks.

Only Deno runs candidate code. Each execution starts a fresh process with a clean
environment and temporary working directory. It receives the request JSON on
stdin plus an internal random completion token. It reports readiness and successful
completion with that token on stdout. Candidate stdout/stderr is streamed between
those frames; the supervisor removes frames and caps retained output. Runtime
errors use stderr. The token is framing, not a security or anti-cheating mechanism.

The runtime loads only local assets. Read permission is scoped to those assets
during bootstrap and revoked before the candidate runs. Network/module downloads,
host writes, subprocesses, FFI and system queries are denied. Python filesystem
operations use an in-memory filesystem; no host directories are mounted. Python
import guards, restricted builtins and an audit hook enforce the configured
allowlists. No benchmark solution data is loaded.

The Python supervisor kills the process group on timeout, memory termination or
cancellation. Linux/macOS RSS monitoring includes startup and samples every 20 ms;
it is not a hard OS memory allocation limit and may observe spikes late. V8 also
caps its heap and individual WebAssembly memories. If initialization cannot fit
the configured memory budget, the call fails. Initialization has a separate
30-second watchdog, so it does not consume the candidate execution time.

Docker remains a SWE-bench dependency. Neither MBPP CI nor this execution path
starts or requires Docker.
