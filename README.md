*This project has been created as part of the 42 curriculum by mnestere, obachuri, iiunusov.*

---

# Agent Smith
Building an agentic framework capable of autonomously solving coding challenges (create and fix code on Python).

Autonomous reasoning, code generation, and execution.

## Description

### System architecture

### Agent loop explanation

### Sandbox design

### Tool implementation details

### Benchmark results and analysis

#### MBPP (Mostly Basic Python Problems) benchmark

#### SWE-bench (Software Engineering Benchmark) 

## Instructions

###  Requirement

Install uv, Deno and Node.js/npm for MBPP. Docker is required only for SWE-bench.

Docker install - https://docs.docker.com/engine/install/

Install uv: 
```
pip install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
# check 
uv --version
```

Install deno:
```
curl -fsSL https://deno.land/install.sh | sh
```

### MBPP MCP server

This server checks a Python solution against the assertions passed to
`run_tests(code, test_list)`. It runs the solution with Python 3.10.2 through
Pyodide, a Python interpreter compiled to WebAssembly, inside a Deno process.

Run these commands from the repository root:

```bash
# Install the MCP server's Python packages using uv.lock.
uv sync --locked

# Download the Pyodide interpreter and standard-library files using package-lock.json.
npm ci --ignore-scripts
```

The npm command stores Pyodide in `node_modules/pyodide/`. Each test call starts a
fresh, permission-restricted Deno process that loads these local files and runs
the supplied code and assertions. MBPP does not use Docker. Downloads happen
during installation; test calls need no network access, model API keys or
benchmark datasets.

```bash
# stdio: the MCP client launches this command
uv run python mcp_tools_mbpp.py --sandbox-config src/sandbox_template.json

# Streamable HTTP: connect to http://127.0.0.1:8000/mcp
uv run python mcp_tools_mbpp.py --transport http \
  --sandbox-config src/sandbox_template.json
```

The MCP API remains `run_tests(code: str, test_list: list[str]) -> str`.
Include any required `test_imports` in `code`. Results are JSON strings containing
`success` and `output`. The metadata resource `agent://mbpp/task` and the
`mbpp_solver` prompt are also available. `final_answer` belongs to the agent
sandbox.

Execution policy comes from `SandboxConfig`, loaded once at server startup.
The defaults are 30 seconds of candidate execution, 512 MiB for the whole Deno
process and 8192 UTF-8 output bytes. These values are not MCP tool arguments.
Imports follow `authorized_imports`; `allowed_directories` refers to a fresh,
in-memory Python filesystem with no host mounts. Deno denies network access,
subprocesses, host writes and native addons, and revokes bootstrap read/environment
permissions before running the candidate. Cancellation and timeouts kill and reap
the runtime process. Failed assertions, exceptions and early exits report failure.

The runtime supports Linux and macOS. Process RSS is sampled every 20 ms, including
startup, and execution terminates when the memory limit is exceeded. This is a
watchdog, not a kernel-enforced hard memory ceiling; short allocation spikes may
exceed the limit between samples. V8 heap and WebAssembly memory have additional
caps. Runtime startup has a separate 30-second watchdog; the configured execution
time begins when the runtime is ready. Output is bounded even on failure or
truncation.

See the [sandbox contract](docs/mbpp-sandbox-contract.md) for configuration fields,
request/result models and the sandbox integration handoff. The interactive sandbox
CLI and agent loop still need their own integration with this executor.

```bash
uv run make test
make lint
```

Tests require Deno and the installed npm assets; missing runtime dependencies fail
explicitly. CI installs the assets and runs the real executor and MCP transport
checks without Docker.

## Resources
- MBPP (Mostly Basic Python Problems) benchmark:
	- https://mcpbr.org/mbpp
	- https://github.com/google-research/google-research/tree/master/mbpp
	- https://huggingface.co/datasets/google-research-datasets/mbpp
- SWE-bench https://www.swebench.com/original.html

- Model Context Protocol Python SDK documentation: https://modelcontextprotocol.github.io/python-sdk/
- MCP Server/Client architecture https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture 
- OpenAI-compatible chat completions API documentation: https://platform.openai.com/docs/api-reference/chat


### AI Usage

Tools Used: ChatGPT (GPT-5, GPT-6, Gemini 2.5 and 3.0)

AI assisted with architecture discussions, README structure, CI configuration,
MBPP MCP/runtime implementation and tests. Changes were reviewed and exercised
with automated checks; the moulinette was not modified.

## License

This project was created as part of the 42 curriculum.

It is intended for educational purposes.
