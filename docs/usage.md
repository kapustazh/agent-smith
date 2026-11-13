# Run and test the MCP servers

Run these commands from the repository root. Install `uv` and Node.js/npm first.

```bash
make install
```

This installs Python dependencies, Deno and the pinned Pyodide assets.
Standalone MCP servers support macOS and Linux.

## Test MBPP

```bash
uv run fastmcp call mcp_tools_mbpp.py run_tests \
  code='def distance(a, b):
    difference = a - b
    return math.fabs(difference)' \
  test_imports='["import math"]' \
  test_list='["assert distance(2, 5) == 3", "assert distance(4, 4) == 0"]'
```

Expected tool output, wrapped by the CLI's `result` field,

```json
{"success": true, "output": "All 2 tests passed."}
```

Change the first expected value from `3` to `99` to check failure handling.
The result should contain `success=false`, `<test 1>` and `AssertionError`.

## Test SWE-bench

This runtime API requires PR #19. Start Docker Desktop, then check the engine
and build the small fixture containing a deliberately broken `add` function.

```bash
docker info
docker build -t agent-smith-swe-fixture \
  -f tests/docker/Dockerfile tests/docker
```

Keep all calls in one MCP connection so they share the same task session.

```bash
uv run python - <<'PY'
import asyncio
from pathlib import Path
from fastmcp import Client

async def main():
    async with Client(Path("mcp_tools_swebench.py")) as client:
        async def call(name, **arguments):
            result = await client.call_tool(name, arguments)
            text = result.content[0].text
            print(f"\n{name}\n{text}")
            return text

        await call(
            "start_task",
            docker_image="agent-smith-swe-fixture",
            eval_script='python -c "from calc import add; assert add(2, 3) == 5"',
        )
        await call("read_file", filepath="/testbed/calc.py")
        await call("run_tests")
        await call(
            "edit_file",
            filepath="/testbed/calc.py",
            old_str="return a - b",
            new_str="return a + b",
        )
        await call("run_tests")
        await call("get_patch")

asyncio.run(main())
PY
```

- The first test run has `exit_code=1` and `AssertionError`.
- The second has `exit_code=0`.
- The patch replaces `a - b` with `a + b`.

For a real task, pull its `docker_image` and pass that image and its `eval_script`
to `start_task`. Only the task's repository files are copied into the workspace.

## Connect another MCP client

Replace `/absolute/path/to/repo` with your checkout. Cursor and Claude Desktop
accept this server map. Claude Code also needs `"type": "stdio"` in the entry.

```json
{
  "mcpServers": {
    "agent-smith-mbpp": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/repo",
               "python", "mcp_tools_mbpp.py"]
    }
  }
}
```

For SWE-bench, use `mcp_tools_swebench.py`. Set the server environment variable
`TESTBED_PATH` to an absolute checkout path for host repository operations.
Call `start_task` after connecting to use Docker.

## Connect the project sandbox

The sandbox requires Linux x86-64 with Landlock and seccomp. Pass the same
configuration file to the sandbox and MCP server, then restart after changes.

```bash
uv run sandbox src/sandbox_template.json \
  --mcp-stdio "uv run python mcp_tools_mbpp.py --sandbox-config src/sandbox_template.json"
```

Both servers also accept `--transport http`, `--host` and `--port`.
The default HTTP endpoint is `http://127.0.0.1:8000/mcp`.

## Checks and common errors

```bash
make lint
make test
make test-docker
```

| Error | Action |
| --- | --- |
| Docker socket missing | Start Docker Desktop and wait until `docker info` succeeds. |
| Pyodide assets missing | Run `make install` from the repository root. |
| No task started | Call `start_task` before SWE-bench `run_tests`. |
| Docker image missing | Pull or build the image before `start_task`. |
