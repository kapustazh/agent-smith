# Connect the MCP servers

Run `make install` from the checkout root. The project sandbox requires Linux x86-64.

## 1. Start SWE-bench

On the machine running MCP, start Docker Desktop or Docker Engine.
Replace `IMAGE_FROM_TASK` with `docker_image` from the task JSON.

```bash
docker info
docker pull IMAGE_FROM_TASK
uv run sandbox src/sandbox_template.json \
  --mcp-stdio "uv run python mcp_tools_swebench.py --sandbox-config src/sandbox_template.json"
```

The launcher reads the task JSON and forwards its image and script through the
sandbox before the first model turn. Replace both placeholders with task values.

```python
print(start_task(docker_image="IMAGE_FROM_TASK", eval_script="SCRIPT_FROM_TASK"))
```

Wait for successful setup. MCP creates the container. Keep the same connection
for the entire task. Then use `read_file`, `edit_file`, `run_command`, `run_tests`
and `get_patch`. Print tool results for model observations.

## 2. Connect MBPP

Use the same sandbox command with `mcp_tools_mbpp.py`. Pass the candidate source
and the task's `test_list` and `test_imports` to `run_tests`.

```python
print(run_tests(code="def add(a, b): return a + b",
                test_list=["assert add(2, 3) == 5"], test_imports=[]))
```

## 3. Read results and finish

Sandbox tool results currently arrive as `{"result": "..."}`. Extract `result`.
Parse MBPP test JSON and check `success`. For SWE-bench, inspect test logs,
`exit_code`, `timed_out` and `truncated`. Pass the complete patch to `final_answer`.

```python
patch = get_patch()
final_answer(patch["result"] if isinstance(patch, dict) else patch)
```

For MBPP, finish with `final_answer(solution_code)`. Save the solution before
closing the connection. Allow enough client wait time for Docker and test calls.
The agent's `SandboxClient.run_tool` currently defaults to two seconds.

[Runnable tests and expected output](usage.md)
