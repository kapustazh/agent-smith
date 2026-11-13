# SWE-bench MCP server

The moulinette and the sandbox launch the server with no arguments:

```bash
uv run sandbox --mcp-stdio "python mcp_tools_swebench.py"
```

`TESTBED_PATH` is the host repository root for that process. Repository tools,
`run_command`, and `get_patch` use it when no container is running. `run_command`
treats the default workdir `/testbed` as that checkout. `run_tests` needs an
eval script, so it reports that no task has been started.

Read `docker_image` and `eval_script` from the task JSON, then call
`start_task(docker_image, eval_script)` through the connected MCP client.
It closes any current container and starts a new one from those two fields.

Inside the container the repository stays at `/testbed`, with no network and a
128-process limit. `run_tests` runs `eval_script` in a disposable copy and
returns logs plus an exit code. `get_patch` builds a `git -c core.fileMode=false`
diff, including new files. SIGTERM still runs container cleanup, including after
a task switch.
