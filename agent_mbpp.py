import sys
import time
import os
import json
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from typing import Annotated
import typer

from src.llm_connector import ProviderConfig, LLMConnector
from src.config_model import MBPPTaskInput
from src.sandbox_connector import SandboxClient
from src.agent_utils import mpc_arguments_to_python

from openai.types.chat import ChatCompletionMessageParam

# import fire

# from src.config_model import AgConfig

# Init Typer app
app = typer.Typer(
    add_completion=False,
    help="Agent Smith CLI (MBPP)"
)


@app.command(name="uv run -m agent_mbpp")
def main(
    task_file: Annotated[
        Path,
        typer.Option(
            "--task-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the JSON file containing tasks (must exist)."
        )
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            writable=True,
            help="Path to the JSON file for saving the generated solutions."
        )
    ],
    model_name: Annotated[
        str,
        typer.Option(
            "--model-name",
            help="LLM Model name"
        )
    ],
    provider_url: Annotated[
        str,
        typer.Option(
            "--provider-url",
            help="Provider API URL."
        )
    ],
    provider_api_key: Annotated[
        list[str],
        typer.Option(
            "--provider-api-key",
            help="Provider API KEY.",
        )
    ] = [],
    provider_api_keys_file: Annotated[
        Path | None,
        typer.Option(
            "--provider-api-keys-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help=("Path to .env file containing API Keys"
                  " (LLM_API_KEYS='[\"key_1\", \"key_2\"]').")
        )
    ] = None,
) -> None:

    typer.echo("-"*20)
    typer.echo("Agent Smith CLI (MBPP)")
    typer.echo(f"📂 Task file: {task_file}")
    typer.echo(f"📥 Output file: {output}")
    typer.echo(f"🤖 Model name: {model_name}")
    typer.echo(f"🌐 Provider URL: {provider_url}")
    typer.echo("-"*20)

    if not provider_api_key:
        if provider_api_keys_file is not None:
            typer.echo(f"Reading API keys from file: {provider_api_keys_file}")
            # 1. Parse the file into a dictionary
            env_dict = dotenv_values(provider_api_keys_file)
            if not env_dict:
                typer.secho("Error: Failed to parse .env file."
                            " Check for syntax errors!",
                            fg=typer.colors.RED, err=True)
            elif (not ("LLM_API_KEYS" in env_dict)
                  and not ("OPENROUTER_API_KEY" in env_dict)):
                typer.secho(
                    "Error: Failed to find variable LLM_API_KEYS"
                    " in .env file!",
                    fg=typer.colors.RED, err=True)
            else:
                load_dotenv(provider_api_keys_file, override=True)
        else:
            load_dotenv()
        raw_keys = os.getenv("LLM_API_KEYS",
                             os.getenv("OPENROUTER_API_KEY", ""))
        if raw_keys.startswith("[") and raw_keys.endswith("]"):
            # If it looks like a JSON array, decode it and convert to a tuple
            try:
                api_keys = tuple(json.loads(raw_keys))
            except Exception as ex:
                typer.secho(
                    "Error: Wrong value LLM_API_KEYS!"
                    f"{ex}",
                    fg=typer.colors.RED, err=True)
                typer.secho(
                    "(Example: LLM_API_KEYS='[\"key_1\","
                    " \"key_2\", \"key_3\"]' "
                    "or LLM_API_KEYS=my_only_one_key) \n",
                    err=True)
        else:
            # If it is just a single string key,
            # wrap it in a single-element tuple
            api_keys = (raw_keys,)
    elif isinstance(provider_api_key, str):
        api_keys = (provider_api_key.strip(),)
    else:
        api_keys = tuple(provider_api_key)

    if not api_keys:
        typer.secho("Error: Missing API Key!",
                    fg=typer.colors.RED, err=True)
        typer.secho(
            "Provide --provider-api-key  or --provider-api-keys-file \n"
            "For more information run: \n" + sys.argv[0] + " --help \n",
            err=True)
        # raise typer.Exit(code=1)
        typer.Exit(code=1)
        return

    provider = ProviderConfig(
        name=os.getenv("LLM_PROVIDER_NAME", "Provider 1"),
        base_url=provider_url,
        model=model_name,
        api_keys=api_keys
    )

    # print("Provider param:", provider)
    # print("Keys:", provider.api_keys)

    # read the task from file
    try:
        task_json = Path(task_file).read_text(encoding="utf-8")
        the_task = MBPPTaskInput.model_validate_json(task_json)
    except Exception as ex:
        typer.secho(f"Error: Can't read task from file '{task_file}'!\n{ex}",
                    fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    # print("Task:", the_task)
    # print("-"*20)

    try:
        llm = LLMConnector(provider)
    except Exception as ex:
        typer.secho(
            f"Error: Can`t connect to LLM Provider ({provider})'!\n{ex}",
            fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    # Connect to sandbox
    try:
        sandbox = SandboxClient('--mcp-stdio "uv run mcp_tools_mbpp.py"')
        sandbox.read_std(5)  # clear stdout from sandbox
        sandbox_manual = sandbox.request("manual()", 5)
    except Exception as ex:
        typer.secho(
            f"Error: Can`t run sandbox'!\n{ex}",
            fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    if not sandbox_manual:
        typer.secho(
            "Error: Can`t get manual from sandbox '!\n",
            fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    print("-"*10, " Sandbox manual:\n", sandbox_manual)

    # t_str = sandbox.request("show_tools()", 5)
    # print("-- test:", t_str)

    try:
        sandbox_tools_list = sandbox.run_tool("show_tools()").sandbox_output
    except Exception as ex:
        typer.secho(
            f"Error: Can`t get show_tools() in sandbox'!\n{ex}",
            fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    print("-"*10, " Sandbox tools:\n", sandbox_tools_list, "\n")

    # return

    # Record start time
    start_time = time.perf_counter()
    typer.echo("🚀 Start...")
    # -----------

    ts: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a Python coding assistant.\n"
            "You must use MCP server via Sandbox.\n"+sandbox_manual,
        },
        {
            "role": "user",
            "content": f"The task: {the_task.task_definition}\n"
            f"Function definition: {the_task.function_definition}\n"
            "Tests: \n ```" + "\n".join(the_task.test_list) + "```\n",

        },
    ]

    # print("-"*10, " Sandbox manual:\n", sandbox_manual)

    print("-"*10, " Prompt:")

    print(ts)

    try:
        response = llm.chat(
            ts,
            temperature=0,
            extra_body={"reasoning": {"effort": "none"}},
        )
    except Exception as ex:
        typer.secho(
            "Error: Can`t get llm response'!\n",
            fg=typer.colors.RED, err=True)
        typer.secho(ex, err=True)
        sandbox.close()
        typer.Exit(code=1)
        return

    usage = response.usage
    if usage is not None:
        print("Model used:", response.model)
        print("Input tokens:", usage.prompt_tokens)
        print("Output tokens:", usage.completion_tokens)
        print("Total tokens:", usage.total_tokens)
        print("-"*20)

    print("LLM content:")
    print(response.choices[0].message.content)

    # -----------
    print("+"*10)
    print("LLM reasoning:")
    print(response.choices[0].message.reasoning)

    print("+"*10, "full llm response:")
    print(response)
    print("+"*10)

    # -----------
    # print("+"*10)
    print("LLM tool_calls:")
    print(response.choices[0].message.tool_calls)
    print("+"*10)

    # If llm request tools via standard interface
    for tool_call in response.choices[0].message.tool_calls:
        print("-- Tool:", tool_call)

        if tool_call.type == 'function':
            call_id = tool_call.id
            tool_name = str(tool_call.function.name).strip()
            if tool_name not in sandbox_tools_list:
                print(f"Tool (name={tool_name}) not found!")
                continue
            arg_str = ""
            if tool_call.function.arguments:
                arg_str = mpc_arguments_to_python(tool_call.function.arguments)
            sb_call = f"{tool_name}({arg_str})"
            print(" - call_id:", call_id)
            print(" - call:", sb_call)
            tool_result = sandbox.run_tool(sb_call).sandbox_output
            print(" - result:", tool_result)
        else:
            print("LLM try to call not supported type of call tool "
                  f"(type={tool_call.type}).")

    # -----------
    typer.echo("✅ Done!")
    end_time = time.perf_counter()
    # Calculate duration
    duration = int(end_time - start_time)
    typer.echo(f"⏱ Execution time: {duration // 60}:{duration % 60}"
               " (minutes:seconds)")
    typer.echo("-"*20)

    sandbox.close()


if __name__ == "__main__":
    sys.argv[0] = "uv run -m agent_mbpp"
    app()
