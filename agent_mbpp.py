import sys
import time
import os
import json
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from typing import Annotated, cast
import typer

from src.llm_connector import ProviderConfig, LLMConnector
from src.config_model import MBPPTaskInput, StepMetrics, SolutionOutput
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

    messages: list[ChatCompletionMessageParam] = [
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

    system_prompt = str(messages)

    # print("-"*10, " Sandbox manual:\n", sandbox_manual)

    solution = ""
    llm_step_number = 0
    llm_finish = False
    llm_tested_code = False
    steps: list[StepMetrics] = []

    while llm_step_number < 5 and not llm_finish:
        llm_step_number += 1
        llm_finish = True
        llm_input_tokens = 0
        llm_output_tokens = 0
        llm_model_name = model_name
        llm_response_raw_text = ""
        sandbox_input = ""
        sandbox_output = ""

        # print("="*10, f" Prompt (step {llm_step}):")
        typer.secho(f"==========  Prompt (step {llm_step_number}):",
                    fg=typer.colors.BLUE)

        print(messages)

        llm_start_time = time.perf_counter()

        try:
            response_raw = llm.chat_raw(
                messages,
                temperature=0,
                extra_body={"reasoning": {"effort": "none"}},
            )
            llm_response_raw_text = ""
            response = response_raw.parse()
            # add current answer to list
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            # print("mmmmmmmmmmm1")
            # print("msG:",llm_response_raw_text)
            # print("mmmmmmmmmmm2")
            # print(f"Status Code: {response_raw.status_code}")
            # print(f"Headers: {response_raw.headers}")
            # # print(assistant_message.model_dump_json(indent=2))
            # print("mmmmmmmmmmm3")
        except Exception as ex:
            typer.secho(
                "Error: Can`t get llm response'!\n",
                fg=typer.colors.RED, err=True)
            typer.secho(ex, err=True)
            sandbox.close()
            typer.Exit(code=1)
            return

        llm_finish_time = time.perf_counter()

        usage = response.usage
        if usage is not None:
            llm_input_tokens = usage.prompt_tokens
            llm_output_tokens = usage.completion_tokens
            llm_model_name = response.model
            print("Model used:", response.model)
            print("Input tokens:", usage.prompt_tokens)
            print("Output tokens:", usage.completion_tokens)
            print("Total tokens:", usage.total_tokens)
            print("-"*20)

        if response.choices[0].message.content:
            llm_response_raw_text += response.choices[0].message.content
        if response.choices[0].message.reasoning:
            llm_response_raw_text += ("\n<reasoning>" +
                                      response.choices[0].message.reasoning +
                                      "\n</reasoning>\n")

        print("LLM content:")
        print(response.choices[0].message.content)
        # -----------
        print("+"*10)
        print("LLM reasoning:")
        print(response.choices[0].message.reasoning)

        # print("+"*10, "full llm response:")
        # print(response)
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
                call_id = str(tool_call.id)
                tool_name = str(tool_call.function.name).strip()
                llm_response_raw_text += (
                    "\n<tool_call>\n"
                    f"  ID: {call_id}\n"
                    f"  Name: {tool_name}\n")
                if tool_call.function.arguments:
                    llm_response_raw_text += (
                        "  Arguments: "
                        f"{tool_call.function.arguments}\n")
                llm_response_raw_text += "</tool_call>\n"
                if tool_name not in sandbox_tools_list:
                    print(f"Tool (name={tool_name}) not found!")
                    messages.append(
                        cast(ChatCompletionMessageParam,
                             {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": ("Error: Tool (name="
                                            f"{tool_name}) not found!")
                             }))
                    llm_finish = False
                    continue
                arg_str = ""
                if tool_call.function.arguments:
                    arg_str = mpc_arguments_to_python(
                        tool_call.function.arguments)
                sb_call = f"{tool_name}({arg_str})"
                print(" - call_id:", call_id)
                print(" - call:", sb_call)
                sandbox_input += sb_call + "\n"
                result_ = sandbox.run_tool(sb_call)
                tool_result = result_.sandbox_output
                sandbox_output += str(tool_result) + "\n"
                print(" - result:", tool_result)
                messages.append(
                    cast(ChatCompletionMessageParam,
                         {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result
                         }))
                # print("*********** message:", messages)
                if tool_name == 'run_tests' and result_.success:
                    print("$$$$$:", tool_result)
                    llm_tested_code = True
                    llm_finish = False
                elif tool_name == 'final_answer' and result_.success:
                    solution = str(tool_result)
                    # the end of task
                    if not llm_tested_code:
                        pass  # test in sandbox if LLM not tested it
                    llm_finish = True
                else:
                    llm_finish = False
            else:
                print("LLM try to call not supported type of call tool "
                      f"(type={tool_call.type}).")

        # Calculate duration in milliseconds accurately
        llm_duration = int((llm_finish_time - llm_start_time) * 1000)
        current_step = StepMetrics(
            step=llm_step_number,
            input_tokens=llm_input_tokens,
            output_tokens=llm_output_tokens,
            request_time_ms=llm_duration,
            api_url=provider_url,
            model_name=llm_model_name,
            llm_output=llm_response_raw_text,
            sandbox_input=sandbox_input.strip(),
            sandbox_output=sandbox_output.strip(),
            )
        steps.append(current_step)
        print("==== ", current_step)

    # -----------
    # -----------
    typer.echo("✅ Done!")
    typer.echo(f"  Steps: {llm_step_number}")
    end_time = time.perf_counter()
    # Calculate duration
    duration = int(end_time - start_time)
    typer.echo(f"⏱ Execution time: {duration // 60}:{duration % 60}"
               " (minutes:seconds)")
    typer.echo("-"*20)
    sandbox.close()
    task_solution = SolutionOutput(
        task_id=str(the_task.task_id),
        benchmark='mbpp',
        success=llm_tested_code,
        solution=solution,
        iterations=llm_step_number,
        steps=steps,
        total_requests=0,  # !!!
        total_input_tokens=0,
        total_output_tokens=0,
        total_time_seconds=duration,
        system_prompt=system_prompt
    )

    # output

    # print("output:", task_solution)

    try:
        with open(output, "w", encoding="utf-8") as f:
            f.write(task_solution.model_dump_json(indent=4))
    except Exception as ex:
        typer.secho(
            f"Error: Can`t write output json in file \"{output}\"!\n{ex}",
            fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return


if __name__ == "__main__":
    sys.argv[0] = "uv run -m agent_mbpp"
    app()
