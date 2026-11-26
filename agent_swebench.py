# Agent Smith CLI (SWE-bench)

import sys
import time
import os
import json
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from typing import Annotated, cast, Any
import typer

from src.llm_connector import ProviderConfig, LLMConnector
from src.config_model import SWEBenchTaskInput, StepMetrics, SolutionOutput
from src.sandbox_connector import SandboxClient
from src.agent_utils import mpc_arguments_to_python, it_is_python_code
from src.agent_utils import it_is_tool_call, mpc_result_to_txt
# from src.agent_utils import extract_function_code
from src.agent_utils import normalize_json_arguments, check_str_is_json_dict
from src.agent_utils import parse_ernie_tool_calls

from openai.types.chat import ChatCompletionMessageParam

# import fire

# from src.config_model import AgConfig

# Init Typer app
app = typer.Typer(
    add_completion=False,
    help="Agent Smith CLI (SWE-bench)"
)


@app.command(name="uv run -m agent_swebench")
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
    ] = "",
    provider_url: Annotated[
        str,
        typer.Option(
            "--provider-url",
            help="Provider API URL."
        )
    ] = "",
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
    print_debug: Annotated[
        bool,
        typer.Option(
            "--print-debug",
            help="Print additional information during run.",
        )
    ] = False,
    is_reasoning: Annotated[
        bool,
        typer.Option(
            "--reasoning",
            help="Model can't work without reasoning.",
        )
    ] = False,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            min=1,
            max=900,
            help="Maximum steps to solve the task.",
        )
    ] = 30,
    max_in_tokens: Annotated[
        int,
        typer.Option(
            "--max-in-tokens",
            min=100,
            max=2000000,
            help="Maximum input tokens.",
        )
    ] = 300000,
    max_out_tokens: Annotated[
        int,
        typer.Option(
            "--max-out-tokens",
            min=100,
            max=1000000,
            help="Maximum output tokens.",
        )
    ] = 10000,
    max_time_s: Annotated[
        int,
        typer.Option(
            "--timeout",
            min=1,
            max=2000,
            help="Timeout - Maximum time in seconds.",
        )
    ] = 900,
    tool_sandbox: Annotated[
        bool,
        typer.Option(
            "--tool-sandbox",
            help=("Expose to LLM "
                  "\"execute_sandbox(python_code:str)\""
                  " in standart tools list."),
        )
    ] = False,
    disable_tool_validation: Annotated[
        bool,
        typer.Option(
            "--disable-tool-validation",
            help=("Not generate error "
                  "if LLM try to call tool not from tools list."),
        )
    ] = False,
    print_model_list: Annotated[
        bool,
        typer.Option(
            "--print-model-list",
            help=("Print Provider model list."),
        )
    ] = False,

) -> None:

    final_test = False

    def final_chek(f_str: str) -> str:
        # nonlocal final_test
        # if "run_tests" not in sandbox_tools_list:
        #     #  can`t check solution without this tool
        #     return ""
        # sb_call = (
        #     'run_tests(\n'
        #     f'    code={repr(f_str)},\n'
        #     f'    test_list={str(the_task.test_list)}'
        # )
        # if the_task.test_imports:
        #     sb_call += f',\n    test_imports={str(the_task.test_imports)}'
        # sb_call += ')'

        # # print("final test:", sb_call)
        # result_ = sandbox.run_tool(f"print({sb_call})")
        # tool_result = mpc_result_to_txt(result_.sandbox_output)
        # if result_.success and not (result_.truncated):
        #     final_test = True
        #     return ""
        # # someting wrong:
        # txt_ = ('{"test_result":\n'
        #         f'{str(tool_result)},\n'
        #         '"test_run_code":\n'
        #         f'{sb_call}\n'
        #         "}"
        #         )
        # print("!!!!! Final test error:", txt_)
        # return txt_
        return ""

    typer.echo("-"*20)
    typer.echo("Agent Smith CLI (SWE-bench)")
    typer.echo(f"📂 Task file: {task_file}")
    typer.echo(f"📥 Output file: {output}")

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

    if not provider_url:
        provider_url = os.getenv("LLM_PROVIDER_URL", "")
        if not provider_url:
            load_dotenv()
            provider_url = os.getenv("LLM_PROVIDER_URL", "")
    if not provider_url:
        typer.secho("Error: Missing LLM_PROVIDER_URL!",
                    fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    if not model_name:
        model_name = os.getenv("LLM_MODEL",
                               os.getenv("LLM_MODEL_NAME", ""))
        if not model_name:
            load_dotenv()
            model_name = os.getenv("LLM_MODEL",
                                   os.getenv("LLM_MODEL_NAME", ""))

    typer.echo(f"🤖 Model name: {model_name}")
    typer.echo(f"🌐 Provider URL: {provider_url}")
    typer.echo(f" Print debug: {print_debug}")
    typer.echo("-"*20)

    provider = ProviderConfig(
        name=os.getenv("LLM_PROVIDER_NAME", f"Provider({provider_url})"),
        base_url=provider_url,
        model=model_name,
        api_keys=api_keys
    )

    # connect to LLM Provide
    try:
        llm = LLMConnector(provider)
    except Exception as ex:
        typer.secho(
            f"Error: Can`t connect to LLM Provider ({provider})'!\n{ex}",
            fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    if print_model_list and llm.model_list():
        print("-"*10, " Provider model list:")
        i = 1
        for lm_ in llm.model_list():
            print(f"\n{i}): ", lm_.id)
            print(lm_)
            i += 1
        print("-"*20)
        if not model_name:
            typer.echo(
                "Specify the LLM model name using "
                "the --model-name parameter or the "
                "LLM_MODEL_NAME variable!")
            return

    if not model_name:
        typer.secho("Error: Missing LLM_MODEL_NAME!",
                    fg=typer.colors.RED, err=True)
        typer.echo(
            "Specify the LLM model name using "
            "the --model-name parameter or the "
            "LLM_MODEL_NAME variable!")
        typer.Exit(code=1)
        return

    # read the task from file
    try:
        task_json = Path(task_file).read_text(encoding="utf-8")
        the_task = SWEBenchTaskInput.model_validate_json(task_json)
    except Exception as ex:
        typer.secho(f"Error: Can't read task from file '{task_file}'!\n{ex}",
                    fg=typer.colors.RED, err=True)
        typer.Exit(code=1)
        return

    if not the_task.docker_image:
        typer.secho(f"Error: Check the task '{task_file}'!\n",
                    fg=typer.colors.RED, err=True)
        typer.echo("Docker image must be specified!")
        typer.Exit(code=1)
        return

    if not the_task.problem_statement:
        typer.secho(f"Error: Check the task '{task_file}'!\n",
                    fg=typer.colors.RED, err=True)
        typer.echo("Problem statement must be specified!")
        typer.Exit(code=1)
        return

    if not the_task.eval_script:
        typer.secho(f"Error: Check the task '{task_file}'!\n",
                    fg=typer.colors.RED, err=True)
        typer.echo("Eval script must be specified!")
        typer.Exit(code=1)
        return

    typer.echo(f" Task id: {the_task.instance_id}")
    typer.echo(f" Docker: {the_task.docker_image}")

    # Connect to sandbox
    try:
        sandbox = SandboxClient('--mcp-stdio "uv run mcp_tools_swebench.py"')
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
        sandbox.close()
        typer.Exit(code=1)
        return

    try:
        sandbox_tools_list_ = sandbox.run_tool("show_tools()").sandbox_output
    except Exception as ex:
        typer.secho(
            f"Error: Can`t get show_tools() in sandbox'!\n{ex}",
            fg=typer.colors.RED, err=True)
        sandbox.close()
        typer.Exit(code=1)
        return
    if isinstance(sandbox_tools_list_, str):
        sandbox_tools_list = [sandbox_tools_list_]
    else:
        sandbox_tools_list = list(sandbox_tools_list_)

    if print_debug:
        print("-"*10, " Sandbox tools:\n", sandbox_tools_list, "\n")

    # Start Docker !!!!!!!
    startup_tool = "start_task_not_a_tool_for_manual"
    if startup_tool not in sandbox_tools_list:
        typer.secho(
            f"Error: SWE-bench MCP is missing {startup_tool}().",
            fg=typer.colors.RED, err=True)
        sandbox.close()
        typer.Exit(code=1)
        return

    try:
        start_res = sandbox.run_tool(
            f"print({startup_tool}("
            f"docker_image={repr(the_task.docker_image)}, "
            f"eval_script={repr(the_task.eval_script)}"
            "))",
            660)
    except Exception as ex:
        typer.secho(
            "Error: MCP server can't start docker image "
            f"{repr(the_task.docker_image)} for the task!\n"
            f"{ex}",
            fg=typer.colors.RED, err=True)
        sandbox.close()
        typer.Exit(code=1)
        return

    if not start_res.success:
        typer.secho(
            "Error: MCP server can't start docker image "
            f"{repr(the_task.docker_image)} for the task!\n",
            fg=typer.colors.RED, err=True)
        typer.echo(start_res.sandbox_output)
        sandbox.close()
        typer.Exit(code=1)
        return

    if print_debug:
        print("Start staus:", start_res.sandbox_output)

    # Record start time
    start_time = time.perf_counter()
    typer.echo("🚀 Start...")
    # ----------- assistant

    if tool_sandbox:
        tools_list = [
            {
                "type": "function",
                "function": {
                    "name": "execute_sandbox",
                    "description": (
                        "Execute MCP function as Python code in the sandbox. "
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to execute in the sandbox"
                                )
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False
                    }
                }
            }
        ]
    else:
        tools_list = []

    hints = str(the_task.hints_text).strip()

    if hints:
        hints = "\nHints: " + hints + "\n"

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are an autonomous Python coding agent.\n"
            "You solve SWE-bench task "
            "on real Python project repositories"
            " inside Docker container.\n"
            "Your objective is to fix the reported issue "
            "in the target repository and return a valid git patch.\n"
            "To work with repository you must use MCP tools via Sandbox.\n"
            "The Sandbox is ONLY an MCP tool interface! "
            "And only MCP tools have access to project repository.\n"
            "Sandbox purpose is to call the available MCP functions "
            "to interact with the target repository.\n"
            "The only permitted operations in sandbox code "
            "are calls to the available MCP tools.\n"
            "In Sandbox code you MUST NOT:\n"
            "- Import project libraries or third-party packages.\n"
            "- Derectly execute the user's reproduction example.\n"
            "- Run arbitrary Python scripts.\n"
            "- Try to read or write repository files "
            "using Python filesystem functions.\n"
            "\n" + sandbox_manual + "\n"
            "\n"
            "# Coding rules:\n"
            "- Never assume a file or directory exists without checking it.\n"
            "- Prefer minimal changes consistent with the"
            " existing code style.\n"
            "- Do not modify unrelated code.\n"
            "- Inspect existing implementations and tests before"
            " deciding how to it.\n"
            "- Consider edge cases and existing behavior.\n"
            "- Do not claim tests passed unless you actually"
            " ran them and received a successful exit code.\n"
            "\n"
            "# Mandatory workflow:\n"
            "- Search for relevant functions, classes, error messages,"
            " or expressions using search_code"
            " or search_function_or_class_definition_in_code.\n"
            "There are a lot of files. "
            "Try to narrow down your search as much as possible.\n"
            "- Use the exact file paths returned by MCP tools."
            " Never guess filenames or paths.\n"
            "- Read only the relevant source sections using read_file.\n"
            "- Make the smallest correct change that fixes"
            " the underlying issue.\n"
            "- Use run_tests() to test solutions.\n"
            "- Call final_answer(get_patch())"
            " when the solution is ready.\n"
            "\n"
            "# Interaction rules:\n"
            "- Produce exactly ONE Python style code block per response "
            "to call Sandbox MCP functions "
            "and nothing else!\n"
            "- Do not include explanations, introductions,"
            " Markdown lists, or reasoning.\n"
            "- Do not repeat previously executed operations"
            " unless necessary.\n"
            "- Use returned MCP observations to decide the next action.\n"
            "- Do not attempt several speculative fixes in one step.\n"
            "- Do not use external sources, GitHub issues, "
            "pull requests, or memorized patches.\n"
            "\n",
        },
        {
            "role": "user",
            "content": (
                "The task (problem to solve): "
                f"{the_task.problem_statement}\n"
                f"{hints}")
        },
    ]

    system_prompt = str(messages)

    if print_debug:
        print("-"*10, "Start prompt:")
        for m_ in messages:
            for k, v in m_.items():
                print(f"{k} : \"{str(v).strip()}\" ")

    solution = ""
    error_txt = ""
    final_test = False
    reasoning = ""
    llm_step_number = 0
    llm_finish = False
    # llm_tested_code = False
    steps: list[StepMetrics] = []
    total_input_tokens = 0
    total_output_tokens = 0
    python_text: list[str] = []
    max_tool_retries = 3
    tool_call_error = 0

    while (
            llm_step_number < max_iterations
            and not llm_finish
            and total_input_tokens < max_in_tokens
            and total_output_tokens < max_out_tokens
            and (int(time.perf_counter() - start_time) < max_time_s)
    ):
        llm_step_number += 1
        llm_finish = True
        llm_input_tokens = 0
        llm_output_tokens = 0
        llm_model_name = model_name
        llm_response_raw_text = ""
        sandbox_input = ""
        sandbox_output = ""
        llm_retries = 0

        # print("="*10, f" Prompt (step {llm_step}):")
        typer.secho(f"==========  Prompt (step {llm_step_number}):",
                    fg=typer.colors.BLUE)

        if print_debug and (llm_step_number > 1):
            if len(messages) < 3:
                for m_ in messages:
                    print(m_)
            else:
                print(messages[-2])
                print(messages[-1])

        llm_start_time = time.perf_counter()
        extra_body: dict[str, Any] = {}
        if not is_reasoning:
            extra_body = {"reasoning": {"effort": "none"}}
        if disable_tool_validation:
            extra_body["disable_tool_validation"] = True

        try:
            if extra_body:
                response_raw = llm.chat_raw(
                    messages,
                    tools=tools_list,
                    extra_body=extra_body,
                )
            else:
                response_raw = llm.chat_raw(
                    messages,
                    tools=tools_list,
                )
            llm_response_raw_text = ""
            response = response_raw.parse()
            # add current answer to list
            assistant_message = response.choices[0].message
        except Exception as ex:
            typer.secho(
                "Error: Can`t get llm response'!\n",
                fg=typer.colors.RED, err=True)
            typer.secho(ex, err=True)
            # sandbox.close()
            # typer.Exit(code=1)
            # return
            error_txt += f"Error: Can`t get llm response'!\n{str(ex)}"
            continue

        if response.choices[0].message.tool_calls:
            # Check if model sent correct tool calls
            # Cohere modeles sometimes send not json as tool arguments

            # Make a mutable copy of the assistant message.
            assistant_message = response.choices[0].message.model_dump(
                exclude_none=True)
            for call in response.choices[0].message.tool_calls:
                if call.type != 'function':
                    continue
                call_id = call.id
                function = call.function
                if function is None:
                    continue
                tool_name = function.name
                arg = function.arguments
                if arg is None:
                    continue
                arg = str(arg).strip()
                if check_str_is_json_dict(arg):
                    continue
                try:
                    arg_dict = normalize_json_arguments(arg)
                    # Always store normalized JSON string arguments.
                    function.arguments = json.dumps(
                        arg_dict, ensure_ascii=False)
                    print("test func arg:", function.arguments)
                except Exception:
                    function.arguments = "{}"

        messages.append(assistant_message)

        llm_finish_time = time.perf_counter()
        llm_retries = llm.retries
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
        total_input_tokens += llm_input_tokens
        total_output_tokens += llm_output_tokens

        msg = response.choices[0].message
        if msg.content:
            llm_response_raw_text += msg.content
        reasoning = (getattr(msg, "reasoning", "")
                     or (msg.model_extra or {}).get("reasoning", ""))
        if reasoning:
            llm_response_raw_text += ("\n<reasoning>" +
                                      reasoning +
                                      "\n</reasoning>\n")

        if print_debug:
            print("LLM content:")
            print(response.choices[0].message.content)
            # -----------
            print("+"*10)
            print("LLM reasoning:")
            print(reasoning)
            # -----------
            print("+"*10)
            print("LLM tool_calls:")
            print(response.choices[0].message.tool_calls)

            print("+"*10)

        # If llm request tools via standard interface
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
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
                    if ((tool_name not in sandbox_tools_list
                         and tool_name != 'execute_sandbox')):
                        print(f"Tool (name={tool_name}) not found!")
                        messages.append(
                            cast(ChatCompletionMessageParam,
                                 {
                                     "role": "tool",
                                     "tool_call_id": call_id,
                                     "content": ("Error: Tool (name="
                                                 f"{tool_name}) not found!")
                                 }))
                        tool_call_error += 1
                        if tool_call_error < max_tool_retries:
                            llm_finish = False
                        continue
                    arg_str = ""
                    if tool_name == 'execute_sandbox':
                        try:
                            arg_dict = normalize_json_arguments(
                                tool_call.function.arguments)
                        except Exception:
                            arg_dict = {}
                        arg_str = arg_dict.get("code", "")
                        if arg_str and it_is_python_code(arg_str):
                            sb_call = arg_str
                        else:
                            messages.append(
                                cast(
                                    ChatCompletionMessageParam,
                                    {
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": (
                                            "Error: parsing tool "
                                            "arguments.\n"
                                            f'tool_name="{tool_name}",\n'
                                            'as arguments "code" must be '
                                            'Python style call of function'
                                            f'from list: {sandbox_tools_list}"'
                                        )
                                    }))
                            tool_call_error += 1
                            if tool_call_error < max_tool_retries:
                                llm_finish = False
                            continue
                    elif (tool_call.function.arguments
                          and tool_call.function.arguments.strip()):
                        arg_str = mpc_arguments_to_python(
                            tool_call.function.arguments)
                        if not arg_str:
                            messages.append(
                                cast(
                                    ChatCompletionMessageParam,
                                    {
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "content": (
                                            "Error: parsing tool "
                                            "arguments from json!\n"
                                            f'tool_name="{tool_name}",\n'
                                            'arguments="'
                                            f'{tool_call.function.arguments}"'
                                        )
                                    }))
                            tool_call_error += 1
                            if tool_call_error < max_tool_retries:
                                llm_finish = False
                            continue
                    if tool_name != 'execute_sandbox':
                        sb_call = f"print({tool_name}({arg_str}))"
                    if print_debug:
                        print("Tool call:")
                        print(" - call_id:", call_id)
                        print(" - call:", sb_call)
                    tool_call_error = 0
                    sandbox_input += sb_call + "\n"
                    # print("=====", sandbox_input)
                    result_ = sandbox.run_tool(sb_call)
                    # print("-===:sandbox return:", result_)
                    tool_result = mpc_result_to_txt(result_.sandbox_output)
                    sandbox_output += str(tool_result) + "\n"
                    messages.append(
                        cast(ChatCompletionMessageParam,
                             {
                                 "role": "tool",
                                 "tool_call_id": call_id,
                                 "content": tool_result
                             }))
                    llm_finish = False
                    if print_debug:
                        print(" - result:", tool_result)
                    if result_.success and result_.final_answer:
                        solution = str(result_.final_answer)
                        llm_finish = True
                else:
                    print("LLM try to call not supported type of call tool "
                          f"(type={tool_call.type}).")
        else:
            #  chek if model ask tool in python style
            # for example for model_name: nvidia/nemotron-3.5-super-vl:free

            # (a) Python code blocks (primary) — “‘python ... “‘ <end_code>
            python_text = []
            if response.choices[0].message.content:
                str_ = str(response.choices[0].message.content).strip()

                # try find XML tool calls in Anthropic-style
                if str_.find("tool_call"):
                    tools_calls, error_txt = parse_ernie_tool_calls(str_)
                    if not tools_calls and error_txt:
                        messages.append(
                            cast(
                                ChatCompletionMessageParam,
                                {
                                    "role": "user",
                                    "content": (
                                        "Tool call parsing failed.\n"
                                        "Your previous tool call was "
                                        "malformed. "
                                        "No command had been executed "
                                        "on this step."
                                        "Regenerate the complete tool call "
                                        "using valid XML.\n"
                                        f"Error: {error_txt}"), }))
                        tool_call_error += 1
                        if tool_call_error < max_tool_retries:
                            llm_finish = False
                        continue
                    if tools_calls:
                        for call_xml in tools_calls:
                            if print_debug:
                                print("-"*10)
                                print("XML Tool call:")
                                print(call_xml)

                            tool_name = call_xml.get("name", "")
                            arg_json = call_xml.get("arguments", "")
                            if ((tool_name not in sandbox_tools_list
                                 and tool_name != 'execute_sandbox')):
                                print(f"Tool (name={tool_name}) not found!")
                                messages.append(
                                    cast(ChatCompletionMessageParam,
                                         {
                                             "role": "user",
                                             "content": (
                                                 "Error: Tool (name="
                                                 f"{tool_name}) not found!")
                                         }))
                                tool_call_error += 1
                                if tool_call_error < max_tool_retries:
                                    llm_finish = False
                                continue

                            if tool_name != 'execute_sandbox':
                                arg_str = ""
                                if arg_json:
                                    arg_str = mpc_arguments_to_python(arg_json)
                                    if not arg_str:
                                        messages.append(
                                            cast(
                                                ChatCompletionMessageParam,
                                                {
                                                    "role": "user",
                                                    "content": (
                                                        "Error: parsing tool "
                                                        "arguments!\n"
                                                        "tool_name="
                                                        f'"{tool_name}",\n'
                                                        'arguments="'
                                                        f'{str(arg_json)}"'
                                                    )
                                                }))
                                        tool_call_error += 1
                                        if tool_call_error < max_tool_retries:
                                            llm_finish = False
                                        continue
                                sb_call = f"print({tool_name}({arg_str}))"
                            else:
                                try:
                                    arg_dict = normalize_json_arguments(
                                        arg_json)
                                except Exception:
                                    arg_dict = {}
                                arg_str = arg_dict.get("code", "")
                                if arg_str and it_is_python_code(arg_str):
                                    sb_call = arg_str
                                else:
                                    messages.append(
                                        cast(
                                            ChatCompletionMessageParam,
                                            {
                                                "role": "user",
                                                "content": (
                                                    "Error: parsing tool "
                                                    "arguments.\n"
                                                    "tool_name="
                                                    f'"{tool_name}",\n'
                                                    'Argument "code"'
                                                    " must be "
                                                    'Python style call of '
                                                    'a function from list: '
                                                    f'{sandbox_tools_list}".'
                                                )
                                            }))
                                    tool_call_error += 1
                                    if tool_call_error < max_tool_retries:
                                        llm_finish = False
                                    continue

                            tool_call_error = 0
                            if print_debug:
                                print("Tool call:")
                                print(" - call:", sb_call)
                            sandbox_input += sb_call + "\n"
                            # print("=====", sandbox_input)
                            result_ = sandbox.run_tool(sb_call)
                            # print("-===:sandbox return:", result_)
                            tool_result = mpc_result_to_txt(
                                result_.sandbox_output)
                            sandbox_output += str(tool_result) + "\n"
                            messages.append(
                                cast(ChatCompletionMessageParam,
                                     {
                                         "role": "user",
                                         "content": (
                                             "Tool call result: "
                                             f"{tool_result}")
                                     }))
                            tool_call_error = 0
                            llm_finish = False
                            if print_debug:
                                print(" - result:", tool_result)
                            if result_.success and result_.final_answer:
                                solution = str(result_.final_answer)
                                llm_finish = True
                        continue

                # try to find Python code
                start = 0
                while (start >= 0):
                    start = str_.find("```python", start)
                    if start >= 0:
                        end = str_.find("```", start + 9)
                        if end > start:
                            python_text.append(str_[start+9:end])
                            start = max(start, end) + 3
                if not python_text:
                    start = 0
                    while (start >= 0):
                        start = str_.find("```", start)
                        if start >= 0:
                            end = str_.find("```", start + 3)
                            if end > start:
                                code_txt = str_[start+3:end]
                                start = max(start, end) + 3
                                if it_is_python_code(code_txt):
                                    python_text.append(code_txt)

            # if not find python code in content - try to find it in reasoning
            if (not python_text) and reasoning:
                str_ = str(reasoning).strip()
                start = 0
                while (start >= 0):
                    start = str_.find("```python", start)
                    if start >= 0:
                        end = str_.find("```", start + 9)
                        if end > start:
                            python_text.append(str_[start+9:end])
                            start = max(start, end) + 3
                if not python_text:
                    start = 0
                    while (start >= 0):
                        start = str_.find("```", start)
                        if start >= 0:
                            end = str_.find("```", start + 3)
                            if end > start:
                                code_txt = str_[start+3:end]
                                start = max(start, end) + 3
                                if it_is_python_code(code_txt):
                                    python_text.append(code_txt)

            if not python_text and response.choices[0].message.content:
                # May be full model answer is Python code
                str_ = str(response.choices[0].message.content).strip()
                if it_is_python_code(str_):
                    python_text.append(str_)

            # try to find tools call
            if python_text:
                # At first look on all blocks as one long Python code
                code_txt = "\n".join(python_text)
                if it_is_python_code(code_txt):
                    tools_ = it_is_tool_call(code_txt, sandbox_tools_list)
                    if tools_:
                        # try to call sandbox with this code
                        if print_debug:
                            print("== sandbox call:")
                            print(code_txt)
                            print("== end")
                        sandbox_input += code_txt + "\n"
                        result_ = sandbox.run_tool(code_txt)
                        tool_result = mpc_result_to_txt(result_.sandbox_output)
                        sandbox_output += str(tool_result) + "\n"
                        if print_debug:
                            print("== sandbox result (success:",
                                  result_.success, "):")
                            print(result_)
                            print("== end")
                        llm_finish = False
                        tool_call_error = 0
                        if result_.success:
                            if tool_result:
                                messages.append({
                                    "role": "user",
                                    "content": ("Sandbox answer:\n"
                                                f"{tool_result}"),
                                })
                            else:
                                messages.append({
                                    "role": "user",
                                    "content": ("Sandbox answer: success\n"
                                                "# Use print() if you need "
                                                "to receive MCP tool result."),
                                })
                            if result_.final_answer:
                                solution = str(result_.final_answer)
                                llm_finish = True
                        else:
                            messages.append({
                                "role": "user",
                                "content": ("Sandbox call error:\n"
                                            f"{tool_result}"),
                            })
                    # else:
                    #     # Perhaps an LLM does not call tools,
                    #     # but rather proposes solutions.
                    #     txt_code_solution = extract_function_code(
                    #         code_txt,
                    #         str(the_task.function_definition).strip())

                    #     if txt_code_solution:
                    #         solution = txt_code_solution
                    #         llm_finish = True
                else:
                    # try to find one block with tool call or solution
                    for code_txt in python_text:
                        tools_ = it_is_tool_call(code_txt, sandbox_tools_list)
                        if tools_:
                            # try to call sandbox with this code
                            if print_debug:
                                print("== sandbox call:")
                                print(code_txt)
                                print("== end")
                            sandbox_input += code_txt + "\n"
                            result_ = sandbox.run_tool(code_txt)
                            tool_result = mpc_result_to_txt(
                                result_.sandbox_output)
                            sandbox_output += str(tool_result) + "\n"
                            if print_debug:
                                print("== sandbox result (success:",
                                      result_.success, "):")
                                print(result_)
                                print("== end")
                            llm_finish = False
                            tool_call_error = 0
                            if result_.success:
                                if tool_result:
                                    messages.append({
                                        "role": "user",
                                        "content": ("Sandbox answer:\n"
                                                    f"{tool_result}"),
                                    })
                                else:
                                    messages.append({
                                        "role": "user",
                                        "content": ("Sandbox answer: success\n"
                                                    "# Use print() if you need"
                                                    " to receive MCP tool "
                                                    " result."),
                                    })
                                if result_.final_answer:
                                    solution = str(result_.final_answer)
                                    llm_finish = True
                            else:
                                messages.append({
                                    "role": "user",
                                    "content": ("Sandbox call error:\n"
                                                f"{tool_result}"),
                                })
                            continue
                        # else:
                        #     # Perhaps an LLM does not call tools,
                        #     # but rather proposes solutions.
                        #     txt_code_solution = extract_function_code(
                        #         code_txt,
                        #         str(the_task.function_definition).strip())
                        #     if txt_code_solution:
                        #         solution = txt_code_solution
                        #         llm_finish = True
                        #         break

        if not solution and llm_finish and tool_call_error == 0:
            tool_call_error += 1
            messages.append(
                cast(ChatCompletionMessageParam,
                     {
                         "role": "user",
                         "content": ("No solution was provided, "
                                     "and no tool was called on "
                                     "the last step.\n"
                                     "This is the end? \n"
                                     "Do not write explanation. "
                                     "You must work only calling "
                                     "avalible function in Python "
                                     "block. \nExample: "
                                     "```python\nrun_tests()\n``` \n")
                     }))
            llm_finish = False

        if solution and llm_finish:
            test_ = final_chek(solution)
            if test_:   # something wrong
                messages.append(
                    cast(ChatCompletionMessageParam,
                         {
                             "role": "user",
                             "content": ("The solution did not pass" +
                                         "the final test:\n" +
                                         str(test_).strip())
                         }))
                llm_finish = False

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
            retries=llm_retries
        )
        steps.append(current_step)
        # print("==== ", current_step)

    # The last try to find solution in the last LLM response
    # if (not solution) and python_text:
    #     code_txt = "\n".join(python_text)
    #     txt_code_solution = extract_function_code(
    #         code_txt,
    #         str(the_task.function_definition).strip())
    #     if txt_code_solution:
    #         solution = txt_code_solution
    #         final_chek(solution)

    # -----------
    # -----------
    if print_debug:
        typer.echo("-"*20)

    typer.echo("✔ Done!")
    typer.echo(f"  Steps: {llm_step_number}")
    end_time = time.perf_counter()
    # Calculate duration
    duration = int(end_time - start_time)
    typer.echo(f"⏱ Execution time: {duration // 60}:{duration % 60}"
               " (minutes:seconds)")
    typer.echo(f"  Input tokens: {total_input_tokens}")
    typer.echo(f"  Output tokens: {total_output_tokens}")

    final_success = False
    if not solution:
        error_txt += "Error - The solution was not provided!\n"
        if llm_step_number >= max_iterations:
            error_txt += "Maximum number of iterations reached!\n"
    elif not final_test:
        error_txt += "Error - Can not check the solution!\n"
    else:
        final_success = True
    if total_input_tokens > max_in_tokens:
        error_txt += "Maximum number of input tokens reached!\n"
    if total_output_tokens > max_out_tokens:
        error_txt += "Maximum number of output tokens reached!\n"

    if final_success:
        typer.echo('✅ Result: Success')
    else:
        typer.echo(f'❌ Result: {error_txt}')

    typer.echo("-"*20)
    sandbox.close()

    task_solution = SolutionOutput(
        task_id=str(the_task.instance_id),
        benchmark='swebench',
        success=final_success,
        solution=solution,
        iterations=llm_step_number,
        steps=steps,
        total_requests=0,  # !!!
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_time_seconds=duration,
        system_prompt=system_prompt,
        error=error_txt,
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
