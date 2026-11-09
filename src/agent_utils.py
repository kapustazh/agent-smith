import json
from typing import Any


def mpc_arguments_to_python(arg_json: str) -> str:
    """ Convert list of function argument from json in Python style """

    if not arg_json:
        return ""

    arg_json = arg_json.strip()
    if not arg_json:
        return ""

    arg_str = ""
    data: dict[str, Any] = {}
    try:
        data = json.loads(arg_json)
    except Exception:
        if arg_json[-2:] == "}}":
            arg_json_ = arg_json[:-1]
        elif arg_json[-4:] == "}}\n]":
            arg_json_ = arg_json[:-3]
    if not data:
        try:
            data = json.loads(arg_json_)
        except Exception:
            pass
    if not data:
        start = arg_json.find("{")
        if start == -1:
            print(f"Error convert json to dict: json='{arg_json}'"
                  ', error=Not fond "{" ')
            return ""
        depth, end = 0, 0
        in_str = False
        for i, ch in enumerate(arg_json[start:], start):
            if ch == '"' and arg_json[i-1] != '\\':
                in_str = not in_str
            elif not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end == 0:
            print(f"Error convert json to dict: json='{arg_json}'")
            return ""
        try:
            data = json.loads(arg_json[start:end])
        except Exception as ex:
            print(f"Error convert json to dict: json='{arg_json}', error={ex}")
            return ""

    if data:
        for d_key in data.keys():
            if arg_str:
                arg_str += ", "
            d = data.get(d_key, None)
            if d is None:
                arg_str += d_key + "=None"
            else:
                arg_str += d_key + "=" + json.dumps(d)
    return arg_str
