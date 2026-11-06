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
            arg_json = arg_json[:-1]

    if not data:
        try:
            data = json.loads(arg_json)
        except Exception as ex:
            print(f"Error convert json to dict: json='{arg_json}',error={ex}")
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
