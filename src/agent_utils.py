import json
from typing import Any


def mpc_arguments_to_python(arg_json: str) -> str:
    """ Convert list of function argument from json in Python style """

    data: dict[str, Any] = json.loads(arg_json)
    arg_str = ""
    if data:
        for d_key in data.keys():
            if arg_str: 
               arg_str += ", "
            d = data.get(d_key, None)
            if d is None:
                arg_str += d_key + "=None"
            else:
                arg_str += d_key +"="+json.dumps(d)
    return arg_str