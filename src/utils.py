import re
import json
from typing import Dict, List


def trim_beginning_of_string(input_string: str, delimiter: str) -> str:
    """
    Trims the beginning of a string until a substring is found.
    """
    start_index = input_string.find(delimiter)
    if start_index == -1:
        return input_string
    else:
        return input_string[start_index:]

def drop_first_line(s: str) -> str:
    index_of_first_newline = s.find("\n")
    return s[index_of_first_newline + 1:]


def sanitize_json_str_with_backticks(s: str) -> str:
    trimmed = trim_beginning_of_string(s, "```")
    trimmed = trimmed.replace("```json", "")
    trimmed = trimmed.replace("```", "")
    return trimmed


def remove_unescaped(s):
    # Remove any unescaped control characters
    s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    return s

def try_to_parse_json(s):
    """
    Tries to parse a string as JSON by sequentially applying different transformations.
    """

    s = sanitize_json_str_with_backticks(s)
    s = trim_beginning_of_string(s, "{")

    # TODO: BORRAR COMENTARIOS

    try:
        return json.loads(s)
    except:
        print(f"Failed to parse: \n{s}")
        pass
    s = remove_unescaped(s)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"Failed to parse: \n{s}")
        raise e

def find_first_and_last_braces(s):
    index_first = s.find("{")
    index_last = s.rfind("}")
    return index_first, index_last

def try_to_find_and_eval_dict(input_string: str) -> Dict:
    index_first, index_last = find_first_and_last_braces(input_string)
    trimmed = input_string[index_first:index_last+1]
    try:
        return eval(trimmed)
    except:
        raise ValueError(f"Failed to evaluate the following string:\n{trimmed}")