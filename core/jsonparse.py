import json

_OPENERS = {list: "[", dict: "{"}


def parse_json(text: str, expect: type[list] | type[dict]) -> list | dict | None:
    """Return the first JSON value of type `expect` found in a model reply.

    Models wrap JSON in prose or code fences; raw_decode from each candidate
    opening bracket finds the value without caring what surrounds it, and
    brackets inside strings ("RT-2 [v2]") don't confuse it.
    """
    opener = _OPENERS[expect]
    decoder = json.JSONDecoder()
    start = text.find(opener)
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, expect):
                return value
        start = text.find(opener, start + 1)
    return None
