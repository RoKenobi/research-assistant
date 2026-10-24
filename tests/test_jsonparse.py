from core.jsonparse import parse_json


def test_parse_json_handles_brackets_inside_strings():
    assert parse_json('Sure: ["RT-2 [v2]", "Octo"]', list) == ["RT-2 [v2]", "Octo"]


def test_parse_json_returns_first_list_as_is():
    assert parse_json('["Octo", "octo", 3, "", "Pi0"]', list) == ["Octo", "octo", 3, "", "Pi0"]


def test_parse_json_returns_none_on_garbage():
    assert parse_json("no json [here", list) is None
    assert parse_json("nothing at all", dict) is None


def test_parse_json_object_in_code_fence():
    assert parse_json('Here you go:\n```json\n{"a": 1}\n```', dict) == {"a": 1}


def test_parse_json_skips_wrong_type():
    assert parse_json('[1] then {"a": 2}', dict) == {"a": 2}
    assert parse_json('{"a": 2} then [1]', list) == [1]
