# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import json
import urllib.error
from unittest.mock import MagicMock, patch

from demo_vllm_backend import chat, complete, openai_tool_schemas, parse_models_response, probe


def _fake_response(payload: dict):
    cm = MagicMock()
    cm.__enter__.return_value = cm
    cm.read.return_value = json.dumps(payload).encode()
    return cm


def test_parse_models_response_extracts_first_model_id():
    assert parse_models_response({"data": [{"id": "episod/tt-tnt-1024"}]}) == "episod/tt-tnt-1024"


def test_parse_models_response_handles_empty_data_list():
    assert parse_models_response({"data": []}) is None


def test_parse_models_response_handles_missing_data_key():
    assert parse_models_response({}) is None


def test_probe_reports_unreachable_on_connection_error():
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        result = probe()
    assert result.reachable is False
    assert result.served_model_id is None


def test_probe_reports_served_model_id_when_reachable():
    fake = _fake_response({"data": [{"id": "episod/tt-tnt-1024"}]})
    with patch("urllib.request.urlopen", return_value=fake):
        result = probe()
    assert result.reachable is True
    assert result.served_model_id == "episod/tt-tnt-1024"


def test_complete_returns_the_first_choice_text():
    fake = _fake_response({"choices": [{"text": "a story continues"}]})
    with patch("urllib.request.urlopen", return_value=fake):
        assert complete("Once upon a time") == "a story continues"


def _captured_request_payload(fake_response, call, **kwargs):
    """Call `call(**kwargs)` against a mocked urlopen, capturing the actual
    urllib.request.Request object that was sent, and return its decoded JSON body."""
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["request"] = request
        return fake_response

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        call(**kwargs)
    return json.loads(captured["request"].data.decode())


def test_complete_sends_the_real_model_id_in_the_request_payload():
    fake = _fake_response({"choices": [{"text": "a story continues"}]})
    payload = _captured_request_payload(
        fake, complete, prompt="Once upon a time", model="episod/tt-tnt-1024",
    )
    assert payload["model"] == "episod/tt-tnt-1024"


def test_complete_sends_an_empty_model_string_when_none_given():
    fake = _fake_response({"choices": [{"text": "a story continues"}]})
    payload = _captured_request_payload(fake, complete, prompt="Once upon a time")
    assert payload["model"] == ""


def test_complete_never_sends_the_literal_string_default():
    fake = _fake_response({"choices": [{"text": "a story continues"}]})
    payload = _captured_request_payload(
        fake, complete, prompt="Once upon a time", model="episod/tt-tnt-1024",
    )
    assert payload["model"] != "default"


def test_chat_sends_the_real_model_id_in_the_request_payload():
    fake = _fake_response({"choices": [{"message": {"content": "hi"}}]})
    payload = _captured_request_payload(
        fake, chat, messages=[{"role": "user", "content": "hi"}], model="episod/tt-tnt-1024",
    )
    assert payload["model"] == "episod/tt-tnt-1024"


def test_chat_sends_an_empty_model_string_when_none_given():
    fake = _fake_response({"choices": [{"message": {"content": "hi"}}]})
    payload = _captured_request_payload(
        fake, chat, messages=[{"role": "user", "content": "hi"}],
    )
    assert payload["model"] == ""


def test_chat_payload_includes_tools_only_when_tools_are_passed():
    fake = _fake_response({"choices": [{"message": {"content": "hi"}}]})
    tools = [{"type": "function", "function": {"name": "witty_response"}}]
    with_tools = _captured_request_payload(
        fake, chat, messages=[{"role": "user", "content": "hi"}], tools=tools,
    )
    assert with_tools["tools"] == tools
    assert with_tools["tool_choice"] == "auto"

    without_tools = _captured_request_payload(
        fake, chat, messages=[{"role": "user", "content": "hi"}],
    )
    assert "tools" not in without_tools
    assert "tool_choice" not in without_tools


def test_probe_returns_unreachable_when_response_has_invalid_json():
    fake = MagicMock()
    fake.__enter__.return_value = fake
    fake.read.return_value = b"not json"
    with patch("urllib.request.urlopen", return_value=fake):
        result = probe()
    assert result.reachable is False
    assert result.served_model_id is None
    assert "JSONDecodeError" in result.error or "Expecting value" in result.error


def test_complete_raises_runtime_error_on_http_error():
    url = "http://localhost:8000/v1/completions"
    exc = urllib.error.HTTPError(url, 400, "Bad Request", {}, None)
    with patch("urllib.request.urlopen", side_effect=exc):
        try:
            complete("test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "HTTP 400" in str(e)
            assert url in str(e)


def test_openai_tool_schemas_derives_all_four_tools_from_train_tool_calling():
    schemas = openai_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {
        "factual_response", "witty_response", "absurdist_response", "misunderstood_question",
    }


def test_openai_tool_schemas_includes_enum_constraints():
    schemas = openai_tool_schemas()
    witty = next(s for s in schemas if s["function"]["name"] == "witty_response")
    assert witty["function"]["parameters"]["properties"]["technique"]["enum"] == [
        "pun", "wordplay", "reference",
    ]
    factual = next(s for s in schemas if s["function"]["name"] == "factual_response")
    assert "enum" not in factual["function"]["parameters"]["properties"]["answer"]
