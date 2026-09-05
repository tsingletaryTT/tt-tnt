# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import json
import urllib.error
from unittest.mock import MagicMock, patch

from demo_vllm_backend import complete, parse_models_response, probe


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
