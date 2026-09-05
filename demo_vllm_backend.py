# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""HTTP client for the Gradio demo's optional "live on Blackhole" mode, mirroring
scripts/story_tools.py's request pattern (stdlib urllib, no new dependency). This
module never opens a Tenstorrent device itself -- it only talks to a vLLM server the
user already launched separately via `tt-model serve`
(docs/serving-with-tt-kernel.md). `probe()` is what lets app.py show "vLLM not
reachable" instead of silently falling back to CPU and looking like a live result.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_BASE = "http://localhost:8000"
_PROBE_TIMEOUT = 2.0


@dataclass(frozen=True)
class ProbeResult:
    reachable: bool
    served_model_id: Optional[str] = None
    error: Optional[str] = None


def _get(url: str, timeout: float) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post(url: str, payload: dict, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:500]}") from exc


def parse_models_response(data: Dict[str, Any]) -> Optional[str]:
    """Pull the first served model id out of a `/v1/models` response body, or None if
    the shape doesn't match what vLLM returns."""
    entries = data.get("data")
    if not entries:
        return None
    first = entries[0]
    if not isinstance(first, dict):
        return None
    model_id = first.get("id")
    return model_id if isinstance(model_id, str) else None


def probe(base: str = DEFAULT_BASE, *, timeout: float = _PROBE_TIMEOUT) -> ProbeResult:
    try:
        data = _get(f"{base}/v1/models", timeout)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        return ProbeResult(reachable=False, error=str(exc))
    return ProbeResult(reachable=True, served_model_id=parse_models_response(data))


def complete(prompt: str, *, model: Optional[str] = None, base: str = DEFAULT_BASE,
             max_tokens: int = 60, temperature: float = 0.8, top_p: float = 0.95,
             timeout: float = 60.0) -> str:
    payload = {
        "model": model or "", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p,
    }
    data = _post(f"{base}/v1/completions", payload, timeout)
    return data["choices"][0]["text"]


def chat(messages: List[Dict[str, str]], *, tools: Optional[List[Dict[str, Any]]] = None,
         model: Optional[str] = None, base: str = DEFAULT_BASE, max_tokens: int = 80,
         temperature: float = 0.8, timeout: float = 60.0) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model or "", "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return _post(f"{base}/v1/chat/completions", payload, timeout)


def openai_tool_schemas() -> List[Dict[str, Any]]:
    """Build OpenAI-style tool schemas straight from train.tool_calling.TOOLS, so the
    demo's vLLM request always matches what the model was actually trained on -- never
    a hand-duplicated copy of the schema that can drift out of sync."""
    from train.tool_calling import TOOLS

    schemas = []
    for name, spec in TOOLS.items():
        required = spec["required_args"]
        enum_args = spec["enum_args"]
        properties: Dict[str, Any] = {}
        for arg in required:
            prop: Dict[str, Any] = {"type": "string"}
            if arg in enum_args:
                prop["enum"] = list(enum_args[arg])
            properties[arg] = prop
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required),
                },
            },
        })
    return schemas
