# tt-tnt Gradio Demo + tt-discolike Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Gradio demo app (`app.py`) showing off tt-tnt's chat/completion
behavior, its tool-calling roles, its known failure modes, and its historical research
findings, discoverable locally via a `.disco/app.yaml` manifest and portable to a
HuggingFace Space with no fork.

**Architecture:** Three small backend modules (`demo_checkpoints.py` for
checkpoint/Hub-repo resolution, `demo_hf_backend.py` for CPU-direct generation with a
small LRU model cache, `demo_vllm_backend.py` for an optional HTTP client to an
already-running vLLM server) plus `demo_findings.py` for reading committed measurement
JSON, all wired together by one `app.py` with four Gradio tabs. A one-time CLI
(`scripts/prepare_demo_checkpoints.py`) converts the two `.pkl`-only checkpoints the
demo needs into HF format.

**Tech Stack:** Python 3.10+, `gradio>=4.44,<5`, `transformers`, `torch` (CPU),
`huggingface_hub`. No `ttnn`/`ttml` import anywhere in this feature — the demo process
never opens a Tenstorrent device.

**Spec:** [docs/superpowers/specs/2026-09-04-gradio-demo-design.md](../specs/2026-09-04-gradio-demo-design.md)

## Global Constraints

- The demo process (`app.py`) never imports `ttnn`/`ttml` and never opens a
  Tenstorrent device. vLLM mode is a passive HTTP client only.
- Port **7862** (7860/7861 are already claimed by `tt-animatediff`/`tt-vjepa2`'s
  `.disco/app.yaml` manifests).
- `.disco/app.yaml` carries **no `chips:` field** — see spec's manifest section.
- Every checkpoint/file lookup degrades gracefully (missing dropdown entry, disabled
  tab, clear error text) — never a crash, never a silent fallback that looks like
  success.
- `demo_hf_backend.generate()` mirrors `scripts/chat.py`'s exact generation call
  (context clamping, `pad_token_id=tok.pad_token_id`, returning only the new
  continuation) rather than inventing new generation parameters.
- `demo_vllm_backend`'s HTTP calls use stdlib `urllib` only, mirroring
  `scripts/story_tools.py` — no new HTTP client dependency.
- One `app.py` must work unmodified on a bare local clone (fewer checkpoints, so
  fewer dropdown entries) and on a HuggingFace Space (checkpoint resolution falls back
  to the published Hub repo id; vLLM `probe()` naturally reports unreachable).

---

## Task 1: `demo_checkpoints.py` — checkpoint/Hub-repo registry

**Files:**
- Create: `demo_checkpoints.py`
- Test: `tests/test_demo_checkpoints.py`

**Interfaces:**
- Produces: `CheckpointEntry` (dataclass: `label: str`, `local_dir: Path`,
  `hub_repo_id: Optional[str]`), `CHECKPOINTS: List[CheckpointEntry]`,
  `resolve(label: str, checkpoints: Optional[List[CheckpointEntry]] = None) ->
  Optional[str]`, `list_available(checkpoints: Optional[List[CheckpointEntry]] = None)
  -> List[str]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_demo_checkpoints.py
from pathlib import Path

from demo_checkpoints import CheckpointEntry, list_available, resolve


def _entry(label, tmp_path, name, config_present, hub_repo_id=None):
    d = tmp_path / name
    d.mkdir()
    if config_present:
        (d / "config.json").write_text("{}")
    return CheckpointEntry(label, d, hub_repo_id)


def test_resolve_returns_local_dir_when_config_json_present(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=True)]
    assert resolve("a", entries) == str(tmp_path / "a")


def test_resolve_falls_back_to_hub_repo_when_local_dir_missing_config(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=False, hub_repo_id="org/model")]
    assert resolve("a", entries) == "org/model"


def test_resolve_returns_none_when_neither_local_nor_hub_available(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=False)]
    assert resolve("a", entries) is None


def test_resolve_returns_none_for_unknown_label(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=True)]
    assert resolve("does-not-exist", entries) is None


def test_list_available_includes_only_resolvable_labels(tmp_path):
    entries = [
        _entry("has-local", tmp_path, "x", config_present=True),
        _entry("has-hub-fallback", tmp_path, "y", config_present=False, hub_repo_id="org/m"),
        _entry("has-neither", tmp_path, "z", config_present=False),
    ]
    assert list_available(entries) == ["has-local", "has-hub-fallback"]


def test_default_checkpoints_list_is_a_nonempty_list_of_entries():
    from demo_checkpoints import CHECKPOINTS
    assert len(CHECKPOINTS) > 0
    assert all(isinstance(e, CheckpointEntry) for e in CHECKPOINTS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_demo_checkpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo_checkpoints'`

- [ ] **Step 3: Write the implementation**

```python
# demo_checkpoints.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint discovery/registry for the Gradio demo (app.py).

Resolves a friendly label to either a local `artifacts/hf-*` directory or, if that
directory has no `config.json` (missing entirely, or present but not yet converted), a
published Hub repo id. This is what lets the SAME registry work unmodified on: a bare
local clone (fewer converted checkpoints -> fewer dropdown entries), a machine with
every experimental checkpoint converted, and a HuggingFace Space (only the published
label resolves at all, via its Hub repo id -- see this project's
docs/superpowers/specs/2026-09-04-gradio-demo-design.md, "Portability: HuggingFace
Spaces").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


@dataclass(frozen=True)
class CheckpointEntry:
    label: str
    local_dir: Path
    hub_repo_id: Optional[str]  # None if this checkpoint was never published


#: Only checkpoints this demo actually offers. Adding a new tab's checkpoint means
#: adding an entry here, not scattering a hardcoded path through app.py.
CHECKPOINTS: List[CheckpointEntry] = [
    CheckpointEntry(
        "tt-tnt-1024 (production)", ARTIFACTS / "hf-tt-tnt-1024", "episod/tt-tnt-1024",
    ),
    CheckpointEntry(
        "tool-calling-s2", ARTIFACTS / "hf-tt-tnt-1024-tool-calling-s2", None,
    ),
    CheckpointEntry(
        "tool-calling-s3", ARTIFACTS / "hf-tt-tnt-1024-tool-calling-s3", None,
    ),
    CheckpointEntry(
        "editor-blend (broken run)", ARTIFACTS / "hf-tt-tnt-1024-editor-blend", None,
    ),
]


def _is_valid_local_dir(path: Path) -> bool:
    return (path / "config.json").is_file()


def resolve(label: str, checkpoints: Optional[List[CheckpointEntry]] = None) -> Optional[str]:
    """What to pass to `from_pretrained` for `label`: the local directory (as a str) if
    it has a real `config.json`, else the label's Hub repo id if it has one, else None
    if this label can't be satisfied at all right now."""
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    for entry in checkpoints:
        if entry.label != label:
            continue
        if _is_valid_local_dir(entry.local_dir):
            return str(entry.local_dir)
        return entry.hub_repo_id
    return None


def list_available(checkpoints: Optional[List[CheckpointEntry]] = None) -> List[str]:
    """Labels `resolve()` can actually satisfy right now, in registry order. Never
    includes a label with neither a valid local directory nor a Hub repo id."""
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    return [e.label for e in checkpoints if resolve(e.label, checkpoints) is not None]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_checkpoints.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add demo_checkpoints.py tests/test_demo_checkpoints.py
git commit -m "feat(demo): add checkpoint/Hub-repo registry for the Gradio demo"
```

---

## Task 2: `demo_hf_backend.py` — CPU-direct generation with a small LRU cache

**Files:**
- Create: `demo_hf_backend.py`
- Test: `tests/test_demo_hf_backend.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `ModelCache` (class: `__init__(self, load_fn, maxsize=2)`, `.get(key) ->
  object`, `.loaded_keys() -> Tuple[str, ...]`), `generate(model_path_or_repo: str,
  prompt: str, *, max_new_tokens=60, temperature=0.8, top_p=0.95) -> str` (raises on
  failure; callers catch it).

- [ ] **Step 1: Write the failing tests for `ModelCache`**

```python
# tests/test_demo_hf_backend.py
import pytest

from demo_hf_backend import ModelCache


def test_cache_hit_does_not_call_load_fn_again():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("a")
    assert calls == ["a"]


def test_cache_evicts_least_recently_used_when_full():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("b")
    cache.get("c")  # should evict "a", the least recently used
    assert cache.loaded_keys() == ("b", "c")


def test_touching_a_key_protects_it_from_eviction():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("b")
    cache.get("a")  # touch "a" again, making "b" the least recently used
    cache.get("c")  # should evict "b", not "a"
    assert cache.loaded_keys() == ("a", "c")


def test_get_on_an_evicted_key_reloads_it():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("b")
    cache.get("c")  # evicts "a"
    cache.get("a")  # must reload
    assert calls == ["a", "b", "c", "a"]


def test_maxsize_must_be_at_least_one():
    with pytest.raises(ValueError):
        ModelCache(lambda k: k, maxsize=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_demo_hf_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo_hf_backend'`

- [ ] **Step 3: Write the implementation**

```python
# demo_hf_backend.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""CPU-direct load-and-generate backend for the Gradio demo, mirroring
scripts/chat.py's own loading and generation pattern exactly (context clamping,
`pad_token_id=tok.pad_token_id`, returning only the new continuation) -- this project's
own docs are explicit that CPU-direct is currently the *reference*-quality path, not a
degraded fallback, so there is no reason to invent a different generation recipe here.

`ModelCache` caches loaded (model, tokenizer) pairs so switching checkpoints in the UI
doesn't reload from disk on every click, while bounding memory with least-recently-used
eviction. `load_fn` is injected so it can be tested with no real model at all.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Tuple


class ModelCache:
    def __init__(self, load_fn: Callable[[str], object], maxsize: int = 2):
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._load_fn = load_fn
        self._maxsize = maxsize
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def get(self, key: str) -> object:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        value = self._load_fn(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return value

    def loaded_keys(self) -> Tuple[str, ...]:
        return tuple(self._cache.keys())


def _load_model_and_tokenizer(model_path_or_repo: str):
    """Real loader. Not unit tested directly (needs a real checkpoint or network
    access to the Hub) -- exercised manually per this feature's testing section, the
    same way scripts/chat.py itself has no automated test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path_or_repo)
    model = AutoModelForCausalLM.from_pretrained(model_path_or_repo).eval()
    return model, tok


_CACHE = ModelCache(_load_model_and_tokenizer, maxsize=2)


def generate(model_path_or_repo: str, prompt: str, *, max_new_tokens: int = 60,
             temperature: float = 0.8, top_p: float = 0.95) -> str:
    """Generate a continuation for `prompt`, returning ONLY the newly generated text
    (scripts/chat.py prints just the continuation, not the echoed prompt -- matched
    here). Raises ValueError if the prompt already fills the model's context, and
    propagates any transformers/torch error -- app.py's callers catch these and show
    them in the output box rather than letting the process crash."""
    import torch

    model, tok = _CACHE.get(model_path_or_repo)
    ctx = model.config.max_position_embeddings
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] >= ctx:
        raise ValueError(f"prompt is {ids.shape[1]} tokens; context is {ctx}. Shorten it.")
    greedy = temperature <= 0
    room = min(max_new_tokens, ctx - ids.shape[1])
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=room,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_hf_backend.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add demo_hf_backend.py tests/test_demo_hf_backend.py
git commit -m "feat(demo): add CPU-direct generation backend with an LRU model cache"
```

---

## Task 3: `demo_vllm_backend.py` — optional vLLM HTTP client

**Files:**
- Create: `demo_vllm_backend.py`
- Test: `tests/test_demo_vllm_backend.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `ProbeResult` (dataclass: `reachable: bool`, `served_model_id:
  Optional[str] = None`, `error: Optional[str] = None`), `parse_models_response(data:
  dict) -> Optional[str]`, `probe(base=DEFAULT_BASE, *, timeout=2.0) -> ProbeResult`,
  `complete(prompt, *, base=DEFAULT_BASE, max_tokens=60, temperature=0.8, top_p=0.95,
  timeout=60.0) -> str`, `chat(messages, *, tools=None, base=DEFAULT_BASE,
  max_tokens=80, temperature=0.8, timeout=60.0) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_demo_vllm_backend.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_demo_vllm_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo_vllm_backend'`

- [ ] **Step 3: Write the implementation**

```python
# demo_vllm_backend.py
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
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


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
    except (urllib.error.URLError, OSError) as exc:
        return ProbeResult(reachable=False, error=str(exc))
    return ProbeResult(reachable=True, served_model_id=parse_models_response(data))


def complete(prompt: str, *, base: str = DEFAULT_BASE, max_tokens: int = 60,
             temperature: float = 0.8, top_p: float = 0.95, timeout: float = 60.0) -> str:
    payload = {
        "model": "default", "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": top_p,
    }
    data = _post(f"{base}/v1/completions", payload, timeout)
    return data["choices"][0]["text"]


def chat(messages: List[Dict[str, str]], *, tools: Optional[List[Dict[str, Any]]] = None,
         base: str = DEFAULT_BASE, max_tokens: int = 80, temperature: float = 0.8,
         timeout: float = 60.0) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": "default", "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return _post(f"{base}/v1/chat/completions", payload, timeout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_vllm_backend.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add demo_vllm_backend.py tests/test_demo_vllm_backend.py
git commit -m "feat(demo): add optional vLLM HTTP client with a reachability probe"
```

---

## Task 4: `scripts/prepare_demo_checkpoints.py` — one-time checkpoint conversion

**Files:**
- Create: `scripts/prepare_demo_checkpoints.py`
- Test: `tests/test_prepare_demo_checkpoints.py`

**Interfaces:**
- Consumes: `scripts.eval_improv.sft_checkpoint_to_hf(step_pkl, *, warm_start_ckpt,
  tokenizer_dir, out_dir)` (existing, `scripts/eval_improv.py:216`).
- Produces: `latest_sft_checkpoint(checkpoint_dir: Path) -> Optional[Path]`,
  `already_converted(out_dir: Path) -> bool`, `main() -> int` (CLI entry point).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prepare_demo_checkpoints.py
from scripts.prepare_demo_checkpoints import already_converted, latest_sft_checkpoint


def test_latest_sft_checkpoint_picks_highest_step(tmp_path):
    (tmp_path / "step_250.pkl").touch()
    (tmp_path / "step_3000.pkl").touch()
    (tmp_path / "step_1000.pkl").touch()
    assert latest_sft_checkpoint(tmp_path) == tmp_path / "step_3000.pkl"


def test_latest_sft_checkpoint_returns_none_when_empty(tmp_path):
    assert latest_sft_checkpoint(tmp_path) is None


def test_latest_sft_checkpoint_ignores_ttml_naming(tmp_path):
    # ttml's own pretrain naming (tt_tnt_step<N>.pkl) is a DIFFERENT format this
    # function must not mistake for an SFT checkpoint.
    (tmp_path / "tt_tnt_step00000500.pkl").touch()
    assert latest_sft_checkpoint(tmp_path) is None


def test_already_converted_true_when_config_json_present(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert already_converted(tmp_path) is True


def test_already_converted_false_when_directory_missing_config(tmp_path):
    assert already_converted(tmp_path) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prepare_demo_checkpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.prepare_demo_checkpoints'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/prepare_demo_checkpoints.py
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""One-time CLI: convert the two SFT `.pkl` checkpoints the Gradio demo (app.py) wants
but that have no HF conversion on disk -- tool-calling-s3 (the corrected, best-selected
tool-calling run) and editor-blend (the ORIGINAL broken run, kept specifically for its
dramatic single-word repeat-loop collapse in the demo's "Known Limitations" tab).
Idempotent: skips a checkpoint whose output directory already has a config.json.

    python scripts/prepare_demo_checkpoints.py

CPU-only -- no ttnn/ttml import, no device, same as every other conversion path in
this project (see scripts/eval_improv.py's sft_checkpoint_to_hf, which does the actual
conversion work this script just calls with the right paths).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Used ONLY to recover an architecture header (vocab_size, seq_len,
#: transformer_config, ...) for HF conversion -- every SFT run this demo converts is
#: dense and this same 1024 shape, so this checkpoint's header describes them all
#: exactly. Its own weights are never used (see scripts/eval_improv.py's own comment
#: on WARM_START_CKPT for the precedent this follows).
WARM_START_CKPT = ROOT / "artifacts" / "checkpoints-v077-beta2-control" / "tt_tnt_step00010764.pkl"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024"

_STEP_RE = re.compile(r"^step_(\d+)\.pkl$")


class DemoConversion(NamedTuple):
    checkpoint_dir: Path
    out_dir: Path


CONVERSIONS = [
    DemoConversion(
        ROOT / "artifacts" / "checkpoints-1024-tool-calling-s3",
        ROOT / "artifacts" / "hf-tt-tnt-1024-tool-calling-s3",
    ),
    DemoConversion(
        ROOT / "artifacts" / "checkpoints-1024-editor-blend",
        ROOT / "artifacts" / "hf-tt-tnt-1024-editor-blend",
    ),
]


def latest_sft_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Highest-step `step_<N>.pkl` in `checkpoint_dir`, or None if there are none.

    SFTTrainer checkpoints use `step_<N>.pkl`, a different naming scheme from ttml's
    own pretrain checkpoints (`tt_tnt_step<N>.pkl`, handled by
    train.checkpoint.latest_checkpoint) -- that function looks for the wrong prefix
    entirely and would silently find nothing in an SFT checkpoint directory.
    """
    best_step = -1
    best_path: Optional[Path] = None
    if not checkpoint_dir.is_dir():
        return None
    for path in checkpoint_dir.glob("step_*.pkl"):
        m = _STEP_RE.match(path.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def already_converted(out_dir: Path) -> bool:
    return (out_dir / "config.json").is_file()


def main() -> int:
    from scripts.eval_improv import sft_checkpoint_to_hf

    if not WARM_START_CKPT.is_file():
        print(f"ERROR: warm-start header checkpoint missing: {WARM_START_CKPT}", file=sys.stderr)
        return 1

    did_work = False
    for conv in CONVERSIONS:
        if already_converted(conv.out_dir):
            print(f"skip {conv.out_dir} (already converted)")
            continue
        step_pkl = latest_sft_checkpoint(conv.checkpoint_dir)
        if step_pkl is None:
            print(f"skip {conv.checkpoint_dir} (no step_*.pkl found -- not on this machine)")
            continue
        print(f"converting {step_pkl} -> {conv.out_dir} ...")
        sft_checkpoint_to_hf(
            step_pkl, warm_start_ckpt=WARM_START_CKPT, tokenizer_dir=TOKENIZER_DIR,
            out_dir=conv.out_dir,
        )
        did_work = True
        print(f"  done: {conv.out_dir}/config.json")

    if not did_work:
        print("nothing to do -- all demo checkpoints already converted or unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prepare_demo_checkpoints.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the script for real on this machine**

Run: `python scripts/prepare_demo_checkpoints.py`
Expected: prints `converting ... -> artifacts/hf-tt-tnt-1024-tool-calling-s3/ ...` and
`converting ... -> artifacts/hf-tt-tnt-1024-editor-blend/ ...`, each followed by a
`done:` line. Confirm both directories now have a `config.json`:

Run: `test -f artifacts/hf-tt-tnt-1024-tool-calling-s3/config.json && test -f artifacts/hf-tt-tnt-1024-editor-blend/config.json && echo OK`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_demo_checkpoints.py tests/test_prepare_demo_checkpoints.py
git commit -m "feat(demo): add one-time CLI to convert the demo's two missing checkpoints"
```

(`artifacts/` is gitignored — the newly-converted directories themselves are not
committed, only the script that produces them.)

---

## Task 5: `demo_findings.py` — historical research-findings reader

**Files:**
- Create: `demo_findings.py`
- Test: `tests/test_demo_findings.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-4.
- Produces: `FindingCard` (dataclass: `title: str`, `json_path: Path`, `verdict_path:
  Tuple[str, ...]`), `FINDING_CARDS: List[FindingCard]`, `load_findings(cards:
  Optional[List[FindingCard]] = None) -> List[Dict[str, Optional[str]]]` (each dict has
  keys `title`, `verdict`, `source`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_demo_findings.py
import json

from demo_findings import FindingCard, load_findings


def test_load_findings_extracts_verdict_via_dotted_path(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"verdict": "PARTIAL"}))
    cards = [FindingCard("A", p, ("verdict",))]
    result = load_findings(cards)
    assert result == [{"title": "A", "verdict": "PARTIAL", "source": str(p)}]


def test_load_findings_walks_a_nested_dotted_path(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"outer": {"inner": "some verdict text"}}))
    cards = [FindingCard("B", p, ("outer", "inner"))]
    result = load_findings(cards)
    assert result[0]["verdict"] == "some verdict text"


def test_load_findings_skips_a_missing_file(tmp_path):
    cards = [FindingCard("Missing", tmp_path / "nope.json", ("verdict",))]
    assert load_findings(cards) == []


def test_load_findings_skips_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    cards = [FindingCard("Bad", p, ("verdict",))]
    assert load_findings(cards) == []


def test_load_findings_reports_none_when_path_does_not_resolve_to_a_string(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"verdict": {"nested": "dict, not a string"}}))
    cards = [FindingCard("C", p, ("verdict",))]
    result = load_findings(cards)
    assert result[0]["verdict"] is None


def test_default_finding_cards_is_a_nonempty_list():
    from demo_findings import FINDING_CARDS
    assert len(FINDING_CARDS) > 0
    assert all(isinstance(c, FindingCard) for c in FINDING_CARDS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_demo_findings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo_findings'`

- [ ] **Step 3: Write the implementation**

```python
# demo_findings.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Reads this project's own committed docs/measurements/*.json verdicts for the Gradio
demo's "Research Findings" tab. No live inference here -- the checkpoints these files
describe (reach-dial, skits stage 2, the tool-calling/LoRA ablations) are no longer
available to serve locally, so this tab quotes the real, already-published numbers
instead of trying to reproduce them.

Each measurement file was written by a different evaluation script over the course of
this project and has its own ad-hoc JSON shape -- there is no shared "verdict" field
name across them (docs/superpowers/specs/2026-09-04-gradio-demo-design.md's Tab 4
section). Rather than guess at a generic field name, FINDING_CARDS hand-picks the exact
dotted path to a real, human-readable verdict string in each specific file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
MEASUREMENTS = ROOT / "docs" / "measurements"


@dataclass(frozen=True)
class FindingCard:
    title: str
    json_path: Path
    verdict_path: Tuple[str, ...]


FINDING_CARDS: List[FindingCard] = [
    FindingCard(
        "Reach dial (controllable generation)",
        MEASUREMENTS / "reach-dial.json",
        ("headline", "dial_kind_verdict"),
    ),
    FindingCard(
        "Skits stage 2 (five-slot turn structure)",
        MEASUREMENTS / "skits-stage2.json",
        ("verdict",),
    ),
    FindingCard(
        "Tool-calling checkpoint selection",
        MEASUREMENTS / "tool-calling-s3-selection.json",
        ("why",),
    ),
    FindingCard(
        "LoRA vs full-parameter fine-tuning",
        MEASUREMENTS / "lora-vs-full-tool-calling.json",
        ("refuted_prediction",),
    ),
    FindingCard(
        "LoRA anti-forgetting prediction",
        MEASUREMENTS / "evaluation-tt-tnt-1024-vs-editor-lora.json",
        ("prediction_refuted_2026_08_31", "prediction"),
    ),
]


def _walk(data: Any, path: Tuple[str, ...]) -> Optional[str]:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data if isinstance(data, str) else None


def load_findings(cards: Optional[List[FindingCard]] = None) -> List[Dict[str, Optional[str]]]:
    """One dict per card whose JSON file exists and parses, in card order. A card
    whose file is missing (a leaner clone) is silently omitted; a card whose file
    exists but whose verdict_path doesn't resolve to a string still appears, with
    verdict=None, so a schema drift is visible rather than silently disappearing."""
    cards = FINDING_CARDS if cards is None else cards
    result: List[Dict[str, Optional[str]]] = []
    for card in cards:
        if not card.json_path.is_file():
            continue
        try:
            data = json.loads(card.json_path.read_text())
        except json.JSONDecodeError:
            continue
        result.append({
            "title": card.title,
            "verdict": _walk(data, card.verdict_path),
            "source": str(card.json_path),
        })
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_findings.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add demo_findings.py tests/test_demo_findings.py
git commit -m "feat(demo): add reader for committed research-findings JSON"
```

---

## Task 6: `app.py` skeleton + Tab 1 (Story Completion / Chat)

**Files:**
- Create: `app.py`
- Modify: `pyproject.toml` (add `ui` optional-dependency group)

**Interfaces:**
- Consumes: `demo_checkpoints.list_available()`, `demo_checkpoints.resolve()`,
  `demo_hf_backend.generate()`, `demo_vllm_backend.probe()`,
  `demo_vllm_backend.complete()`.
- Produces: `build_app() -> gr.Blocks`, `_cpu_generate(label, prompt, max_new_tokens,
  temperature) -> str` (used by later tasks' tabs too).

- [ ] **Step 1: Add the `ui` optional-dependency group**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
ui = ["gradio>=4.44,<5"]
```

- [ ] **Step 2: Install it**

Run: `pip install -e ".[ui]"`
Expected: installs `gradio` and its dependencies with no error.

- [ ] **Step 3: Write `app.py` with the skeleton and Tab 1**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Gradio demo for tt-tnt: chat, tool-calling roles, known limitations, and research
findings. CPU-only process -- never opens a Tenstorrent device itself; the optional
"vLLM server" mode is a passive HTTP client to a server you launch separately (see
docs/serving-with-tt-kernel.md). Works unmodified on a bare local clone (fewer
converted checkpoints -> fewer dropdown entries) and on a HuggingFace Space (see
docs/superpowers/specs/2026-09-04-gradio-demo-design.md's "Portability" section).

    python app.py
    # open http://localhost:7862
"""
from __future__ import annotations

import gradio as gr

import demo_checkpoints
import demo_hf_backend
import demo_vllm_backend

PORT = 7862


def _cpu_generate(label: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
    target = demo_checkpoints.resolve(label)
    if target is None:
        return f"ERROR: no local checkpoint or published Hub repo for {label!r}."
    try:
        return demo_hf_backend.generate(
            target, prompt, max_new_tokens=int(max_new_tokens), temperature=float(temperature),
        )
    except Exception as exc:  # noqa: BLE001 -- shown to the user, not swallowed
        return f"ERROR generating with {label}: {exc}"


def _vllm_status_message() -> str:
    result = demo_vllm_backend.probe()
    if not result.reachable:
        return "vLLM server not reachable at localhost:8000 -- CPU direct only."
    return f"vLLM is serving: {result.served_model_id or 'unknown model'}"


def build_chat_tab() -> None:
    with gr.Tab("Story Completion / Chat"):
        gr.Markdown(
            "Free-form story completion. This is primarily a *completion* model -- "
            "give it the opening of a simple story for its best behavior."
        )
        vllm_status = gr.Markdown(_vllm_status_message())
        refresh_btn = gr.Button("Refresh vLLM status", size="sm")
        refresh_btn.click(lambda: _vllm_status_message(), outputs=vllm_status)
        with gr.Row():
            available = demo_checkpoints.list_available()
            default = "tt-tnt-1024 (production)" if "tt-tnt-1024 (production)" in available else (
                available[0] if available else None
            )
            checkpoint = gr.Dropdown(choices=available, value=default, label="Checkpoint")
            backend = gr.Radio(["CPU direct", "vLLM server"], value="CPU direct", label="Backend")
        prompt = gr.Textbox(label="Prompt", value="Once upon a time, there was a little")
        with gr.Row():
            max_new_tokens = gr.Slider(8, 200, value=60, step=1, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
        output = gr.Textbox(label="Output", lines=6)
        generate_btn = gr.Button("Generate")

        def _run(label, backend_choice, prompt_text, max_tok, temp):
            if backend_choice == "vLLM server":
                result = demo_vllm_backend.probe()
                if not result.reachable:
                    return "vLLM server not reachable -- switch to CPU direct."
                try:
                    return demo_vllm_backend.complete(
                        prompt_text, max_tokens=int(max_tok), temperature=float(temp),
                    )
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR from vLLM server: {exc}"
            return _cpu_generate(label, prompt_text, max_tok, temp)

        generate_btn.click(
            _run, inputs=[checkpoint, backend, prompt, max_new_tokens, temperature],
            outputs=output,
        )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="tt-tnt demo") as demo:
        gr.Markdown("# tt-tnt demo\nWhat this model is good, weird, and bad at.")
        build_chat_tab()
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=PORT, share=False)
```

**Implementation note — deviation from the spec's literal wording:** the design spec
says the vLLM backend choice should be "disabled... with a tooltip" when unreachable.
Gradio's `gr.Radio` has no per-choice disable — only `interactive=False` for the whole
control, which would also block picking CPU direct. `_run()` above instead always
keeps both choices selectable and returns a clear, immediate "switch to CPU direct"
message on click if vLLM isn't reachable — same intent (never a silent fallback that
looks like a live vLLM result), different mechanism. Flagged here rather than left
unexplained.

- [ ] **Step 4: Manual verification**

Run: `python app.py`
Expected: prints a local URL (`http://0.0.0.0:7862`). Open it in a browser, confirm the
"Story Completion / Chat" tab renders, the checkpoint dropdown lists at least
`tt-tnt-1024 (production)`, and clicking Generate with the default prompt produces
story-like text within a few seconds. Confirm the vLLM status line reads "not
reachable" (no server running yet), and that selecting "vLLM server" + Generate
returns the "switch to CPU direct" message rather than hanging or crashing. Stop the
app with Ctrl-C.

- [ ] **Step 5: Commit**

```bash
git add app.py pyproject.toml
git commit -m "feat(demo): add app.py skeleton with the Story Completion / Chat tab"
```

---

## Task 7: `app.py` Tab 2 (Tool-Calling Roles)

**Files:**
- Modify: `app.py`
- Modify: `demo_vllm_backend.py` (add `openai_tool_schemas()`)
- Test: `tests/test_demo_vllm_backend.py` (extend)

**Interfaces:**
- Consumes: `train.tool_calling.TOOLS` (existing, `train/tool_calling.py:71`),
  `demo_checkpoints.list_available()`, `_cpu_generate()` (Task 6),
  `demo_vllm_backend.probe()`, `demo_vllm_backend.chat()` (Task 3).
- Produces: `demo_vllm_backend.openai_tool_schemas() -> List[Dict[str, Any]]`,
  `build_tool_calling_tab()` (called from `build_app()`).

- [ ] **Step 1: Write the failing test for the schema derivation**

```python
# append to tests/test_demo_vllm_backend.py
from demo_vllm_backend import openai_tool_schemas


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_demo_vllm_backend.py -k openai_tool_schemas -v`
Expected: FAIL with `ImportError: cannot import name 'openai_tool_schemas'`

- [ ] **Step 3: Add `openai_tool_schemas()` to `demo_vllm_backend.py`**

Append to `demo_vllm_backend.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_demo_vllm_backend.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Add the Tool-Calling tab to `app.py`**

Add `import json` to `app.py`'s imports, then add this function and call it from
`build_app()`:

```python
def build_tool_calling_tab() -> None:
    tool_calling_labels = [
        l for l in demo_checkpoints.list_available() if l.startswith("tool-calling-")
    ]
    with gr.Tab("Tool-Calling Roles"):
        gr.Markdown(
            "Trained to answer through one of four tools: `factual_response`, "
            "`witty_response`, `absurdist_response`, `misunderstood_question`. Shows "
            "the raw generated text (literal `<tool_call>` tag included) and, when a "
            "vLLM server is live and actually serving a tool-calling checkpoint, the "
            "real *parsed* structured tool call vLLM's hermes parser extracts from it."
        )
        if not tool_calling_labels:
            gr.Markdown(
                "No tool-calling checkpoint available locally. Run "
                "`python scripts/prepare_demo_checkpoints.py` first."
            )
            return
        checkpoint = gr.Dropdown(
            choices=tool_calling_labels, value=tool_calling_labels[0], label="Checkpoint",
        )
        question = gr.Textbox(label="Question", value="What is the capital of Portugal?")
        ask_btn = gr.Button("Ask")
        raw_output = gr.Textbox(label="Raw generation (CPU direct)", lines=4)
        parsed_output = gr.Textbox(label="Parsed tool call (vLLM, if live)", lines=4)

        def _ask(label, q):
            raw = _cpu_generate(label, f"Q: {q}\nAnswer:", 80, 0.8)
            result = demo_vllm_backend.probe()
            if not result.reachable:
                parsed = "vLLM not reachable -- start `tt-model serve` to see the parsed tool call."
            elif result.served_model_id is None or "tool-calling" not in result.served_model_id:
                parsed = f"vLLM is serving {result.served_model_id!r}, not a tool-calling checkpoint."
            else:
                try:
                    resp = demo_vllm_backend.chat(
                        [{"role": "user", "content": q}],
                        tools=demo_vllm_backend.openai_tool_schemas(),
                    )
                    tool_calls = resp["choices"][0]["message"].get("tool_calls")
                    parsed = json.dumps(tool_calls, indent=2) if tool_calls else "no tool call returned"
                except Exception as exc:  # noqa: BLE001
                    parsed = f"ERROR from vLLM server: {exc}"
            return raw, parsed

        ask_btn.click(_ask, inputs=[checkpoint, question], outputs=[raw_output, parsed_output])
```

In `build_app()`, call `build_tool_calling_tab()` right after `build_chat_tab()`.

- [ ] **Step 6: Manual verification**

Run: `python app.py`, open the browser. Confirm the "Tool-Calling Roles" tab shows
(or, on a machine that skipped Task 4's conversion run, shows the "run
prepare_demo_checkpoints.py" message rather than crashing). Ask a question, confirm
the raw output contains a `<tool_call>`-shaped string. Confirm the parsed-output box
says "vLLM not reachable" when no server is running.

- [ ] **Step 7: Commit**

```bash
git add app.py demo_vllm_backend.py tests/test_demo_vllm_backend.py
git commit -m "feat(demo): add Tool-Calling Roles tab with raw + parsed tool calls"
```

---

## Task 8: `app.py` Tab 3 (Known Limitations) + Tab 4 (Research Findings)

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `demo_checkpoints.list_available()`, `_cpu_generate()` (Task 6),
  `demo_findings.load_findings()` (Task 5).
- Produces: `build_limitations_tab()`, `build_findings_tab()` (both called from
  `build_app()`).

- [ ] **Step 1: Add `import demo_findings` to `app.py`, then add the two tab functions**

```python
_LIMITATION_PRESETS = [
    {
        "title": "Q&A collapse",
        "checkpoint": "tt-tnt-1024 (production)",
        "prompt": "Q: What is the capital of France?\nAnswer:",
        "temperature": 0.0,
        "why": (
            "This model's Q&A ability came from a thin dialogue slice layered onto a "
            "story-completion base. It often can't hold a factual answer and "
            "collapses into repeating a wrong answer rather than admitting it "
            "doesn't know."
        ),
    },
    {
        "title": "Register / genre collapse",
        "checkpoint": "tt-tnt-1024 (production)",
        "prompt": "Once upon a time, there was a little",
        "temperature": 1.2,
        "why": (
            "TinyStories dominates the training corpus. Even sampled at high "
            "temperature for variety, the model tends to fall back into the same "
            "fairy-tale register rather than genuinely diversifying."
        ),
    },
    {
        "title": "Catastrophic repeat loop (editor-blend, broken run)",
        "checkpoint": "editor-blend (broken run)",
        "prompt": "Once upon a time, there was a little",
        "temperature": 0.8,
        "why": (
            "A real, diagnosed bug: an unshifted-labels error in an earlier training "
            "run taught the model to predict the token already at its own position. "
            "Every prompt against this checkpoint collapses into repeating a single "
            "word from the very first generated token (e.g. 'to to to to...')."
        ),
    },
]


def build_limitations_tab() -> None:
    with gr.Tab("Known Limitations (weird & bad)"):
        gr.Markdown(
            "Curated failures this project actually measured and diagnosed -- not "
            "cherry-picked bad luck. CPU direct only; this tab is about the model, "
            "not the serving stack."
        )
        available = set(demo_checkpoints.list_available())
        prompt = gr.Textbox(label="Prompt")
        checkpoint = gr.Dropdown(choices=demo_checkpoints.list_available(), label="Checkpoint")
        temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature")
        why = gr.Markdown()
        for preset in _LIMITATION_PRESETS:
            disabled = preset["checkpoint"] not in available
            label = preset["title"] + (" [unavailable locally]" if disabled else "")
            btn = gr.Button(label, interactive=not disabled)

            def _fill(p=preset):
                return p["prompt"], p["checkpoint"], p["temperature"], p["why"]

            btn.click(_fill, outputs=[prompt, checkpoint, temperature, why])

        output = gr.Textbox(label="Output", lines=6)
        run_btn = gr.Button("Run")
        run_btn.click(
            lambda ck, p, t: _cpu_generate(ck, p, 80, t),
            inputs=[checkpoint, prompt, temperature], outputs=output,
        )


def build_findings_tab() -> None:
    with gr.Tab("Research Findings"):
        gr.Markdown(
            "**Historical findings, not live generation.** These checkpoints are no "
            "longer available to serve -- this reads the actual committed "
            "`docs/measurements/*.json` verdicts and quotes them verbatim."
        )
        cards = demo_findings.load_findings()
        if not cards:
            gr.Markdown("No docs/measurements/*.json files found.")
            return
        for card in cards:
            verdict = card["verdict"] or "(verdict field not found -- check docs/measurements/ schema)"
            gr.Markdown(f"### {card['title']}\n{verdict}\n\n*Source: `{card['source']}`*")
```

In `build_app()`, call `build_limitations_tab()` and `build_findings_tab()` after the
previous two tabs.

- [ ] **Step 2: Manual verification**

Run: `python app.py`, open the browser. On "Known Limitations": click each preset
button, confirm the prompt/checkpoint/temperature/why fields populate, and confirm the
"editor-blend" preset's button is disabled with an "[unavailable locally]" suffix if
Task 4 wasn't run on this machine, or enabled and producing a visible repeat-loop if it
was. On "Research Findings": confirm at least the five cards render with real verdict
text (not `None`, not a `dict` repr).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat(demo): add Known Limitations and Research Findings tabs"
```

---

## Task 9: `.disco/app.yaml` manifest, Spaces `requirements.txt`, and README section

**Files:**
- Create: `.disco/app.yaml`
- Create: `requirements.txt`
- Modify: `README.md`

**Interfaces:** none (no code consumed or produced — this task is packaging/docs only).

- [ ] **Step 1: Create `.disco/app.yaml`**

```yaml
name: tt-tnt
description: tt-tnt demo — chat, tool-calling roles, known limitations, and research findings
port: 7862
launch: .venv/bin/python app.py
```

(No `chips:` field — see the spec's manifest section and this plan's Global
Constraints: the demo process never opens a device.)

- [ ] **Step 2: Create `requirements.txt` for HuggingFace Spaces deployment**

```
# HuggingFace Spaces runtime pin. Copy this file, app.py, demo_checkpoints.py,
# demo_hf_backend.py, demo_vllm_backend.py, demo_findings.py, and docs/measurements/
# into a new Space repo (SDK: Gradio). gradio is pinned exactly and MUST match the
# Space's sdk_version in its README.md front matter -- a drift between the two means
# the Space builds against a different Gradio than this file claims.
gradio==4.44.1
torch>=2.0.0
transformers>=4.52,<6
huggingface_hub>=0.36
safetensors>=0.4.0
```

- [ ] **Step 3: Add a README section documenting both deploy paths**

Add this section to `README.md` (placed near any existing "Usage"/"Serving" section):

```markdown
## Gradio Demo

A demo covering chat/completion, tool-calling roles, known failure modes, and
historical research findings.

### Local

```bash
pip install -e ".[ui]"
python scripts/prepare_demo_checkpoints.py  # one-time, converts 2 missing checkpoints
python app.py
# open http://localhost:7862
```

Discoverable via [tt-discolike](https://github.com/) through `.disco/app.yaml` — no
separate launch step needed if you're already using that catalog.

### HuggingFace Spaces

Only the published production checkpoint (`episod/tt-tnt-1024`) and the static
research-findings tab work on a Space — the tool-calling and editor-blend checkpoints
were never published, and Blackhole hardware isn't reachable from HF infrastructure, so
those parts degrade gracefully rather than erroring. To deploy:

1. Create a new Space (SDK: Gradio).
2. Copy `app.py`, `demo_checkpoints.py`, `demo_hf_backend.py`,
   `demo_vllm_backend.py`, `demo_findings.py`, `requirements.txt`, and
   `docs/measurements/` into the Space repo root.
3. No further configuration — checkpoint resolution falls back to the Hub repo id
   automatically when no local `artifacts/hf-*` directory exists.
```

- [ ] **Step 4: Run the whole test suite to confirm nothing regressed**

Run: `pytest -q`
Expected: all prior tests plus this feature's ~26 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add .disco/app.yaml requirements.txt README.md
git commit -m "docs(demo): add tt-discolike manifest, Spaces requirements, and deploy docs"
```
