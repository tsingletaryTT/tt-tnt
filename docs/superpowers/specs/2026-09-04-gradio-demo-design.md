# tt-tnt Gradio demo + tt-discolike manifest — design

Status: approved, pre-implementation-plan.
Date: 2026-09-04.

## Purpose

tt-tnt has no way to *show* what the model does short of running scripts by hand.
Sibling projects (`tt-animatediff`, `tt-vjepa2`) each ship a Gradio app plus a
`.disco/app.yaml` manifest so `tt-discolike` (a local catalog/launcher,
`~/code/tt-discolike`) can list and start them. This adds the same pair for tt-tnt:
a kitchen-sink demo covering what the model is good, weird, and bad at, discoverable
the same way.

## Non-goals

- Not a production serving frontend. It is a demo for a person exploring the project.
- Not a new evaluation instrument. Tab 4 (Research Findings) renders numbers that
  already exist under `docs/measurements/`; it computes nothing new.
- Not a checkpoint-training or conversion pipeline. `scripts/prepare_demo_checkpoints.py`
  converts two specific existing `.pkl` checkpoints to HF format because the demo wants
  them and they were never converted; it does not generalize into a batch tool.
- Does not manage a vLLM server's lifecycle (start/stop/lease). The demo is a passive
  HTTP client to whatever the user already launched via `tt-model serve`.

## What's actually available locally (checked, not assumed)

`artifacts/hf-*` present on this machine today:
`hf-tt-tnt-1024` (production dialogue model), `hf-tt-tnt-1024-tool-calling`,
`hf-tt-tnt-1024-tool-calling-s2`, plus a long tail of research-only pretraining
variants (`hf-tt-tnt-v1/v3/v4/v5`, `hf-ts10-*`, `hf-ts31-*`, `hf-winpurity-*`) not used
by this demo.

Not present as loadable weights: reach-dial, skits/dialogue-turn, and the editor
checkpoints (`editor`, `editor-blend`, `editor-blend-v2`, `editor-lora` all have raw
`.pkl` under `artifacts/checkpoints-1024-*` but no HF conversion on disk; reach-dial and
skits have no `.pkl` checkpoint directory at all — only their derived corpora and
`docs/measurements/*.json` results survive). This is why Tab 4 is a findings readout,
not live generation, and why Tab 3 needs one checkpoint conversion
(`editor-blend`, the *original* broken run — the one with the dramatic single-word
repeat-loop collapse) done ahead of time rather than live.

## Components

### `app.py` (repo root)

Gradio `Blocks` app, 4 tabs (below), CPU-only process (never opens a Tenstorrent
device itself — see manifest section). Launch: `demo.launch(server_name="0.0.0.0",
server_port=7862, share=False)` guarded by `if __name__ == "__main__":`, matching
`tt-animatediff/app.py`'s pattern. Port 7862 chosen because 7860/7861 are already
claimed by `tt-animatediff`/`tt-vjepa2`'s manifests.

### `demo_checkpoints.py`

A small registry: `CHECKPOINTS: dict[str, Path]` mapping a friendly label (e.g.
`"tt-tnt-1024 (production)"`, `"tool-calling-s2"`, `"tool-calling-s3"`,
`"editor-blend (broken run)"`) to its `artifacts/hf-*` directory, *not* asserted to
exist — a `list_available() -> list[str]` function filters to labels whose
`config.json` is actually present, the same existence check `chat.py` already uses
(`(model_dir / "config.json").is_file()`). Every dropdown in `app.py` is populated from
`list_available()`, so a fresh clone with fewer converted checkpoints degrades to fewer
dropdown entries rather than a crash.

### `demo_hf_backend.py`

Wraps the load-and-generate pattern already in `scripts/chat.py` (`AutoModelForCausalLM`
/ `AutoTokenizer.from_pretrained`, greedy or sampled generation). Adds an
`functools.lru_cache(maxsize=2)`-style cache keyed on the resolved model directory path,
so flipping between two checkpoints in the UI doesn't reload from disk on every click,
while a third distinct checkpoint evicts the least-recently-used one rather than
growing memory unboundedly.

### `demo_vllm_backend.py`

Mirrors `scripts/story_tools.py`'s `_post()` helper (`urllib.request`, no new
dependency): a `complete(prompt, ...)` function hitting `/v1/completions` and a
`chat(messages, tools=...)` function hitting `/v1/chat/completions` (for Tab 2's
structured tool-call path). A `probe(endpoint="http://localhost:8000") -> ProbeResult`
does a short-timeout `GET /v1/models`; `ProbeResult` carries `reachable: bool` and, if
reachable, the served model id, so the UI can say *which* checkpoint vLLM is currently
serving rather than just "online". No lease, no subprocess — this module never touches
`/dev/tenstorrent/*`.

### `scripts/prepare_demo_checkpoints.py`

One-time CLI, run by hand (documented in this app's README section / `--help`), not
invoked from any Gradio callback. Converts, if their HF output directory is missing, the highest-step `.pkl` under
`artifacts/checkpoints-1024-tool-calling-s3/` and under
`artifacts/checkpoints-1024-editor-blend/` (selected the same way `latest_checkpoint()`
already does elsewhere in this project — highest zero-padded step in the filename, never
assumed or hardcoded), both through
`scripts/eval_improv.py`'s existing `sft_checkpoint_to_hf(step_pkl, warm_start_ckpt=...,
tokenizer_dir=..., out_dir=...)`. CPU-only, no ttnn/ttml import, consistent with every
other conversion path in this project. Idempotent: skips a checkpoint whose `out_dir`
already has a `config.json`.

## The four tabs

### Tab 1 — Story Completion / Chat

- Checkpoint dropdown: `demo_checkpoints.list_available()`, default `tt-tnt-1024
  (production)`.
- Backend radio: `CPU direct` / `vLLM server`. `vLLM server` is disabled (not hidden —
  visibly present but non-interactive, with a tooltip) unless `demo_vllm_backend.probe()`
  succeeds; when it succeeds, the label shows which checkpoint is actually being served,
  since that may not match the dropdown's selection (vLLM only ever serves one model).
- Prompt textbox (prefilled with one of the completions from `chat.py`'s own docstring,
  e.g. `"Once upon a time, there was a little"`), max-new-tokens slider, temperature
  slider, Generate button, output textbox.
- CPU path calls `demo_hf_backend`; vLLM path calls `demo_vllm_backend.complete()`
  against whatever the dropdown resolves to being irrelevant — vLLM always answers with
  whatever it's serving, and the UI says so rather than pretending the dropdown selected it.

### Tab 2 — Tool-Calling Roles

- Checkpoint dropdown restricted to whichever of `tool-calling-s2` / `tool-calling-s3`
  is available (from `demo_checkpoints.list_available()`, filtered to that pair).
- Question textbox (e.g. `"What is the capital of Portugal?"`), Ask button.
- Always shows the raw CPU-direct completion text verbatim, literal `<tool_call>` tags
  and all — this is the honest baseline, matching how `train/tool_calling.py`'s format
  actually looks pre-parsing.
- If `probe()` succeeds *and* the served model id matches a tool-calling checkpoint,
  additionally calls `demo_vllm_backend.chat()` with the four tool schemas
  (`factual_response` / `witty_response` / `absurdist_response` /
  `misunderstood_question`, mirroring `train/tool_calling.py`'s definitions) and renders
  the real parsed `tool_calls` JSON object beside the raw text — otherwise that half of
  the panel says "vLLM isn't currently serving a tool-calling checkpoint" rather than
  faking it.

### Tab 3 — Known Limitations ("weird & bad")

- A fixed list of curated (prompt, checkpoint, one-line why) triples pulled from this
  project's own documented findings, e.g.:
  - `"Q: What is the capital of France?\nAnswer:"` against `tt-tnt-1024` — reliably
    collapses into a repeated wrong-answer loop; documented Q&A weakness.
  - A children's-story opening against `tt-tnt-1024` sampled at higher temperature —
    illustrates the measured register/genre-collapse behavior (tinystories-flavored
    prose regardless of prompt).
  - Any short prompt against `editor-blend (broken run)` — the catastrophic
    single-word repeat loop (`"to to to to..."`) from the mis-shifted-labels bug,
    included specifically because it is a real, dramatic, reproducible failure this
    project diagnosed and documented rather than a contrived one.
- Each triple is a preset button that fills the prompt box and picks the right
  checkpoint automatically; a free-text box lets the user try their own prompt against
  any available checkpoint from the same dropdown. CPU-direct only — no vLLM toggle,
  since this tab is about model behavior, not the serving stack.

### Tab 4 — Research Findings

- No live inference. Reads a small fixed list of `docs/measurements/*.json` files
  (`reach-dial.json`, `skits-stage2.json`, `tool-calling-s3-selection.json`,
  `lora-vs-full-tool-calling.json`, `evaluation-tt-tnt-1024-vs-editor-lora.json`) at
  app startup, and renders each as a card: title, one-line verdict (pulled from the
  file's own recorded verdict field, not re-derived), and — where the JSON itself
  carries one — a verbatim sample completion already quoted in that file or in
  CLAUDE.md's corresponding log entry.
- Explicitly labeled in the UI as historical findings from checkpoints no longer
  available to serve, not something you can regenerate by clicking a button.
- If a listed file is missing (e.g. a leaner clone), that card is simply omitted, not
  an error.

## Portability: HuggingFace Spaces

Unlike `tt-animatediff`'s separate `spaces/app.py` (needed there because Blackhole's
video pipeline has a genuinely different CPU code path), tt-tnt's CPU-direct path *is*
the reference-quality path already (`chat.py`'s own docstring: the vLLM device path is
currently the worse of the two). So this ships as **one `app.py`**, not a fork — it runs
on a Space by leaning on error-handling this design already requires:

- `demo_checkpoints.list_available()` resolves a label against a local
  `artifacts/hf-*` directory first and, if absent, against a **published Hub repo id**
  (`episod/tt-tnt-1024` for the production label) via the same
  `AutoModelForCausalLM.from_pretrained` call `demo_hf_backend` already makes —
  `from_pretrained` accepts a repo id or a local path interchangeably, so no branch is
  needed in the loading code itself, only in what path/id gets offered.
- `demo_vllm_backend.probe()` naturally reports unreachable on a Space (Blackhole isn't
  reachable from HF infrastructure), which already disables the vLLM half of Tabs 1 and
  2 per the existing error-handling rules — no Spaces-specific code path.
- Tab 2 (Tool-Calling) has nothing to load on a Space: those checkpoints were never
  promoted/published (per this project's "PARTIAL results aren't shipped" convention).
  It degrades to the same "not available, checkpoint missing" state as a lean local
  clone — already specified, not a new case.
- Tab 3's `editor-blend` preset can't run on a Space for the same reason; it degrades to
  showing the actual verbatim collapse transcript already quoted in CLAUDE.md as static
  text, labeled "captured locally, not reproducible here" — the same move
  `tt-animatediff`'s Space makes with its pre-rendered gallery for what real hardware
  produces versus what the Space itself can show.
- Tab 4 (Research Findings) is already static-file reads; the Space just needs
  `docs/measurements/*.json` copied into the Space repo alongside `app.py`.
- Deploy: create a Gradio-SDK Space, copy `app.py` + its support modules
  (`demo_checkpoints.py`, `demo_hf_backend.py`, `demo_vllm_backend.py`) + the
  `docs/measurements/*.json` files referenced by Tab 4 into the Space repo root. No
  extra runtime dependencies beyond what CPU-direct generation already needs
  (`gradio`, `torch`, `transformers`, `safetensors`, `huggingface_hub`) — no `ttnn`, no
  device.

## `.disco/app.yaml`

```yaml
name: tt-tnt
description: tt-tnt demo — chat, tool-calling roles, known limitations, and research findings
port: 7862
launch: .venv/bin/python app.py
```

No `chips:` field. Per `tt-discolike`'s own README, `chips` exists to wrap the launch
command in `gozer run --chips N ...` for an app whose *own process* needs a device.
This app's process never does — CPU-direct generation needs none, and vLLM mode is an
HTTP client to a server the user leases and launches separately via `tt-model serve`
(`docs/serving-with-tt-kernel.md`). Setting `chips` here would request a lease this
process never uses.

## Error handling (summary, detail is above per-tab)

- Missing checkpoint directory → dropdown simply excludes it; if that leaves a tab with
  zero valid checkpoints, the tab shows a message pointing at
  `scripts/prepare_demo_checkpoints.py` instead of a blank/broken control.
- vLLM unreachable → toggle/section disabled with a message; never a silent fallback to
  CPU that looks like a successful vLLM call.
- Generation-time exceptions (OOM, tokenizer error) are caught per-request and shown in
  the output box; they do not crash the Gradio process.

## Testing

- `tests/test_demo_checkpoints.py` — pure logic, no model/device: `list_available()`
  against a tmp directory tree with some `config.json`s present and some absent.
- `tests/test_demo_vllm_backend.py` — `probe()`'s response-parsing logic against a fake
  HTTP handler (stdlib `http.server` on localhost, or a monkeypatched `urlopen`); no real
  vLLM needed.
- Everything else (the actual Gradio UI, real generation quality, real vLLM
  integration) is manual verification: start the app, exercise each tab in a browser,
  confirm degraded-but-graceful behavior with vLLM not running and with
  `prepare_demo_checkpoints.py` not yet run.
