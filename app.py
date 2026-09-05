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

import json
from typing import List, Optional

import gradio as gr

import demo_checkpoints
import demo_findings
import demo_hf_backend
import demo_vllm_backend

PORT = 7862


def _default_label(available: List[str]) -> Optional[str]:
    """The checkpoint label a dropdown should default to: the production checkpoint
    if it's available locally, else the first available label, else None. gradio's
    `Dropdown` does NOT default to the first choice when `value` is omitted, so every
    checkpoint dropdown in this app needs to compute one explicitly."""
    if "tt-tnt-1024 (production)" in available:
        return "tt-tnt-1024 (production)"
    return available[0] if available else None


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
        return (
            f"vLLM server not reachable at {demo_vllm_backend.DEFAULT_BASE} -- "
            "CPU direct only."
        )
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
            checkpoint = gr.Dropdown(
                choices=available, value=_default_label(available), label="Checkpoint",
            )
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
                        prompt_text, model=result.served_model_id,
                        max_tokens=int(max_tok), temperature=float(temp),
                    )
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR from vLLM server: {exc}"
            return _cpu_generate(label, prompt_text, max_tok, temp)

        generate_btn.click(
            _run, inputs=[checkpoint, backend, prompt, max_new_tokens, temperature],
            outputs=output,
        )


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
                        model=result.served_model_id,
                    )
                    tool_calls = resp["choices"][0]["message"].get("tool_calls")
                    parsed = json.dumps(tool_calls, indent=2) if tool_calls else "no tool call returned"
                except Exception as exc:  # noqa: BLE001
                    parsed = f"ERROR from vLLM server: {exc}"
            return raw, parsed

        ask_btn.click(_ask, inputs=[checkpoint, question], outputs=[raw_output, parsed_output])


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
        "title": "TinyStories lexical habit / coherence slips",
        "checkpoint": "tt-tnt-1024 (production)",
        "prompt": "Once upon a time, there was a little",
        "temperature": 1.2,
        # NOTE: this preset used to claim the model "falls back into the same
        # fairy-tale register" at high temperature -- but the prompt itself is a
        # fairy-tale opening, so getting fairy-tale output back proves nothing; a
        # register-collapse claim needs a control (a non-fairy-tale prompt at the
        # same temperature) to mean anything, and this preset has none. Verified
        # directly against this checkpoint instead (5 samples at T=1.2, same prompt):
        # TinyStories-specific names (Lily x3/5, Sue x1/5) and the stock phrase
        # "One day," appeared in every single completion regardless of the story
        # that had actually started, and one completion introduces a rescuing
        # "truck" that becomes a "dog" one sentence later with no reintroduction --
        # a character/object drifting or appearing without having been set up. That
        # is a genuine, showable lexical-habit and coherence failure and does not
        # need a control condition to demonstrate.
        "why": (
            "Even when the story's own setup doesn't call for it, the model keeps "
            "reaching for the same handful of TinyStories names (Lily, Sue) and "
            "stock phrases ('One day,'), and can lose track of a character or "
            "object it just introduced -- e.g. a rescuing truck that becomes a dog "
            "one sentence later with no explanation. A lexical habit and a "
            "coherence slip, not a deliberate creative choice."
        ),
    },
    {
        "title": "Catastrophic repeat loop (editor-blend, broken run)",
        "checkpoint": "editor-blend (broken run)",
        # NOTE: deliberately NOT an ordinary TinyStories-style prompt. This project's
        # own incident record (CLAUDE.md, "The base-blend follow-up's first attempt
        # trained a second, worse-broken checkpoint") measured that this checkpoint
        # stays fluent on an ordinary opening like "Once upon a time, there was a
        # little" -- the collapse was reproduced only on the "spine"-register frozen
        # prompts (docs/evaluation_prompts_b.json, id prefix "b-spine-"). Verified
        # directly against this converted checkpoint before shipping this preset:
        # this exact prompt collapses into "to to to to..." from the first generated
        # token, matching the historical record; an ordinary story-opening prompt does
        # not trigger the collapse at all and would have shown fluent prose instead.
        "prompt": (
            "The beetle came back to the same square of wall each evening, and I "
            "began to"
        ),
        "temperature": 0.8,
        "why": (
            "A real, diagnosed bug: an unshifted-labels error in an earlier training "
            "run taught the model to predict the token already at its own position. "
            "This checkpoint collapses into repeating a single word from the very "
            "first generated token (e.g. 'to to to to...') on unusual, non-TinyStories "
            "openings -- an ordinary story prompt stays fluent, which is itself part "
            "of the finding."
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
        available_list = demo_checkpoints.list_available()
        available = set(available_list)
        prompt = gr.Textbox(label="Prompt", value=_LIMITATION_PRESETS[0]["prompt"])
        checkpoint = gr.Dropdown(
            choices=available_list, value=_default_label(available_list), label="Checkpoint",
        )
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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="tt-tnt demo") as demo:
        gr.Markdown("# tt-tnt demo\nWhat this model is good, weird, and bad at.")
        build_chat_tab()
        build_tool_calling_tab()
        build_limitations_tab()
        build_findings_tab()
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=PORT, share=False)
