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


def build_app() -> gr.Blocks:
    with gr.Blocks(title="tt-tnt demo") as demo:
        gr.Markdown("# tt-tnt demo\nWhat this model is good, weird, and bad at.")
        build_chat_tab()
        build_tool_calling_tab()
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=PORT, share=False)
