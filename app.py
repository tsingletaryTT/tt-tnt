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
