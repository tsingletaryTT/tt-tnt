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
