# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from __future__ import annotations

import numpy as np

from scripts.train_tool_calling import (
    MAX_SEQ_LEN,
    build_base_blend_example,
    build_category_lists,
    build_tool_calling_example,
    sample_base_blend_examples,
    stratified_split,
)
from train.tool_calling import ToolCallExample, render_tool_call


class _FakeTok:
    """Same word-level fake tokenizer as tests/test_train_editor.py, so lengths are checkable
    by hand without a real tokenizer."""

    def encode(self, text, add_special_tokens=True):
        n = len(text.split())
        return ([0] if add_special_tokens else []) + list(range(1, n + 1))


def test_max_seq_len_matches_the_published_models_real_context():
    """This constant must track the PUBLISHED model's context, not be copy-pasted or left
    behind by a registry change.

    It has now been wrong in both directions, which is why the test pins it rather than
    trusting it: written as 2048 when the 2048-context line was live, and stale at 2048 for
    exactly as long as it took this test to go red after that line was reverted on 2026-08-29
    (see CLAUDE.md). Padding every training example to a cap the model was never trained to is
    silent -- nothing downstream would have raised.
    """
    from train.sizes import get_size

    assert MAX_SEQ_LEN == 512
    assert MAX_SEQ_LEN == get_size("1024").max_sequence_length, (
        "MAX_SEQ_LEN drifted from the size registry -- update both together"
    )


def test_build_tool_calling_example_masks_the_question_and_supervises_the_tool_call():
    example = ToolCallExample(
        question="a b c", tool="factual_response",
        arguments={"answer": "x", "confidence": "high"},
    )
    tok = _FakeTok()
    ex = build_tool_calling_example(example, tok, pad_token_id=99)

    prompt = "Q: a b c\nAnswer:"
    completion = render_tool_call(example.tool, example.arguments)
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)

    assert ex["input_ids"][: len(p_ids)] == p_ids
    assert ex["input_ids"][len(p_ids):] == c_ids
    # Every label up to the last prompt position is masked...
    assert ex["labels"][: len(p_ids) - 1] == [-100] * (len(p_ids) - 1)
    # ...the last prompt position's label is the FIRST completion token (the trained
    # transition), and the completion's own labels equal its own ids shifted by one position
    # relative to input, i.e. labels[len(p_ids)-1:] == c_ids + [-100].
    assert ex["labels"][len(p_ids) - 1:] == c_ids + [-100]
    assert ex["labels"][-1] == -100


def test_build_tool_calling_example_returns_none_when_question_alone_exceeds_max_seq_len(monkeypatch):
    monkeypatch.setattr("scripts.train_tool_calling.MAX_SEQ_LEN", 5)
    example = ToolCallExample(
        question=" ".join(f"w{i}" for i in range(20)), tool="factual_response",
        arguments={"answer": "x", "confidence": "high"},
    )
    ex = build_tool_calling_example(example, _FakeTok(), pad_token_id=99)
    assert ex is None


def test_build_base_blend_example_shifts_labels_by_one_not_unshifted():
    """Same bug class this project has hit twice before (scripts/train_editor.py's own
    docstring, and the 2026-08-20 improv-thinking entry): labels[t] must equal
    input_ids[t+1], never input_ids[t]."""
    window = list(range(10, 21))
    ex = build_base_blend_example(window)
    assert ex["input_ids"] == list(range(10, 20))
    assert ex["labels"] == list(range(11, 21))
    assert ex["input_ids"] != ex["labels"]
    for t in range(len(ex["input_ids"]) - 1):
        assert ex["labels"][t] == ex["input_ids"][t + 1]
    assert -100 not in ex["labels"]


def test_sample_base_blend_examples_reads_real_windows_at_2048(tmp_path):
    arr = np.arange(5000, dtype=np.uint32)
    token_path = tmp_path / "train_ids.npy"
    np.save(token_path, arr)

    examples = sample_base_blend_examples(token_path, 3, seed=0)  # default seq_len=MAX_SEQ_LEN
    assert len(examples) == 3
    for ex in examples:
        assert len(ex["input_ids"]) == MAX_SEQ_LEN
        assert len(ex["labels"]) == MAX_SEQ_LEN
        start = ex["input_ids"][0]
        assert ex["input_ids"] == list(range(start, start + MAX_SEQ_LEN))


def test_build_category_lists_drops_zero_supervision_examples(monkeypatch):
    # 18 is chosen to sit between the short example's full length (15 tokens with the fake
    # tokenizer) and the long example's PROMPT-ALONE length (23 tokens) -- so the short one
    # survives intact and the long one truncates to zero real supervision, rather than both
    # being dropped by an arbitrarily tiny cap that wouldn't distinguish the two cases.
    monkeypatch.setattr("scripts.train_tool_calling.MAX_SEQ_LEN", 18)
    corpus = [
        ToolCallExample(question="a b c", tool="factual_response",
                        arguments={"answer": "x", "confidence": "high"}),
        ToolCallExample(question=" ".join(f"w{i}" for i in range(20)), tool="factual_response",
                        arguments={"answer": "x", "confidence": "high"}),
    ]
    categories = build_category_lists(corpus, _FakeTok(), pad_token_id=99)
    # The long-question example's prompt alone exceeds MAX_SEQ_LEN=3, so it must be dropped,
    # not silently truncated into a zero-supervision example.
    assert len(categories["tool_calling"]) == 1


def test_stratified_split_holds_out_an_even_share_from_every_category():
    categories = {
        "tool_calling": [{"input_ids": [i], "labels": [i]} for i in range(20)],
        "base_blend": [{"input_ids": [i], "labels": [i]} for i in range(20)],
    }
    train, val = stratified_split(categories, val_size=8)
    assert len(val) == 8
    # 4 from each category (val_size // len(categories))
    assert len(train) == 32


def test_stratified_split_with_zero_val_size_returns_everything_as_train():
    categories = {"tool_calling": [{"input_ids": [1], "labels": [1]}]}
    train, val = stratified_split(categories, val_size=0)
    assert val == []
    assert train == categories["tool_calling"]
