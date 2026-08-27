# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Example-building and masking only. No `ttml`/`ttnn` import, no device -- these tests run
in a bare CPU environment, same convention as tests/test_corrupt.py.
"""
from scripts.train_editor import build_editor_example, build_poetry_example


class _FakeTok:
    """Minimal tokenizer stand-in: word-level 'encoding' so lengths are checkable by hand."""

    def encode(self, text, add_special_tokens=True):
        n = len(text.split())
        return ([0] if add_special_tokens else []) + list(range(1, n + 1))


def test_build_editor_example_masks_the_draft_and_supervises_the_edit():
    pair = {"draft": "a b c", "better": "x y"}
    tok = _FakeTok()
    ex = build_editor_example(pair, tok, pad_token_id=99)
    prompt = "\nDraft: a b c\nEdit: "
    p_ids = tok.encode(prompt)
    assert ex["input_ids"][: len(p_ids)] == p_ids
    # every label up to the last prompt position is masked...
    assert all(l == -100 for l in ex["labels"][: len(p_ids) - 1])
    # ...and the supervised region is exactly len(completion_ids) long
    c_ids = tok.encode("x y", add_special_tokens=False)
    supervised = [l for l in ex["labels"] if l != -100]
    assert len(supervised) == len(c_ids)


def test_build_editor_example_input_ids_and_labels_are_pre_shift_aligned_length():
    pair = {"draft": "one two three", "better": "four five"}
    tok = _FakeTok()
    ex = build_editor_example(pair, tok, pad_token_id=99)
    assert len(ex["input_ids"]) == len(ex["labels"])


def test_build_poetry_example_continuation_prompt_masks_input_and_supervises_target():
    pair = {"kind": "continuation", "input": "a b", "arg": None, "target": "c d"}
    tok = _FakeTok()
    ex = build_poetry_example(pair, tok, pad_token_id=99)
    prompt = "\nContinue this poem:\na b\nContinuation:\n"
    p_ids = tok.encode(prompt)
    assert ex["input_ids"][: len(p_ids)] == p_ids
    assert all(l == -100 for l in ex["labels"][: len(p_ids) - 1])
    c_ids = tok.encode("c d", add_special_tokens=False)
    supervised = [l for l in ex["labels"] if l != -100]
    assert len(supervised) == len(c_ids)


def test_build_poetry_example_keywords_prompt_includes_the_keyword_list():
    pair = {"kind": "keywords", "input": None, "arg": ["moon", "shadow"], "target": "x y z"}
    tok = _FakeTok()
    ex = build_poetry_example(pair, tok, pad_token_id=99)
    prompt = "\nWrite a poem about: moon, shadow\nPoem:\n"
    p_ids = tok.encode(prompt)
    assert ex["input_ids"][: len(p_ids)] == p_ids
