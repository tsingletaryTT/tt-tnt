# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Example-building and masking only. No `ttml`/`ttnn` import, no device -- these tests run
in a bare CPU environment, same convention as tests/test_corrupt.py.
"""
from scripts.train_editor import (
    MAX_SEQ_LEN,
    _pad_to_max_seq_len,
    build_editor_example,
    build_poetry_example,
)


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


def test_build_editor_example_truncates_from_the_end_to_max_seq_len():
    """A completion long enough to push the combined length past MAX_SEQ_LEN gets sliced
    down to exactly MAX_SEQ_LEN, with the SAME prefix as the untruncated example -- i.e.
    truncation happens at the end, never touching the prompt.
    """
    tok = _FakeTok()
    draft = "a b c"
    # Build a completion long enough that prompt + completion exceeds MAX_SEQ_LEN.
    prompt = f"\nDraft: {draft}\nEdit: "
    p_len = len(tok.encode(prompt))
    completion_words = MAX_SEQ_LEN + 50 - p_len
    better = " ".join(f"w{i}" for i in range(completion_words))
    pair = {"draft": draft, "better": better}

    # Compute the untruncated example by hand for comparison.
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(better, add_special_tokens=False)
    untruncated_input_ids = p_ids + c_ids
    untruncated_labels = [-100] * (len(p_ids) - 1) + c_ids + [-100]
    assert len(untruncated_input_ids) > MAX_SEQ_LEN  # sanity: this pair really overflows

    ex = build_editor_example(pair, tok, pad_token_id=99)
    assert ex is not None
    assert len(ex["input_ids"]) == MAX_SEQ_LEN
    assert len(ex["labels"]) == MAX_SEQ_LEN
    assert ex["input_ids"] == untruncated_input_ids[:MAX_SEQ_LEN]
    assert ex["labels"] == untruncated_labels[:MAX_SEQ_LEN]
    # There is still real supervision left after truncation.
    assert any(l != -100 for l in ex["labels"])


def test_build_editor_example_short_example_is_not_truncated():
    pair = {"draft": "a b c", "better": "x y"}
    tok = _FakeTok()
    ex = build_editor_example(pair, tok, pad_token_id=99)
    assert len(ex["input_ids"]) < MAX_SEQ_LEN


def test_build_editor_example_returns_none_when_prompt_alone_exceeds_max_seq_len():
    """`_FakeTok` makes one token per word (+1 bos), so a 513-word draft alone tokenizes to
    514 prompt tokens -- already over MAX_SEQ_LEN before any completion token is added.
    Truncating to MAX_SEQ_LEN then leaves every label masked (-100), so the example must be
    dropped (None) rather than trained on with zero supervised signal.
    """
    tok = _FakeTok()
    draft = " ".join(f"d{i}" for i in range(513))
    pair = {"draft": draft, "better": "x y"}
    ex = build_editor_example(pair, tok, pad_token_id=99)
    assert ex is None


def test_pad_to_max_seq_len_pads_short_example_to_exactly_the_cap():
    example = {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]}
    padded = _pad_to_max_seq_len(example, pad_token_id=99)
    assert len(padded["input_ids"]) == MAX_SEQ_LEN
    assert len(padded["labels"]) == MAX_SEQ_LEN
    assert padded["input_ids"][:3] == [1, 2, 3]
    assert padded["input_ids"][3:] == [99] * (MAX_SEQ_LEN - 3)
    assert padded["labels"][:3] == [-100, -100, 3]
    assert padded["labels"][3:] == [-100] * (MAX_SEQ_LEN - 3)


def test_pad_to_max_seq_len_is_a_noop_on_an_already_full_example():
    example = {"input_ids": list(range(MAX_SEQ_LEN)), "labels": [-100] * MAX_SEQ_LEN}
    padded = _pad_to_max_seq_len(example, pad_token_id=99)
    assert padded["input_ids"] == example["input_ids"]
    assert padded["labels"] == example["labels"]


def test_build_poetry_example_truncates_from_the_end_to_max_seq_len():
    tok = _FakeTok()
    poem_input = "a b"
    prompt = f"\nContinue this poem:\n{poem_input}\nContinuation:\n"
    p_len = len(tok.encode(prompt))
    completion_words = MAX_SEQ_LEN + 50 - p_len
    target = " ".join(f"w{i}" for i in range(completion_words))
    pair = {"kind": "continuation", "input": poem_input, "arg": None, "target": target}

    p_ids = tok.encode(prompt)
    c_ids = tok.encode(target, add_special_tokens=False)
    untruncated_input_ids = p_ids + c_ids
    untruncated_labels = [-100] * (len(p_ids) - 1) + c_ids + [-100]
    assert len(untruncated_input_ids) > MAX_SEQ_LEN

    ex = build_poetry_example(pair, tok, pad_token_id=99)
    assert ex is not None
    assert len(ex["input_ids"]) == MAX_SEQ_LEN
    assert ex["input_ids"] == untruncated_input_ids[:MAX_SEQ_LEN]
    assert ex["labels"] == untruncated_labels[:MAX_SEQ_LEN]


def test_build_poetry_example_returns_none_when_prompt_alone_exceeds_max_seq_len():
    tok = _FakeTok()
    poem_input = " ".join(f"d{i}" for i in range(513))
    pair = {"kind": "continuation", "input": poem_input, "arg": None, "target": "c d"}
    ex = build_poetry_example(pair, tok, pad_token_id=99)
    assert ex is None
