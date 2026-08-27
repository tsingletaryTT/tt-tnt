# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Example-building and masking only. No `ttml`/`ttnn` import, no device -- these tests run
in a bare CPU environment, same convention as tests/test_corrupt.py.
"""
import numpy as np

from scripts.train_editor import (
    MAX_SEQ_LEN,
    _pad_to_max_seq_len,
    build_base_blend_example,
    build_editor_example,
    build_poetry_example,
    sample_base_blend_examples,
    stratified_split,
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


def test_build_base_blend_example_shifts_labels_by_one_not_unshifted():
    # window has length seq_len+1 = 11; input_ids drops the last token, labels drops
    # the first -- labels[t] must equal input_ids[t+1] (ttml's convention), NOT
    # input_ids[t] (the HF-style bug this function originally shipped with and that
    # broke a real training run -- see this function's docstring).
    window = list(range(10, 21))
    ex = build_base_blend_example(window)
    assert ex["input_ids"] == list(range(10, 20))
    assert ex["labels"] == list(range(11, 21))
    assert ex["input_ids"] != ex["labels"]
    for t in range(len(ex["input_ids"]) - 1):
        assert ex["labels"][t] == ex["input_ids"][t + 1]
    assert -100 not in ex["labels"]


def test_build_base_blend_example_returns_independent_lists():
    window = [1, 2, 3]
    ex = build_base_blend_example(window)
    ex["input_ids"].append(99)
    assert ex["labels"] == [2, 3]


def test_sample_base_blend_examples_reads_real_windows_from_disk(tmp_path):
    arr = np.arange(1000, dtype=np.uint32)
    token_path = tmp_path / "train_ids.npy"
    np.save(token_path, arr)

    examples = sample_base_blend_examples(token_path, 5, seq_len=32, seed=0)
    assert len(examples) == 5
    for ex in examples:
        assert len(ex["input_ids"]) == 32
        assert len(ex["labels"]) == 32
        # every window must be a real contiguous slice of arr, not fabricated, and
        # labels must be input_ids shifted by exactly one position (arange makes
        # this checkable: labels[t] == input_ids[t] + 1 for every t)
        start = ex["input_ids"][0]
        assert ex["input_ids"] == list(range(start, start + 32))
        assert ex["labels"] == list(range(start + 1, start + 33))


def test_sample_base_blend_examples_is_deterministic_given_the_same_seed(tmp_path):
    arr = np.arange(1000, dtype=np.uint32)
    token_path = tmp_path / "train_ids.npy"
    np.save(token_path, arr)

    a = sample_base_blend_examples(token_path, 5, seq_len=32, seed=7)
    b = sample_base_blend_examples(token_path, 5, seq_len=32, seed=7)
    assert a == b


def test_sample_base_blend_examples_raises_if_array_shorter_than_seq_len(tmp_path):
    arr = np.arange(10, dtype=np.uint32)
    token_path = tmp_path / "train_ids.npy"
    np.save(token_path, arr)

    try:
        sample_base_blend_examples(token_path, 1, seq_len=32, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_stratified_split_holds_out_from_every_category_not_just_the_tail():
    # A "skits"-only tail-of-one-list split (the first run's real bug) would put ALL
    # held-out examples in one category. This fixture makes that failure visible: if
    # any category contributes ZERO val examples, the split is not actually stratified.
    categories = {
        "editor": [{"input_ids": [i], "labels": [i]} for i in range(20)],
        "poetry": [{"input_ids": [i], "labels": [i]} for i in range(100, 120)],
        "skits": [{"input_ids": [i], "labels": [i]} for i in range(200, 220)],
        "base_blend": [{"input_ids": [i], "labels": [i]} for i in range(300, 320)],
    }
    train, val = stratified_split(categories, val_size=8)
    val_ids = {e["input_ids"][0] for e in val}
    for name, exs in categories.items():
        category_ids = {e["input_ids"][0] for e in exs}
        assert val_ids & category_ids, f"category {name!r} contributed zero val examples"
    # every example is in exactly one of train/val
    train_ids = {e["input_ids"][0] for e in train}
    assert not (train_ids & val_ids)
    assert len(train_ids) + len(val_ids) == sum(len(v) for v in categories.values())


def test_stratified_split_val_size_zero_returns_everything_as_train():
    categories = {"editor": [{"input_ids": [1], "labels": [1]}]}
    train, val = stratified_split(categories, val_size=0)
    assert train == categories["editor"]
    assert val == []


def test_stratified_split_tiny_category_contributes_everything_to_train():
    # A category too small to safely hold out from (len // 4 == 0) must not be emptied
    # into val -- it should keep all its examples in train instead.
    categories = {
        "editor": [{"input_ids": [i], "labels": [i]} for i in range(2)],
        "poetry": [{"input_ids": [i], "labels": [i]} for i in range(100, 140)],
    }
    train, val = stratified_split(categories, val_size=8)
    editor_val = [e for e in val if e["input_ids"][0] < 100]
    assert editor_val == []
    assert len(train) + len(val) == 42
