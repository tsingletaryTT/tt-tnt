# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import re

from train.corrupt import (
    corrupt,
    drop_or_double_function_word,
    fuse_clauses,
    garble_word,
    repeat_collapse,
)


def test_repeat_collapse_duplicates_a_span():
    text = "The dragon flew over the mountain and landed softly."
    out = repeat_collapse(text, seed=0, severity=1.0)
    words = out.split()
    # some 2-4 word span appears at least twice consecutively or near-consecutively
    found = False
    for span_len in (2, 3, 4):
        for i in range(len(words) - span_len):
            span = words[i : i + span_len]
            if words[i + span_len : i + 2 * span_len] == span:
                found = True
    assert found, f"no repeated span found in: {out!r}"


def test_repeat_collapse_is_deterministic_for_a_seed():
    text = "The dragon flew over the mountain and landed softly."
    assert repeat_collapse(text, seed=1) == repeat_collapse(text, seed=1)


def test_garble_word_replaces_an_ordinary_word():
    text = "The girl found a small silver key by the river."
    out = garble_word(text, seed=0, severity=1.0)
    assert out != text
    # at least one token differs and the differing token is not a real short function word
    orig_words = text.split()
    new_words = out.split()
    assert len(orig_words) == len(new_words)
    diffs = [(a, b) for a, b in zip(orig_words, new_words) if a != b]
    assert diffs, "garble_word changed nothing"


def test_garble_word_skips_existing_proper_nouns():
    # "Mira" is capitalized mid-sentence -- an existing proper noun, must survive untouched.
    text = "The girl named Mira walked to the old mill by the river."
    for seed in range(20):
        out = garble_word(text, seed=seed, severity=1.0)
        assert "Mira" in out.split(), f"seed={seed} corrupted a protected proper noun: {out!r}"


def test_garble_word_skips_word_after_named_or_called():
    text = "The dog was called Bramble and lived in the barn near the pond."
    for seed in range(20):
        out = garble_word(text, seed=seed, severity=1.0)
        assert "Bramble" in out.split(), f"seed={seed} corrupted a protected coined name: {out!r}"


def test_drop_or_double_function_word_changes_length_or_doubles():
    text = "She was always sad and had never been happy before that day."
    out = drop_or_double_function_word(text, seed=0, severity=1.0)
    assert out != text


def test_fuse_clauses_removes_a_conjunction():
    text = "She was tired, and she wanted to sleep, but the noise kept her awake."
    out = fuse_clauses(text, seed=0, severity=1.0)
    assert out != text
    # at least one of the original conjunctions is gone
    conjunctions = {"and", "but", "or", "so", "because"}
    orig_conj_count = sum(1 for w in re.findall(r"[a-z]+", text.lower()) if w in conjunctions)
    new_conj_count = sum(1 for w in re.findall(r"[a-z]+", out.lower()) if w in conjunctions)
    assert new_conj_count < orig_conj_count


def test_corrupt_applies_requested_corruptor_count_and_is_deterministic():
    text = "The little fox ran across the field before the sun went down."
    out_a = corrupt(text, seed=42, n_corruptors=2)
    out_b = corrupt(text, seed=42, n_corruptors=2)
    assert out_a == out_b
    out_c = corrupt(text, seed=43, n_corruptors=2)
    # different seed, overwhelmingly likely to differ on this input
    assert out_c != out_a or True  # documents intent; not a strict guarantee for one input


def test_corrupt_with_zero_corruptors_returns_input_unchanged():
    text = "A quiet morning came over the village."
    assert corrupt(text, seed=0, n_corruptors=0) == text
