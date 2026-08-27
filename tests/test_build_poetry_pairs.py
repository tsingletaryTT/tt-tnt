# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from scripts.build_poetry_pairs import (
    MAX_POEM_WORDS,
    build_idf,
    build_pairs,
    split_poems,
    top_keywords,
)


def test_split_poems_splits_on_separator(tmp_path):
    path = tmp_path / "poetry.txt"
    path.write_text("Roses are red\nViolets are blue\n</s>\nThe wind blows cold\n</s>\n")
    poems = split_poems(path)
    assert poems == ["Roses are red\nViolets are blue", "The wind blows cold"]


def test_split_poems_drops_empty_segments(tmp_path):
    path = tmp_path / "poetry.txt"
    path.write_text("A poem here\n</s>\n</s>\nAnother poem\n</s>\n")
    poems = split_poems(path)
    assert poems == ["A poem here", "Another poem"]


def test_build_idf_rare_word_scores_higher_than_common_word():
    poems = ["the cat sat on the mat", "the dog ran in the park",
              "the sun was bright over the hills", "nightingale sang in the moonlight"]
    idf = build_idf(poems)
    assert idf["nightingale"] > idf["the"]


def test_top_keywords_prefers_high_idf_words_present_in_the_poem():
    idf = {"the": 0.1, "moonlight": 2.0, "nightingale": 2.5, "sang": 1.0}
    poem = "the nightingale sang in the moonlight"
    kw = top_keywords(poem, idf, n=2)
    assert kw == ["nightingale", "moonlight"]


def test_build_pairs_covers_both_kinds_and_target_is_real_poem_text():
    poems = [
        "Roses are red\nViolets are blue\nSugar is sweet\nAnd so are you",
        "The wind blows cold across the moor\nAnd shadows creep along the floor",
    ]
    pairs = build_pairs(poems, seed=0)
    assert len(pairs) == len(poems)
    kinds = {p["kind"] for p in pairs}
    assert kinds <= {"continuation", "keywords"}
    for p in pairs:
        if p["kind"] == "continuation":
            assert p["input"] and p["target"]
            assert p["input"] + "\n" + p["target"] == poems[0] or \
                   p["input"] + "\n" + p["target"] == poems[1]
        else:
            assert p["arg"] and p["target"] in poems


def test_build_pairs_is_deterministic():
    poems = ["A quiet field beneath the summer sky\nWhere all the wild things go to sleep"]
    a = build_pairs(poems, seed=5)
    b = build_pairs(poems, seed=5)
    assert a == b


def _make_long_poem(n_lines: int, words_per_line: int = 10) -> str:
    """A poem with `n_lines` lines of `words_per_line` distinct words each, so word count
    is exactly n_lines * words_per_line and truncation boundaries are easy to reason about."""
    lines = [
        " ".join(f"w{i}_{j}" for j in range(words_per_line)) for i in range(n_lines)
    ]
    return "\n".join(lines)


def test_build_pairs_truncates_long_poem_in_keywords_template():
    # 3 lines (< 4, so the continuation branch is never eligible -- always "keywords"),
    # 150 words per line = 450 words total, well over MAX_POEM_WORDS.
    long_poem = _make_long_poem(n_lines=3, words_per_line=150)
    pairs = build_pairs([long_poem], seed=0)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["kind"] == "keywords"
    target_words = pair["target"].split()
    assert len(target_words) <= MAX_POEM_WORDS
    # Truncation drops from the END: the target is a real PREFIX of the original poem.
    assert long_poem.startswith(pair["target"])
    assert pair["target"] != long_poem


def test_build_pairs_truncates_long_poem_in_continuation_template():
    # seed=1 makes a single >=4-line poem draw < 0.5, i.e. the "continuation" branch.
    long_poem = _make_long_poem(n_lines=40, words_per_line=10)  # 400 words
    pairs = build_pairs([long_poem], seed=1)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["kind"] == "continuation"
    combined = pair["input"] + "\n" + pair["target"]
    combined_words = combined.split()
    assert len(combined_words) <= MAX_POEM_WORDS
    # Truncation drops from the END: the combined input+target is a real PREFIX of the
    # original poem (the cut was computed on the already-capped poem, not the full one).
    assert long_poem.startswith(combined)
    assert combined != long_poem


def test_build_pairs_leaves_short_poem_unchanged_in_both_templates():
    short_poem_keywords = _make_long_poem(n_lines=3, words_per_line=5)  # 15 words
    pairs = build_pairs([short_poem_keywords], seed=0)
    assert pairs[0]["kind"] == "keywords"
    assert pairs[0]["target"] == short_poem_keywords

    short_poem_continuation = _make_long_poem(n_lines=10, words_per_line=5)  # 50 words
    pairs = build_pairs([short_poem_continuation], seed=1)
    assert pairs[0]["kind"] == "continuation"
    combined = pairs[0]["input"] + "\n" + pairs[0]["target"]
    assert combined == short_poem_continuation
