# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from scripts.build_poetry_pairs import build_idf, build_pairs, split_poems, top_keywords


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
