# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from pathlib import Path

from scripts.build_editor_pairs import build_pairs, sample_clean_sentences


def test_sample_clean_sentences_reads_real_lines(tmp_path):
    corpus = tmp_path / "source.txt"
    corpus.write_text(
        "The fox ran across the field.\n"
        "</s>\n"
        "A quiet morning came over the village.\n"
        "She opened the old wooden box.\n"
    )
    sentences = sample_clean_sentences([corpus], n=2, seed=0)
    assert len(sentences) == 2
    for s in sentences:
        assert s.strip() and s.strip() != "</s>"


def test_sample_clean_sentences_is_deterministic():
    corpus = Path(__file__).parent / "fixtures" / "editor_pairs_source.txt"
    corpus.parent.mkdir(exist_ok=True)
    corpus.write_text("One line here.\nAnother line here.\nA third line here.\n")
    a = sample_clean_sentences([corpus], n=2, seed=7)
    b = sample_clean_sentences([corpus], n=2, seed=7)
    assert a == b


def test_build_pairs_draft_differs_from_better():
    sentences = [
        "The little fox ran across the field before the sun went down.",
        "She opened the old wooden box and found a silver key inside.",
    ]
    pairs = build_pairs(sentences, seed=0)
    assert len(pairs) == len(sentences)
    for p in pairs:
        assert set(p.keys()) == {"draft", "better"}
        assert p["draft"] != p["better"]
        assert p["better"] in sentences


def test_build_pairs_is_deterministic():
    sentences = ["A quiet morning came over the village and the birds began to sing."]
    a = build_pairs(sentences, seed=3)
    b = build_pairs(sentences, seed=3)
    assert a == b
