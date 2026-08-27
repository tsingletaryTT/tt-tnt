# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from pathlib import Path

from scripts.build_editor_pairs import build_pairs, sample_clean_sentences, select_corpus_files


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


def test_build_pairs_never_emits_draft_equal_to_better_on_a_retriable_short_sentence():
    """"Other websites." is a real short/atypical corpus line (a caption-style fragment):
    at some seed offsets every corruptor in the randomly-chosen set hits its documented
    no-op precondition (`repeat_collapse` on <3 words, etc.) and would have returned the
    sentence unchanged under the old single-attempt code -- silently emitting
    `draft == better`. With the retry loop, across many seeds this sentence must never
    appear in the output with `draft == better`; it is either retried into a genuine change
    (the common case here, since `garble_word` can touch "websites") or dropped."""
    atypical = "Other websites."
    ordinary = "The little fox ran across the field before the sun went down."
    for seed in range(25):
        pairs = build_pairs([atypical, ordinary], seed=seed)
        for p in pairs:
            assert p["draft"] != p["better"]


def test_build_pairs_drops_a_sentence_no_corruptor_can_ever_touch():
    """"Hi ok." has no word 3+ letters long (garble_word's floor), no function/auxiliary
    word (drop_or_double_function_word's requirement), no conjunction (fuse_clauses'
    requirement), and fewer than 3 words (repeat_collapse's floor) -- every corruptor
    no-ops on it regardless of which one(s) `corrupt()` picks or at what seed offset, so no
    number of retries can ever produce a real change. `build_pairs` must drop it rather than
    emit `draft == better`, while still emitting a pair for the ordinary sentence beside it."""
    untouchable = "Hi ok."
    ordinary = "The little fox ran across the field before the sun went down."
    for seed in range(10):
        pairs = build_pairs([untouchable, ordinary], seed=seed)
        assert all(p["better"] != untouchable for p in pairs)
        assert any(p["better"] == ordinary for p in pairs)


def test_select_corpus_files_excludes_known_aggregates(tmp_path):
    (tmp_path / "blend.txt").write_text("aggregate content\n")
    (tmp_path / "corpus.txt").write_text("legacy aggregate content\n")
    (tmp_path / "tinystories.txt").write_text("a per-source file\n")
    (tmp_path / "wikipedia_simple.txt").write_text("another per-source file\n")

    selected = select_corpus_files(tmp_path)

    names = {p.name for p in selected}
    assert names == {"tinystories.txt", "wikipedia_simple.txt"}
    assert "blend.txt" not in names
    assert "corpus.txt" not in names
