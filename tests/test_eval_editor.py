# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from scripts.eval_editor import _word_overlap, recovers_real_words, score_recovery

_VOCAB = {"the", "girl", "found", "a", "small", "silver", "key", "by", "river"}


def test_recovers_real_words_true_for_all_real_words():
    assert recovers_real_words("the girl found a small silver key", _VOCAB) is True


def test_recovers_real_words_false_for_a_fake_word():
    assert recovers_real_words("the girl found a smallury key", _VOCAB) is False


def test_score_recovery_reports_word_overlap_and_fake_word_flag():
    better = "the girl found a small silver key by the river"
    edited_good = "the girl found a small silver key"
    edited_bad = "the girl found a smallury spleck"
    good = score_recovery(better, edited_good, _VOCAB)
    bad = score_recovery(better, edited_bad, _VOCAB)
    assert good["has_fake_word"] is False
    assert bad["has_fake_word"] is True
    assert good["word_overlap"] > bad["word_overlap"]


def test_word_overlap_is_the_shared_helper_behind_score_recovery():
    better = "the girl found a small silver key by the river"
    assert _word_overlap(better, better) == 1.0
    assert _word_overlap(better, "completely unrelated text") == 0.0
    # score_recovery's word_overlap is exactly _word_overlap(better, edited) --
    # the baseline (draft vs better) computed in main() uses the same function so the
    # two numbers this check reports are directly comparable, not two different metrics.
    assert score_recovery(better, "the girl found a small silver key", set())[
        "word_overlap"
    ] == _word_overlap(better, "the girl found a small silver key")
