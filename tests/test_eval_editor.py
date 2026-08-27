# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from scripts.eval_editor import recovers_real_words, score_recovery

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
