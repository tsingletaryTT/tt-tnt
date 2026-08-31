# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import numpy as np
import pytest

from scripts.measure_document_lengths import document_lengths, length_report

SEP = 2


def test_lengths_are_the_gaps_between_separators():
    # three documents of 3, 1 and 4 tokens, separated by id 2
    ids = np.array([9, 9, 9, SEP, 7, SEP, 5, 5, 5, 5, SEP], dtype=np.uint32)
    np.testing.assert_array_equal(document_lengths(ids, SEP), [3, 1, 4])


def test_a_trailing_partial_document_is_not_counted():
    """Text after the last separator is an unterminated fragment, not a document. Counting
    it would inflate the count with a document whose true length is unknown."""
    ids = np.array([9, 9, SEP, 7, 7, 7], dtype=np.uint32)
    np.testing.assert_array_equal(document_lengths(ids, SEP), [2])


def test_tokens_in_long_documents_is_not_the_same_as_documents_that_are_long():
    """Gate 3 is about TOKENS, because a corpus can be 99% short documents by count while
    most of its tokens live in a few long ones -- which is exactly what tokens-v4 looks
    like (median 113, mean 1031)."""
    ids = np.array([1] * 10 + [SEP] + [1] * 990 + [SEP], dtype=np.uint32)
    rep = length_report(document_lengths(ids, SEP), thresholds=[100])
    assert rep["docs_at_least"][100] == pytest.approx(0.5)
    assert rep["tokens_in_docs_at_least"][100] == pytest.approx(990 / 1000)


def test_report_states_the_distribution_not_just_a_mean():
    """A mean is not a finding here: tokens-v4's mean is 1031 and its median is 113, and
    only the median describes a typical document."""
    rep = length_report(np.array([1, 1, 1, 1000]), thresholds=[])
    assert rep["count"] == 4
    assert rep["median"] == pytest.approx(1.0)
    assert rep["mean"] == pytest.approx(250.75)


def test_an_array_with_no_separators_reports_no_documents_rather_than_one():
    """A corpus with no separators is the pre-2026-08-14 bug, not a single huge document."""
    rep = length_report(document_lengths(np.array([1, 1, 1]), SEP), thresholds=[10])
    assert rep["count"] == 0
    assert rep["tokens_in_docs_at_least"][10] == 0.0
