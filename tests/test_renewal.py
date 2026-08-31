# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import pytest

from train.renewal import RenewalIndex, RenewalRecord, admissible, verify


def _index():
    # A renewal index is a set of (normalised title, author surname, original year).
    return RenewalIndex([
        ("the puppet masters", "heinlein", 1951),
        ("foundation", "asimov", 1951),
    ])


def test_a_work_present_in_the_renewal_index_is_renewed_and_inadmissible():
    r = verify("The Puppet Masters", "Robert A. Heinlein", 1951, _index())
    assert r.renewed is True
    assert not admissible(r)


def test_a_work_absent_from_the_index_is_not_renewed_and_is_admissible():
    r = verify("Second Variety", "Philip K. Dick", 1953, _index())
    assert r.renewed is False
    assert admissible(r)


def test_a_year_outside_the_renewal_window_is_UNKNOWN_not_absent():
    """The index only covers 1929-1963. Absence from it says nothing about a 1971 work, and
    treating 'not in the index' as 'not renewed' would license the entire modern era."""
    r = verify("Something", "Someone", 1971, _index())
    assert r.renewed is None
    assert not admissible(r)


def test_unknown_is_rejected_rather_than_assumed_free():
    """The whole point of this gate: uncertainty rejects. A hedge is not upgraded to a claim."""
    assert not admissible(RenewalRecord("t", "a", 1955, None, "no record consulted"))


def test_matching_ignores_case_punctuation_and_leading_articles():
    r = verify("THE Puppet-Masters", "heinlein, robert a.", 1951, _index())
    assert r.renewed is True


def test_every_record_carries_its_evidence_string():
    """A verdict with no evidence is unauditable, and this gate exists to be audited."""
    r = verify("Second Variety", "Philip K. Dick", 1953, _index())
    assert r.evidence and "1953" in r.evidence


def test_a_suffix_with_no_comma_does_not_swallow_the_surname():
    """'Robert A. Heinlein Jr.' must still match a renewal record filed under 'Heinlein' --
    the old surname-is-last-token rule would extract 'Jr' instead and silently admit a
    work that was genuinely renewed (the dangerous false-negative direction)."""
    r = verify("The Puppet Masters", "Robert A. Heinlein Jr.", 1951, _index())
    assert r.renewed is True


def test_a_multiword_particle_surname_normalises_the_same_with_or_without_a_comma():
    """'L. Sprague de Camp' and 'de Camp, L. Sprague' must resolve to the same surname --
    the old rule would take only the last token ('Camp') from the no-comma form, which
    diverges from the comma form ('de Camp') and would miss a real index entry."""
    index = RenewalIndex([("lest darkness fall", "de camp", 1941)])
    r_no_comma = verify("Lest Darkness Fall", "L. Sprague de Camp", 1941, index)
    r_comma = verify("Lest Darkness Fall", "de Camp, L. Sprague", 1941, index)
    assert r_no_comma.renewed is True
    assert r_comma.renewed is True
