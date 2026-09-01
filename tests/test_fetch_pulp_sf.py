# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""No network here: these tests check selection logic and the registry entry, never fetch a
URL. `pulp_sf` is registered with target_share=0.0 and carries no documents until the real
CCE/Stanford renewal index is ingested -- see the module docstring of
scripts/fetch_pulp_sf.py for why that is correct and not a placeholder oversight."""
import json

import pytest

from scripts.fetch_pulp_sf import select_admissible
from train.corpus import SOURCES
from train.renewal import RenewalIndex


def _index():
    return RenewalIndex([("foundation", "asimov", 1951)])


CANDIDATES = [
    {"title": "Foundation", "author": "Isaac Asimov", "year": 1951, "url": "https://x/1"},
    {"title": "Second Variety", "author": "Philip K. Dick", "year": 1953, "url": "https://x/2"},
    {"title": "Neuromancer", "author": "William Gibson", "year": 1984, "url": "https://x/3"},
]


def test_only_verified_non_renewals_are_selected():
    kept, records = select_admissible(CANDIDATES, _index())
    assert [k["title"] for k in kept] == ["Second Variety"]


def test_a_renewed_work_is_rejected_and_the_rejection_is_recorded():
    _, records = select_admissible(CANDIDATES, _index())
    foundation = next(r for r in records if r.title == "Foundation")
    assert foundation.renewed is True and not foundation.evidence == ""


def test_a_post_window_work_is_rejected_as_UNKNOWN_not_kept():
    """1984 is outside 1929-1963, so the index says nothing about it. Keeping it would
    license the entire modern era on an absence of evidence."""
    _, records = select_admissible(CANDIDATES, _index())
    neuromancer = next(r for r in records if r.title == "Neuromancer")
    assert neuromancer.renewed is None


def test_every_candidate_produces_a_record_even_when_rejected():
    """The audit trail is the point: a reader must be able to see what was excluded and why,
    not only what survived."""
    _, records = select_admissible(CANDIDATES, _index())
    assert len(records) == len(CANDIDATES)


def test_empty_candidate_list_yields_no_kept_and_no_records():
    kept, records = select_admissible([], _index())
    assert kept == [] and records == []


def test_pulp_sf_is_registered_with_no_target_share_and_a_url_fetch_kind():
    """The slice is gated and registered but must not silently claim any share of the blend
    until the real renewal index exists -- there are no admissible documents to back it."""
    s = SOURCES["pulp_sf"]
    assert s.target_share == 0.0
    assert s.fetch_kind == "url"


def test_pulp_sf_license_note_states_the_verified_non_renewal_basis():
    s = SOURCES["pulp_sf"]
    note = s.license_note.lower()
    assert "renewal" in note
    assert "renewal_records.jsonl" in s.license_note
    assert "host" in note or "assert" in note


def test_pulp_sf_license_id_is_empty_pending_verification():
    """No SPDX identifier applies to 'public domain via unrenewed copyright' any more than
    one applies to a statutory basis -- and here there isn't even a verified basis yet."""
    assert SOURCES["pulp_sf"].license_id == ""
