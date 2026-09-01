# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Registry anti-drift tests.

The registry is the single place a source's licence, slice and fetch spec live. A source
missing any of them is a source whose provenance cannot be stated, which this project
treats as a defect rather than an omission.
"""
import pytest
from train.corpus import SLICES, SOURCES, CorpusSource, get_source, total_target_share

ALL = sorted(SOURCES)


@pytest.mark.parametrize("name", ALL)
def test_every_source_declares_a_licence(name):
    """``pulp_sf`` is the one deliberate exception: it has no admissible documents yet (no
    real CCE/Stanford renewal index has been ingested -- see train/corpus.py and
    scripts/fetch_pulp_sf.py), so there is nothing to license and no SPDX identifier for
    "verified non-renewal, pending" -- ``license_id`` is empty on purpose. Its basis must
    still be stated in words in ``license_note``, which this test checks instead."""
    src = SOURCES[name]
    if name == "pulp_sf":
        assert src.license_id == ""
        assert "renewal" in src.license_note.lower()
        return
    assert src.license_id, f"{name}: no license_id"
    assert src.license_url, f"{name}: no license_url"


@pytest.mark.parametrize("name", ALL)
def test_every_source_has_a_known_slice(name):
    assert SOURCES[name].slice in SLICES


@pytest.mark.parametrize("name", ALL)
def test_every_source_has_a_resolvable_fetch_spec(name):
    src = SOURCES[name]
    if src.fetch_kind == "url":
        # A url-kind source pins by being a fixed URL rather than a dataset revision --
        # "mission" is fetched from several such URLs (scripts/fetch_mission.py's
        # MISSION_DOCUMENTS), of which source_url is one representative anchor, not the
        # whole fetch spec. The resolvability requirement this test enforces still holds:
        # a real, https, non-empty URL rather than a placeholder.
        assert src.source_url, f"{name}: fetch_kind='url' with no source_url"
        assert src.source_url.startswith("https://"), (
            f"{name}: source_url '{src.source_url}' is not https"
        )
        return
    assert src.hf_repo, f"{name}: no hf_repo"
    assert src.hf_revision, f"{name}: no pinned revision — fetches must be reproducible"
    # Reject branch refs like "main", "master", "HEAD" — must be a pinned commit sha
    branch_like = {"main", "master", "head", "develop", "development"}
    assert src.hf_revision.lower() not in branch_like, (
        f"{name}: hf_revision '{src.hf_revision}' is a branch ref, not a pinned commit"
    )
    # Assert it looks like a 40-character hex sha
    assert len(src.hf_revision) == 40 and all(
        c in "0123456789abcdef" for c in src.hf_revision.lower()
    ), f"{name}: hf_revision '{src.hf_revision}' is not a 40-char commit sha"


@pytest.mark.parametrize("name", ALL)
def test_upsample_is_at_least_one(name):
    assert SOURCES[name].upsample >= 1


def test_target_shares_sum_to_one():
    total = total_target_share()
    assert abs(total - 1.0) < 1e-9, f"target shares sum to {total:.4f}, not 1.0"


def test_get_source_rejects_unknown_with_a_helpful_message():
    with pytest.raises(KeyError) as excinfo:
        get_source("not-a-source")
    assert "registered sources" in str(excinfo.value)


def test_flavour_sources_are_capped():
    """Upsampled flavour sources risk memorisation; the cap is the control."""
    for name, src in SOURCES.items():
        if src.slice == "flavour":
            assert src.upsample <= 8, f"{name}: upsample {src.upsample} exceeds the cap"


def test_corpus_module_does_no_io():
    """The registry is data and arithmetic. I/O belongs in scripts/."""
    import inspect
    import train.corpus as m
    src = inspect.getsource(m)
    for forbidden in ("open(", "requests.", "urllib", "load_dataset", "snapshot_download"):
        assert forbidden not in src, f"train/corpus.py performs I/O: {forbidden}"


def test_spine_is_broad_enough_to_avoid_heavy_repetition():
    """spine had 53 books against a 12% share -- 10x repetition, over the cap of 8.

    Every author here was verified present in the Gutenberg catalogue before being added.
    The count guards against the slice silently narrowing again.
    """
    spine = SOURCES["spine"]
    assert len(spine.authors) >= 17, (
        f"spine has only {len(spine.authors)} author selectors; it was broadened to avoid "
        f"needing 10x upsample"
    )
    for required in ("Fabre, Jean-Henri", "Fort, Charles", "Thoreau, Henry David",
                     "Darwin, Charles", "Jefferies, Richard", "Flammarion, Camille"):
        assert required in spine.authors, f"spine lost its {required!r} selector"


def test_spine_and_folklore_do_not_share_selectors():
    """Andrew Lang belongs to folklore. Listing him in both would double-count him."""
    overlap = set(SOURCES["spine"].authors) & set(SOURCES["folklore"].authors)
    assert not overlap, f"spine and folklore share author selectors: {sorted(overlap)}"


def test_spine_and_weird_do_not_share_selectors():
    """Browne belongs to weird. Listing him in both would double-count him.

    KNOWN, DELIBERATE GAP: this only checks for a shared SELECTOR, not a shared BOOK. Gutenberg
    text_id 30092, "Lords of the Housetops: Thirteen Cat Tales", is a 14-contributor anthology
    matched independently by spine's "Hudson, W. H." and weird's "Blackwood, Algernon" -- no
    single author name is in both lists, so this test correctly cannot see the overlap. Impact
    is one book out of ~296 selected, and the blend built from these selectors is already
    frozen (artifacts/corpus/blend.txt, blend_manifest.json) -- this is a documented, accepted
    duplicate, not an oversight to fix by re-selecting or re-blending.
    """
    overlap = set(SOURCES["spine"].authors) & set(SOURCES["weird"].authors)
    assert not overlap, f"spine and weird share author selectors: {sorted(overlap)}"


# --- share formatting: ":.0%" rendered the smallest slice as "0%".


def test_format_share_keeps_a_fraction_of_a_percent():
    from train.corpus import format_share
    assert format_share(0.005) == "0.5%"
    assert format_share(0.135) == "13.5%"
    assert format_share(0.00575) == "0.575%"   # flavour's arithmetic ceiling at 4x


def test_format_share_leaves_whole_percentages_whole():
    from train.corpus import format_share
    assert format_share(0.31) == "31%"
    assert format_share(0.04) == "4%"
    assert format_share(1.0) == "100%"


def test_no_registered_share_renders_as_zero():
    """A slice that reads as 0% reads as "contributes nothing".

    The guard is against a NONZERO share that rounds away to nothing (``flavour``'s original
    2.00% bug this test was written for). A source whose ``target_share`` is *exactly* 0.0 is
    a different case: a deliberately unsettled placeholder for a source newly registered but
    not yet given a real share (``longform``, staged by the 2026-08-31 long-context-corpus
    plan's Task 4, awaiting Task 7's re-settle). That state is meant to read as "0%" -- it
    IS zero -- so it is excluded here rather than made to pass some other way.
    """
    from train.corpus import SOURCES, format_share
    for name, src in SOURCES.items():
        if src.target_share == 0.0:
            continue
        assert format_share(src.target_share) != "0%", name


def test_describe_shows_a_fractional_share():
    from train.corpus import SOURCES
    assert "0.5%" in SOURCES["flavour"].describe()


# --- rationale anti-drift.
#
# Every share in this registry was settled against a measurement, and the rationale is
# where that reasoning is written down. Three separate reviews on this branch found
# rationales still quoting superseded numbers -- availability from a retired tokenizer,
# an upsample computed at a share the source no longer holds. Prose is the only part of
# the registry nothing else checks, so check it here.

def test_a_rationale_that_cites_availability_cites_the_CURRENT_availability():
    """Historical figures are fine and useful -- they show how a share was arrived at --
    but the number in force has to be in there too, or the rationale describes a
    measurement that no longer exists."""
    import json
    from pathlib import Path
    from train.corpus import SOURCES

    report = (Path(__file__).resolve().parents[1]
              / "docs" / "measurements" / "corpus_availability.json")
    available = json.loads(report.read_text())["available"]
    for name, src in SOURCES.items():
        text = src.rationale.lower()
        if "availability" not in text and "measured tokens" not in text:
            continue
        assert f"{available[name]:,}" in src.rationale, (
            f"{name}'s rationale cites availability but not the current "
            f"{available[name]:,} tokens from {report.name}"
        )


def test_no_rationale_claims_an_upsample_the_registry_does_not_declare():
    """`upsample=N` written into prose next to a different declared N is how a reader
    learns to distrust the whole file."""
    import re
    from train.corpus import SOURCES
    for name, src in SOURCES.items():
        for claimed in re.findall(r"upsample\s*=\s*(\d+)", src.rationale):
            # A rationale may recount that a LOWER factor was tried and failed, but must
            # not present one as the factor in force.
            assert int(claimed) <= src.upsample, (
                f"{name}'s rationale claims upsample={claimed}, registry declares "
                f"{src.upsample}"
            )


def test_longform_is_registered_with_an_open_licence_and_a_pinned_revision():
    s = SOURCES["longform"]
    assert s.fetch_kind == "hf"
    assert s.hf_revision, "an unpinned fetch is not reproducible"
    assert s.license_id, "a source with no licence id cannot be rendered into the model card"


def test_longform_exists_for_document_LENGTH_and_says_so():
    """A rationale that does not state why the source is here is prose, not provenance --
    and this repo has a gate that fails when a rationale goes stale."""
    r = SOURCES["longform"].rationale.lower()
    assert "long" in r and ("2048" in r or "document" in r)


#: Measured 2026-08-31 (scripts/measure_document_lengths.py over artifacts/corpus/*.txt):
#: every one of these is a slice of whole books, 56k-97k tokens median, 100% of their tokens
#: in documents >= 2048. They are the only sources that can carry gate 3.
BOOK_SOURCES = ("folklore", "spine", "weird", "gutenberg_children", "grimoire")


def test_the_book_slices_hold_the_share_gate_3_needs():
    """Gate 3 needs >=40% of TOKENS in documents >=2048. Only whole-book sources supply those:
    longform manages 43.7% of its own tokens, wikipedia_simple 22.3%, poetry and tinystories
    0.0%. If the book slices are small, the gate cannot pass however the rest is arranged."""
    books = sum(SOURCES[n].target_share for n in BOOK_SOURCES if n in SOURCES)
    assert books >= 0.40, f"book slices hold {books:.1%}, need >=40%"


def test_tinystories_is_no_longer_the_largest_slice():
    """It was 31% at 198 tokens median and 0% above threshold -- the single biggest obstacle to
    the gate. This does not mandate a particular value, only that it stopped dominating."""
    ts = SOURCES["tinystories"].target_share
    assert ts <= max(SOURCES[n].target_share for n in BOOK_SOURCES if n in SOURCES)
