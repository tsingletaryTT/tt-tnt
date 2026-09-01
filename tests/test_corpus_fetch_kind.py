import pytest

from train.corpus import SOURCES, CorpusSource


def _src(**kw):
    base = dict(name="x", slice="s", target_share=0.01,
                hf_repo="", hf_revision="", license_id="CC0-1.0")
    base.update(kw)
    return CorpusSource(**base)


def test_fetch_kind_defaults_to_hf_so_every_existing_source_is_unchanged():
    """Written when ``fetch_kind`` was added and every registered source was still an HF
    dataset -- it asserted that adding the field changed nothing for them. Task 5 of the
    2026-08-31 long-context-corpus plan registered ``mission``, the first (and, by design,
    only) source that is genuinely NOT a HuggingFace dataset -- NASA pages fetched by
    ``scripts/fetch_mission.py``. Task 6 registered ``pulp_sf``, gated on a per-work
    verified-renewal check rather than a dataset fetch, as a second exclusion. Both
    exclusions are the point of their respective tasks, not a
    regression of the property this test checks for every other source.
    """
    assert all(
        s.fetch_kind == "hf"
        for name, s in SOURCES.items()
        if name not in ("mission", "pulp_sf")
    )


def test_a_url_source_carries_its_url():
    s = _src(fetch_kind="url", source_url="https://example.gov/a.txt")
    assert s.fetch_kind == "url" and s.source_url.startswith("https://")


def test_an_unknown_fetch_kind_is_rejected_at_construction():
    """A typo here would otherwise surface as a silent empty fetch, and an empty slice looks
    exactly like a source that legitimately had no rows."""
    with pytest.raises(ValueError, match="fetch_kind"):
        _src(fetch_kind="ftp")


def test_a_url_source_without_a_url_is_rejected():
    with pytest.raises(ValueError, match="source_url"):
        _src(fetch_kind="url")


def test_an_hf_source_still_requires_a_pinned_revision():
    """The pinning rule is not weakened by adding a second fetch kind."""
    with pytest.raises(ValueError, match="revision"):
        _src(hf_repo="some/repo", hf_revision="")
