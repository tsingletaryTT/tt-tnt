import pytest

from train.corpus import SOURCES, CorpusSource


def _src(**kw):
    base = dict(name="x", slice="s", target_share=0.01,
                hf_repo="", hf_revision="", license_id="CC0-1.0")
    base.update(kw)
    return CorpusSource(**base)


def test_fetch_kind_defaults_to_hf_so_every_existing_source_is_unchanged():
    assert all(s.fetch_kind == "hf" for s in SOURCES.values())


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
