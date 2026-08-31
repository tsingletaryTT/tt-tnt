# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""No network here: these tests check the registered document list and the licence
declaration, never fetch a URL."""
from scripts.fetch_mission import MISSION_DOCUMENTS
from train.corpus import SOURCES


def test_every_mission_document_is_on_a_us_government_host():
    """The licence basis for this slice is 17 USC 105 -- US Government works are public
    domain. That basis holds only for material actually produced by the government, and the
    host is the cheapest available check that it is."""
    for label, url in MISSION_DOCUMENTS:
        assert url.startswith("https://"), label
        assert ".nasa.gov/" in url or ".gov/" in url, f"{label}: {url} is not a .gov host"


def test_the_slice_states_its_licence_basis_rather_than_a_licence_id():
    """There is no SPDX identifier for 'US Government work'. Saying CC0 would be a claim we
    cannot support; the note has to carry the reasoning."""
    s = SOURCES["mission"]
    assert s.fetch_kind == "url"
    assert "17 USC 105" in s.license_note or "Government" in s.license_note


def test_mission_documents_are_distinct():
    urls = [u for _, u in MISSION_DOCUMENTS]
    assert len(urls) == len(set(urls))


def test_mission_documents_are_a_nonempty_list_of_label_url_pairs():
    assert MISSION_DOCUMENTS
    for label, url in MISSION_DOCUMENTS:
        assert isinstance(label, str) and label
        assert isinstance(url, str) and url


def test_strip_tags_removes_markup_but_keeps_prose_and_timestamps():
    """The ruling for this task: HTML tags are markup and must go; mission-elapsed-time
    stamps are content and must not."""
    from scripts.fetch_mission import strip_tags

    raw = (
        "<html><head><style>.x{color:red}</style></head><body>"
        "<p>00 00 01 02  CC  Apollo 11, Houston. You&#39;re go.</p>"
        "<script>var x = 1;</script>"
        "</body></html>"
    )
    out = strip_tags(raw)
    assert "<" not in out and ">" not in out
    assert "00 00 01 02" in out
    assert "Apollo 11, Houston. You're go." in out
    assert "var x = 1" not in out
    assert "color:red" not in out
