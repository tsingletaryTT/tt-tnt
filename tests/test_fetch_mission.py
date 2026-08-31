# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""No network here: these tests check the registered document list and the licence
declaration, never fetch a URL."""
from scripts.fetch_mission import MISSION_DOCUMENTS
from train.corpus import SOURCES


def test_every_mission_document_is_on_a_us_government_host():
    """The licence basis for this slice is 17 USC 105 -- US Government works are public
    domain. That basis holds only for material actually produced by the government, and the
    host is the cheapest available check that it is -- but it is NECESSARY, not SUFFICIENT.
    The Apollo Lunar Surface Journal pages this slice used to include were also on a .gov
    host and were still privately-copyrighted editorial commentary; the second, independent
    check on the document's own text (``assert_no_third_party_copyright_notice``, tested
    below) is what actually catches that, and this test must not be treated as redundant
    with it."""
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


def test_a_document_with_a_copyright_notice_is_refused():
    """The real gate: a .gov host is not proof of US Government authorship. This is what
    caught the Apollo Lunar Surface Journal pages -- privately-authored commentary hosted on
    nasa.gov, each carrying its own explicit copyright line."""
    import pytest
    from scripts.fetch_mission import (
        ThirdPartyCopyrightNoticeError,
        assert_no_third_party_copyright_notice,
    )

    alsj_style = (
        "00 00 01 02 CC Apollo 11, Houston.\n"
        "Corrected Transcript and Commentary Copyright \u00a9 1995 by\n"
        "Eric M. Jones . All rights reserved."
    )
    with pytest.raises(ThirdPartyCopyrightNoticeError) as excinfo:
        assert_no_third_party_copyright_notice(alsj_style, "fixture_doc")
    assert "copyright" in str(excinfo.value).lower()
    assert "Eric M. Jones" in str(excinfo.value)


def test_a_document_with_no_copyright_notice_is_accepted():
    """The negative case: real government transcript text with no third-party claim over it
    must pass cleanly, or the gate would be indistinguishable from one that rejects
    everything."""
    from scripts.fetch_mission import assert_no_third_party_copyright_notice

    clean = (
        "00 00 18 23 CDR Roger. Reading you loud and clear. Our insertion checklist "
        "is complete, and we have no abnormalities."
    )
    assert_no_third_party_copyright_notice(clean, "fixture_doc")  # must not raise


def test_the_gate_catches_every_marker_case_insensitively():
    """Three independent markers, and case must not be a loophole."""
    from scripts.fetch_mission import (
        ThirdPartyCopyrightNoticeError,
        assert_no_third_party_copyright_notice,
    )
    import pytest

    for text in (
        "some prose. COPYRIGHT 2012 by someone.",
        "some prose. \u00a9 2012 by someone.",
        "some prose. ALL RIGHTS RESERVED.",
    ):
        with pytest.raises(ThirdPartyCopyrightNoticeError):
            assert_no_third_party_copyright_notice(text, "fixture_doc")


def test_mission_documents_no_longer_include_the_removed_alsj_pages():
    """Anti-regression for the fix itself: the eight Apollo Lunar Surface Journal pages
    (Eric M. Jones's privately-authored commentary) must not silently come back."""
    labels = {label for label, _ in MISSION_DOCUMENTS}
    for removed in (
        "apollo11_landing",
        "apollo11_first_step",
        "apollo11_eva_mobility",
        "apollo11_eva_closeout",
        "apollo11_eva_prep",
        "apollo11_post_eva",
        "apollo11_launch",
        "apollo11_contingency_sample",
    ):
        assert removed not in labels, f"{removed} was removed for carrying a copyright notice"
