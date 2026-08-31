#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Fetch NASA mission-transcript documents into ``artifacts/raw/mission/text.jsonl``.

The ``mission`` source (``train/corpus.py::SOURCES["mission"]``) is not a HuggingFace
dataset: it is a small, hand-verified list of individual pages on a NASA web host. Unlike
every other ``fetch_kind="url"`` source, which is one document at one URL,
``MISSION_DOCUMENTS`` below is a LIST of ``(label, url)`` pairs, each fetched and written as
its own document -- ``scripts/fetch_corpus.py``'s generic ``_iter_url_rows`` only knows how
to fetch a single ``source.source_url``, so this source is fetched by this dedicated script
rather than through the generic path.

A page that is HTML wrapped around real content only needs its tags removed; it does not
need a readability/extraction library (and this project adds no new dependencies), so
``strip_tags`` below is a plain regex-based stripper, not a heuristic content extractor.

CORRECTED 2026-08-31: the first version of this slice included eight Apollo Lunar Surface
Journal pages alongside the raw transcript, on the strength of a single check -- ".gov host,
and the one page I opened by hand was prose, not navigation." That check was NECESSARY but
not SUFFICIENT: it verifies the page is real content, not that the government produced it.
A .gov host can serve someone else's copyrighted writing just as easily as it can serve a
government work, and seven of those eight pages carry an explicit, unambiguous notice in
their own text -- "Corrected Transcript and Commentary Copyright (c) 1995 by Eric M. Jones.
All rights reserved." (the eighth, the contingency-sample page, carries the 2012 variant
crediting Rene Cantin and Eric M. Jones). The Apollo Lunar Surface Journal is Eric M. Jones's
privately-authored editorial project -- annotated commentary and "corrected" transcripts --
merely HOSTED on nasa.gov; 17 USC 105 does not reach it just because of where it is served
from. Removed. See ``assert_no_third_party_copyright_notice`` below, which now makes this
class of mistake fail loudly instead of silently, and
``.superpowers/sdd/2026-08-31-long-context-corpus/task-5-report.md`` for the full account.

Two deliberate choices, both directives rather than defaults, for the text that remains:

- Mission-elapsed-time stamps ("00 00 01 02") are KEPT. They are part of the document's real
  structure, not markup, and removing them would be an editorial change to source material
  this project would then have to describe honestly in its provenance. They are also exactly
  the kind of low-information-but-real context this project has repeatedly found small models
  are good at learning to ignore.
- HTML tags themselves ARE stripped, because they are markup, not content, and
  ``scripts/prepare_corpus.py`` would otherwise treat ``<`` and ``>`` runs as ordinary text.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_corpus import write_documents  # noqa: E402
from train.paths import shared_dir  # noqa: E402

#: (label, url) pairs -- ONLY documents verified to be free of a third-party copyright
#: notice, per ``assert_no_third_party_copyright_notice`` below, in addition to being on a
#: .gov host. The eight Apollo Lunar Surface Journal pages this list used to carry (Eric M.
#: Jones's privately-authored editorial commentary, merely hosted on nasa.gov) were removed
#: for exactly that reason -- see the module docstring's "CORRECTED" note. What remains is
#: the raw Technical Air-to-Ground Voice Transcription: a verbatim transcription of the
#: actual air-to-ground radio traffic, with no separate editorial authorship claimed over it
#: anywhere in its own text, which is what a genuine US Government work looks like.
#:
#: A .gov host is necessary but never sufficient on its own -- see the two-check gate in
#: ``iter_mission_rows``. Do not add a document here on host alone.
MISSION_DOCUMENTS: List[Tuple[str, str]] = [
    (
        "apollo11_technical_air_to_ground",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11transcript_tec.html",
    ),
]

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_RUN_OF_SPACES_RE = re.compile(r"[ \t\f\v]+")

#: Case-insensitive markers of a third-party copyright claim. Any one of these appearing in
#: fetched text means someone other than the US Government is asserting authorship over it,
#: regardless of what host served the page -- see ``assert_no_third_party_copyright_notice``.
_COPYRIGHT_MARKERS = ("copyright", "©", "all rights reserved")


class ThirdPartyCopyrightNoticeError(ValueError):
    """Raised when fetched text carries its own copyright claim.

    A .gov host only tells you who SERVED a page, not who WROTE it -- the Apollo Lunar
    Surface Journal pages this slice used to include were served from nasa.gov and were
    still Eric M. Jones's own copyrighted commentary. This exception is the second,
    independent check: it reads the document's own words rather than trusting its address.
    """


def assert_no_third_party_copyright_notice(text: str, label: str) -> None:
    """Refuse ``text`` if it carries an explicit copyright notice.

    This is deliberately a SEPARATE check from the .gov-host test in
    ``tests/test_fetch_mission.py``, not a replacement for it: the host test is necessary
    (17 USC 105 only ever applies to something the government produced) but not sufficient
    (a .gov server can and does host privately-authored, separately-copyrighted material).
    Do not delete the host test on the theory that this one makes it redundant -- a document
    could in principle carry no notice at all yet still not be a government work, and the
    host test is what stands watch over that case.
    """
    lowered = text.lower()
    for marker in _COPYRIGHT_MARKERS:
        idx = lowered.find(marker.lower())
        if idx != -1:
            start = max(0, idx - 40)
            end = min(len(text), idx + len(marker) + 60)
            raise ThirdPartyCopyrightNoticeError(
                f"{label}: found a copyright notice ({marker!r}) in the fetched text -- "
                f"this document is NOT admissible as a US Government work under 17 USC 105 "
                f"purely because it was served from a .gov host. Matched text: "
                f"{text[start:end]!r}"
            )


def strip_tags(raw_html: str) -> str:
    """Strip HTML markup from one mission-transcript page, keeping the prose (and the
    mission-elapsed-time stamps) intact.

    Deliberately simple: this is real content wrapped in tags, not an article buried in page
    furniture, so a plain regex removal is enough -- reaching for a readability/extraction
    library would be solving a problem this source does not have.
    """
    text = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    lines = [_RUN_OF_SPACES_RE.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _fetch(url: str, timeout: int = 120) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        raw = fh.read()
    return raw.decode("utf-8", errors="replace")


def iter_mission_rows(limit: int = 0) -> Iterator[Dict[str, object]]:
    """Fetch and strip each of ``MISSION_DOCUMENTS``, yielding ``{"text": ...}`` rows.

    Every document is checked with ``assert_no_third_party_copyright_notice`` before being
    yielded -- a hard refusal (the exception propagates), not a skip-and-warn, because a
    document that fails this check has no business anywhere in the corpus, not merely in
    this run's output.

    ``limit`` caps how many documents are fetched (0 = all) -- for smoke-testing the fetch
    and strip before committing to the whole slice, per Step 5 of the task brief.
    """
    docs = MISSION_DOCUMENTS[:limit] if limit else MISSION_DOCUMENTS
    for label, url in docs:
        raw_html = _fetch(url)
        text = strip_tags(raw_html)
        if not text:
            print(f"WARNING: {label} ({url}) produced no text after stripping", file=sys.stderr)
            continue
        assert_no_third_party_copyright_notice(text, label)
        yield {"text": text}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=0,
                   help="Fetch only the first N documents (0 = all). For smoke tests.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override the destination path (default: artifacts/raw/mission/text.jsonl).")
    args = p.parse_args()

    dest = args.out or (shared_dir("raw") / "mission" / "text.jsonl")
    n = write_documents(iter_mission_rows(args.limit), dest)
    print(f"mission: {n:,} documents -> {dest}")
    if n == 0:
        print("WARNING: mission produced no documents", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
