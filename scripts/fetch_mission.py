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

Every URL is a real NASA (``.nasa.gov``) page, verified to resolve and to contain prose (not
navigation boilerplate) before being added here -- see the module docstring below and
``.superpowers/sdd/2026-08-31-long-context-corpus/task-5-report.md`` for exactly how. A page
that is HTML wrapped around real content only needs its tags removed; it does not need a
readability/extraction library (and this project adds no new dependencies), so
``strip_tags`` below is a plain regex-based stripper, not a heuristic content extractor.

Two deliberate choices, both directives rather than defaults:

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

#: (label, url) pairs. Every url is a NASA history page fetched and confirmed, by hand, to
#: return real prose before being added -- not guessed from a path pattern. Most of the
#: mission-transcript tree that used to live on nasa.gov has since moved to
#: apollojournals.org (not a .gov host, so out of scope for this slice); of the mission
#: directories checked, only Apollo 11's pages are still live at these paths -- every
#: sibling path tried for Apollo 12/14/15/16/17 returned HTTP 404. A smaller, verified list
#: beats a longer, guessed one.
MISSION_DOCUMENTS: List[Tuple[str, str]] = [
    (
        "apollo11_technical_air_to_ground",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11transcript_tec.html",
    ),
    (
        "apollo11_landing",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11.landing.html",
    ),
    (
        "apollo11_first_step",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11.step.html",
    ),
    (
        "apollo11_eva_mobility",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11.mobility.html",
    ),
    (
        "apollo11_eva_closeout",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11.clsout.html",
    ),
    (
        "apollo11_eva_prep",
        "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11.evaprep.html",
    ),
    (
        "apollo11_post_eva",
        "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11.posteva.html",
    ),
    (
        "apollo11_launch",
        "https://www.nasa.gov/wp-content/uploads/static/history/alsj/a11/a11.launch.html",
    ),
    (
        "apollo11_contingency_sample",
        "https://www.nasa.gov/wp-content/uploads/static/history//alsj/a11/a11ContingencySample.html",
    ),
]

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_RUN_OF_SPACES_RE = re.compile(r"[ \t\f\v]+")


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

    ``limit`` caps how many documents are fetched (0 = all) -- for smoke-testing the fetch
    and strip before committing to the whole slice, per Step 5 of the task brief.
    """
    docs = MISSION_DOCUMENTS[:limit] if limit else MISSION_DOCUMENTS
    for label, url in docs:
        raw_html = _fetch(url)
        text = strip_tags(raw_html)
        if text:
            yield {"text": text}
        else:
            print(f"WARNING: {label} ({url}) produced no text after stripping", file=sys.stderr)


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
