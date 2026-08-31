# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Was this work's copyright renewed? The gate `pulp_sf` admits texts through.

US works published 1929-1963 required a renewal in their 28th year. NYPL's Catalog of
Copyright Entries project found that only about 25% of registered books were renewed, which
is why this window is worth opening at all -- and why it must be opened per work rather than
per collection.

WHY THIS EXISTS RATHER THAN TRUSTING A HOST. Project Gutenberg hosts much of the material
this slice wants and asserts it is public domain. That assertion has been documented (Locus,
2010) as resting on a theory that is correct for some works and wrong for others, with no
distinction drawn between them. The texts may well be fine; PG's say-so is not the basis on
which this project may use them. A verified renewal record is.

UNKNOWN REJECTS. `renewed=None` -- a year outside the window, a work the index cannot speak
to -- is not admissible. Absence of evidence is not evidence of absence, and the failure mode
this guards is publishing a model trained on copyrighted fiction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple

#: The renewal requirement applies to works published in these years. Outside it, absence
#: from the index means nothing.
RENEWAL_WINDOW = (1929, 1963)

_ARTICLES = ("the ", "a ", "an ")


def _norm_title(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
    return s


def _norm_author(s: str) -> str:
    """Surname only, lowercased. Handles both 'Robert A. Heinlein' and 'Heinlein, Robert A.'"""
    s = s.strip()
    surname = s.split(",")[0] if "," in s else s.split()[-1] if s.split() else s
    return re.sub(r"[^a-z]+", "", surname.lower())


@dataclass(frozen=True)
class RenewalRecord:
    title: str
    author: str
    year: int
    #: True = renewed (still in copyright), False = verified not renewed, None = unknown.
    renewed: Optional[bool]
    evidence: str


class RenewalIndex:
    """Normalised (title, author-surname, year) triples of works KNOWN to be renewed."""

    def __init__(self, entries: Iterable[Tuple[str, str, int]]) -> None:
        self._e: Set[Tuple[str, str, int]] = {
            (_norm_title(t), _norm_author(a), int(y)) for t, a, y in entries
        }

    def __contains__(self, key: Tuple[str, str, int]) -> bool:
        return key in self._e

    def __len__(self) -> int:
        return len(self._e)


def verify(title: str, author: str, year: int, index: RenewalIndex) -> RenewalRecord:
    lo, hi = RENEWAL_WINDOW
    if not (lo <= year <= hi):
        return RenewalRecord(title, author, year, None,
                             f"{year} is outside the renewal window {lo}-{hi}; this index "
                             f"cannot speak to it")
    key = (_norm_title(title), _norm_author(author), int(year))
    if key in index:
        return RenewalRecord(title, author, year, True,
                             f"renewal record found for {title!r} ({year})")
    return RenewalRecord(title, author, year, False,
                         f"no renewal record for {title!r} ({year}) in an index of "
                         f"{len(index)} entries covering {lo}-{hi}")


def admissible(record: RenewalRecord) -> bool:
    """Only a VERIFIED non-renewal admits a work. Unknown rejects."""
    return record.renewed is False
