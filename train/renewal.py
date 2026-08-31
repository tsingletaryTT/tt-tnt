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

DESIGN NOTE ON AUTHOR MATCHING (read this before touching `_surname_candidates`). Three
rounds of review each found a real false-negative in a single-committed-surname-guess
parser: a trailing suffix swallowing the surname ("Robert A. Heinlein Jr." -> "jr"), a
particle dropped from a compound surname ("L. Sprague de Camp" -> "camp" instead of "de
camp"), and a real surname that collides with a suffix word ("Naomasa Ii" -> "naomasa", and
even after a guard narrowly aimed at THAT case, "Naomasa T. Ii" -> "t"). Every fix closed the
reported case and missed the class, because free-text author names do not have one correct
parse without name-frequency data this module does not have.

So `verify` does not commit to one surname. `_surname_candidates` returns every plausible
surname reading of a name, and a work is treated as renewed if ANY candidate matches the
index for that title and year. This is deliberately biased toward the cheap error: this
gate's two error directions are not symmetric. A false POSITIVE (treating an unrenewed work
as renewed) costs one usable text. A false NEGATIVE (treating a renewed work as not renewed)
admits copyrighted fiction into a corpus for a published model. Trying every plausible
reading and rejecting on any hit makes the parser's remaining imperfection harmless rather
than load-bearing -- which is the property three rounds of patching a single-guess parser
could not achieve.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

#: The renewal requirement applies to works published in these years. Outside it, absence
#: from the index means nothing.
RENEWAL_WINDOW = (1929, 1963)

_ARTICLES = ("the ", "a ", "an ")

#: Generational/honorific suffixes that are NOT part of a surname. Case-insensitive, with or
#: without a trailing period (checked after lowercasing and stripping trailing dots).
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}

#: Lowercase name particles that can lead a compound surname (e.g. "de Camp", "van
#: Beethoven", or a chain like "de la Cruz"). Used only to build a CANDIDATE surname reading
#: on the query side -- see `_particle_chain` -- never to commit to a single interpretation.
_PARTICLES = {"de", "van", "von", "du", "della", "la", "le"}


def _norm_title(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
    return s


def _normalize_tokens(tokens: Iterable[str]) -> str:
    """Join tokens with a space, lowercase, and strip everything but letters."""
    return re.sub(r"[^a-z]+", "", " ".join(tokens).lower())


def _normalize_author_key(s: str) -> str:
    """Normalise an author string that is ALREADY a bare surname, as `RenewalIndex` entries
    are expected to be (see its own docstring) -- e.g. "Heinlein" or "de Camp". No
    suffix-stripping or particle-chain guessing is applied here: an index entry does not
    need a surname guessed out of it, because it already is one. Guessing is only needed on
    the query side, where a full free-text author name is given -- see
    `_surname_candidates`.
    """
    return _normalize_tokens(s.strip().split())


def _strip_all_suffix_tokens(tokens: List[str]) -> List[str]:
    """Drop ALL trailing generational/honorific tokens (Jr., III, PhD, ...), however many,
    with no minimum-remaining-tokens guard.

    Earlier rounds of this module tried to make this safe by refusing to strip below some
    token count (Round 2's `>= 3` guard) -- and it was still wrong, because a middle initial
    ("Naomasa T. Ii") pads the token count without changing which token is the real surname.
    The guard is unnecessary now: this function's result is only ONE entry in a candidate
    SET (`_surname_candidates`), and the un-stripped tokens remain available as other
    candidates regardless of what this function does. It is safe to strip aggressively
    because nothing downstream treats this as the only reading.
    """
    tokens = list(tokens)
    while tokens:
        bare = tokens[-1].strip(".").lower()
        if bare in _SUFFIXES:
            tokens = tokens[:-1]
        else:
            break
    return tokens


def _particle_chain(tokens: List[str]) -> Optional[List[str]]:
    """Return the trailing span starting at the first of a run of lowercase particles
    immediately before the final token -- e.g. ["de", "la", "Cruz"] out of
    ["Maria", "de", "la", "Cruz"], or ["de", "Camp"] out of ["L.", "Sprague", "de", "Camp"].
    Returns None if the token before the last one isn't a particle (including when there
    are fewer than two tokens at all).

    Walking back through a whole RUN of particles (not just one lookback) is what closes the
    multi-particle gap earlier rounds of this module documented as a known limitation
    ("Ludwig van der Berg", "de la Cruz") rather than fixed.
    """
    if len(tokens) < 2:
        return None
    start = len(tokens) - 1
    while start - 1 >= 0 and tokens[start - 1].strip(".").lower() in _PARTICLES:
        start -= 1
    if start == len(tokens) - 1:
        return None
    return tokens[start:]


def _surname_candidates(author: str) -> Set[str]:
    """Every plausible normalised surname reading of a full author name.

    At minimum, and unconditionally (see the module docstring for why unconditionally):
    - the whole name (or, with a comma, everything before it) joined as one string --
      covers a comma-form surname verbatim, e.g. "de Camp, L. Sprague" -> "de Camp";
    - the last token alone -- covers an ordinary "Given Surname" name, and also covers a
      short surname that happens to collide with a suffix word ("Naomasa Ii", or with a
      middle initial in between, "Naomasa T. Ii" -- the last token is always tried, whatever
      else precedes it);
    - the last token after stripping trailing generational/honorific suffixes -- covers
      "Robert A. Heinlein Jr." -> "heinlein";
    - a chain of lowercase particles immediately before the final token, with and without
      suffix-stripping applied first -- covers "L. Sprague de Camp" -> "de camp" and
      "Maria de la Cruz" -> "de la cruz".

    Every candidate is tried independently in `verify`; a match on ANY of them is enough.
    """
    author = author.strip()
    if "," in author:
        tokens = author.split(",", 1)[0].split()
    else:
        tokens = author.split()
    if not tokens:
        return {re.sub(r"[^a-z]+", "", author.lower())}

    candidates: Set[str] = set()
    candidates.add(_normalize_tokens(tokens))
    candidates.add(_normalize_tokens(tokens[-1:]))

    stripped = _strip_all_suffix_tokens(tokens)
    if stripped:
        candidates.add(_normalize_tokens(stripped[-1:]))

    chain = _particle_chain(tokens)
    if chain:
        candidates.add(_normalize_tokens(chain))
    stripped_chain = _particle_chain(stripped)
    if stripped_chain:
        candidates.add(_normalize_tokens(stripped_chain))

    candidates.discard("")
    return candidates


@dataclass(frozen=True)
class RenewalRecord:
    title: str
    author: str
    year: int
    #: True = renewed (still in copyright), False = verified not renewed, None = unknown.
    renewed: Optional[bool]
    evidence: str


class RenewalIndex:
    """Normalised (title, author-surname, year) triples of works KNOWN to be renewed.

    Author entries are expected to already be bare surnames and are normalised with
    `_normalize_author_key`, NOT the candidate-guessing `_surname_candidates` used on the
    query side in `verify` -- an index entry doesn't need a surname guessed out of it,
    because it already is one.
    """

    def __init__(self, entries: Iterable[Tuple[str, str, int]]) -> None:
        self._e: Set[Tuple[str, str, int]] = {
            (_norm_title(t), _normalize_author_key(a), int(y)) for t, a, y in entries
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
    title_norm = _norm_title(title)
    candidates = sorted(_surname_candidates(author))
    for candidate in candidates:
        if (title_norm, candidate, year) in index:
            return RenewalRecord(title, author, year, True,
                                 f"renewal record found for {title!r} ({year}): author "
                                 f"{author!r} matches via surname candidate {candidate!r}")
    return RenewalRecord(title, author, year, False,
                         f"no renewal record for {title!r} ({year}) in an index of "
                         f"{len(index)} entries covering {lo}-{hi}; tried surname "
                         f"candidates {candidates!r} from author {author!r}")


def admissible(record: RenewalRecord) -> bool:
    """Only a VERIFIED non-renewal admits a work. Unknown rejects."""
    return record.renewed is False
