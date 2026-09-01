<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Long-Context Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a corpus whose documents are long enough that a 2048-token window is usually
one document, and prove it with a measured gate before any training is commissioned.

**Architecture:** Three new slices are added to the existing `train/corpus.py` registry —
`longform` (openly licensed bulk long documents), `mission` (NASA/Apollo, public domain by
statute), and `pulp_sf` (1950–63 works whose copyright non-renewal is *verified* per work).
The existing fetch → prepare → measure → blend pipeline is reused unchanged wherever possible;
two extensions are needed, because two of the three slices are not HuggingFace datasets and
because `pulp_sf` needs a licence gate no existing source needs.

**Tech Stack:** Python 3.12, `datasets`, `numpy`, `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-31-long-context-corpus-design.md`

## Scope

This plan covers **the corpus and gate 3 only**. The spec's gate 1 (does the model use the
window) and gate 2 (scored Q&A coherence) commission GPU runs and a new evaluation signal;
they are a separate subsystem and get their own plan, written **only if gate 3 passes**. That
ordering is the spec's "stop at the first failure" rule applied to planning as well as to
running: a plan for two 4-hour training arms is wasted work if the corpus cannot clear a
measurement that takes seconds.

## Global Constraints

- **Every new source's licence is recorded in `README.md`'s provenance section in the same
  change that adds the source.** Not afterwards. This is a standing repo rule.
- **A hedge is never upgraded into a claim.** Where a licence basis is uncertain, the wording
  stays uncertain.
- **`pulp_sf` admits a work only on a verified non-renewal record.** "Project Gutenberg hosts
  it" is not a licence basis; PG is documented as voiding copyrights on this material with a
  theory that is right for some works and wrong for others.
- **`hf_revision` is never `None`.** An unpinned fetch is not reproducible.
- **`convert/` must not import `ttnn` or `ttml`.** Unchanged from the existing constraint.
- **No new dependencies.** `numpy`, `datasets`, `pytest` only.
- **A token budget is denominated in a unit the tokenizer defines.** Do not retrain the
  tokenizer in this plan; every measurement here is in the existing 32k vocabulary.

---

### Task 1: Document-length measurement (gate 3's instrument)

Gate 3 is the cheapest gate and runs first, so its instrument is built first — and building it
first also reproduces the number the spec argues from, making that number checkable rather than
quoted.

**Files:**
- Create: `scripts/measure_document_lengths.py`
- Test: `tests/test_measure_document_lengths.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `document_lengths(ids: np.ndarray, separator_id: int = 2) -> np.ndarray`, and
  `length_report(lengths: np.ndarray, thresholds: Sequence[int]) -> Dict[str, Any]` returning
  keys `count`, `mean`, `median`, `p75`, `p90`, `p95`, `docs_at_least` (a
  `Dict[int, float]` of threshold → fraction of *documents*), and `tokens_in_docs_at_least`
  (a `Dict[int, float]` of threshold → fraction of *tokens*). Task 7 asserts gate 3 against
  `tokens_in_docs_at_least[2048]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_measure_document_lengths.py
import numpy as np
import pytest

from scripts.measure_document_lengths import document_lengths, length_report

SEP = 2


def test_lengths_are_the_gaps_between_separators():
    # three documents of 3, 1 and 4 tokens, separated by id 2
    ids = np.array([9, 9, 9, SEP, 7, SEP, 5, 5, 5, 5, SEP], dtype=np.uint32)
    np.testing.assert_array_equal(document_lengths(ids, SEP), [3, 1, 4])


def test_a_trailing_partial_document_is_not_counted():
    """Text after the last separator is an unterminated fragment, not a document. Counting
    it would inflate the count with a document whose true length is unknown."""
    ids = np.array([9, 9, SEP, 7, 7, 7], dtype=np.uint32)
    np.testing.assert_array_equal(document_lengths(ids, SEP), [2])


def test_tokens_in_long_documents_is_not_the_same_as_documents_that_are_long():
    """Gate 3 is about TOKENS, because a corpus can be 99% short documents by count while
    most of its tokens live in a few long ones -- which is exactly what tokens-v4 looks
    like (median 113, mean 1031)."""
    ids = np.array([1] * 10 + [SEP] + [1] * 990 + [SEP], dtype=np.uint32)
    rep = length_report(document_lengths(ids, SEP), thresholds=[100])
    assert rep["docs_at_least"][100] == pytest.approx(0.5)
    assert rep["tokens_in_docs_at_least"][100] == pytest.approx(990 / 1000)


def test_report_states_the_distribution_not_just_a_mean():
    """A mean is not a finding here: tokens-v4's mean is 1031 and its median is 113, and
    only the median describes a typical document."""
    rep = length_report(np.array([1, 1, 1, 1000]), thresholds=[])
    assert rep["count"] == 4
    assert rep["median"] == pytest.approx(1.0)
    assert rep["mean"] == pytest.approx(250.75)


def test_an_array_with_no_separators_reports_no_documents_rather_than_one():
    """A corpus with no separators is the pre-2026-08-14 bug, not a single huge document."""
    rep = length_report(document_lengths(np.array([1, 1, 1]), SEP), thresholds=[10])
    assert rep["count"] == 0
    assert rep["tokens_in_docs_at_least"][10] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_measure_document_lengths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.measure_document_lengths'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""How long are this corpus's documents, in tokens?

Gate 3 of docs/superpowers/specs/2026-08-31-long-context-corpus-design.md. The question is
not "how many documents are long" but "how many TOKENS live in long documents", because a
training window samples tokens, not documents. On artifacts/tokens-v4 the two differ wildly:
the median document is 113 tokens and the mean is 1031, so a handful of books hold a large
share of the tokens while almost every document is short.

Documents are the spans between `</s>` (id 2). Text after the final separator is an
unterminated fragment and is not counted -- its true length is unknown, and counting it
would report a document shorter than it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: `</s>` in artifacts/tokenizer -- an added special token, so byte-level BPE can neither
#: split it nor absorb a neighbour.
DEFAULT_SEPARATOR_ID = 2

#: Window sizes worth asking about. 512 is what tt-tnt-1024 trains at today; 2048 is the
#: target the spec gates on.
DEFAULT_THRESHOLDS = (512, 1024, 2048, 4096)


def document_lengths(ids: np.ndarray, separator_id: int = DEFAULT_SEPARATOR_ID) -> np.ndarray:
    """Lengths of the complete documents in `ids`, in tokens, excluding the separators."""
    idx = np.flatnonzero(np.asarray(ids) == separator_id)
    if idx.size < 2:
        return np.zeros(0, dtype=np.int64)
    return (np.diff(idx) - 1).astype(np.int64)


def length_report(lengths: np.ndarray, thresholds: Sequence[int] = DEFAULT_THRESHOLDS
                  ) -> Dict[str, Any]:
    """Distribution summary, plus the two fractions gate 3 cares about."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total_tokens = int(lengths.sum())
    rep: Dict[str, Any] = {
        "count": int(lengths.size),
        "total_tokens": total_tokens,
        "mean": float(lengths.mean()) if lengths.size else 0.0,
        "median": float(np.median(lengths)) if lengths.size else 0.0,
        "p75": float(np.percentile(lengths, 75)) if lengths.size else 0.0,
        "p90": float(np.percentile(lengths, 90)) if lengths.size else 0.0,
        "p95": float(np.percentile(lengths, 95)) if lengths.size else 0.0,
        "docs_at_least": {},
        "tokens_in_docs_at_least": {},
    }
    for t in thresholds:
        long = lengths[lengths >= t]
        rep["docs_at_least"][t] = float(long.size / lengths.size) if lengths.size else 0.0
        rep["tokens_in_docs_at_least"][t] = (
            float(long.sum() / total_tokens) if total_tokens else 0.0
        )
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tokens", type=Path,
                    help="a .npy token array, e.g. artifacts/tokens-v4/train_ids.npy")
    ap.add_argument("--separator-id", type=int, default=DEFAULT_SEPARATOR_ID)
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only the first N tokens (0 = all)")
    ap.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args()

    ids = np.load(args.tokens, mmap_mode="r")
    ids = np.asarray(ids[: args.limit]) if args.limit else np.asarray(ids)
    rep = length_report(document_lengths(ids, args.separator_id))
    rep["source"] = str(args.tokens)
    rep["tokens_scanned"] = int(ids.size)

    print(f"{rep['count']:,} documents over {rep['tokens_scanned']:,} tokens")
    print(f"  mean {rep['mean']:.0f}  median {rep['median']:.0f}  "
          f"p75 {rep['p75']:.0f}  p90 {rep['p90']:.0f}  p95 {rep['p95']:.0f}")
    for t in sorted(rep["docs_at_least"]):
        print(f"  >= {t:5}: {rep['docs_at_least'][t]*100:6.2f}% of documents, "
              f"{rep['tokens_in_docs_at_least'][t]*100:6.2f}% of TOKENS")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_measure_document_lengths.py -v`
Expected: 5 passed

- [ ] **Step 5: Reproduce the spec's baseline number**

Run: `python scripts/measure_document_lengths.py artifacts/tokens-v4/train_ids.npy --limit 40000000`

Expected: median 113, mean 1031, `>= 2048` about 1.08% of documents. If these disagree with
the spec's table, **stop and report** — the spec's argument rests on them, and a disagreement
means either the measurement or the spec is wrong.

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_document_lengths.py tests/test_measure_document_lengths.py
git commit -m "feat(corpus): measure document lengths, the instrument gate 3 uses

Gate 3 asks how many TOKENS live in long documents, not how many documents
are long -- a training window samples tokens. On tokens-v4 the two differ
wildly: median document 113 tokens, mean 1031.

Text after the final separator is an unterminated fragment and is not counted;
its true length is unknown."
```

---

### Task 2: A source that is not a HuggingFace dataset

`CorpusSource` assumes `hf_repo`/`hf_revision`, and `scripts/fetch_corpus.py::iter_source_rows`
calls `load_dataset` unconditionally. Two of the three new slices are not HF datasets: NASA
transcripts are files on a government web host, and `pulp_sf` is a per-work selection. This
task adds the smallest extension that admits them without weakening the pinning rule.

**Files:**
- Modify: `train/corpus.py` (add `fetch_kind` and `source_url` to `CorpusSource`)
- Modify: `scripts/fetch_corpus.py:138-179` (`iter_source_rows` dispatches on `fetch_kind`)
- Test: `tests/test_corpus_fetch_kind.py`

**Interfaces:**
- Consumes: `CorpusSource` from `train/corpus.py`.
- Produces: `CorpusSource.fetch_kind: str` (one of `"hf"`, `"url"`), defaulting to `"hf"`, and
  `CorpusSource.source_url: str` (default `""`). Tasks 4–6 set these.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus_fetch_kind.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus_fetch_kind.py -v`
Expected: FAIL — `CorpusSource.__init__() got an unexpected keyword argument 'fetch_kind'`

- [ ] **Step 3: Write minimal implementation**

Add to `CorpusSource`, after the `hf_split` field:

```python
    #: How this source is fetched. "hf" is a HuggingFace dataset (the original and still the
    #: common case); "url" is a direct download, needed for sources that are files on a web
    #: host rather than a dataset -- NASA's mission transcripts are the motivating case.
    #: Adding a kind must never weaken the pinning rule: an "hf" source still requires a
    #: revision, and a "url" source pins by being a fixed URL to an archived document.
    fetch_kind: str = "hf"
    #: For fetch_kind="url": where to get it. Empty for "hf".
    source_url: str = ""
```

and add a `__post_init__` to the dataclass:

```python
    def __post_init__(self) -> None:
        if self.fetch_kind not in ("hf", "url"):
            raise ValueError(
                f"{self.name}: fetch_kind must be 'hf' or 'url', got {self.fetch_kind!r}. "
                f"A typo here fetches nothing, and an empty slice is indistinguishable from "
                f"a source that legitimately had no rows."
            )
        if self.fetch_kind == "url" and not self.source_url:
            raise ValueError(f"{self.name}: fetch_kind='url' needs a source_url")
        if self.fetch_kind == "hf" and self.hf_repo and not self.hf_revision:
            raise ValueError(
                f"{self.name}: hf_repo without hf_revision -- an unpinned fetch is not "
                f"reproducible, and shipping an exact recipe is the point"
            )
```

In `scripts/fetch_corpus.py::iter_source_rows`, before the `load_dataset` call:

```python
    if source.fetch_kind == "url":
        yield from _iter_url_rows(source, limit_rows)
        return
```

and add above it:

```python
def _iter_url_rows(source: CorpusSource, limit_rows: int = 0) -> Iterator[Dict[str, object]]:
    """Rows from a plain-text document at `source.source_url`.

    One row per document, so `rows_per_document` stays 1: the whole point of these sources
    is that a document is long, and splitting one into rows here would hand
    `prepare_corpus.py` a separator every few lines -- the exact defect
    `rows_per_document` exists to prevent for the poetry source.
    """
    import urllib.request

    with urllib.request.urlopen(source.source_url, timeout=120) as fh:
        text = fh.read().decode("utf-8", errors="replace")
    if text.strip():
        yield {"text": text}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus_fetch_kind.py tests/test_corpus.py -v`
Expected: all pass. `tests/test_corpus.py` must be run too — `__post_init__` now validates
every existing source at import time, and a pre-existing source that violates the rule would
surface here.

- [ ] **Step 5: Commit**

```bash
git add train/corpus.py scripts/fetch_corpus.py tests/test_corpus_fetch_kind.py
git commit -m "feat(corpus): admit sources that are not HuggingFace datasets

NASA mission transcripts are files on a government host, not a dataset.
fetch_kind dispatches; the pinning rule is unchanged, and an unknown kind or a
url source with no url now raises at construction rather than fetching nothing
-- an empty slice is indistinguishable from a source that legitimately had no
rows."
```

---

### Task 3: Copyright-renewal verification for `pulp_sf`

The spec's hardest constraint: a 1950–63 work enters the corpus only if its non-renewal is
verified. This task builds the check and its record; Task 6 uses it.

**Files:**
- Create: `train/renewal.py`
- Create: `scripts/verify_renewal.py`
- Test: `tests/test_renewal.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `RenewalRecord` (a dataclass with `title: str`, `author: str`, `year: int`,
  `renewed: Optional[bool]`, `evidence: str`), and
  `verify(title: str, author: str, year: int, index: RenewalIndex) -> RenewalRecord`.
  `renewed=None` means **unknown**, which Task 6 treats as a rejection.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_renewal.py
import pytest

from train.renewal import RenewalIndex, RenewalRecord, admissible, verify


def _index():
    # A renewal index is a set of (normalised title, author surname, original year).
    return RenewalIndex([
        ("the puppet masters", "heinlein", 1951),
        ("foundation", "asimov", 1951),
    ])


def test_a_work_present_in_the_renewal_index_is_renewed_and_inadmissible():
    r = verify("The Puppet Masters", "Robert A. Heinlein", 1951, _index())
    assert r.renewed is True
    assert not admissible(r)


def test_a_work_absent_from_the_index_is_not_renewed_and_is_admissible():
    r = verify("Second Variety", "Philip K. Dick", 1953, _index())
    assert r.renewed is False
    assert admissible(r)


def test_a_year_outside_the_renewal_window_is_UNKNOWN_not_absent():
    """The index only covers 1929-1963. Absence from it says nothing about a 1971 work, and
    treating 'not in the index' as 'not renewed' would license the entire modern era."""
    r = verify("Something", "Someone", 1971, _index())
    assert r.renewed is None
    assert not admissible(r)


def test_unknown_is_rejected_rather_than_assumed_free():
    """The whole point of this gate: uncertainty rejects. A hedge is not upgraded to a claim."""
    assert not admissible(RenewalRecord("t", "a", 1955, None, "no record consulted"))


def test_matching_ignores_case_punctuation_and_leading_articles():
    r = verify("THE Puppet-Masters", "heinlein, robert a.", 1951, _index())
    assert r.renewed is True


def test_every_record_carries_its_evidence_string():
    """A verdict with no evidence is unauditable, and this gate exists to be audited."""
    r = verify("Second Variety", "Philip K. Dick", 1953, _index())
    assert r.evidence and "1953" in r.evidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_renewal.py -v`
Expected: FAIL — `No module named 'train.renewal'`

- [ ] **Step 3: Write minimal implementation**

```python
# train/renewal.py
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
```

`scripts/verify_renewal.py` loads the CCE/Stanford renewal data into a `RenewalIndex` and
reports admissibility for a candidate list, writing one `RenewalRecord` per work to
`artifacts/pulp_sf/renewal_records.jsonl`. Write it to accept `--renewals <path>` (the
downloaded index) and `--candidates <path>` (a JSONL of `{title, author, year, url}`), and to
exit non-zero if **any** candidate is inadmissible, so the caller cannot ignore it.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_renewal.py -v`
Expected: 6 passed

- [ ] **Step 5: Mutation-check the gate**

Change `admissible` to `return record.renewed is not True` (i.e. treat unknown as free),
re-run, confirm `test_unknown_is_rejected_rather_than_assumed_free` and
`test_a_year_outside_the_renewal_window_is_UNKNOWN_not_absent` go RED, then revert. A licence
gate that has never been seen to reject is a claim, not a check.

- [ ] **Step 6: Commit**

```bash
git add train/renewal.py scripts/verify_renewal.py tests/test_renewal.py
git commit -m "feat(corpus): verify copyright non-renewal per work for pulp_sf

US works published 1929-1963 needed a 28th-year renewal and only ~25% got one.
That window is what makes 1950s SF reachable -- per work, not per collection.

Project Gutenberg hosts much of this material and asserts it is public domain;
that assertion is documented as resting on a theory right for some works and
wrong for others, with no distinction drawn. The texts may be fine; PG's say-so
is not a licence basis.

UNKNOWN REJECTS: a year outside the window, or a work the index cannot speak to,
is inadmissible. Mutation-checked -- treating unknown as free turns two tests red."
```

---

### Task 4: The `longform` slice

The bulk of the length fix, and the easiest: an existing openly licensed HF dataset.

**Files:**
- Modify: `train/corpus.py` (add `longform` to `SOURCES`)
- Modify: `README.md` (provenance section — **same change**, per the global constraint)
- Test: `tests/test_corpus.py` (extend the existing registry tests)

**Interfaces:**
- Consumes: `CorpusSource` incl. `fetch_kind` from Task 2.
- Produces: `SOURCES["longform"]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_corpus.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -k longform -v`
Expected: FAIL with `KeyError: 'longform'`

- [ ] **Step 3: Write minimal implementation**

Add to `SOURCES` in `train/corpus.py`. Resolve the exact revision with
`huggingface_hub.dataset_info(...).sha` before committing — **do not invent one**; an
unresolvable revision is a loud failure at fetch time by design.

```python
    "longform": CorpusSource(
        name="longform",
        slice="spine",
        target_share=0.0,   # set by the re-settle in Task 7
        hf_repo="HuggingFaceFW/fineweb-edu",
        hf_revision="<resolve with huggingface_hub.dataset_info>",
        hf_config="sample-10BT",
        license_id="ODC-By-1.0",
        license_url="https://opendatacommons.org/licenses/by/1-0/",
        attribution="FineWeb-Edu (HuggingFaceFW), ODC-By 1.0",
        license_note=(
            "ODC-By covers the DATABASE. The underlying web pages carry their own rights; "
            "FineWeb-Edu is a filtered Common Crawl derivative and this project does not "
            "redistribute it."
        ),
        rows_per_document=1,
        rationale=(
            "Bulk long documents. The corpus's median document is 113 tokens and only 1.08% "
            "reach 2048, so a 2048-token window holds ~18 unrelated documents and the model "
            "cannot learn to use distant context. This slice exists for LENGTH, not voice."
        ),
    ),
```

Add the matching entry to `README.md`'s provenance section in the same commit, stating ODC-By
and the database-versus-contents distinction in the `license_note` above.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add train/corpus.py README.md tests/test_corpus.py
git commit -m "feat(corpus): add the longform slice for document LENGTH

The corpus's median document is 113 tokens and 1.08% reach 2048, so a
2048-token window holds ~18 unrelated documents. This slice is bulk long
documents under ODC-By; provenance recorded in README in the same change,
including that ODC-By covers the database and not the underlying pages."
```

---

### Task 5: The `mission` slice

**Files:**
- Create: `scripts/fetch_mission.py`
- Modify: `train/corpus.py`, `README.md`
- Test: `tests/test_fetch_mission.py`

**Interfaces:**
- Consumes: `fetch_kind="url"` from Task 2.
- Produces: `SOURCES["mission"]`; `scripts/fetch_mission.py::MISSION_DOCUMENTS`, a list of
  `(label, url)` pairs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_mission.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fetch_mission.py -v`
Expected: FAIL — `No module named 'scripts.fetch_mission'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/fetch_mission.py` with a `MISSION_DOCUMENTS` list of `(label, url)` pairs for
Apollo/Gemini air-to-ground transcripts and NASA technical reports, each on a `.gov` host, and
a `main()` that writes them to `artifacts/raw/mission/text.jsonl` one JSON object per document
via `scripts/fetch_corpus.py::write_documents`.

Register the source:

```python
    "mission": CorpusSource(
        name="mission",
        slice="procedural",
        target_share=0.0,   # set by the re-settle in Task 7
        hf_repo="", hf_revision="",
        fetch_kind="url",
        source_url="",      # per-document; see scripts/fetch_mission.py
        license_id="",      # no SPDX id exists for a US Government work
        license_url="https://www.copyright.gov/title17/92chap1.html#105",
        attribution="NASA mission transcripts and technical reports (US Government work)",
        license_note=(
            "17 USC 105: works of the US Government are not subject to copyright in the "
            "United States. This is a statutory basis, not a licence, and it covers only "
            "material the government itself produced -- hence every document is fetched "
            "from a .gov host and that is asserted by test."
        ),
        rows_per_document=1,
        rationale=(
            "The only unambiguously clean post-1950 source with period voice. Transcripts "
            "are extremely long documents, and their content -- technical dialogue between "
            "people solving hard problems under pressure -- is further from this corpus's "
            "existing children's fiction than anything else available."
        ),
    ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fetch_mission.py -v`
Expected: 3 passed

- [ ] **Step 5: Fetch and measure a single document before committing to the slice**

Run `python scripts/fetch_mission.py --limit 1` and check the resulting text is prose, not a
navigation-heavy HTML dump. If it is HTML boilerplate, add extraction to `fetch_mission.py`
and re-check — a slice of page furniture teaches page furniture.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_mission.py train/corpus.py README.md tests/test_fetch_mission.py
git commit -m "feat(corpus): add the mission slice -- NASA transcripts, PD by statute

The only unambiguously clean post-1950 source with period voice: 17 USC 105
puts US Government works outside copyright. That is a statutory basis rather
than a licence, and it covers only what the government produced -- so every
document is fetched from a .gov host and a test asserts it.

Transcripts are very long documents, which is the other reason they are here."
```

---

### Task 6: The `pulp_sf` slice, gated on verified non-renewal

**Files:**
- Create: `scripts/fetch_pulp_sf.py`
- Modify: `train/corpus.py`, `README.md`
- Test: `tests/test_fetch_pulp_sf.py`

**Interfaces:**
- Consumes: `train.renewal.verify` / `admissible` (Task 3), `fetch_kind="url"` (Task 2).
- Produces: `SOURCES["pulp_sf"]`; `artifacts/pulp_sf/renewal_records.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_pulp_sf.py
import json

import pytest

from scripts.fetch_pulp_sf import select_admissible
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fetch_pulp_sf.py -v`
Expected: FAIL — `No module named 'scripts.fetch_pulp_sf'`

- [ ] **Step 3: Write minimal implementation**

```python
def select_admissible(candidates, index):
    """(kept, records). Every candidate yields a record; only verified non-renewals are kept."""
    from train.renewal import admissible, verify

    kept, records = [], []
    for c in candidates:
        rec = verify(c["title"], c["author"], int(c["year"]), index)
        records.append(rec)
        if admissible(rec):
            kept.append(c)
    return kept, records
```

`main()` writes every record to `artifacts/pulp_sf/renewal_records.jsonl` (kept and rejected
alike), fetches only `kept`, and prints the kept/rejected counts.

Register `pulp_sf` with `fetch_kind="url"`, `license_id=""`, and a `license_note` stating that
the basis is verified non-renewal under the pre-1964 US renewal requirement, that records are
in `artifacts/pulp_sf/renewal_records.jsonl`, and that no work is admitted on a host's
assertion.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fetch_pulp_sf.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_pulp_sf.py train/corpus.py README.md tests/test_fetch_pulp_sf.py
git commit -m "feat(corpus): add pulp_sf, admitted only on verified non-renewal

Every candidate produces a record, kept or rejected, so the audit trail shows
what was excluded and why. A work outside 1929-1963 is UNKNOWN and rejected --
keeping it would license the modern era on an absence of evidence."
```

---

### Task 7: Re-settle shares, rebuild the blend, and run gate 3

**Files:**
- Modify: `train/corpus.py` (`target_share` for every source)
- Modify: `docs/corpus_blend.md` (regenerated)
- Test: `tests/test_corpus.py` (share sum), `tests/test_measure_corpus.py`

**Interfaces:**
- Consumes: every source from Tasks 4–6, `length_report` from Task 1.
- Produces: `artifacts/corpus/blend.txt`, `artifacts/tokens-v5/`, and
  `docs/measurements/document-lengths-v5.json`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_corpus.py
def test_target_shares_sum_to_one():
    assert abs(sum(s.target_share for s in SOURCES.values()) - 1.0) < 1e-9


def test_the_three_new_slices_hold_a_real_share_not_a_token_gesture():
    """Gate 3 needs >=40% of TOKENS in documents >=2048. The long-document sources cannot
    deliver that from a 1% share -- if these are small, the gate cannot pass and the two
    training runs behind it are wasted."""
    long_sources = sum(SOURCES[n].target_share for n in ("longform", "mission", "pulp_sf"))
    assert long_sources >= 0.40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -k share -v`
Expected: FAIL — shares still sum to 1.0 across the old nine, new sources at 0.0

- [ ] **Step 3: Re-settle the shares**

Reduce the existing nine proportionally to free ≥40% for the three new slices, keeping every
existing source's share relative to the others unchanged — the same proportional method the
TinyStories-reduction experiment used, so the change is single-variable. Run
`python scripts/measure_corpus.py` and honour its exit code: it fails when a slice cannot
reach its share within the upsample cap. **Reduce the share; do not raise the cap.**

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_corpus.py tests/test_measure_corpus.py -v`
Expected: all pass

- [ ] **Step 5: Rebuild the blend and tokenize**

```bash
python scripts/fetch_corpus.py
python scripts/fetch_mission.py
python scripts/fetch_pulp_sf.py
python scripts/prepare_corpus.py
python scripts/blend_corpus.py
python -m train.tokenization --out artifacts/tokens-v5
```

Do **not** retrain the tokenizer: a token budget is denominated in a unit the tokenizer
defines, and retraining it would re-denominate every measurement in this plan.

- [ ] **Step 6: RUN GATE 3**

```bash
python scripts/measure_document_lengths.py artifacts/tokens-v5/train_ids.npy \
  --out docs/measurements/document-lengths-v5.json
```

**Pass:** `tokens_in_docs_at_least[2048] >= 0.40`.
**Fail:** STOP. Do not commission the training runs. Report the achieved figure, and the
per-source breakdown showing which slice fell short. A failed gate 3 costs seconds and saves
two multi-hour runs — that is what it is for.

- [ ] **Step 7: Commit**

```bash
git add train/corpus.py docs/corpus_blend.md docs/measurements/document-lengths-v5.json tests/
git commit -m "feat(corpus): re-settle shares for long documents, and run gate 3

The three long-document slices take >=40% of the blend, with the existing nine
reduced proportionally so their ratios to each other are unchanged -- a
single-variable change.

Gate 3 measured on tokens-v5 and recorded. The prior corpus had 1.08% of
documents at >=2048 tokens and a median of 113."
```

---

## Self-Review

**Spec coverage.** Corpus slices → Tasks 4–6. Renewal verification → Task 3. Licence recorded
in the same change → the `README.md` modification inside Tasks 4–6, and a Global Constraint.
Gate 3 → Tasks 1 and 7. Reuse of existing machinery → Tasks 4–7 modify `train/corpus.py` and
run the existing scripts rather than replacing them. **Gates 1 and 2 are deliberately absent**
and the Scope section says so: they are the next plan, written only if gate 3 passes.

**Placeholder scan.** Two intentional non-placeholders, both flagged in-line as *resolve before
committing* rather than *decide later*: `longform`'s `hf_revision` (must be resolved with
`huggingface_hub.dataset_info` — inventing one is worse than leaving it) and the contents of
`MISSION_DOCUMENTS` (a real URL list, checked by test to be `.gov`). `target_share=0.0` in
Tasks 4–6 is a real value that Task 7 replaces, and Task 7's test fails until it does.

**Type consistency.** `document_lengths`/`length_report` (Task 1) are used by name in Task 7.
`RenewalIndex`/`RenewalRecord`/`verify`/`admissible` (Task 3) are used by name in Task 6.
`fetch_kind`/`source_url` (Task 2) are set in Tasks 5 and 6. `write_documents` is the existing
function in `scripts/fetch_corpus.py:58`.

---

### Task 8: The `if_fiction` slice — interactive fiction, per-work licensed

**Runs AFTER gate 3 passes.** Added 2026-08-31 at the user's request, deliberately outside the
gate-3 sequence: Task 7 settles shares, and adding a slice mid-flight is how manifest drift
starts. If gate 3 fails, this task does not run — the corpus work stops there by design.

**Why it is here.** `mission` was cut to one document by a licence defect (see the note below),
so the post-1950 voice the corpus was meant to carry now rests almost entirely on `pulp_sf`.
Interactive fiction adds a register nothing else in the blend has — second person, present
tense, terse and strange — and it is post-1950 by construction.

**The trap this task exists to avoid.** ClubFloyd (426 transcripts, ~590 games, on HuggingFace
tagged `license:mit`) is the obvious source and is NOT usable. That MIT tag is the packaging of
someone's scrape; the transcripts reproduce the games' own copyrighted output text. Same shape
as Project Gutenberg's pulp-SF claim and as the Apollo Lunar Surface Journal pages that had to
be dropped from `mission`: **a host asserting a licence it is not in a position to grant.**
Three for three on this project. This slice admits a work only on its AUTHOR's licence.

**Licence rule.** CC-BY and CC-BY-SA only. `NC` and `ND` are excluded — they fail the Open
Knowledge Foundation's Open Definition 2.1, which is the same bar Common Pile v0.1 applies, and
much IF is released BY-NC-SA. Do not reason about whether a research model that is never sold
clears an NC clause; exclude it.

**Files:**
- Create: `scripts/fetch_if_fiction.py`
- Modify: `train/corpus.py`, `README.md`
- Test: `tests/test_fetch_if_fiction.py`

**Interfaces:**
- Consumes: `fetch_kind="url"` and `source_url` (Task 2); the copyright-notice scanner added to
  the fetch path by Task 5's fix — reuse it, do not write a second one.
- Produces: `SOURCES["if_fiction"]`; `IF_WORKS`, a list of dicts
  `{"title": str, "author": str, "license": str, "license_url": str, "url": str}`;
  `admissible_licence(license_id: str) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetch_if_fiction.py
import pytest

from scripts.fetch_if_fiction import IF_WORKS, admissible_licence
from train.corpus import SOURCES


def test_only_open_definition_licences_are_admissible():
    assert admissible_licence("CC-BY-4.0")
    assert admissible_licence("CC-BY-SA-4.0")


def test_noncommercial_and_noderivatives_are_refused():
    """NC and ND fail the Open Definition, which is the bar Common Pile v0.1 applies. Much IF
    is BY-NC-SA, so this is the common case and not an edge case."""
    assert not admissible_licence("CC-BY-NC-SA-4.0")
    assert not admissible_licence("CC-BY-NC-4.0")
    assert not admissible_licence("CC-BY-ND-4.0")


def test_an_unknown_or_empty_licence_is_refused_rather_than_assumed_open():
    """Same rule as the renewal gate: uncertainty rejects. A work with no stated licence has
    not been released, it has merely been published."""
    assert not admissible_licence("")
    assert not admissible_licence("unknown")
    assert not admissible_licence("MIT-but-actually-a-scrape")


def test_every_registered_work_carries_an_admissible_licence_and_a_url_for_it():
    """A licence with no URL is an assertion. The URL is what makes it checkable."""
    assert IF_WORKS, "the slice must not ship empty"
    for w in IF_WORKS:
        assert admissible_licence(w["license"]), f"{w['title']}: {w['license']}"
        assert w["license_url"].startswith("https://"), w["title"]
        assert w["author"], w["title"]


def test_the_slice_records_that_a_hosts_claim_is_not_a_licence():
    note = SOURCES["if_fiction"].license_note
    assert "author" in note.lower()
    assert "clubfloyd" in note.lower() or "transcript" in note.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fetch_if_fiction.py -v`
Expected: FAIL — `No module named 'scripts.fetch_if_fiction'`

- [ ] **Step 3: Write minimal implementation**

```python
#: Licences that meet the Open Definition. Deliberately a small allow-list rather than a
#: pattern: "CC-BY-NC-SA-4.0" contains "CC-BY", so anything matching on prefix admits exactly
#: the licences this slice exists to exclude.
_ADMISSIBLE = frozenset({
    "CC-BY-4.0", "CC-BY-3.0", "CC-BY-SA-4.0", "CC-BY-SA-3.0", "CC0-1.0",
})


def admissible_licence(license_id: str) -> bool:
    """True only for a licence on the allow-list. Unknown and empty reject."""
    return license_id in _ADMISSIBLE
```

`IF_WORKS` starts with Emily Short's *Counterfeit Monkey* (released in full under CC-BY 4.0)
and grows from IFWiki's Open Source IF list. Every entry is verified by fetching it and
checking BOTH that the author's licence is on the allow-list AND that the text passes Task 5's
copyright-notice scanner. A work that fails either is not added.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fetch_if_fiction.py -v`
Expected: 5 passed

- [ ] **Step 5: Mutation-check the licence gate**

Change `admissible_licence` to `return license_id.startswith("CC-BY")` — the plausible-looking
shortcut — and confirm `test_noncommercial_and_noderivatives_are_refused` goes RED, then
revert. That mutation is the exact mistake this allow-list exists to prevent, and a licence
gate never seen to reject is a claim rather than a check.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_if_fiction.py train/corpus.py README.md tests/test_fetch_if_fiction.py
git commit -m "feat(corpus): add if_fiction, admitted only on the author's open licence"
```

---

## REVISION 2026-08-31 — supersedes Task 7's Step 1, and adds Task 9

See the spec's "CORRECTION" section for the measurement that forced this. In short: the four
public-domain book slices already in the corpus run 56k–97k tokens median with **100%** of their
tokens in documents ≥2048, while `longform` is 582 tokens median and 43.7%. The corpus does not
lack long documents; it under-weights the ones it has. `tinystories` is 31% of the blend at 198
tokens median and 0% above threshold.

### Task 7, Step 1 — REPLACEMENT test

The original test asserted `longform + mission + pulp_sf >= 0.40`. That is now wrong: those are
not the slices that can supply long documents. Use instead:

```python
# append to tests/test_corpus.py
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
```

Task 7's remaining steps are unchanged: re-settle proportionally, honour `measure_corpus.py`'s
exit code (reduce a share, never raise the cap), rebuild, tokenize to `artifacts/tokens-v5`,
then run gate 3 and STOP if it fails.

**Two stale citations to correct in the same commit**, both mine: `longform`'s `rationale` and
`README.md` still say the median document is "113 tokens"; the corrected figure is **112** (the
earlier number counted the terminating separator as content). And the spec and this plan
describe `mission` as "Apollo/Gemini transcripts" plural when it is now one document. This repo
has a gate for stale citations; it should not be the thing that catches our own text.

---

### Task 9: The `grimoire` slice — pre-1929 occult and esoteric books

**Runs after Task 7's gate 3 passes**, alongside Task 8.

**Why.** `weird` is the smallest slice in the corpus — 55 books, ~7M tokens — and is exactly the
register wanted. Its documents are 85,025 tokens median with 100% above threshold, so it is also
the best material in the corpus for gate 3. Growing it serves both goals at once, which nothing
else in this plan does.

**Licence basis.** Pre-1929 US publication, public domain. This is the one basis that has not
failed today: Project Gutenberg's pulp-SF *post-1929* claim is documented as unreliable, but its
pre-1929 holdings are not in dispute — the dispute is entirely about works published after the
1929 boundary. Restrict to pre-1929 imprints and the question does not arise.

**Files:**
- Modify: `train/corpus.py` (add `grimoire` to `SOURCES`), `README.md`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: the existing Gutenberg batch fetch (`scripts/fetch_corpus.py::fetch_gutenberg_batch`)
  and `CorpusSource.bookshelves` / `authors` selection — this slice needs NO new fetch code.
- Produces: `SOURCES["grimoire"]`.

- [ ] **Step 1: Write the failing test**

```python
def test_grimoire_is_a_pre_1929_public_domain_book_source():
    s = SOURCES["grimoire"]
    assert s.fetch_kind == "hf", "reuses the existing Gutenberg batch fetch"
    assert s.hf_revision, "an unpinned fetch is not reproducible"
    assert s.bookshelves or s.authors, "must select, not take the whole of Gutenberg"


def test_grimoire_states_the_pre_1929_basis_rather_than_a_bare_public_domain_claim():
    """'Public domain' with no boundary is the claim that failed on the pulp-SF material. The
    note must say WHY these are public domain, not that they are."""
    note = SOURCES["grimoire"].license_note.lower()
    assert "1929" in note
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_corpus.py -k grimoire -v`
- [ ] **Step 3: Register the source**, selecting on Gutenberg bookshelves for the occult and
  esoteric, with `rows_per_document=1` (a Gutenberg row is a whole book) and a `license_note`
  stating the pre-1929 boundary and why it, unlike the post-1929 claim, is not disputed.
- [ ] **Step 4: Run to verify it passes**, plus the whole of `tests/test_corpus.py`.
- [ ] **Step 5: Measure what it actually yields** with
  `python scripts/measure_document_lengths.py` against the prepared slice, and record the median
  and the ≥2048 fraction in the report. A slice that does not clear 2048 is not doing its job and
  should be reported as such rather than registered on faith.
- [ ] **Step 6: Commit** with the README provenance entry in the same change.
