#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Verify copyright non-renewal for a candidate list, one record per work.

DELIBERATELY DEFERRED: the real CCE/Stanford renewal data is not on disk and its exact
on-disk format (NYPL's Catalog of Copyright Entries scans, or Stanford's derived index) has
not been inspected by this task. Writing a parser against a format nobody here has looked at
would be worse than writing none -- it would appear to work, and would silently admit the
wrong works if the imagined format turned out to differ from the real one. So this script
does not ingest CCE/Stanford data directly. It consumes an ALREADY-NORMALISED renewal index:
a JSONL file, one object per line, ``{"title": str, "author": str, "year": int}``, one line
per renewal record. Producing that normalised file from the real CCE/Stanford source is a
separate, later task, once the real file is in hand and its format can actually be read.

Usage::

    python scripts/verify_renewal.py \\
        --renewals path/to/renewals.jsonl \\
        --candidates path/to/candidates.jsonl \\
        --out artifacts/pulp_sf/renewal_records.jsonl

``--candidates`` is a JSONL file, one object per line, of
``{"title": str, "author": str, "year": int, "url": str}`` -- the works under consideration
for the ``pulp_sf`` corpus slice. One ``RenewalRecord`` is written to ``--out`` per candidate,
in candidate order. The script exits non-zero if ANY candidate is inadmissible (renewed, or
unknown), so a caller cannot silently ignore a failed gate by only checking stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.renewal import RenewalIndex, RenewalRecord, admissible, verify  # noqa: E402


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def load_renewal_index(path: Path) -> RenewalIndex:
    """Load an already-normalised renewal index: one {"title","author","year"} per line."""
    entries = []
    for obj in _read_jsonl(path):
        entries.append((str(obj["title"]), str(obj["author"]), int(obj["year"])))
    return RenewalIndex(entries)


def load_candidates(path: Path) -> List[dict]:
    """Load the candidate list: one {"title","author","year","url"} per line."""
    candidates = list(_read_jsonl(path))
    for i, c in enumerate(candidates):
        for key in ("title", "author", "year"):
            if key not in c:
                raise ValueError(f"{path}: candidate {i} is missing required key {key!r}")
    return candidates


def verify_candidates(candidates: List[dict], index: RenewalIndex) -> List[RenewalRecord]:
    return [
        verify(str(c["title"]), str(c["author"]), int(c["year"]), index)
        for c in candidates
    ]


def write_records(records: List[RenewalRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "title": r.title,
                "author": r.author,
                "year": r.year,
                "renewed": r.renewed,
                "evidence": r.evidence,
                "admissible": admissible(r),
            }) + "\n")


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renewals", type=Path, required=True,
                       help="Already-normalised renewal index JSONL "
                            "({title, author, year} per line).")
    parser.add_argument("--candidates", type=Path, required=True,
                       help="Candidate works JSONL ({title, author, year, url} per line).")
    parser.add_argument("--out", type=Path, required=True,
                       help="Where to write one RenewalRecord per candidate (JSONL).")
    args = parser.parse_args(argv)

    index = load_renewal_index(args.renewals)
    candidates = load_candidates(args.candidates)
    records = verify_candidates(candidates, index)
    write_records(records, args.out)

    n_admissible = sum(1 for r in records if admissible(r))
    n_inadmissible = len(records) - n_admissible
    print(f"{len(records)} candidates checked against {len(index)} renewal records: "
          f"{n_admissible} admissible, {n_inadmissible} inadmissible.")
    if n_inadmissible:
        for r in records:
            if not admissible(r):
                print(f"  INADMISSIBLE: {r.title!r} by {r.author!r} ({r.year}): {r.evidence}",
                      file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
