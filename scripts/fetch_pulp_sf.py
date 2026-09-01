#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Select and fetch the `pulp_sf` slice: 1950-63 American science fiction, admitted only on
a VERIFIED renewal check, never on a host's say-so.

WHY THIS EXISTS. US works published 1929-1963 are public domain only if their copyright was
never renewed in the 28th year -- see `train/renewal.py`'s module docstring for the full
history and the reason `verify`/`admissible` are deliberately biased toward the cheap error
(rejecting a possibly-admissible work) rather than the expensive one (admitting a renewed,
still-copyrighted one). This module is the SELECTION layer built on top of that gate: given a
list of (title, author, year, url) candidates, decide which are admissible, and record why
every single one -- kept or rejected -- came out the way it did.

THIS SLICE IS REGISTERED WITH NO DOCUMENTS, AND THAT IS CORRECT. `train.renewal.RenewalIndex`
is populated from the CCE/Stanford renewal records, and ingesting that real dataset was
explicitly deferred by Task 3's own ruling -- it does not exist on disk in this repository.
Without a real index, `select_admissible` cannot admit anything: `admissible()` requires a
VERIFIED non-renewal (`renewed is False`), and every candidate checked against an empty or
placeholder index comes back `renewed is None` (UNKNOWN, i.e. rejected) unless it happens to
literally match a fabricated entry -- which this module never fabricates. So `pulp_sf` is
registered with `target_share=0.0` and `main()` below refuses to run at all without a real
index supplied, rather than silently writing zero documents and looking like a successful
empty fetch. A slice that produces nothing must fail loudly, not quietly.

WHAT "REAL" MEANS HERE. A future task that has the actual CCE/Stanford renewal-index file in
hand loads it into a `RenewalIndex`, passes candidate metadata (title/author/year/url) mined
from a real pulp-SF bibliography, and re-runs this exact selection logic unchanged -- nothing
in `select_admissible` needs to know where the index or the candidates came from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.renewal import RenewalIndex, RenewalRecord, admissible, verify  # noqa: E402

#: Where every candidate's renewal record -- kept and rejected alike -- is written. This is
#: the audit trail: a reader must be able to see what was excluded and why, not only what
#: survived. Not under artifacts/raw/ like a fetched-text source, because this file records
#: a LEGAL determination about candidates, not fetched document text.
RECORDS_PATH = ROOT / "artifacts" / "pulp_sf" / "renewal_records.jsonl"


def select_admissible(
    candidates: Sequence[Dict[str, object]], index: RenewalIndex
) -> Tuple[List[Dict[str, object]], List[RenewalRecord]]:
    """(kept, records). Every candidate yields a record; only verified non-renewals are kept.

    `admissible()` (see `train/renewal.py`) is exactly `record.renewed is False` -- a
    candidate outside the 1929-1963 renewal window, or one the index has no record of, comes
    back `renewed is None` (UNKNOWN) and is rejected, not kept on the absence of evidence.
    """
    kept: List[Dict[str, object]] = []
    records: List[RenewalRecord] = []
    for c in candidates:
        rec = verify(str(c["title"]), str(c["author"]), int(c["year"]), index)
        records.append(rec)
        if admissible(rec):
            kept.append(c)
    return kept, records


def write_records(records: Sequence[RenewalRecord], dest: Path = RECORDS_PATH) -> None:
    """Write every record -- kept and rejected -- as one JSON object per line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({
                "title": rec.title,
                "author": rec.author,
                "year": rec.year,
                "renewed": rec.renewed,
                "evidence": rec.evidence,
            }, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--renewal-index", type=Path, default=None,
                   help="Path to a real CCE/Stanford renewal-index file. Required -- there "
                        "is no bundled index, and this script refuses to run without one "
                        "rather than silently producing an empty slice that looks like a "
                        "successful fetch.")
    args = p.parse_args()

    if args.renewal_index is None:
        print(
            "pulp_sf: no renewal index supplied (--renewal-index). This slice cannot admit "
            "any work without a verified CCE/Stanford renewal index, and ingesting that "
            "index was explicitly deferred (see train/renewal.py and this module's "
            "docstring). Refusing to run rather than silently writing zero documents.",
            file=sys.stderr,
        )
        return 1

    print(
        "pulp_sf: --renewal-index was supplied but this script does not yet know how to "
        "parse a real CCE/Stanford renewal-index file or mine pulp-SF candidate metadata -- "
        "both are follow-up work. Refusing to fabricate either.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
