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
    """Lengths of the complete documents in `ids`, in tokens, excluding the separators.

    A document is the span of tokens ending right before a separator: the first document
    runs from index 0 up to (not including) the first separator, and each subsequent one
    runs from just after the previous separator up to (not including) the next. Text after
    the *last* separator is an unterminated fragment and is never counted.
    """
    idx = np.flatnonzero(np.asarray(ids) == separator_id)
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    first = idx[:1]
    rest = np.diff(idx) - 1
    return np.concatenate([first, rest]).astype(np.int64)


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
