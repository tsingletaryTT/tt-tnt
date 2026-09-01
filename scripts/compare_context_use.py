#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does one model use context better than another? A paired, same-windows comparison.

`scripts/probe_context_use.py` answers "does THIS model use its context", reporting a bucketed
position-wise loss curve with a marginal SEM per bucket. That is the right instrument for one
model and the wrong one for comparing two, for a reason this project has now been bitten by in
both directions: the buckets share windows, so their marginal SEMs contain between-window
variance that a *paired* contrast cancels. Reading a paired effect against an unpaired floor
once stamped a t~6 result NOT INTERPRETABLE; the mirror-image error read a between-seed spread
as refuting a paired effect.

So this script pairs twice over, and says which pairing it is doing at every step:

1. **Within a model**, `delta_late` is one number per window -- the drop in cross-entropy from
   an early band of positions to a late one. Per window, so the window stays the sampling unit.
2. **Between models**, the two are evaluated on the SAME windows (one seeded draw, shared), so
   the difference of deltas is paired and per-window noise cancels.

The declared statistic (docs/superpowers/specs/2026-09-01-window-purity-control.md) is
`delta_late = CE[64,128) - CE[448,512)`: positive means the model is still gaining from more
context at the far end of its window. The spec's claim under test is that this is ~0.

**Both models must be probed on the same tokens.** Each arm's own validation split is the tail
of a different blend, and comparing across different held-out mixtures is exactly the confound
that produced a +0.4288-nat phantom in the TinyStories-reduction experiment. `--tokens` is a
single path used for both, not one per model.

    python scripts/compare_context_use.py \
        --model-a artifacts/hf-winpurity-v4-s5489 \
        --model-b artifacts/hf-winpurity-v5-s5489 \
        --tokens artifacts/tokens-v4/val_ids.npy --out docs/measurements/winpurity.json

CPU only: transformers on the host, no ttnn, no ttml, no device, no lease.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_context_use import (  # noqa: E402
    per_window_position_losses,
    sample_windows,
)

#: The declared bands. Early sits just past where tt-tnt-v1's curve goes flat (~64); late is
#: the last 64 positions of a 512 window. Kept as module constants, not CLI defaults buried in
#: argparse, because the spec declared them before any run existed and a reader checking the
#: claim needs to find them without reading the CLI.
EARLY_BAND: Tuple[int, int] = (64, 128)
LATE_BAND: Tuple[int, int] = (448, 512)


def delta_late(per_position: np.ndarray,
               early: Tuple[int, int] = EARLY_BAND,
               late: Tuple[int, int] = LATE_BAND) -> np.ndarray:
    """Per-window `CE(early) - CE(late)`, from a ``(n_windows, seq_len)`` loss array.

    One number per window, never pooled over positions first: the window is the exchangeable
    unit, and averaging positions into a single grand mean would discard the pairing this
    whole comparison rests on.
    """
    if per_position.ndim != 2:
        raise ValueError(f"expected (n_windows, seq_len), got shape {per_position.shape}")
    seq_len = per_position.shape[1]
    for name, (lo, hi) in (("early", early), ("late", late)):
        if not 0 <= lo < hi <= seq_len:
            raise ValueError(
                f"{name} band [{lo},{hi}) does not fit a {seq_len}-position window"
            )
    return (per_position[:, early[0]:early[1]].mean(axis=1)
            - per_position[:, late[0]:late[1]].mean(axis=1))


def sign_test_p(n_positive: int, n: int) -> float:
    """Exact two-sided sign test. `math.comb`, so no new dependency."""
    if n == 0:
        return 1.0
    k = min(n_positive, n - n_positive)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def paired_stats(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired summary of `b - a`, element i of each being the same window.

    Reports the sign test beside mean/sd/t deliberately: the t says how large the difference
    was, the sign test how consistently it pointed one way, and this project has had a result
    where those two disagreed.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired inputs: {a.shape} vs {b.shape}")
    d = b - a
    n = int(d.size)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    n_pos = int((d > 0).sum())
    return {
        "n": n,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "paired_mean": float(d.mean()), "paired_sd": sd, "paired_sem": sem,
        "t": float(d.mean() / sem) if sem else 0.0,
        "n_positive": n_pos, "n_negative": int((d < 0).sum()),
        "sign_test_p": sign_test_p(n_pos, n),
        # The smallest effect this design could resolve, at the conventional 1.96 SEM. Printed
        # so a null is reported as "smaller than we can see" rather than as "zero".
        "minimum_detectable": 1.96 * sem,
    }


def load_model(path: Path):
    """Float32 on CPU, matching every other host-side measurement in this repo."""
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32).eval()


def probe(path: Path, x: np.ndarray, y: np.ndarray, batch_size: int) -> np.ndarray:
    return per_window_position_losses(load_model(path), x, y, batch_size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", type=Path, required=True, help="baseline arm")
    ap.add_argument("--model-b", type=Path, required=True, help="candidate arm")
    ap.add_argument("--tokens", type=Path, required=True,
                    help="ONE token file, used for BOTH models -- see the module docstring")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-windows", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    label_a = args.label_a or args.model_a.name
    label_b = args.label_b or args.model_b.name

    ids = np.asarray(np.load(args.tokens, mmap_mode="r"))
    # ONE draw, shared. This is the line that makes the comparison paired.
    x, y = sample_windows(ids, args.seq_len, args.n_windows, np.random.default_rng(args.seed))

    print(f"probing {label_a} and {label_b} on the SAME {args.n_windows} windows "
          f"of {args.tokens} (seed {args.seed})")
    pa, pb = probe(args.model_a, x, y, args.batch_size), probe(args.model_b, x, y,
                                                               args.batch_size)
    da, db = delta_late(pa), delta_late(pb)
    st = paired_stats(da, db)

    print(f"\ndelta_late = CE{list(EARLY_BAND)} - CE{list(LATE_BAND)}, per window")
    print(f"  {label_a}: {st['mean_a']:+.4f}")
    print(f"  {label_b}: {st['mean_b']:+.4f}")
    print(f"  paired difference (b - a): {st['paired_mean']:+.4f}  "
          f"sd {st['paired_sd']:.4f}  sem {st['paired_sem']:.4f}  t {st['t']:+.2f}")
    print(f"  positive on {st['n_positive']}/{st['n']} windows, "
          f"sign test p = {st['sign_test_p']:.3g}")
    print(f"  minimum detectable at this n: {st['minimum_detectable']:.4f}")

    rep = {
        "model_a": str(args.model_a), "model_b": str(args.model_b),
        "label_a": label_a, "label_b": label_b,
        "tokens": str(args.tokens), "tokens_shared_by_both_models": True,
        "seq_len": args.seq_len, "n_windows": args.n_windows, "seed": args.seed,
        "early_band": list(EARLY_BAND), "late_band": list(LATE_BAND),
        "delta_late": st,
        "mean_position_curve": {
            label_a: pa.mean(axis=0).tolist(), label_b: pb.mean(axis=0).tolist()},
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
