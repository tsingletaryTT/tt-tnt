#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Reclaim disk by keeping only the FINAL ``.pkl`` of each training run.

Dry-run by default. ``--apply`` deletes, and only then, and always writes a manifest of
exactly what it removed so the deletion is auditable after the fact.

**Why keeping one checkpoint per run is not a loss of evidence.** Three things this repo
actually depends on all survive:

1. **Trajectories and the seed-noise floor read `val_losses.jsonl`, never the weights.**
   ``scripts/evaluate.py`` derives its loss floor from
   ``artifacts/checkpoints-tt-tnt-{v3,v5}/val_losses.jsonl`` and pairs candidate runs from
   the same file. This script never touches a non-``.pkl`` file, so every trajectory, run
   manifest and training log stays exactly where it was.
2. **Published measurements are already in `docs/measurements/`.** They are results, not
   derivations that get re-run; nothing under ``docs/`` is touched.
3. **No code depends on an intermediate checkpoint.** The only ``.pkl`` paths referenced
   from ``tests/``, ``scripts/``, ``train/`` or ``convert/`` are final ones
   (``checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl``,
   ``checkpoints-1024-dialogue/tt_tnt_step00010764.pkl``); every other match is a
   ``tmp_path`` fixture or a comment. ``train.checkpoint.latest_checkpoint`` and
   ``tests/test_numpy_parity.py`` both select the highest-step file, which is the one kept.

**And for SFT runs, the intermediates are literally redundant.** Until 2026-08-31 every
``step_*.pkl`` in an SFT run held the same weights — ``SFTTrainer``'s default saver read
each bf16 parameter at FULL precision and was handed ``AutocastTensor``'s stale fp32 cache
(see ``train.checkpoint.save_sft_checkpoint`` and upstream-asks entry 8). Deleting the
earlier ones of those discards duplicate bytes, not history.

What this cannot do is restore an intermediate later: re-deriving one means retraining.
That is the trade being made, and it is made once, deliberately, with the list in hand.

    python scripts/prune_checkpoints.py                 # show the plan, delete nothing
    python scripts/prune_checkpoints.py --apply         # delete, and write the manifest
    python scripts/prune_checkpoints.py --keep 2        # keep the final two per run
    python scripts/prune_checkpoints.py --exclude checkpoints-tt-tnt-1024a
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
GIB = 1024 ** 3

#: Never removed, whatever else happens. These are the derivation inputs (val_losses.jsonl
#: feeds every trajectory and the seed-noise floor) and the run provenance.
PROTECTED_SUFFIXES = (".jsonl", ".json", ".log", ".txt", ".md", ".npy", ".yaml", ".safetensors")


def run_directories(root: Path) -> List[Tuple[Path, List[Path]]]:
    """Every directory directly containing ``.pkl`` checkpoints, with those files."""
    found: List[Tuple[Path, List[Path]]] = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir() and not p.is_symlink()):
        pkls = sorted(p for p in d.glob("*.pkl") if p.is_file() and not p.is_symlink())
        if pkls:
            found.append((d, pkls))
    return found


def split_keep_drop(pkls: List[Path], keep: int) -> Tuple[List[Path], List[Path]]:
    """Highest-step ``keep`` files are kept; the rest are dropped.

    Ordering is by filename, which is how both naming schemes this repo has used encode the
    step with zero padding (``tt_tnt_step00010764.pkl``) — the same ordering
    ``train.checkpoint.latest_checkpoint`` and ``tests/test_numpy_parity.py`` use to pick
    "the" checkpoint. ``step_3000.pkl``-style SFT names are NOT zero-padded, so they are
    ordered by the integer in the name when every file parses, and lexically otherwise.
    """
    def step_of(p: Path):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else None

    steps = [step_of(p) for p in pkls]
    ordered = (sorted(pkls, key=step_of) if all(s is not None for s in steps)
               else sorted(pkls, key=lambda p: p.name))
    if keep <= 0:
        raise ValueError("--keep must be at least 1")
    return ordered[-keep:], ordered[:-keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ARTIFACTS,
                    help="directory tree to scan (default: artifacts/)")
    ap.add_argument("--keep", type=int, default=1,
                    help="how many of the newest checkpoints to keep per run (default: 1)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="run directory name to leave completely alone (repeatable)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete. Without this nothing is removed.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="where to write the record of what was deleted "
                         "(default: <root>/pruned-<utc>.json)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"no such directory: {args.root}", file=sys.stderr)
        return 2

    plan: List[Dict] = []
    reclaim = 0
    for d, pkls in run_directories(args.root):
        rel = d.relative_to(args.root)
        if d.name in args.exclude or str(rel) in args.exclude:
            continue
        kept, dropped = split_keep_drop(pkls, args.keep)
        if not dropped:
            continue
        size = sum(p.stat().st_size for p in dropped)
        reclaim += size
        plan.append({
            "run": str(rel),
            "keep": [p.name for p in kept],
            "drop": [p.name for p in dropped],
            "bytes": size,
        })

    plan.sort(key=lambda e: -e["bytes"])
    print(f"{'run directory':<52}{'drop':>5}{'keeping':>26}{'reclaim':>10}")
    for e in plan:
        print(f"{e['run']:<52}{len(e['drop']):>5}{e['keep'][-1][:24]:>26}"
              f"{e['bytes'] / GIB:>9.2f}G")

    dropped_total = sum(len(e["drop"]) for e in plan)
    print(f"\n{len(plan)} run directories, {dropped_total} files, "
          f"{reclaim / GIB:.1f}G reclaimable (keeping the newest {args.keep} per run)")

    # Nothing outside .pkl is ever in the plan; assert it rather than trusting the glob.
    for e in plan:
        for name in e["drop"]:
            assert name.endswith(".pkl") and not name.endswith(PROTECTED_SUFFIXES), name

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply to remove these files.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = args.manifest or (args.root / f"pruned-{stamp}.json")
    removed, freed = 0, 0
    for e in plan:
        d = args.root / e["run"]
        for name in e["drop"]:
            target = d / name
            freed += target.stat().st_size
            target.unlink()
            removed += 1
    manifest.write_text(json.dumps({
        "schema": "tt-tnt/checkpoint-prune/1",
        "pruned_at": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root),
        "keep_per_run": args.keep,
        "excluded": args.exclude,
        "files_removed": removed,
        "bytes_freed": freed,
        "runs": plan,
    }, indent=2) + "\n")
    print(f"\nremoved {removed} files, freed {freed / GIB:.1f}G")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
