#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Aggregate the window-purity control arm and apply its pre-declared gates.

Spec: docs/superpowers/specs/2026-09-01-window-purity-control.md, written before any run
existed. The thresholds below are read from that declaration, not chosen here:

1. the arm-level difference must exceed **1.2x** the seed-only floor, where the floor is the
   spread of ``delta_late`` across seeds *within* an arm;
2. it must clear its own paired minimum-detectable difference.

Failing either is NOT INTERPRETABLE. Both gates, because this project has made the mistake in
both directions -- a delta quoted against sampling error when run-to-run error was the floor
(the refuted LR-decay register finding, 1.03x), and a large ratio over a tiny denominator that
could not clear its own MDE (the 1024 run's engagement, 2.99x but +0.0198 against 0.0275).

The two validation splits are reported SEPARATELY and never pooled: they are tails of
different blends, so a conclusion holding under only one of them is reported as exactly that.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEAS = ROOT / "docs" / "measurements" / "winpurity"

#: From scripts/evaluate.py. A delta at or below this multiple of the run-to-run floor is not
#: interpretable whatever its confidence interval says.
FLOOR_RATIO_MIN = 1.2
SEEDS = (5489, 20260815, 8191)


def load(seed: int, valset: str) -> dict:
    p = MEAS / f"context-use-s{seed}-on-{valset}val.json"
    if not p.is_file():
        raise SystemExit(f"missing {p} -- run scripts/analyse_window_purity.sh first")
    return json.loads(p.read_text())


def report(valset: str) -> dict:
    rows = [(s, load(s, valset)) for s in SEEDS]
    a = [r["delta_late"]["mean_a"] for _, r in rows]      # tokens-v4 arm
    b = [r["delta_late"]["mean_b"] for _, r in rows]      # tokens-v5 arm

    print(f"\n{'=' * 78}\nProbed on artifacts/tokens-{valset}/val_ids.npy "
          f"(both arms, identical windows)\n{'=' * 78}")
    print(f"{'seed':>10} {'v4 Δ_late':>11} {'v5 Δ_late':>11} {'paired':>9} "
          f"{'t':>7} {'sign p':>8} {'MDE':>8}")
    for (s, r) in rows:
        d = r["delta_late"]
        print(f"{s:>10} {d['mean_a']:>+11.4f} {d['mean_b']:>+11.4f} {d['paired_mean']:>+9.4f} "
              f"{d['t']:>+7.2f} {d['sign_test_p']:>8.3f} {d['minimum_detectable']:>8.4f}")

    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    effect = mean_b - mean_a
    # The floor: how much delta_late moves on SEED ALONE, within an arm. Pooled across the two
    # arms because both estimate the same run-to-run variability.
    floor = statistics.mean([statistics.stdev(a), statistics.stdev(b)])
    ratio = abs(effect) / floor if floor else float("inf")

    # Across-seed paired test (n=3): the seed is the exchangeable unit for an ARM-level claim,
    # not the window. Low power by construction and reported as such.
    per_seed = [bb - aa for aa, bb in zip(a, b)]
    sd = statistics.stdev(per_seed)
    sem = sd / math.sqrt(len(per_seed))
    mde = 1.96 * sem
    t = effect / sem if sem else 0.0
    same_sign = sum(1 for d in per_seed if d > 0)

    print(f"\n  arm means      v4 {mean_a:+.4f}   v5 {mean_b:+.4f}")
    print(f"  effect (v5-v4) {effect:+.4f}")
    print(f"  seed-only floor (sd of Δ_late across seeds, within arm) {floor:.4f}")
    print(f"  ratio to floor {ratio:.2f}x   (gate: > {FLOOR_RATIO_MIN}x)")
    print(f"  across-seed paired: sd {sd:.4f} sem {sem:.4f} t {t:+.2f} "
          f"MDE {mde:.4f}  same-direction {same_sign}/3")

    gate_floor = ratio > FLOOR_RATIO_MIN
    gate_mde = abs(effect) > mde
    if gate_floor and gate_mde:
        verdict = "REAL: clears both the seed floor and its own detection threshold"
    elif not gate_floor and not gate_mde:
        verdict = "NOT INTERPRETABLE: inside the seed floor AND below paired detection"
    elif not gate_floor:
        verdict = f"NOT INTERPRETABLE: {ratio:.2f}x the seed floor, at or below the {FLOOR_RATIO_MIN}x rule"
    else:
        verdict = "BELOW PAIRED DETECTION: clears the floor but not its own MDE"
    print(f"\n  VERDICT: {verdict}")
    return {"valset": valset, "arm_mean_v4": mean_a, "arm_mean_v5": mean_b,
            "effect": effect, "seed_floor": floor, "ratio_to_floor": ratio,
            "across_seed_sd": sd, "across_seed_sem": sem, "across_seed_t": t,
            "minimum_detectable": mde, "same_direction": f"{same_sign}/3",
            "gate_floor_passed": gate_floor, "gate_mde_passed": gate_mde,
            "verdict": verdict,
            "per_seed": [{"seed": s, **r["delta_late"]} for s, r in rows]}


def main() -> int:
    out = {"spec": "docs/superpowers/specs/2026-09-01-window-purity-control.md",
           "floor_ratio_min": FLOOR_RATIO_MIN, "seeds": list(SEEDS),
           "statistic": "delta_late = CE[64,128) - CE[448,512), per window",
           "splits_never_pooled": True,
           "results": [report(v) for v in ("v4", "v5")]}
    p = ROOT / "docs" / "measurements" / "window-purity-control.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
