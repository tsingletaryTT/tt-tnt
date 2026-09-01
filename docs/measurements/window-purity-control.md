<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Window purity does not move context use — a null

**Result: NULL.** Six runs, 4h52m, three seeds per arm.
Spec (declared before any run): [`2026-09-01-window-purity-control.md`](../superpowers/specs/2026-09-01-window-purity-control.md).
Data: [`window-purity-control.json`](window-purity-control.json).

## The question

The corpus re-weighting on `feat/long-context-corpus` produced a real, large change in the
artifact. At the 512-token window these arms train at:

| | mean separators/window | single-document windows |
|---|---:|---:|
| `tokens-v4` (shipped) | 1.067 | 57.9% |
| `tokens-v5` (re-weighted) | 0.520 | **74.2%** |

Cross-document contamination was roughly halved. After
[gate 3 turned out to be non-discriminating](gate3-document-length.json) and the spec's
document-fragmentation premise was refuted, this was the only remaining justification for the
branch: **does window purity reach the model?**

## The answer

`delta_late = CE[64,128) - CE[448,512)`, per window — positive means the model is still
gaining from context at the far end. Both arms probed on **identical windows**, both
validation splits reported separately and never pooled.

| probed on | v4 arm | v5 arm | effect | ×seed floor | across-seed t | same direction | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `tokens-v4` val | +0.1231 | +0.1291 | +0.0060 | 0.57× | +0.57 | 1/3 | **NOT INTERPRETABLE** |
| `tokens-v5` val | +0.1822 | +0.1924 | +0.0102 | 1.89× | +1.65 | 2/3 | **BELOW PAIRED DETECTION** |

Neither split clears both declared gates. Across all six seed×split cells the paired
difference is positive in **three** — a coin flip, not a direction.

**Both gates earned their place, one each.** On `v4`'s split the effect is inside the seed
floor. On `v5`'s split it clears the floor at 1.89× but is smaller than its own
minimum-detectable difference (0.0102 vs 0.0121) — the mirror-image error the spec named in
advance, a respectable ratio over a small denominator. A single-gate design would have
reported the second row as a finding.

## Why the early read looked different

A first pass at **256** windows gave seed 5489 on `v4`'s split `+0.0129, t = +0.75`. At
**512** windows the same cell is `−0.0012, t = −0.10` — the sign flips on doubling the
sample. It was reported as "not a finding" at the time, and doubling the sample is what
demonstrated that rather than argued it.

## Neither arm uses context notably well

Both sit at Δ_late ≈ +0.12 to +0.19, against +0.11 for `tt-tnt-v1` and +0.10 for `v3`/`v4`
from the committed context-use probes. **The re-weighting did not produce a model that reads
its window differently; it produced a model that reads it the same way.**

## The null is real, not an instrument artifact

Checked, because identical-looking numbers are where this project's rule says suspect the
instrument first:

- all six `model.safetensors` hash **distinct** (6/6);
- the probe's mean position curves **differ between arms** on both splits (max abs 0.3020 and
  0.6129), so it was genuinely looking at two models;
- the arms' own final validation losses differ substantially (v4 2.78–2.85, v5 3.50–3.57 —
  a val-mixture artifact, see below, but proof the runs are not duplicates).

## Limitations, stated plainly

- **The seed floor is estimated from n=3.** An sd from three points carries roughly 50%
  relative uncertainty, which is why the two splits disagree about the floor gate (0.0107 vs
  0.0054) more than they disagree about the effect (+0.0060 vs +0.0102). Neither floor is
  solid; the verdict does not turn on which is right, because the effect fails a gate either
  way.
- **The raw validation losses are not comparable across arms** and are not used here. Each arm
  holds out the tail of a *different* blend: v4 holds out ~29% TinyStories, the easiest text in
  the corpus; v5 holds out ~10%. A model of identical quality scores worse on v5's split. This
  is why the primary measurement is on shared windows.
- One architecture, one context length, one constant-LR schedule, 1.000 epoch. The probe
  measures position-wise held-out loss, which is a proxy for "uses context", not a
  demonstration that any particular long-range dependency was learned.

## What this retires

The spec said in advance what a null would mean, and it means it: **window purity is not the
mechanism either.** The corpus work changed the artifact measurably and did not reach the
model. Combined with the refuted fragmentation premise, the corpus-*shape* line is closed —
`if_fiction` and `grimoire` (plan tasks 8 and 9) would have added more long documents against
a hypothesis that has now been tested and failed.

What remains unexplained is the original observation: the model stops using context past
position ~64. Four mechanisms have now been eliminated (capacity, context length, document
fragmentation, window purity). The standing candidate is the one this branch never tested —
**2.87 tokens/param against Chinchilla's 20**, the subject of
[the data scale-up draft](../superpowers/specs/2026-09-01-data-scale-up-design.md).
