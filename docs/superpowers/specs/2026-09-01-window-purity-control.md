<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# The window-purity control arm

**Declared 2026-09-01, before any run existed.** Thresholds and readings are fixed here so
they cannot be chosen after seeing which side of them the data landed on — the rule this
project broke once (importing a looser threshold would have manufactured two findings) and has
since held.

## The question

Gate 3 passed and proved nothing: it clears on the unchanged corpus too
([`gate3-document-length.json`](../../measurements/gate3-document-length.json)). The spec's
premise — that a training window holds ~18 unrelated documents — is refuted; the median window
on `tokens-v4` already held **zero** document boundaries.

What survived is a real, measured difference between the two corpora. At the 512-token window
these arms train at:

| | mean separators/window | single-document windows |
|---|---:|---:|
| `tokens-v4` (shipped) | 1.067 | 57.9% |
| `tokens-v5` (re-weighted) | 0.520 | 74.2% |

**So: does window purity move the model?** Nothing in this project has established that it
does. This arm tests it directly, and it is the only remaining justification for the corpus
work on this branch.

## Design

Two arms, identical in every respect except `--tokens-dir`. Verified by `--dry-run` on both
before launch: same size, seq_len, batch, steps, seed, ddp, model-impl, optimizer, LR schedule
and `stochastic_rounding: True`.

```
python train/run.py --size 1024 --seq-len 512 --batch-size 64 --steps 10719 \
  --seed <SEED> --ddp 4 --model-impl python \
  --config train/configs/nanollama3_bpe_v2.yaml \
  --tokens-dir artifacts/tokens-{v4,v5} --val-every 500 --save-every 2000
```

**Why 512 and not 2048.** The contamination gap is proportionally *larger* at 512 (57.9% →
74.2%) than at 2048 (53.0% → 63.7%); 512 is the size's registered and published context, so no
config moves; it covers the claim under test exactly, which is about positions ~64 to 511; and
it is 4x cheaper. It also avoids the trap that produced two wasted 4.5-hour runs — holding
steps fixed while raising seq_len silently quadrupled the epochs.

**Epoch arithmetic, recomputed rather than inherited.** 64 x 512 = 32,768 tokens/step. 10,719
steps = 351.2M tokens = **1.000 epoch on v5, 0.996 on v4** (the corpora differ in size by
0.4%). Steps are matched, not epochs: matched steps means the same number of optimizer updates
at the same point in a constant-LR schedule, and 0.4% of an epoch is not worth desynchronising
that for.

**Seeds: 5489, 20260815, 8191** — the three the TinyStories-reduction experiment used, so its
seed-spread is a prior for this shape of run. Three per arm, six runs total. This is not
optional padding: that experiment's per-seed effect ran 0.92x, 1.74x, 2.70x, and seed 5489
alone — the seed every single-run experiment here has used — would have been written up as a
null. Runs are sequential; each needs all four chips.

## The primary measurement, and its threshold

`scripts/probe_context_use.py`, run for **both arms against the identical token file**. This is
the design's one non-negotiable: each arm's own val split is the tail of a *different* blend,
and comparing across different held-out mixtures is the confound that cost the TinyStories
experiment a +0.4288-nat phantom. Probing both arms on the same bytes removes it by
construction rather than correcting for it afterwards.

Both arms are probed on **both** val sets (`tokens-v4/val_ids.npy` and `tokens-v5/val_ids.npy`),
reported separately. A conclusion that holds under only one of them is reported as such.

**Statistic.** `Δ_late = CE(positions 64–127) − CE(positions 448–511)`, in nats. Positive means
the model still gains from more context; the spec's fact 1 is that it is ≈0. The comparison is
`Δ_late(v5) − Δ_late(v4)`, paired by window.

**Thresholds, both required** (the standing two-gate rule):
1. the difference must exceed **1.2x** the seed-only floor, computed as the spread of Δ_late
   across the three seeds *within* an arm;
2. it must clear its own paired minimum-detectable difference at the same n.

A result failing either is `NOT INTERPRETABLE`. A result whose CI spans zero is `no change`,
never a small improvement.

## Secondary, reported but not headline

- **Held-out loss**, matched-window, via `scripts/evaluate.py`. ⚠️ Confounded by the val
  mixture: the arms hold out tails of different blends. The expected mixture-only delta is
  computed and subtracted before the residual is read, exactly as the TinyStories run did —
  the null hypothesis for loss in a corpus experiment is not zero.
- **Behavioural signals** via `scripts/evaluate.py` against the committed seed floor.
- **Per-source loss** via `scripts/eval_per_source.py`.

## What a null means

If Δ_late does not move, the honest conclusion is that **window purity is not the mechanism**
either — and the corpus work on this branch bought a real change in the artifact that does not
reach the model. That is a publishable negative result and it retires the branch's premise
rather than the branch's measurements. It does not license a third mechanism hunt without a
cheaper test first.

## Costs and limits

Six runs at roughly 45 minutes each. No hardware is touched without a `gozer` lease. Nothing
under `artifacts/tokens-*`, `artifacts/corpus/`, or any existing `artifacts/checkpoints-*` is
written; this experiment's checkpoints go to `artifacts/checkpoints-winpurity-{v4,v5}-s<SEED>`.

Known limits, stated now rather than discovered later: one architecture, one context length,
one LR schedule; three seeds bound the seed floor loosely at n=3; and the probe measures
position-wise loss on held-out text, which is a proxy for "uses context", not a demonstration
that any particular long-range dependency was learned.
