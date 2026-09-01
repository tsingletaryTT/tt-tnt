<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Long context, earned from the corpus first

**Status:** design approved 2026-08-31, not yet implemented.
**Scope:** the corpus and the experiment that gates everything after it. The model reshape is
deliberately **out of scope** and is described only well enough to say what would justify it.

## The ask

Make tt-tnt right-sized for a single Blackhole p300c and substantially better across four,
where "better across four" means **longer context at the same weights and the same quality** —
not a bigger model, not merely faster. One chip serves N tokens of context; four chips serve
roughly 4N, because tensor parallelism shards the KV cache by head and GQA (`num_groups: 4`)
has already made that cache four times smaller.

## Why this is a corpus project and not an architecture project

Two facts, both measured, both in this repo.

**1. The model does not use the window it already has.** `scripts/probe_context_use.py`
measures per-token cross-entropy bucketed by position. On `tt-tnt-v1` it stops improving at
position ~64 and is flat, or fractionally worse, out to 511 — on a book source as much as on a
short one.

**2. The corpus cannot teach it to.** Measured 2026-08-31 over the first 40M tokens of
`artifacts/tokens-v4/train_ids.npy`, counting the `</s>` separator (id 2):

| statistic | value |
|---|---|
| mean document length | 1,030 tokens |
| **median** | **112 tokens** |
| p75 / p90 / p95 | 223 / 404 / 609 |
| documents ≥ 512 tokens | 6.88% |
| documents ≥ 1024 tokens | 2.17% |
| documents ≥ 2048 tokens | 1.08% |

A document's length **excludes** its terminating `</s>`. An earlier draft of this table was
one token higher throughout because the measurement counted the separator as content; the
figures above are the corrected ones, and `scripts/measure_document_lengths.py` pins the
convention by test. The ratio the gate actually uses is unaffected — 1.08% of documents reach
2048 tokens under either convention.

The mean is dragged by a handful of Gutenberg books; the median is the honest number. A
2048-token training window therefore holds on the order of **eighteen unrelated documents**.
The model was never asked to carry information 2000 tokens; it was asked to ignore seventeen
documents' worth of noise, and it correctly learned to.

**This retires an open mystery.** `train/configs/model/tt-tnt-1024.yaml` records that
`max_sequence_length` was raised to 2048 on 2026-08-28 and reverted on 2026-08-29 because every
2048-context checkpoint answered questions worse, and that **the mechanism was never isolated**
— the epoch-inflation hypothesis was tested by a six-checkpoint sweep and came back
non-monotone and NOT SIGNIFICANT at n=32. Document length is a mechanism that predicts the
observed result and was not previously considered. It is a hypothesis, not yet a finding; the
control arm in §4 is what would confirm it.

**Consequence.** Reshaping for tensor parallelism before fixing this would build a model that
*can* hold 2048 tokens and has never learned to use more than about a hundred. The corpus work
comes first, and gate 1 decides whether the reshape happens at all.

## The licensing position

This repo's standing rule is that adding a corpus means adding its licence in the same change,
and that a hedge is never quietly upgraded into a claim. The requested era makes that rule
bite.

- **1929–1963 US books: roughly 75% are public domain.** NYPL's Catalog of Copyright Entries
  project found that of ~642,000 registered book copyrights, only ~162,000 were renewed. The
  renewal records are machine-readable (NYPL's `catalog_of_copyright_entries_project`;
  Stanford's Copyright Renewal Database covers Class A renewals received 1950–1992). This
  window contains the American SF pulp boom.
- **1964 onward is closed.** Renewal became automatic. Openly licensed 1970s and 1980s fiction
  does not exist at corpus scale. **We cannot deliver 70s/80s fiction and should not pretend
  to.** What we can deliver from that era is US Government work, which is public domain by
  statute.
- **Project Gutenberg's pulp-SF collection is not an acceptable provenance on its own.** PG has
  been documented (Locus, 2010) voiding copyrights on exactly this material using a theory that
  is correct for some works and wrong for others, and PG does not distinguish them. The texts
  are usable; PG's assertion is not the basis on which we may use them.

**The gate this implies:** a work enters `pulp_sf` only if its non-renewal is **verified against
the CCE/Stanford renewal record**, per work, with the lookup recorded in the provenance
manifest. "PG hosts it" is not a licence. This is the price of the slice and it is worth
paying; it also converts a disputed claim into a defensible one.

## Corpus design

Three new slices, all chosen to be long-document, all post-1950 where law permits.

| slice | era | licence basis | why it is here |
|---|---|---|---|
| `pulp_sf` | 1950–63 | PD by **verified** non-renewal | the requested sci-fi flare; novelettes run 8–30k tokens, so it serves length and voice at once |
| `mission` | 1958–75 | US Government work, PD by statute | Apollo/Gemini air-to-ground transcripts and NASA technical reports: unambiguously clean, genuine period voice, extremely long documents, and real dialogue between people solving hard problems under pressure |
| `longform` | modern | Common Pile v0.1 / FineWeb-Edu, ODC-By and CC BY/BY-SA | bulk long documents; post-1950 by construction |

`mission` is the slice to defend hardest. It is the only *unambiguously* clean post-1950 source
with period character, its documents are enormous, and its content — technical dialogue under
pressure — is closer to what this model demonstrably lacks than more children's fiction.

Note on Common Pile: its books are pre-1929 by its own stated principle, so it supplies
**length**, not era. It does not solve the 1950+ requirement and is not claimed to.

Existing machinery is reused rather than rebuilt: `train/corpus.py`'s share planner and
rationale registry, `scripts/measure_corpus.py`'s availability gate (which exits non-zero when
a slice cannot reach its target share within the upsample cap), `scripts/prepare_corpus.py`'s
per-source `rows_per_document`, and the provenance manifest. Two known traps apply directly and
are called out for the implementer: `re.IGNORECASE` applies to every character class in a
pattern including the `[A-Z]` written *because* case matters, and a token budget is denominated
in a unit the tokenizer defines — retraining the tokenizer re-denominates every measurement
taken before it.

## CORRECTION 2026-08-31: the long documents were already here

The corpus design above was written without measuring the per-source document length it depends
on. Measured afterwards, on the prepared corpus:

| source | documents | median | tokens in documents ≥2048 |
|---|---:|---:|---:|
| `folklore` | 199 | **97,276 tok** | **100.0%** |
| `spine` | 241 | **89,810 tok** | **100.0%** |
| `weird` | 55 | **85,025 tok** | **100.0%** |
| `gutenberg_children` | 482 | **56,347 tok** | **100.0%** |
| `longform` (FineWeb-Edu) | — | 582 tok | 43.7% |
| `wikipedia_simple` | 81,391 | 136 tok | 22.3% |
| `poetry` | 46,314 | 593 tok | 0.0% |
| `tinystories` | 133,615 | 198 tok | 0.0% |

**The four public-domain book slices are already perfect for the gate, and they are the four
smallest.** Whole books run 56k–97k tokens; every one of them clears 2048. `longform`, proposed
above as the load-bearing long source, is two orders of magnitude worse on the exact axis the
gate measures — at 43.7% it would have to be ~91% of the blend to carry gate 3 alone, which is
a replacement rather than a blend.

**What changes.** The corpus problem is not "we lack long documents". It is **"the long
documents we have are 3% of the blend while `tinystories`, at 198 tokens median and 0% above
threshold, is 31%."** The fix is re-weighting, not acquisition. ~88M tokens of book text at 2×
upsample gives ~176M against the ~160M a 400M-token budget needs at 40%.

**What this costs the era goal, stated plainly.** Every post-1950 source with real character
failed a rights check: Project Gutenberg's pulp-SF claim is documented as unreliable, the Apollo
Lunar Surface Journal pages carry private copyright, and ClubFloyd's `license:mit` tag is
packaging over copyrighted game text. Pre-1929 public domain is the one place where character
and clean provenance coexist. The corpus will sound older than originally intended, and that is
a consequence of copyright rather than of the design.

**Revised slice list.** `mission` stays at one document (~173k words), honestly described.
`pulp_sf` stays, under its per-work renewal gate. `longform` is **demoted, not dropped** — it
still beats `tinystories` 43.7% to 0% and supplies the only modern non-narrative register here.
A new `grimoire` slice adds pre-1929 occult, esoteric and Fortean books, extending `weird`.

⚠️ **Gate 3's 40% threshold does not move.** It was declared before this measurement, and the
measurement says it is reachable. Moving a pre-declared threshold after seeing which side you
landed on is the one thing this project cannot do.

## The experiment

Three gates. They are **numbered by importance and executed in reverse** — gate 3 is
nearly free and runs first, gate 1 is the decisive one. **Stop at the first failure.**

### Gate 3 (runs first — it is nearly free)

Measured on the finished token array before any training: **≥40% of tokens live in documents of
≥2048 tokens.** Today the figure is 1.1% of *documents*. This gates the two expensive runs
behind it and takes seconds.

### Gate 1 — does the model use the window?

Two arms on the new blend at **matched epochs, not matched steps**, same seed: arm A at
`seq_len` 512, arm B at 2048. Matched epochs is a hard requirement and not a preference:
holding steps constant while quadrupling `seq_len` is exactly what silently took the reverted
2048 runs from 1.00 to 4.00 epochs, and cost two 4.5-hour runs before anyone did the
arithmetic.

**Plus a control arm: the OLD corpus at 2048.** Without it, an improvement cannot be attributed
to the corpus rather than to anything else that changed. This is the arm that turns the
document-length hypothesis into a finding or refutes it.

**Pass:** mean cross-entropy in the `[1024, 2048)` position bucket is below the `[256, 512)`
bucket by more than this probe's **own** noise floor.

⚠️ **That floor must be measured, not borrowed.** This project's existing seed floor (sd
0.1944) is a property of *validation-loss trajectories* and says nothing about
position-bucketed cross-entropy — they are different instruments on different quantities.
Reusing it here would repeat, in a new place, the error `scripts/evaluate.py` was built to
refuse and already carries a ⚠️ about (a floor applied across windows it was not measured at).
Gate 1's floor comes from a seed replicate of arm B, or failing that from the probe's own
per-window standard errors, and the artifact must record which.
**Fail:** loss still flat past 512 on a corpus of genuinely long documents. That is a real
result: it says the ceiling is capacity or data volume rather than window, retires long context
for the price of two short runs, and **the reshape does not happen.**

### Gate 2 — did we re-break question answering?

The ctx2048 regression was caught by *reading* greedy completions; this repo records that the
discriminating evidence "was READ rather than scored". That gap closes **before** the run: a
scored Q&A coherence signal is added to `scripts/evaluate.py`, so gate 2 is a number with an
interval. Verdicts obey the existing rules — `FLOOR_RATIO_MIN = 1.2`, and both the floor and
the paired minimum-detectable difference must pass.

## What is out of scope

The model reshape. It is described here only to state what would justify it: tensor parallelism
splits **width**, and `tt-tnt-1024`'s width was tuned to fill exactly one grid —
`intermediate 2816 → 88 tiles → 8×11 = 88 cores`, 80% of a harvested 11×10 p300c. Divided four
ways that is 22 tiles, **20% per chip**. The shape chosen to fill one chip is by construction
the shape that empties four, and each block costs two all-reduces regardless of the work between
them, so "wider and shallower" is the direction that scales. None of that is worth doing until
gate 1 says the model can use a long window at all.

Also out of scope, and recorded so it is not re-proposed: **sparse MoE.** It is measured to work
here (from scratch, pooled over two seeds, 0.0417 nats better than dense at 3.62× total
parameters for 0.989× active compute) but it answers a different question — more capability per
unit compute, not longer context — and expert parallelism needs a 32-node mesh this box does not
have.

## Risks

- **Long-form sources are the scarcest kind.** The corpus already holds only ~651.5M unique
  tokens, and this competes for the same budget as the undertraining problem
  (2.87 tokens/parameter against Chinchilla's 20; 1.458 bits/byte against GPT-2's 0.977 at
  equal parameter count). Fixing document length does not fix data volume, and may worsen it if
  long sources are upsampled hard.
- **Renewal verification is per-work and will reject material.** The slice may come in under
  target share. `measure_corpus.py`'s gate is designed to fail loudly in exactly that case; the
  share should be reduced rather than the gate relaxed.
- **Register shift is real and may not be wanted.** Pulp SF and Apollo transcripts are a long
  way from TinyStories. This project's register metric will move, and that is the point — but
  it is a change to what the model sounds like, not only to what it can hold.
- **Gate 1 may fail.** That is a designed outcome, not a project failure.

## Success criteria

1. Gate 3: ≥40% of tokens in documents ≥2048.
2. Gate 1: `[1024,2048)` bucket loss below `[256,512)` by more than the seed floor, with the
   old-corpus control arm failing the same test.
3. Gate 2: no confirmed Q&A regression under the existing floor-and-interval rules.
4. Every new slice's licence recorded in the README provenance section **in the same change**,
   with `pulp_sf`'s per-work renewal lookups in the manifest.
