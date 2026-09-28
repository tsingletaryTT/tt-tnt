<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Storytelling register, phase two: a fresh pretrain on a richer narrative blend

**Status: DRAFT, not approved, nothing implemented.** Drafted 2026-09-28 at the user's
request, following brainstorming. No task in it has been started, no fetch has run.

## 1. What this is, and what it explicitly is not

This project has a designated model (`tt-tnt-1024`, Stage A + Stage B, promoted 2026-09-24)
with a base that reads reasonably well (bits/byte 1.2356 vs. GPT-2's 0.977) but whose
storytelling **register** — voice, vocabulary richness, freedom from filler — has only ever
been *restored to parity*, never advanced. StoryCloze after Stage B is not significantly
different from the pre-Stage-A baseline (p = 0.326); MMLU is flat across every data scale
tried, which this project already treats as a closed question, not something this spec
reopens.

The one intervention that *did* move register (the TinyStories-share reduction, 2026-08-27)
did so with the existing nine/ten-source curated blend, at 1.79x/1.36x the seed floor on two
signals — and cost termination rate as a side effect, mechanically traced to TinyStories
supplying 73% of the blend's document separators. It was never adopted into a published
checkpoint. This spec is the follow-through: bring in a genuinely larger, lexically richer
narrative source, fix the termination-rate mechanism this time instead of accepting it as a
cost, and retrain from scratch with the new blend present from the start rather than bolted on
as a continued-training afterthought.

**This is not a capacity or knowledge project.** MMLU is expected to stay flat, same as every
Stage A/B measurement; that is not a failure of this run, it is the standing result. This spec
targets *register* specifically: the story-frame/lexical-habit collapse split, the register
margin against TinyStories, and (new) whether the model can take an abstract, evocative prompt
and go somewhere with it — as opposed to defaulting to "Once upon a time, there was a little
girl named Lily."

## 2. What is already eliminated, so this spec does not re-litigate it

- **Capacity.** 22M → 123M parameters moved held-out loss 0.011 nats against a 0.194-nat seed
  floor. Holding the model at 123.0M.
- **Knowledge via scale.** MMLU 0.2295 → 0.2297 → 0.2292 across 352.7M → 1.117B → 2.53B
  training tokens. Dead flat. Not a target of this run.
- **Context length.** Two 2048-context retrains were reverted; a chat-template window cap
  solved the serving problem the raise existed for. Staying at 512.
- **Anti-forgetting by freezing (LoRA) as a register fix.** LoRA at 100% task data still
  regressed repetition 18.46x the seed floor — freezing base weights constrains the rank of an
  update, not its effect. Not relevant to a from-scratch pretrain, noted only so it isn't
  re-proposed as a lever here.
- **The document-separator/register mechanism is understood, not guessed.** The
  2026-08-14 fix ("the corpus had no document boundaries at all") and the 2026-08-27
  TinyStories-reduction entry both trace termination-rate effects to separator *density per
  token*, not to register itself. §4 below designs around this directly rather than
  discovering it again after a wasted run.

## 3. Sources

Three components in one blend, each doing a different job:

| component | role | approx. scale | licence |
|---|---|---:|---|
| `deepmind/pg19` (new) | register/vocabulary richness — full novels, pre-1919 public domain | up to ~2B tokens available, downsampled to fit budget | **Apache-2.0** (dataset card), underlying texts public domain |
| existing 9/10-source curated blend, TinyStories share cut further | the registers PG19 doesn't cover — modern simple prose, dialogue Q&A, poetry, the deliberately strange slices (`weird`, `spine`, `flavour`) | ~325–350M tokens (comparable to today's 352.7M) | as already registered in `train/corpus.py` |
| `longform` (FineWeb-Edu) | general fluency / commonsense retention — Stage A already proved this is what drives loss and commonsense gains, and that MMLU doesn't move either way | large minority share, smaller than Stage A's 2.53B alone | ODC-By-1.0, already registered |
| `tortoise` (new, small) | flavour-class register injection — see §5 | ~25–35K raw tokens before upsampling | titles: not independently copyrightable short phrases; vignettes: original text written for this project |

**PG19 due diligence, not yet done, blocking Gate 0:** the dataset card's Apache-2.0 label
covers DeepMind's packaging; the underlying 19th-century texts are public domain by age, which
is the stronger and more durable claim. Both must be recorded in the README provenance section
in the same change that adds the source, per this project's standing licensing rule — a
credit is not the same thing as a licence, and this project has been burned once already
(Mini-LLM) by not checking that a licence label actually grants what it claims to.

**PG19's register is not neutral, and this is a risk, not a defect to fix.** DeepMind's own
card and this project's own research pass both flag 19th-century diction and social attitudes.
Widening vocabulary and shifting era-voice are not the same thing, and Gate 6 (§6) needs to look
at generation samples specifically for period-piece drift, not just trust an aggregate register
number moving in the "less TinyStories-like" direction — that number would also move if the
model just started writing like Dickens instead of like a children's book, and the *first* of
those was the intent, the second is a side effect worth naming honestly if it shows up.

## 4. PG19 must be chunked — a document-boundary problem, designed around before it happens

PG19's documents are whole novels — tens of thousands of words. Treated as one document each,
`</s>` becomes far rarer per token than even the TinyStories-reduction experiment, which
already cost termination rate by thinning separators from TinyStories' short-story cadence.
Left unfixed, this run would very likely reproduce that regression *worse*, for a mechanical
reason this project has already diagnosed twice.

**Fix: chunk each PG19 book into paragraph-aligned sub-documents (~2,000–4,000 words each)**
during `prepare_corpus.py`, each terminated with its own `</s>`, reading order preserved within
a book. This is the same fix already built once for the poetry corpus (`rows_per_document`,
where "one row = one document" was an assumption about the upstream dataset, not a property of
jsonl) applied to the opposite failure mode — there, rows were *smaller* than a document; here,
PG19's rows are *larger*. `CorpusSource` already has the machinery (`rows_per_document`); PG19
needs the inverse operation, a book-splitting step, which does not yet exist and is new code.

**A gate exists for this, not just an intention (see Gate 1 in §6).** Measure `</s>`-per-token
density on the assembled blend before training, compare against the currently-shipped blend's
own density, and refuse to proceed if it falls outside a declared band.

## 5. The Tortoise slice and prompt set — sized honestly

The project's own name is understood, per this conversation, as TT (Tenstorrent) + Tortoise's
1998 album *TNT* — confirmed, not invented: the band's real 1998 track "A Simple Way to Go
Faster Than Light That Does Not Work" is the origin of the faster-than-light canary prompt this
project has run at every checkpoint since the qualitative-canary convention started. This spec
treats "earning the namesake" as a real, if secondary, design goal.

**Titles, verified, not fabricated** (cross-checked against Wikipedia/Discogs/AllMusic in this
conversation's research pass): the full discography across seven studio albums (*Tortoise*
1994, *Millions Now Living Will Never Die* 1996, *TNT* 1998, *Standards* 2001, *It's All Around
You* 2004, *Beacons of Ancestorship* 2009, *The Catastrophist* 2016). Two tracks confirmed to
carry vocals (*Rock On*, *Yonder Blue*, both off *The Catastrophist*) are excluded from an
"instrumental titles" framing. From the full list, ~28 titles were shortlisted for being
genuinely abstract/imagistic rather than plain-noun or track-numbering titles — *Djed*,
*Gigantes*, *The Suspension Bridge at Iguazú Falls*, *Prepare Your Coffin*, *Onions Wrapped in
Rubber*, *Salt The Skies*, *Tesseract*, *The Lithium Stiffs*, *Ten-Day Interval*, and others —
the exact list to be finalized in the implementation plan.

**Two separate artifacts, not one, because they serve different purposes and have very
different scale requirements:**

1. **Corpus slice (`tortoise`, a new entry in `SLICES`)**: 1–3 short (a few hundred words)
   hand-written vignettes per shortlisted title, using the title as a narrative seed, written
   in an evocative/imagistic register deliberately contrasting with TinyStories. Total raw text
   ~25–35K tokens. **Stated honestly, matching the `flavour` precedent**: this project's own
   upsample cap (4x, hit once already at the "flavour" source's 2%→0.5% arithmetic-ceiling
   correction) means this slice's real blend share will land well under 0.5%, plausibly under
   0.2%. `measure_corpus.py`'s gate decides the real number; this spec does not pre-declare an
   aspirational share for it, because the whole point of that gate is refusing to let a source
   claim more presence than its actual text supports.
2. **Frozen eval prompt set ("set C")**, `docs/evaluation_prompts_c.json`, following set A/B's
   exact schema: one entry per title (rendered as an opening line or scene seed, not just the
   bare title, so the prompt has something for the model to continue), a new `id` prefix
   (`c-tortoise-*`), a new `probe` tag (`evocative-continuation`) alongside the existing set.
   **Never pooled with sets A or B** — same rule those two already follow with each other.
   This is the instrument that actually measures "does it earn its namesake," not the tiny
   corpus share, which cannot move an aggregate metric at this size.

## 6. Gates, cheapest first, each able to fail

Following this project's own established pattern (a gate every arm clears is decoration; a
gate that fails cheap has repeatedly saved a multi-hour run) — ordered cheapest-first:

**Gate 0 — licence (free, before any fetch).** PG19's Apache-2.0 packaging licence and the
underlying-text public-domain claim both recorded in `train/corpus.py` and the README
provenance section in the same change. FAILS if either cannot be established from primary
source text (the dataset card, not a summary of it).

**Gate 1 — separator density (minutes, after prepare, before tokenize).** Measure `</s>` per
1,000 tokens on the assembled (chunked) blend; compare against the currently-shipped blend's
own figure. FAILS if density falls outside ±25% of the current blend — at which point the
PG19 chunk size in §4 needs revising before any training time is spent, exactly the kind of
cheap-first check this project's history says pays for itself.

**Gate 2 — tokenizer fertility (minutes).** Retrain the 48K BPE tokenizer on the full new
blend; verify the achieved vocabulary actually reaches 48,000 (per the standing "`vocab_size`
is a ceiling, not a promise" rule) rather than shipping a shortfall silently.

**Gate 3 — scale on disk (~hours, scales with fetch/tokenize time).** Confirm the assembled
blend reaches its target token budget (~2.5B, matching Stage A's scale) without needing
upsampling beyond the declared caps. FAILS if it falls meaningfully short — the honest response
is to reduce the target, not to upsample further, since uncontrolled repetition is exactly what
the richer-corpus approach exists to avoid.

**Gate 4 — the real one: register (post-training, ~hours).** Story-frame and lexical-habit
collapse rate, and the TinyStories register margin, on sets A and B, against the current
`tt-tnt-1024` checkpoint, through `scripts/evaluate.py`'s existing seed-floor-gated methodology
— both the ratio-over-floor gate and the paired-minimum-detectable-difference gate must clear,
per this project's own "both gates must pass" rule. **At least 2 seeds.** A result below both
gates on both seeds is a real, reportable null, not a failure of the spec.

**Gate 5 — guardrails (must not regress, not targets).** Termination rate, 4-gram repeat rate,
StoryCloze accuracy (must hold parity with the pre-run baseline, not necessarily beat it).
FAILS this gate (not the whole run) if any of these regresses beyond its seed floor — in which
case the register gain, if any, is reported alongside the cost, exactly as the editor-training
line's own history did when a real gain came with a real regression.

**Gate 6 — set C, qualitative (minutes, human read).** Generate on all ~28 Tortoise-title
prompts at greedy and T=0.8, read for coherence and whether the completion goes somewhere
interesting versus defaulting to a TinyStories attractor. This gate has no numeric pass/fail
threshold — it is a qualitative report, explicitly not dressed up as a quantitative finding,
matching this project's standing distinction between a calibrated instrument and a canary read.

## 7. Decisions, and what each costs if wrong

**Retrain the tokenizer at 48K.** Cost if wrong: every tokenizer-dependent artifact
(evaluation prompt digests, `evaluate.py`'s instruments) needs re-deriving — a known, bounded
cost this project has paid before (corpus-assembly branch), not a new risk.

**Chunk PG19 to 2,000–4,000-word sub-documents rather than one document per book.** Cost if
wrong: Gate 1 catches a bad chunk size before training, not after — this is exactly why that
gate exists as a pre-training check rather than a post-hoc explanation.

**Fresh pretrain, not continued training on the existing checkpoint.** Cost if wrong: this is
the more expensive path (a full ~2.5B-token run vs. a continued-training pass), chosen
deliberately over the cheaper option to avoid a stage-order/forgetting confound in the result —
already decided in brainstorming, not reopened here.

**The Tortoise corpus slice's share is NOT pre-declared.** Cost if wrong: none — this is the
same honesty this project already applies to `flavour`, and pre-declaring a share for a source
this small would just be inventing a number `measure_corpus.py`'s gate would refuse anyway.

**Two seeds minimum for Gate 4.** Cost if wrong: doubles the training-time cost of the register
evaluation. Justified by this project's own repeated single-seed failures (the TinyStories
reduction arm's per-seed effect was monotone 0.92x→2.70x; seed 5489 alone would have read as a
null on that exact question).

## 8. Budget

| | |
|---|---|
| PG19 fetch + book-level chunking + prepare | ~1–2 h (streaming; scale depends on how much of the ~2B available tokens are used) |
| existing-blend re-share (TinyStories cut further) + re-measure | ~30 min |
| tokenizer retrain (48K) + re-derive prompt-set digests | ~30 min |
| tokenize full blend | ~1 h |
| Gate 1/2/3 checks | ~15 min |
| training, 2 seeds × ~2.5B tokens at ~169.4k tok/s (4-chip DDP) | ~4 h × 2 = ~8 h |
| Tortoise vignette writing (creative, not automated) | writing-plans/implementation time, not device time |
| Gates 4–6 evaluation | ~2 h |
| **wall clock** | **~13–15 h**, similar order to Stage A+B combined |
| disk | current free: **78 GB** (post-2026-08-31 prune). PG19 raw/prepared intermediates must be deleted immediately after each stage consumes them, same discipline as every prior corpus-assembly pass — no second uncleaned generation left on disk. |

## 9. What this does not do

It does not touch MMLU/knowledge — expected flat, not a target. It does not revisit context
length, capacity, or LoRA/editor-objective fine-tuning; those remain exactly where their own
specs left them, to be re-run on top of whatever checkpoint this stage produces if the project
continues in that direction afterward. It does not address the 4-chip serving quality
regression or the tt-metal v0.78.0 adoption question. It does not promise the Tortoise slice
moves any aggregate metric — that is explicitly the eval prompt set's job, not the corpus
share's.

## 10. Risks

- **PG19 register drift (archaic voice, not just richer vocabulary)** — named in §3, checked
  qualitatively in Gate 6, not just via the aggregate register-margin number, which cannot
  distinguish "richer" from "just old-fashioned."
- **The tokenizer→availability→shares circularity** (documented repeatedly in this project's
  own history): settle shares once against measured availability, accept being "one revision
  behind," do not chase re-convergence.
- **The book-chunking code is new** (§4) — nothing in `train/corpus.py`'s existing
  `rows_per_document` machinery does the inverse operation (splitting an oversized document),
  so this is real new code, not a parameter change, and should be planned and tested as such.
- **Two full training runs (2 seeds) at Stage-A scale is real compute**, not a rounding error —
  ~8 hours of device time before any evaluation. Confirm this budget is acceptable before
  writing the implementation plan.
- **A confirmed null is a valid, reportable outcome.** If Gate 4 does not clear both its floor
  and paired-detection thresholds on either seed, that is this project's own standing
  convention for a real result, not a failure to fix by loosening the gate after seeing the
  data — the one thing this project's history says it must never do.
