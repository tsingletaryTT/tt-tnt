<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Data scale-up: train tt-tnt-1024 on enough tokens to be worth its parameters

**Status: DRAFT, not approved, nothing implemented.** Drafted 2026-09-01 at the user's
request while the window-purity control arm drains. No task in it has been started.

## 1. The constraint, measured

Every capability intervention this project has run — improv thinking, skits, the reach dial,
the editor objective, tool calling, LoRA — has been a fine-tune of a base model that has not
read enough. The numbers are already in the repo and they are not close:

| | value | source |
|---|---|---|
| parameters | 123.0M | `train/sizes.py` |
| tokens trained on | 352.7M | `10,764 × 64 × 512` |
| **tokens/param** | **2.87** | derived |
| Chinchilla-optimal tokens/param | ~20 | Hoffmann et al. 2022 |
| tokens needed at 123.0M params | **2.46B** | derived |
| bits/byte, WikiText | **1.458** | `docs/measurements/external-*.json` |
| GPT-2 (124.4M params — the same size) | **0.977** | same |
| MMLU / ARC-Challenge | at or below chance | same |

**The corpus is the binding constraint, not compute.** Total *unique* tokens across all
thirteen registered sources is **870,125,123** — **35.4%** of Chinchilla for this model, i.e.
7.07 tokens/param even if every token were read exactly once. And half of it is one source:

| source | unique tokens | share of unique |
|---|---:|---:|
| `tinystories` | 447,943,902 | **51.5%** |
| `longform` (FineWeb-Edu) | 218,337,930 | 25.1% |
| `wikipedia_simple` | 68,469,002 | 7.9% |
| the other ten | 135,374,289 | 15.6% |

Compute is not the limit and this is measured, not assumed: 4-chip DDP runs at **169.4k
tok/s**, so 2.46B tokens is **~4.0 hours** and 10B is **~16.4 hours**.

## 2. What is already eliminated, so this spec does not re-litigate it

- **Capacity.** 22M → 123M moved held-out loss 0.011 nats against a 0.194-nat seed floor.
- **Context length.** Two 2048-context retrains were reverted; a chat-template window cap
  solved the problem the raise existed for.
- **Document fragmentation.** Refuted 2026-08-31: the median 2048-token window on the shipped
  corpus already held *zero* document boundaries
  ([`gate3-document-length.json`](../../measurements/gate3-document-length.json)).
- **Window purity.** Under test now; the early single-seed read is below detection.
- **Anti-forgetting by freezing.** LoRA at 100% task data still regressed repetition 18.46x
  the seed floor: freezing `W` constrains the *rank* of the update, not its effect.
- **Fine-tuning objectives generally.** Five have produced form without knowledge. The
  standing conclusion — "a small curated slice buys form, not knowledge" — is a statement
  about the base, and this spec is the only proposal that addresses the base.

## 3. Approach: two stages, not one bigger blend

**Stage A — scale.** A large, licence-clean, mostly-web corpus, trained to at least
Chinchilla. **Stage B — character.** The existing curated nine-source blend as a smaller
continued-training stage on top.

**Why not one blend at 2.46B.** The curated slices are what make this model interesting —
`weird`, `spine`, `folklore`, `flavour`, the occult and pulp material. They hold 135M unique
tokens between them. In a single 2.46B blend they are **5.5%** and in a 10B blend **1.4%**:
diluted to nothing, having been deliberately assembled. Separating the stages keeps scale and
character as independent knobs, and lets Stage B be re-run cheaply when the character changes
without repaying for scale.

**This is not a new mechanism — it is the one thing here that has already worked.** The
editor line's third run added a majority base-blend refresh and turned two confirmed
regressions into one null and one real improvement (4-gram repeat *better* at 3.00x the
floor). Stage B is that finding generalised: a minority curated share against a majority
base, with the mixing ratio a declared parameter rather than an accident.

## 4. Sources

**The scale-up is mostly "fetch more of what is already audited", not new acquisition.**
`longform` is FineWeb-Edu under **ODC-By-1.0**, already registered, already streamed by
`scripts/fetch_corpus.py`, already carrying a rationale — and the current fetch took 200,000
rows of a sample holding billions of tokens. That single source can supply the whole gap.

| source | licence | role | headroom |
|---|---|---|---|
| `longform` — FineWeb-Edu | ODC-By-1.0 | Stage A bulk | effectively unlimited |
| `wikipedia_simple` → full English Wikipedia | CC-BY-SA | Stage A reference prose | ~4B tokens |
| the ten curated slices | as registered | Stage B | already fetched |

**Licence discipline is unchanged and is a gate, not a step.** Two facts this project already
records stay binding: a host's licence tag on a compilation grants nothing about the
underlying content, and share-alike data licences (`CDLA-Sharing-1.0`, `CC-BY-SA`) are *not*
permissive — the README's provenance hedge must not be quietly upgraded into a claim. Adding
full Wikipedia adds a share-alike source at *scale* rather than at 7.9%, which is a
materially different position on weight release and must be decided explicitly, in writing,
before it is fetched. If that decision is uncomfortable, FineWeb-Edu alone closes the gap and
Wikipedia is optional.

## 5. Gates, cheapest first, each able to fail

Ordered so the cheapest disqualifier runs first. This project's own record is that a failed
gate costing seconds has twice saved multi-hour runs, and that a gate every arm clears is
decoration — so each gate below names what would make it *fail*.

**Gate 0 — licence (free, before any fetch).** Every token traceable to a licence permitting
training and weight release, recorded in `train/corpus.py` and the README provenance section
*in the same change*. FAILS if any source's rights cannot be established from the licence
text itself. No fetching before this passes.

**Gate 1 — tokenizer fertility (minutes).** The 32k BPE was trained on a TinyStories-heavy
blend; Stage A is web text. Measure tokens/word on FineWeb-Edu against the current corpus.
FAILS if fertility on Stage A text is more than **15%** worse than on the current blend — at
which point the tokenizer decision in §6 flips and must be re-argued, because a 15% fertility
penalty is a 15% tax on every token of a 2.46B-token run.

**Gate 2 — LR schedule (~2 hours, 2 runs at current scale).** Training runs at a **constant
3e-4 with no decay and no warmup**. A constant-LR run cannot converge; it asymptotes. Compare
`--lr-schedule cosine` against `constant` at today's 10,719 steps, same seed, same corpus.
FAILS to justify a change if the improvement does not exceed the 0.194-nat seed floor. ⚠️ An
LR decay tail was tested before and refuted **for register**; its effect on *loss* is
untested, and conflating those two results is exactly the error this gate exists to avoid.
Run this *before* the big run: scaling data 7x while leaving a known-suboptimal schedule in
place would under-deliver and confound the result.

**Gate 3 — scale on disk (~4 hours).** ≥2.46B unique tokens tokenized. FAILS if the achieved
unique count falls short — and the honest response is to reduce the target, never to upsample,
since repetition is what this whole spec exists to escape.

**Gate 4 — the real one (~4 hours training).** Bits/byte on WikiText via the existing
external-benchmark tool, so the number is comparable to the committed 1.458. Pre-declared:
**must improve by ≥0.15 bits/byte** (1.458 → ≤1.308), roughly 31% of the 0.481 gap to GPT-2.
⚠️ **No seed floor exists for bits/byte in this repo.** It must be established from a seed
replicate before the result is a headline, and until it exists the gate is provisional. A
result inside an unmeasured floor is NOT INTERPRETABLE, exactly as `scripts/evaluate.py`
would rule it.

**Gate 5 — knowledge (minutes).** MMLU and ARC-Challenge, currently at or below chance. This
is the gate that distinguishes "lower loss" from "knows more", and it is the one that would
make the model *useful*. FAILS if both stay within noise of chance — which would be a real
and publishable finding: that 20 tokens/param is not where knowledge appears at this size.

## 6. Decisions, and what each costs if wrong

**Keep the existing 32k tokenizer** (subject to gate 1). Retraining it re-denominates every
measurement in the repo — the corpus-assembly branch learned this the expensive way: "a token
budget is denominated in a unit the tokenizer defines," and retraining moved measured
availability by 6–24% per source. *Cost if wrong:* a fertility tax on every Stage A token,
bounded and measured by gate 1 rather than guessed.

**Store token arrays as `uint16`, not `uint32`.** The vocabulary is 32,000, well inside
65,536, and the current arrays are `uint32` — exactly 2x larger than they need to be. At 2.46B
tokens that is 9.8GB versus **4.9GB**. *Cost if wrong:* none identified; a vocabulary above
65,536 would break it, and a test should assert the dtype against the tokenizer's real size
rather than hardcoding either.

**Hold the model at 123.0M.** The brief that opened this line was "right-sized for a single
chip". At 2.46B tokens 123M is Chinchilla-optimal; at 10B it is deliberately over-trained,
which is the correct trade for a model meant to be *served*. *Cost if wrong:* if gate 4 passes
easily, the next question is a larger model, and that is a different spec.

**Stage B's mixing ratio is a declared parameter, not a default.** The editor line's evidence
is for a 60% base share; whether that transfers to a curated-character stage is untested.

## 7. Budget

| | |
|---|---|
| fetch ~2.5B tokens (streaming, observed ~57 MB/min) | ~3.5 h |
| prepare + tokenize | ~1–2 h |
| gate 2 (LR ablation, 2 runs) | ~2 h |
| Stage A training, 2.46B tokens at 169.4k tok/s | ~4 h |
| Stage B + evaluation | ~2 h |
| **wall clock** | **~13–15 h** |
| disk: raw + prepared + `uint16` arrays | **~30 GB** |

⚠️ **Disk is the live risk, not time.** `/` is at 98%. The 2026-08-31 prune reclaimed 112.5GB
and there is ~88GB free before the window-purity experiment's ~16GB lands. 30GB fits, with no
room for a second uncleaned generation of anything. Every fetch/prepare/tokenize step needs
its predecessor's intermediates removed — and per the prune's own lesson, **the record of what
was deleted has to live outside `artifacts/`**, which is gitignored.

## 8. What this does not do

It does not make the model creative, and it is not a capability objective. It buys a base
worth fine-tuning; every shelved capability line (skits, the reach dial, the editor) would
need re-running on top of it, and their earlier null results would need re-testing rather than
inheriting. It does not address the 4-chip serving quality regression, the SFT overfitting
question, or the tt-metal v0.78.0 adoption. And it will not reach GPT-2's 0.977: GPT-2 saw
~81 tokens/param, so matching it needs ~10B tokens, not 2.46B — which is why gate 4's
threshold is a 0.15 improvement and not parity.

## 9. Risks

- **The most likely failure is gate 4 passing and gate 5 failing**: loss improves, knowledge
  does not. That would say 123M parameters cannot hold the knowledge these benchmarks probe at
  any data scale, and it is the outcome that should most change the roadmap. Worth pre-agreeing
  that it is an acceptable result, because it retires "just add data" as this project's
  standing explanation for everything.
- **FineWeb-Edu is filtered for educational content**, which is a narrower distribution than
  WebText. It may cap what a WikiText bits/byte comparison can reach, and that ceiling is not
  measurable in advance from anything in this repo.
- **Character loss.** Stage A is 96%+ of tokens. If Stage B cannot restore the curated voice,
  the model gets better and less interesting — and "less interesting" has no gate above, which
  is itself a gap. `scripts/score_behaviour.py`'s register signal is the closest existing
  instrument and should be wired into Stage B before it runs, not after.
- **One architecture, one tokenizer, one seed per arm** unless gate 4's floor work says
  otherwise.
