---
license: apache-2.0
library_name: transformers
tags:
  - tenstorrent
  - blackhole
  - llama
---

# tt-tnt-1024

A 123M-parameter Llama-3-style model that is *of* Tenstorrent hardware, not merely
trained on it. Trained from random initialization on Blackhole with `ttml`
(tt-train) over a nine-source corpus with a small instruction slice, served
through the Tenstorrent vLLM plugin, and packaged with
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager).

It is small on purpose. One epoch takes about an hour on a single p300c, which is
what makes it useful as an instrument rather than a product.

⚠️ **This card's weights were retrained on 2026-08-28 to a 2048-token context** (up from
512), to push a real growing-conversation KV-cache crash — confirmed as a generic
tt-metal/vLLM defect, not specific to this model (see
`docs/upstream-tt-metal-asks.md` entry 6) — far out of ordinary reach, and now ships a chat
template (baked into `tokenizer_config.json`) that caps rendered conversation history to
the last 5 messages as a backstop. Matched-window (512) loss improved -0.2318 nats against
the prior checkpoint this card otherwise describes; no behavioural signal moved past its
noise floor in either direction (n=1 seed — no replicate exists yet). Full comparison:
`docs/measurements/evaluation-tt-tnt-1024-dialogue-vs-tt-tnt-1024-ctx2048.md`.
**The feature-support table and experiments below describe the prior 512-context
checkpoint this one was retrained from** (same corpus, same architecture, same warm-start
lineage) — none of MoE routing, die-region routing, thinking, skits, or the reach dial has
been re-verified against these specific retrained weights.

## What it does best

**Routing by physical die address.** Tokens can be assigned to experts by *where
they live on the harvested 11×10 Tensix grid* rather than by a learned gate, and
freezing routing to that geometry costs only **0.0118 nats** against a gate free
to learn (|t| 5.1, 14/15 signs). Source-characteristic tokens occupy measurably
distinct die regions — cell purity 0.546 against a 0.231 permutation floor. This
part requires a real harvested grid to be *about*.

**Sparse routing that pays.** A Mixture of Enthusiasts beats the dense baseline
from scratch, replicated at two seeds: pooled **+0.0417 nats**, |t| 8.0, 39/44
paired signs, separating late in training in both runs.

**Thinking on demand.** Asked to plan before it speaks, it emits well-formed
five-slot think-blocks in **98%** of generations where a control arm emits none.

## Feature support

| capability | state | note |
|---|---|---|
| Training on Blackhole (`ttml`) | ✅ | one epoch ≈ 65 min, one p300c |
| Multi-chip DDP | ✅ `[1,2]` / ⚠️ `[1,4]` | the 4-chip mesh froze the host |
| Sparse MoE (Mixture of Enthusiasts) | ✅ | replicated at two seeds |
| Die-region expert routing | ✅ | 0.0118 nats to freeze to physical address |
| Thinking (five-slot think-blocks) | ✅ format / ⚠️ effect | steers measurably; does not govern |
| Per-core Gumbel sampling on device | ✅ | custom kernels, per-core RNG |
| vLLM serving (TT plugin) | ✅ | OpenAI-compatible |
| `tt-model` packaging | ✅ v4 / ⏳ v5 | v5 needs wheels assembled |
| CPU-portable HF export | ✅ | runs without Tenstorrent hardware |
| Tool calling / chat template | ➖ | base completion model by design |
| Skits (multi-turn improv) | ✅ | five-turn scenes, real two-voice dialogue |
| Reach dial (controllable surprise) | ⚠️ measured, small | +0.060 residualised; plateaus; see below |

This card also records where the model fails, because a card that does not is
not useful. It is not a claim that the model is good.

## Shape

| | |
|---|---|
| parameters | 122,962,944 |
| hidden size / layers / heads | 1024 / 8 / 16 (4 KV heads) |
| context | 2048 |
| vocabulary | 32,000 (BPE, trained on this corpus) |
| training | 10,764 steps, batch 64, seq 2048, 4-chip DDP on one p300c |
| final validation loss (2048-token window) | 2.4504 |
| loss at matched 512-token window | 2.5408 (vs. 2.7726 for the prior 512-context checkpoint) |

4 KV heads means it shards across 1, 2 or 4 chips without violating
head-divisibility.

## What it does

It continues text, and — unlike earlier checkpoints in this project — it will
answer a question in the shape of an answer. Under greedy decoding:

> **Q: What is the capital of France?** → Answer: The capital of France is the city of Paris.
>
> **Q: What is the capital of Italy?** → Answer: The capital of Italy is the city of Rome.

Both correct. The corpus carries `wikipedia_simple` at 15%, which is where facts
of that kind live.

## What it gets wrong

> **Q: How many legs does a spider have?** → Answer: The body of a spider is a frog.
>
> **Q: What color is the sky?** → Answer: The sky is a blue color that is blue because it is blue because it is blue…

The second is a measured regression, not an anecdote. Against `tt-tnt-1024a`
(same architecture, same steps, corpus without the dialogue slice):

| signal | delta | vs seed floor | verdict |
|---|---|---|---|
| 4-gram repeat rate | +0.0074 | 3.32× | **worse** |
| termination rate | −0.0076 | 0.52× | not interpretable |
| genre collapse | −0.0035 | 0.06× | not interpretable |
| loss at matched window | +0.0102 | — | no floor for this instrument |

Nine of ten behavioural signals came back NOT INTERPRETABLE against this
project's 0.1944-nat seed-only noise floor. The one finding that cleared both
gates is that repetition got **worse**. A prediction that short question-answer
documents would improve termination was not supported.

Full comparison:
`docs/measurements/evaluation-tt-tnt-1024a-vs-tt-tnt-1024-dialogue.md`.

## Experiments this checkpoint is the base for

Two things were measured on top of this checkpoint on 2026-08-20. Both are recorded with their
limits, because both are easy to overstate.

**Sparse routing (Mixture of Enthusiasts) beats dense from scratch.** Replacing the
feed-forward with `ttml`'s sparse MoE and training both arms one epoch from init, paired on
seed 5489: validation **2.8098 for MoE against 2.8748 for dense** (mean delta +0.0481,
|t| 7.3, 20 of 22 signs), and the gap widens across training. Read it as the ordinary MoE
bargain — the configuration carries **3.62× total parameters at 0.989× active compute**, so
more parameters for the same compute helped. It is *not* evidence about the die-region routing
below. **Replicated at a second seed** (8191: +0.0354, |t| 4.5, 19/22 signs; pooled +0.0417 over
44 points), with the same late-separating trajectory in both runs, so treat ~0.04 as the
estimate.

**Routing by physical die address is nearly free.** Tokens can be routed to experts by where
they live on the harvested Tensix grid rather than by a learned gate. Freezing the gate to that
geography — never letting it learn — costs only **0.0118 nats** against a freely-learned gate
(|t| 5.1, 14/15 signs). Seeding the gate from the die map and then letting it move buys nothing
measurable (+0.0044, signs 8+/7−), even though the seeding demonstrably works as a classifier
(61.2% region recovery against a 10% chance floor). The geometry is real; the loss does not care
where the gate starts, only where it may end up.

**A five-slot think-block can be learned, and does not yet help.** Fine-tuned to emit
`offer / accept / add / stakes / handback` before continuing a story — one slot per improv
failure mode (escalating to the worst place, blocking with the dullest next step, drifting too
far out) — the model produces well-formed blocks in **98%** of generations (784/800; the
no-think control produces them 0% of the time). Substituting another story's block changes
**100%** of continuations, so the block steers rather than decorates. But it moves **none** of
the four failure-mode scores at α = 0.01, and one of those four is saturated on the real
co-occurrence table and cannot discriminate at all. Stage 1 is *partial*.

The generations explain the null better than the scores do. Asked to continue a story, the
model planned `add: dance` / `handback: dance` and then wrote a scary dog; another block set
`stakes: up` and the scene resolved into contentment. The syntax of intention is perfect and the
intention is not honoured. Read alongside the swap test that names it precisely: the block is
*context the model conditions on, not an instruction it obeys* — change it and the output moves,
ask it to mean something and it shrugs. On the same opening the no-think arm writes plainly
better prose. A plainer contributing reason: the slots are telegraphese (*loved play outside
friends*), because derivation lifts content words and drops the rest, so the model was asked to
produce a register nothing in 400M tokens of storybook prose resembles — and then to let that
register steer one it knows fluently.

Next unit is a **skit**: two or more turns with a partner who answers. A single continuation
gives `handback` nothing to hand back to, so the slot that encodes "make your partner look good"
cannot pay off or fail. Close reading in
[`episod-log.md`](https://github.com/tsingletaryTT/tt-tnt/blob/main/episod-log.md), 2026-08-21.

One process note kept deliberately: an earlier pass reported 0% adherence, from a run in which
all 17 RMSNorm gammas were provably frozen because `stochastic_rounding` defaults off on the SFT
path. With the gammas free, 0% became 98%. Both runs are preserved in the repo's measurement
files.

## The reach dial, and why this line of work is paused

**2026-08-23/24.** The last experiment on this checkpoint asked whether the model's declared
plan can be turned into a *control*: force a `reach` slot to `near` / `mid` / `far` and see
whether the model reaches a correspondingly distant word.

**It works, and it is small.** Forcing the dial moves the realised semantic distance of the
model's `add` word monotonically — `near` < `mid` < `far` — scene-paired over 826 held-out
scenes, and it **survives frequency control**. NPMI is not frequency-neutral (rare co-occurring
pairs score high, common words are capped low), so a raw effect would partly be "`far` picked a
commoner word". Residualising on log document frequency:

| contrast | raw | frequency-residualised |
|---|---|---|
| `near` < `mid` | +0.0839 (t 16.2) | +0.0324 (t 7.4) |
| `mid` < `far` | +0.0456 (t 13.9) | +0.0281 (t 9.0) |
| `near` < `far` | +0.1295 (t 23.3) | **+0.0604 (t 12.5)** |

About **53% of the raw effect is word frequency; ~47% survives.** A control arm that never saw a
`reach` slot shows nothing (t 2.36 / −0.30 / 2.35, not one step significant) — and two of those
sit between stage 1's threshold and this eval's, so importing the old constant would have turned
the *negative control* positive.

**The pre-declared EUREKA criterion was not met**, on one gate: the `add` slot-hit rate does not
hold across settings (worst-vs-best shortfall 0.0896). The failure is a *`near`-side dip*, not the
`far`-side collapse the gate was written to catch — `far` is +0.030 **above** `near`. The gate was
in the code before the data, so it stands as written rather than being narrowed afterwards.

### Three explanations eliminated, with evidence

- **More training does not help.** At 3× the budget (9000 steps) the effect is +0.060392 against
  +0.060438 — a **0.08% change**, with 38.8% of continuations differing, so it is genuinely a
  different checkpoint. These effects are a **plateau, not a floor**.
- **The arms are not undertrained.** The adherence gate got *worse* at 9000 steps (0.0896 →
  0.1062). Undertraining is refuted as the explanation. Separately: the recipe uses a **constant
  1e-5 learning rate with no decay**, so "never converged" describes the recipe, not a shortage of
  steps.
- **The vocabulary is not fixable by filtering.** The `add` slot was a discourse-particle
  vocabulary (`look`, `please`, `hi`, `hello` — 18.5% of observations in the top 25). A validated
  content-word filter (precision 0.949, recall 0.942) removed every particle — and the slot got
  **more** concentrated (top-25 share 0.185 → 0.218, distinct 6,442 → 4,846) with a **worse**
  frequency confound (spearman +0.208 → +0.233). The particle mass collapsed onto common verbs.

### Why it is paused: the constraint is the corpus

The dial reaches for distant words. This corpus does not contain many. TinyStories is **13,777
distinct words, with the top 1,000 covering 90.9% of all tokens** — `dragon` 959, `castle` 1,383,
`volcano` 208, `comet` 176, but `cathedral` 0, `submarine` 0, `meteor` 1, `orchestra` 2. `far`
collapsing to **88 distinct words** against `near`'s 265 is a model faithfully reflecting a world
with about a thousand usable words in it.

The mechanism is real and portable. The next investment is a corpus with range, not more training
and not a better filter. One narrower experiment — a noun-preferring rank key — was started and
stopped unfinished when this line of work was shelved.

Everything above re-derives from `docs/measurements/reach-dial.json` via
`scripts/eval_reach.py --rescore-from`, with **no model, tokenizer, or device** — verified
byte-identical, with gold distances reproducing at max absolute error 0.0.

## What it cannot do

No instruction tuning beyond a 2% slice of `databricks-dolly-15k`. No chat
template. No system prompt. It repeats under greedy decoding. It has 512 tokens
of context. It is a small model trained for one epoch on 352.6M tokens, and it
should be treated as an artifact of a hardware-and-tooling project rather than as
a useful assistant.

## Corpus

Nine sources, blended to a 400M-token budget and shipped as a **recipe** rather
than as text, because 46% of it is share-alike under two mutually incompatible
copyleft terms. Reconstruct it from
[`episod/tt-tnt-corpus`](https://huggingface.co/datasets/episod/tt-tnt-corpus).

The dialogue slice is `databricks-dolly-15k` (CC-BY-SA-3.0) at 2%, rendered as
plain `Question: … / Answer: …` prose with no role markers — the tokenizer has no
vocabulary for chat scaffolding.

## Serving

Through the Tenstorrent vLLM plugin. Use a plugin at or after `c127c17`: earlier
builds show a decode defect that degrades free-running generation into repetition
within a few tokens. The plugin reports version `0.1.0` either way, so a version
check cannot detect this; the bundle's adapter warns structurally instead.
