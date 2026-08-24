# Skits v2: real dialogue, and a dial for reach

**Date:** 2026-08-23
**Status:** design, approved to execute
**Supersedes in priority:** skits plan tasks 6-8 (TT optimization, explicitly non-gating)
**Builds on:** `docs/superpowers/specs/2026-08-21-skits-design.md`, and the stage-2 result in
`docs/measurements/skits-stage2.json`

## The one thing stage 2 proved

Stage 2's publishable result is a single slot. `add` beats a context-matched floor by
**+0.326** (and its confound runs the other way, so that is biased *downward*). `accept` survives
at roughly **+0.083**. `stakes` was withdrawn as mostly confounded, and
`handback_anticipation` is not measurable by that design.

The reason `add` is the one that worked is mechanical, and it is the whole basis of this spec:

| slot | how often its value is already visible in the context |
|---|---|
| `accept` | **92.8%** — mostly the protagonist's name |
| `add` | **~40-44%** |

`accept` was near-tautological. `add` is the slot where the model names something *not in front of
it* and then goes and uses it. That is not next-token prediction; it is reaching for an idea and
committing to it. Everything below is an attempt to take that one real mechanism and make it
**controllable**.

## What we are building

A model whose think-block declares **how far it is reaching** for the next beat, and then reaches
that far. Two capabilities, one of which is the headline:

1. **The reach dial (headline).** A new slot, `reach`, taking `near` / `mid` / `far`. It is derived
   from how semantically distant the `add` word is from what is already in play. At inference the
   dial is an *input*: set `reach: far` and the model should produce a more distant beat.
2. **Turns that are actually turns (foundation).** Stage 2's "partner turns" were the same
   narrator's next sentence. Rebuilding derivation on real two-speaker exchanges is what should
   finally give `handback_anticipation` a real ceiling.

### Why this is the interesting direction and not just another slot

The three failure modes this project set out to avoid were: taking the story to the worst place,
taking the most boring next step, and going so far out nobody can follow. Those are not three
problems. They are one axis — reach — with a bad policy at each end. A dial does not merely measure
the axis; it hands it to the co-author.

## The measurement that decides it

**EUREKA criterion.** Conditioning on `reach` changes the realised distance of the `add` word,
monotonically, at Bonferroni-corrected significance — *and* coherence does not collapse at `far`.

Concretely, generate the same held-out scenes three times with the dial forced to `near`, `mid`,
`far`, then measure:

- `realised_reach`: NPMI distance of the generated `add` word to the context.
  Require **near < mid < far**, each step significant.
- `groundedness` (NPMI-based, already in `scripts/score_improv.py`) must not fall below its
  `near` value by more than a pre-declared margin at `far`. This is the "nobody can follow" guard:
  a dial that only produces noise at `far` has failed, not succeeded.
- `add` slot-hit rate must hold at every setting. If the model stops *fulfilling* its plan when the
  plan gets ambitious, the dial is decorative.

A dial that moves distance but destroys coherence is a **negative result**, and gets published as
one. So is a dial that does not move.

### Pre-declared thresholds, fixed before any run

- Family: 3 dial contrasts (near<mid, mid<far, near<far) + 3 coherence guards + 3 adherence
  checks + 2 handback checks = **11 tests**. `BONFERRONI_ALPHA = 0.05/11`, `CRITICAL_T = 2.843`.
  Same family size as stage 2 by coincidence, not by copying — recount if the design changes.
- Do NOT import stage 2's or stage 1's constants. In stage 2 two effects landed between stage 1's
  threshold (2.576) and the correct one (2.843); importing would have manufactured both.
- Coherence margin at `far`: groundedness may fall no more than **0.05** below its `near` value.
  Declared here, before the data exists.

### Amendment, 2026-08-23, before any training run: the dial must beat a frequency confound

**NPMI is not frequency-neutral, and the bias runs opposite to the obvious worry.** Two rare words
that co-occur at all score *high* — their expected co-occurrence under independence is tiny — while
a common word's NPMI with anything is capped low by its own marginal. So a rare `add` word tends to
score NEAR and a common one tends to score FAR. Measured on 20,000 stories:

- `spearman(add-word document frequency, distance) = +0.306` (~9% of rank variance)
- median `add`-word document-frequency rank: **1,100 in `near` vs 444 in `far`** (higher rank = rarer)

Therefore **`near < mid < far` on realised distance is not, by itself, evidence the model reached
further.** It is equally consistent with the model reaching for a *commoner* word. The criterion
above is insufficient as written and is amended:

The EUREKA criterion now requires the monotone dial effect to survive **frequency control**. Report
all three of:

1. **Raw** realised distance per bucket (the original criterion).
2. **Frequency-matched**: distance per bucket over a subsample matched on `add`-word document
   frequency, or equivalently the effect residualised on log document frequency.
3. The **realised document frequency** per bucket, so a reader can see the confound's size next to
   the effect rather than taking our word for it.

A dial that moves raw distance but not frequency-controlled distance is a **frequency dial**, and
must be published as that — it is a real finding about NPMI, not a finding about the model.

Every derived row carries its own `add_df` so an eval can match or covary without rebuilding a
document-frequency table. An eval can control for a number; it cannot control for a warning.

### Amendment: two further instrument problems found during derivation

**Self-inclusion.** The scored story is itself a document in its own association table, which drives
the zero-evidence rate to a structural 0.0000 — every pair the story contains has co-occurred, in
that story. Fixed with leave-one-out: a story's own contribution is removed before its distances are
computed. Any future distance metric over a corpus-built table inherits this bug by default.

**The same-speaker filter cannot catch the case that motivated it.** The controller inspected a real
skit where model turn 0 is the mother and model turn 2 is the daughter, and required a conservative
filter in response. That filter provably cannot catch it: the gap between those utterances carries
narrative rather than being tag-only. The filter still removes tag-only gaps, which are a real
subset, so it stays — but the residual same-voice rate after filtering must be **measured and
published**, not assumed away, and the premise "the partner turn is a different voice" holds only
probabilistically. State the residual rate beside any handback result, since that slot is the one
that depends on the premise.

## Derivation: real dialogue

### Turn extraction

Replace consecutive-sentence slicing with **alternation by position**. A skit needs five turns; take
stories with **≥5 quoted utterances** and assign roles by index parity (`model, partner, model,
partner, model`).

Do **not** attribute speakers by name. A probe that tried found 9.6% of stories usable but got the
speakers *backwards* — vocatives inside the quote ("*Fluffy, the word is 'repeat'*" is Timmy
addressing Fluffy) fool any first-capitalised-name heuristic. A skit needs the **voice to change**,
not to know whose it is, so attribution is unnecessary and dropping it drops the bug with it.

Measured yield: **18.8% of stories usable → ~398,000 at corpus scale → ~80,000 skits** at a
conservative keep rate, roughly 4× stage 2's 18,610. Sample output is real yes-and structure:

```
MODEL  0: Hello, Amy! I am here to help you.
PARTNER 1: Can you help me pick a dream?
MODEL  2: Of course! Let's pick a happy dream for you.
PARTNER 3: Thank you, ghost!
MODEL  4: You're welcome. Now, go back to sleep and enjoy your dream.
```

Compare stage 2, where "PARTNER" was the narrator continuing: *"The cherry tree was envious of the
big trees."*

### The splitter, fixed properly

The current splitter is frozen because its output sits behind the published
`improv-stage1.json`. It is now costing us the experiment: it fragments dialogue-with-attribution,
those skits fail the accept gate, and stage 2's eval population lost **43% of the corpus's
dialogue** relative to the corpus (54.6% → 31.0% of units; 17.6% → 11.6% of sentences). We trained
on the monologic residue.

Both known variants are wrong in opposite directions:

| pattern | `'"It catches the light!" said her friend.'` | `'He said "no." She left.'` |
|---|---|---|
| `(?<=[.!?"])\s+` (current) | 2 — wrong | 2 — right |
| `(?<=[.!?])\s+` (attempted) | 1 — right | 1 — wrong |

**Decision (approved):** write one dialogue-aware splitter that keeps a quoted span together with
its attribution clause, then **re-derive and republish stage 1** with a note recording what moved.
Clean cutover; no second splitter in the tree. `improv-stage1.json` gets a `superseded_by` field and
the new numbers beside the old.

### `reach` derivation

For each model turn, `reach` is the tercile of `max NPMI(add_word, c)` over content words `c`
already in play. Measured distribution over stage-2 skits (n=9,000):

| p5 | p25 | p50 | p75 | p95 | mean | sd |
|---|---|---|---|---|---|---|
| 0.000 | 0.081 | 0.169 | 0.273 | 0.439 | 0.187 | 0.133 |

3,764 distinct values across 9,000 observations. Terciles are therefore **balanced by
construction** — the specific failure that killed `stakes`, which was 85.3% one class and left only
26 points of headroom above a 0.738 chance floor. Tercile cut points are computed on the training
split only and **recorded in the manifest**, so eval cannot silently re-fit them.

`far` = most distant tercile. The naming is deliberate: high NPMI means *closely associated*, so
`reach: far` corresponds to **low** NPMI. Getting this backwards inverts the whole result, so the
derivation must carry a test that a hand-built near pair and a hand-built far pair land in the
expected buckets.

### `stakes`, per the approved decision

Re-derived as a **continuous delta** rather than a three-way label, and tested on magnitude. It is
not a headline slot in this spec; it is carried so the withdrawal in stage 2 has a successor rather
than a silent disappearance.

## Slots, final

| slot | meaning | scored against |
|---|---|---|
| `offer` | what I heard | the preceding partner turn (block 0: the whole 2-sentence prefix — see the erratum in the stage-2 spec) |
| `accept` | what I take from it | my own turn |
| `add` | what I bring that is new | my own turn |
| `reach` | how far I am reaching for it | **realised distance of the generated `add` word** |
| `handback` | what I leave my partner | the next partner turn (now a real second voice) |

`reach` is the only slot whose scoring is a *magnitude* rather than a hit, which is what lets it be
a dial rather than a checkbox.

## Non-goals

- Not a bigger model. 123M, 512-token context, one board, `--ddp 2` at most.
- Not MoE in this spec. MoE beat dense at two seeds and is worth compounding, but changing the
  architecture and the corpus and adding a slot in one step makes the result uninterpretable.
  MoE is the next spec, once the dial is established or refuted.
- Not tt-lang kernels in this spec. Task 6's measure-first rule stands: nothing gets optimised
  until the eval is shown to be generation-bound. `tt-lang` is not currently importable here
  (`find_spec` returns None for `ttlang` and `tt_lang`), which is a prerequisite to sort out first.

## Carried-forward facts, each learned the hard way

- Labels are **pre-shifted**: `labels[t] = input_ids[t+1]`, `-100` elsewhere. Applied positionally
  over the nine segments; the last position of an unsupervised segment takes the **first** token of
  the following supervised segment. Off by one drops every partner→think transition, and a mutant
  that does exactly that passed 1,181 tests.
- **Tile-align to 32** or SDPA backward raises `TT_FATAL ... u_scaler shape mismatch`.
- `stochastic_rounding` defaults **off** on the ttml SFT path and silently froze all 17 RMSNorm
  gammas for 3000 steps. Set it, then **read it back from `trainer.optimizer.get_state_dict()`** and
  raise. Verify the guard by breaking it on purpose.
- Warm start via `train.enthusiasts.warm_start` from
  `artifacts/checkpoints-v077-beta2-control/tt_tnt_step00010764.pkl` (66 tensors, exactly 17 gamma).
- Examples are tokenized **segment-wise**, so the trained sequence carries a per-segment prefix
  space (`['.', 'Ġ<', 'think', '>']`). Eval prompts must be built the same way or the model is asked
  to emit `<think>` after a token it never saw there.
- Both arms consume identical skits in identical order; prove it with a permutation fingerprint in
  the manifest, do not assert it.
- **Never compare the arms' training losses.** They supervise different token sets.
- One board only. A `[1,4]` mesh hard-froze this host with no OOM, panic, or pstore record. No
  reproduction, no fix, do not test it.
- Lease before anything opens `/dev/tenstorrent/*`. A bare `import ttml` is a device open; use
  `importlib.util.find_spec` to ask whether a package exists.

## Testing requirements

This project has shipped **ten** tests that passed against both correct and incorrect code. Nine
were value-level; the tenth was a class — whole decision functions reached only through a driver,
where four mutations each rewrote a published claim while 1,212 tests passed, one of them flipping
the verdict outright.

Therefore, in this spec:

1. Every decision function — bucket assignment, dial verdicts, coherence guards, publication
   selection — is a **named, called function with its own fixture test**. Nothing decides a
   published claim from inside `main()`.
2. Every substantive test is demonstrated **red** against a plausible wrong implementation, with the
   failure output recorded.
3. Fixtures must be built so the assertion can fail. Two fixtures in this project were vacuous
   because one word repeated through every sentence, and a third could not distinguish two candidate
   offer spans because both prefix sentences shared a word.
4. Any property of **scale** (saturation, hub structure, base rates, bucket balance) is tested
   against the real artifact, not a synthetic fixture.
5. Every reported effect ships with its floor, and every degenerate-by-construction metric says so
   **inline in its own row**.

## Deliverables

- `train/reach.py` — bucket derivation, tercile fitting, the `reach` slot
- `scripts/derive_dialogue_skits.py` — dialogue-aware splitter + alternation extraction
- `scripts/train_reach.py` — the dial arm and its control
- `scripts/eval_reach.py` — the three forced-dial generations and the EUREKA test
- `docs/measurements/reach-dial.json` — the result, with floors, and with the drop rate and
  selection bias disclosed (stage 2's artifact omitted both until the final review caught it)
- republished `docs/measurements/improv-stage1.json` with the corrected splitter
