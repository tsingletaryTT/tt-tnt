# StoryCloze benchmark for tt-tnt and tt-tnt-1024 — design

## Why this exists, and why it is not literally "StoryBench"

The prompt was to run the "StoryBench" model benchmark against this project's two published
models. The actual paper with that name (arXiv 2506.13356, "StoryBench: A Dynamic Benchmark for
Evaluating Long-Term Memory with Multi Turns") evaluates frontier LLMs (200k+-token context)
playing branching interactive fiction: read a scene, select from labeled multiple-choice
options, handle correction feedback, and in one mode name the earliest point a mistake occurred.
That is an instruction-following, multi-turn agentic task. Neither published model here can do
it — `episod/tt-tnt` and `episod/tt-tnt-1024` have no instruction tuning, and this project's own
logs record the 1024 line failing plain factual Q&A ("capital of France" → a repetition loop).
Running the literal paper's protocol would measure "does this model follow instructions" (no,
categorically) and tell us nothing about narrative/memory quality specifically — a floor result
with no diagnostic value, decided against explicitly (see brainstorming transcript).

What's actually being adopted here is the closest fit that shares StoryBench's stated concern —
narrative coherence and long-range memory in generative models — while matching what a
23M/123M-parameter base LM can do at all: a **forced-choice, log-likelihood-ranking** task,
the same family as LAMBADA/HellaSwag. Concretely: the **Story Cloze Test** (Mostafazadeh et al.
2016), via its openly-licensed mirror. No literature "StoryBench" exists that targets small base
LMs; this substitution was surfaced and agreed during brainstorming rather than assumed.

The "long memory" angle from the original ask — varying how far back a disambiguating detail
sits — does **not** fit Story Cloze's stories (4 sentences of context, ~60-100 tokens total, too
short to manipulate distance meaningfully). That variant is written down as a follow-up, not
built here (see "Deliberately out of scope").

## Data source and licensing

**Corrected during planning, not assumed:** the obvious first candidate,
`Muennighoff/xstory_cloze`, turned out to be a dead end on direct inspection of its loading
script — its `_LANG` list (`ru, zh, es, ar, hi, id, te, sw, eu, my`) does not include `"en"` at
all, and every language it does support still requires a manual download via the same gated
Google Form as the original 2016 SCT release (`http://goo.gl/forms/aQz39sdDrO`). An HF dataset
card's own auto-generated summary claimed an ungated "en" config existed; it did not, and this
was verified against the loading script's actual source rather than the card's prose before
being written down here.

**`juletxara/xstory_cloze`, config `en`, on the HF Hub — CC BY-SA 4.0.** This mirror **is**
auto-converted to Parquet (confirmed via its dataset viewer) — loadable as
`datasets.load_dataset("juletxara/xstory_cloze", "en")` with no `trust_remote_code` and no
manual-download step, unlike both `Muennighoff/xstory_cloze` (gated, no `en`) and
`Muennighoff/xstory_cloze_data` (explicitly gated behind an access-conditions click-through,
and its own card states no English config is included — it points back at the original
Rochester request form for English). Confirmed columns and one example row via the dataset
viewer: `story_id`, `input_sentence_1..4`, `sentence_quiz1`, `sentence_quiz2`,
`answer_right_ending` (1 or 2, int). **Splits are named `train` (360 rows) and `eval` (1,510
rows)** — not `val`/`test` as an earlier draft of this spec assumed; both are real Story Cloze
items (the `train` split is the 2018 shared-task's small labeled set, `eval` is the larger one),
and both should be used since the model is not trained on either.

Fetched at a **pinned HF revision** (resolved and hardcoded during Task 1 below), cached under
`artifacts/storycloze/` (gitignored, per existing convention), with the revision hash recorded
in every output JSON — the same "a corpus generation is not implied by a model size" discipline
this project already applies to its own tokenized corpora (`train/paths.py`'s docstring, and the
tokens-v3-vs-v4 mistake in the ctx2048 retrain episode).

**README update, in the same change that adds this script**: a new provenance entry for
`juletxara/xstory_cloze` (CC BY-SA 4.0, used for evaluation only — not redistributed, not mixed
into any training corpus, so share-alike has no downstream obligation beyond attributing the
eval data itself). Worth naming plainly in that entry: the original English Story Cloze Test is
still nominally gated behind Rochester's request form; this mirror carries an "en" split anyway
under its own declared CC BY-SA 4.0 license, inherited from the same license the multilingual
XStoryCloze project (Lin et al.) applied to its whole release. This project is relying on that
upstream project's licensing decision, stated here rather than left implicit.

## Scope

- **Models under test**: `episod/tt-tnt` (v3, 22,025,088 params, 384-dim, 2048-token context,
  the protected public baseline) and `episod/tt-tnt-1024` (123M params, 512-token context,
  dialogue-trained, currently `docs/current_model.json`'s designated model for the 1024 line).
  Both stories fit comfortably inside either context window — no truncation concerns.
- **Execution**: CPU-only via `transformers`, no `ttnn`/`ttml` import, no device, no gozer
  lease — matching `scripts/evaluate.py`/`score_behaviour.py`/`probe_context_use.py`'s existing
  convention for anything that doesn't need hardware.
- **Deliverable**: `scripts/eval_storycloze.py`, plus `docs/measurements/storycloze-tt-tnt.json`,
  `docs/measurements/storycloze-tt-tnt-1024.json`, and a comparison artifact
  (`docs/measurements/storycloze-tt-tnt-vs-tt-tnt-1024.json` or `.md`, format decided during
  planning) produced by `--compare`.

## Scoring methodology

For each item, for each candidate ending: concatenate context + ending, run the model
teacher-forced (same technique as `test_hf_parity.py`/`convert/ttml_forward.py`), and take the
summed log-probability of the ending's tokens conditioned on the context. The higher-scoring
ending is the model's choice; compare against `answer_right_ending`.

**Both normalizations are computed and reported, not one assumed correct:**
- **raw summed log-likelihood** over the ending's tokens
- **mean log-probability per token** (length-normalized)

Raw sum structurally favors shorter endings regardless of content — a real, known bias for this
exact task family. The report states, per model, the correlation between "ending chosen" and
"ending is the shorter of the two" for each normalization, and the headline accuracy is drawn
from whichever normalization has the lower length-bias correlation on this data (both numbers
stay in the artifact regardless, so a reader can see what was not chosen — same convention as
`docs/current_model.json`'s `candidates` field before it was retired, and every other measurement
artifact in this project that reports the road not taken).

## Controls and calibration

Story Cloze has a specific, published artifact worth guarding against directly: Schwartz et al.
("Story Cloze Ending Selection Baselines and Data Examination", arXiv 1703.04330) showed
classifiers can score well above chance using **only the candidate endings, never the context**,
because the wrong endings were authored separately from the right ones and carry detectable
stylistic tells (length, sentiment, generic phrasing). A number that doesn't check for this is
exactly the kind of instrument-measures-itself result this project's CLAUDE.md has hit
repeatedly under other names. Required controls, all computed in the same run:

1. **Context-blind control.** Score the two endings' likelihood with no context at all (just the
   ending sentence, unconditioned or conditioned on a fixed empty/BOS prefix). If this alone
   scores well above the class-balance baseline, the full-context number's real narrative-
   understanding contribution is at most the gap between the two — stated explicitly as a ceiling
   on what the headline can claim, not subtracted out or hidden.
2. **Class-balance check.** Report the dataset's actual empirical split of
   `answer_right_ending` (1 vs 2) rather than assume 50/50 — the chance baseline used everywhere
   else in the report is this measured number, not an assumed one.
3. **Not-hollow scorer proof.** A constructed item (real context, one real ending, one
   obviously-garbled wrong ending — e.g. token-shuffled) that the scorer must get right. Mutation-
   tested: reversing which ending's score is treated as "chosen" must fail this test.
4. **v3-vs-1024 paired comparison.** McNemar's test over the identical item set (same items
   scored by both models) — items are the correct paired/exchangeable unit here (same principle
   as pairing by prompt in `score_behaviour.py`, by window in `probe_context_use.py`), not an
   unpaired accuracy-difference or a t-test built for a different kind of eval in this repo.

## CLI and reproducibility

```
scripts/eval_storycloze.py --model {tt-tnt,tt-tnt-1024} [--split train|eval] [--out PATH]
scripts/eval_storycloze.py --rescore-from PATH.json         # no model, tokenizer, or device
scripts/eval_storycloze.py --compare A.json B.json          # paired McNemar, refuses a mismatched pair
```

- `--rescore-from` re-derives every published number (both normalizations, all controls, McNemar
  if given two files) from stored per-item log-likelihoods with no model, tokenizer, or device —
  same convention as `eval_reach.py`/`eval_skits.py`'s `--rescore-from`, verified under an import
  blocker that raises on `torch`/`transformers`/`ttnn`/`ttml`.
- `--compare` refuses to compare two result files scored on different dataset revisions or splits
  rather than compute a number that isn't actually comparable — same refusal shape as
  `evaluate.py`'s window guard and `reach.py`'s cross-arm-set refusal.
- Every output JSON records: HF dataset revision hash, split, model HF repo path + a content
  hash of the weights actually loaded (not just the directory name — this project has shipped a
  wrong-corpus checkpoint under a right-looking path before), and both normalization variants'
  full per-item scores (so `--rescore-from` has something to rescore from).

## Testing

- Not-hollow scorer test (above), mutation-checked (reversing the argmax must fail it).
- Context-blind-control purity test: a mutation that leaks context into the "context-blind"
  scoring path must fail its own test — the control has to be provably blind, not just named
  blind (same shape as this project's repeated "a test that supplies what's missing tests
  nothing" lesson).
- License/provenance test: README contains the CC BY-SA 4.0 entry for `juletxara/xstory_cloze`.
- Import-purity test: `scripts/eval_storycloze.py` never imports `ttnn`/`ttml`.
- `--compare`'s mismatched-revision/split refusal, tested directly (two files built to differ
  only in revision hash must be refused).
- McNemar computation tested against a hand-computed small fixture (not just "runs without
  error").

## Deliberately out of scope (this spec)

- **Long-memory / clue-distance variant.** Story Cloze's stories are too short to manipulate
  disambiguation distance meaningfully. A follow-up eval built from this project's own corpus
  (same shape as `probe_context_use.py`: real documents, disambiguating detail placed at a
  controlled distance from the choice point, forced-choice scoring reused from this spec's
  scorer) is the natural next step, not attempted here.
- **Literal StoryBench (arXiv 2506.13356) protocol.** Considered and rejected — see above. Not
  revisited unless a future instruction-tuned checkpoint changes the capability picture.
- **Generation-based story evaluation** (TinyStories' own GPT-4-judge protocol, or any
  LLM-as-judge scoring). Would introduce an external judge-model dependency this project has so
  far avoided in favor of self-built detectors; not part of this spec.

## Self-review notes

- No placeholders remain; every section states a concrete mechanism.
- The dataset identifier was corrected mid-planning (`Muennighoff/xstory_cloze` → confirmed to
  exclude English and remain gated → `juletxara/xstory_cloze`, confirmed ungated/parquet/CC
  BY-SA 4.0/has an `en` split) by reading the loading script and dataset viewer directly rather
  than trusting an HF card's prose summary — the same "verify the instrument" discipline this
  project applies elsewhere. Every other section already used the generic phrase "openly
  licensed mirror" rather than assuming the first name found was final, so no other section
  needed a matching correction.
- Internal consistency: the "both normalizations reported" decision in Scoring methodology and
  the length-bias control in Controls describe the same measurement from two angles and do not
  contradict each other.
- Scope check: single spec, one implementation plan's worth of work (one script, its tests, one
  README edit, two/three output artifacts). Not decomposed further.
- Ambiguity check: "current_model.json's designated 1024 checkpoint" is named explicitly
  (`artifacts/hf-tt-tnt-1024`, currently the 512-context dialogue weights) rather than left as
  "the current one," since that designation can change; the plan should re-read
  `docs/current_model.json` at implementation time rather than trust this spec's snapshot.
