<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Training an editor: a checkpoint that reviews its own drafts, and stops treating skits and dialogue as separate

## Why this exists

A tool-calling storytelling harness (`scripts/story_tools.py`, 2026-08-27) drove
`tt-tnt-1024-dialogue` one short, judged turn at a time instead of one long unguided
generation, and found two things prompting cannot fix:

1. **Chaining short restarts does not obviously beat one long generation.** Each fresh call
   re-rolls this model's fragile sentence-onset behavior from a cold start.
2. **Prompting the model to "edit" its own draft does nothing.** Framed explicitly as an
   editing task ("Draft: ... / Better version:"), the same base model produced *worse*,
   unrelated fragments — a clean negative result. `tt-tnt-1024-dialogue` was never trained to
   revise text, and prompting alone does not manufacture that capability.

Separately, this project has two training lines that have been kept apart for no principled
reason: the *dialogue* slice (2% databricks-dolly-15k, taught this model to answer a question
in the shape of an answer) and the *skits* checkpoints (a five-slot `offer/accept/add/stakes/
handback` turn schema, `train/skit.py`, trained on a **separate** checkpoint lineage). Nothing
about those two objectives conflicts; they were just never asked to share weights.

This spec designs one continued-training stage that does both: gives the model a real editor
objective, and folds the skits turn-structure into the same checkpoint dialogue already lives
in.

## Scope — what this is not

This does **not** address the `FABRIC_2D_TORUS_XY` 4-chip serving regression found the same
day (`AUTOFIX.md`, `docs/serving-with-tt-kernel.md` §8) — that is a serving/hardware-fabric
correctness bug, unrelated to model weights, and conflating the two tracks would make both
harder to reason about. The editor objective targets this model's *inherent* repetition/
grammar/word-collapse weakness, which is present even on the known-good 2-chip config.

## 1. The corruption module (`train/corrupt.py`)

Four targeted corruptors, each reproducing one **specific, documented** failure mode from this
model's real output — not generic noise:

| corruptor | reproduces | example (from tonight's real output) |
|---|---|---|
| `repeat_collapse` | duplicated span/sentence loop | "The dragon. The dragon, in the dragon." |
| `garble_word` | invented non-word standing in for an ordinary word | "Tryburg", "Alexandary", "Higheriq" |
| `drop_or_double_function_word` | dropped/doubled article-preposition-auxiliary | "she was always had been very special" |
| `fuse_clauses` | run-on clauses with no conjunction | "she had a fairy tale, love, come" |

Each corruptor is a **pure function**: `corrupt(text: str, *, seed: int, severity: float) -> str`,
zero coupling to the training pipeline. This is deliberate — the same module is reusable later
as (a) a deliberate creative "glitch" post-processing mode at inference time, (b) a held-out
eval-set generator independent of the training corpus, (c) a severity dial in the spirit of
the reach-dial mechanism, if that's ever wanted. None of that is built now; the module's API
just doesn't foreclose it.

**`garble_word` and legitimate coined words.** The user explicitly does not want invented
*names* (Tolkien-style) or onomatopoeia flagged as errors — only a fake word standing in for
ordinary vocabulary. `garble_word` therefore skips: any word already capitalized mid-sentence
in the source (an existing proper noun), and any word immediately following "named"/"called"/
"nicknamed". It only ever corrupts common-noun/verb/adjective tokens, so the training signal
is specifically "an ordinary word came out wrong," never "a coined name came out wrong."

**Building the `draft`/`better` pairs.** `better` = a clean sentence drawn from the existing
nine-source corpus (never model-generated — guarantees real, grammatical English by
construction). `draft` = the same sentence after one or more corruptors are applied
(1–2 corruptors per example, severity sampled, so the training set spans mild-to-severe rather
than a single fixed corruption strength).

## 2. Task framing: plain-text delimiters, NOT new special tokens

**Corrected during planning, after the first draft of this spec proposed new added tokens
(`<draft>`/`<edit>`), the same way `</s>` was added.** That precedent doesn't transfer:
`</s>` became an added token during the tokenizer's *original from-scratch training*, when
the full 32000-slot vocab was still being decided. This checkpoint's vocab is already fixed
and fully populated (BPE reached exactly 32000 tokens with no shortfall — see
`convert/tokenizer.py:SPECIAL_TOKENS`, ids 0–3 fixed at that original build). Adding a new
special token *now* means growing the vocab to 32002 and resizing the (tied) embedding/
lm_head table by 2 rows before any continued training can resume — real model surgery, and
a silent way to corrupt a plain `--resume` if done carelessly. Not worth it for this
objective.

Instead: plain-text delimiters, encoded with the existing vocab like any other text —
`\nDraft: ` and `\nEdit: `. Format:

```
\nDraft: {corrupted sentence}\nEdit: {clean sentence}</s>
```

Trade-off, stated plainly: a real sentence in the corpus containing the literal words
"Draft:"/"Edit:" at line-start would collide with the delimiter. Checked against the
existing corpus in Task 2 below (grep for the exact strings) rather than assumed away — the
literal risk is measured, not just judged small in the abstract.

Loss is masked exactly the way the dialogue slice already masks its Q&A pairs
(`labels = [-100]*(len(prompt_ids)-1) + completion_ids + [-100]`, keeping the last prompt
position supervised since its target is the first completion token — see the 2026-08-20 improv-
thinking entry in `CLAUDE.md` for the precedent): the model is only ever scored on producing
the `<edit>` half, so the loss signal is specifically "given a draft, produce the correction,"
not "predict a draft is likely."

## 3. The skits slice, folded into the same stage

`train/skit.py` already derives five-turn (`offer/accept/add/stakes/handback`) examples from
real corpus text. This stage adds those as a second SFT-style slice, same masking convention,
target being the model (narrator) turns. No new derivation work — reusing the existing
`scripts/derive_skits.py` output. This is what makes the "not orthogonal" goal literally true
of the resulting weights: one checkpoint, trained on dialogue (already baked in from the
current checkpoint), skits turn-structure, and now editing, together.

## 4. The blend for this stage

Continued training from `artifacts/checkpoints-1024-dialogue`'s final checkpoint (`--resume`,
same mechanism already used for chunked runs — the AdamW optimizer state carries over, not
reset). Per-step mix:

- **Base-blend refresh** (majority share): resampled from the existing nine-source
  `artifacts/corpus/blend.txt`, at the SAME shares already settled — this is anti-forgetting,
  not a re-litigation of `docs/corpus_blend.md`.
- **Editor pairs** (`train/corrupt.py` output): a minority share, sized the same order as the
  original 2% dialogue slice that measurably worked without destabilizing the base capability.
- **Skits examples** (`derive_skits.py` output): a minority share, likewise modest.

Exact percentages and step count are an implementation-time decision (informed by the same
per-source availability check `measure_corpus.py` already runs), not fixed here — the spec's
job is the objective and the data construction, not tuning the mixing ratio by hand in advance
of seeing real numbers.

## 5. Evaluation — before/after, not vibes

1. **Held-out corruption set**, sentences never used in training. Score whether `<edit>`
   output on a corrupted draft is closer to real English than the draft — using
   `scripts/story_tools.py`'s existing repetition/garble-adjacent checks as a starting point,
   extended with a real-word check (vocabulary/frequency lookup) to catch `garble_word`-style
   fake words specifically.
2. **Re-run `story_tools.py::self_edit()` on the new checkpoint.** It had a clean negative
   result on the current checkpoint tonight — this is the direct before/after proof the
   objective is asking for.
3. **Skits recognizability**: does the model's own unprompted continuation exhibit the
   accept/add/stakes shape, checked the same way `scripts/eval_skits.py` already checks it for
   the separate skits checkpoints.
4. **No regression on dialogue/base capability** — re-run the existing behavioural eval
   (`scripts/evaluate.py`, `scripts/score_behaviour.py`) against the current designated
   checkpoint as the control, per this project's own standing floor-comparison convention
   (`docs/measurements/seed-noise-floor.json`).

## 6. Checkpoint identity

New candidate entry, `tt-tnt-1024-editor`, added to `docs/current_model.json`'s `candidates`
list (following the existing `-dialogue`/`-1024a`/`-2ep` naming convention) — **not** promoted
to `current` until the evaluation in §5 actually shows an improvement without regression, per
this project's own rule that a designation is not a claim of quality in the abstract.

## Risks

- **Corruption realism is only as good as tonight's sample.** The four corruptors are built
  from real output observed in one session; they may not cover every failure mode this
  checkpoint has, only the ones that surfaced tonight. Acceptable for a first run — the
  held-out eval in §5.1 will show if the objective transfers beyond the exact patterns it was
  built from.
- **Mixing-ratio risk shared with every prior continued-training stage in this project**:
  too large an editor/skits share risks displacing base fluency the way TinyStories-share
  changes have in the past; too small risks the 2026-08-27 negative result repeating. Sized
  conservatively (§4) and gated on §5.4's no-regression check.
- **The garble-word skip heuristic (capitalization / "named"/"called") is a heuristic, not a
  guarantee.** It could still occasionally corrupt a genuine coined word that happens not to
  follow those cues, or fail to skip one. Acceptable risk for a first pass; worth revisiting if
  the held-out eval shows it misfiring.
