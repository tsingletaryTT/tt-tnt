#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Stage-2 evaluation. The swap test runs FIRST, then per-slot prediction accuracy.

Order is not cosmetic. If substituting another skit's think-block barely changes the
continuations, the model emits a plan and writes independently of it — the thinking is
decorative, and no amount of good per-slot accuracy afterwards would mean anything. Stage 1
established that the block DOES steer (100% of continuations changed); this re-establishes
it for skits before anything is built on it.

WHAT STAGE 1 GOT WRONG, AND WHAT THIS FILE DOES DIFFERENTLY
----------------------------------------------------------
Stage 1 produced well-formed think-blocks ~98% of the time and moved ZERO of four scorers.
The blocks were context the model conditioned on, not instructions it obeyed, and — the
part that matters here — every number was reported without its floor, so nobody could see
how much of a "rate" was the metric's own chance level. Three consequences run through this
module:

  1. **Every matched rate is printed beside a floor.** Two floors, actually: this run's own
     shuffled-slot control, and the corpus floor measured in task 3 (`CORPUS_FLOORS`).
  2. **`accept` and `add` are ~1.000 on CORPUS-derived blocks BY CONSTRUCTION** —
     `train/skit.py`'s `_slots_for_turn` selects those two slots FROM the very turn they are
     later scored against. On corpus data that is a WIRING CHECK, not evidence of scorer
     power. On MODEL-generated blocks it is a real measurement, because nothing forces the
     model's block to agree with the turn it then writes.
  3. **`stakes` chance is 0.738, not 0.5** — "level" is the majority class (1023/1200 of
     ground-truth blocks). The whole headroom is 26.2 points, and any power calculation or
     "80% accuracy!" claim anchored at a coin flip is measuring against an effect that
     cannot exist.

MULTIPLE COMPARISONS — THE CONSTANTS BELOW ARE THIS EVAL'S OWN
--------------------------------------------------------------
`scripts.eval_improv` carries stage 1's `BONFERRONI_ALPHA = 0.01` / `CRITICAL_T = 2.576` for
its five tests, and its `paired_verdict` CLOSES OVER that module-level `CRITICAL_T`. This
eval runs eleven tests, so importing stage 1's function would silently apply a threshold
almost 0.27 standard errors looser than the one this design chose and could manufacture a
significant result out of a null. `paired_verdict` is therefore reimplemented here (same
zero-scatter ruling, this module's threshold) rather than imported, and
`tests/test_eval_skits.py::test_a_t_between_stage_ones_threshold_and_ours_is_not_significant_here`
is the test that fails if anyone "simplifies" that back into an import.

`swap_verdict` and `SCORER_DIRECTIONS` ARE imported from stage 1 — neither depends on a
threshold constant (swap_verdict's bar is a hardcoded 0.5 fraction-changed, and
SCORER_DIRECTIONS is data), so sharing them keeps the two stages' definitions identical
where identity is what we want.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_improv import (  # noqa: E402
    SCORER_DIRECTIONS,
    generate_batched_from_ids,
    load_hf,
    sft_checkpoint_to_hf,
    swap_verdict,
)
from scripts.score_improv import (  # noqa: E402
    build_association,
    intensity,
    load_closure_lexicon,
    load_harm_lexicon,
    score_pair,
)
from scripts.score_skits import SLOT_NAMES, SlotHits, score_block, slot_accuracy  # noqa: E402
from train.improv import parse_think, render_think, split_sentences  # noqa: E402
from train.skit import MODEL_TURNS  # noqa: E402

#: Four slots + pooled accuracy + four failure-mode scorers + adherence + degeneration.
#: Stated as a division so the count is visible and cannot drift silently.
N_TESTS = 11
BONFERRONI_ALPHA = 0.05 / N_TESTS


def critical_t_for(alpha: float) -> float:
    """Two-sided normal critical value at `alpha`.

    A function, not a magic number, so `CRITICAL_T` below is DERIVED from this module's own
    alpha. A hand-typed constant is how stage 1's pair (0.01 / 2.576) could be copied into a
    module that had chosen a different alpha and still look plausible.
    """
    return st.NormalDist().inv_cdf(1.0 - alpha / 2.0)


#: Two-sided critical value at BONFERRONI_ALPHA (~0.004545) — NOT stage 1's 2.576.
#: The exact normal quantile is 2.83760; the plan pins 2.843, which is 0.005 STRICTER, so
#: keeping the plan's number can only ever refuse a marginal result, never manufacture one.
#: The test asserts both that it matches `critical_t_for(BONFERRONI_ALPHA)` to 0.01 AND
#: that it is never below it — an anti-conservative drift (e.g. someone pasting stage 1's
#: 2.576 back in) fails on the second assertion even if it passed the first.
CRITICAL_T = 2.843

#: A slot score is a HIT RATE, so more is better for all four. Declared explicitly because
#: `paired_verdict` requires a direction and a guessed one inverts the verdict.
SLOT_DIRECTIONS = {name: "higher" for name in SLOT_NAMES}

#: Shuffled-slot floors measured on CORPUS-derived blocks in task 3 (400 skits, each
#: block scored against the NEXT skit's turns). These are the chance levels the model's
#: rates must be read against — see task-3-report.md.
CORPUS_FLOORS = {"accept": 0.051, "add": 0.004, "stakes": 0.738,
                 "handback_anticipation": 0.021}
#: Matched rate of the GROUND-TRUTH blocks on the same 400 skits, for the one slot where it
#: is not ~1.0. handback's derived span reappears in the real next partner turn only 11.9%
#: of the time, so 0.119 is the practical ceiling and 0.021 the floor: the entire detectable
#: band is ~10 points wide, and a null inside it means "within a 10-point band", NOT "no
#: effect".
CORPUS_CEILINGS = {"handback_anticipation": 0.119}
#: `stakes` majority class. Recorded separately from CORPUS_FLOORS so the power calculation
#: cannot silently anchor on 0.5.
STAKES_CHANCE = CORPUS_FLOORS["stakes"]

SKITS_PATH = ROOT / "artifacts" / "skits-200k" / "skits.jsonl"
CKPT_THINK = ROOT / "artifacts" / "skits" / "ckpt-think" / "step_3000.pkl"
CKPT_NOTHINK = ROOT / "artifacts" / "skits" / "ckpt-nothink" / "step_3000.pkl"
MANIFEST_THINK = ROOT / "artifacts" / "skits" / "ckpt-think" / "train_manifest.json"
MANIFEST_NOTHINK = ROOT / "artifacts" / "skits" / "ckpt-nothink" / "train_manifest.json"
DERIVE_MANIFEST_PATH = ROOT / "artifacts" / "skits-200k" / "derive_manifest.json"
#: Architecture header source only — the dense pretraining checkpoint both arms warm-started
#: from. Its WEIGHTS are never used here (see eval_improv.sft_checkpoint_to_hf).
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"

_THINK_RE = re.compile(r"<think>\s*.*?\s*</think>", re.S)
_WORD_RE = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------------------


def paired_verdict(a: Sequence[float], b: Sequence[float], direction: str) -> Dict[str, object]:
    """Paired comparison of two equal-length score series, at THIS module's threshold.

    Deliberately not `scripts.eval_improv.paired_verdict`: that function reads stage 1's
    module-level `CRITICAL_T` (2.576, five tests), which is looser than this eval's 2.843
    (eleven tests). Importing it would apply the wrong threshold with no visible sign.

    The zero-scatter ruling is carried over verbatim from stage 1 because it is correct and
    was hard-won: `sd == 0` with a non-zero mean is every paired point moving by the same
    non-zero amount, which is the STRONGEST signal a paired design can produce, not the
    weakest. It reports `t = inf`. `sd == 0` with a zero mean is two identical series, which
    genuinely carries no information and stays NOT INTERPRETABLE.

    `direction` says which sign of `mean_delta = a - b` (think minus no-think) favours the
    think arm for this scorer: "higher" (hit rates, groundedness, affordance) or "lower"
    (escalation, new_harm, degeneration). There is no default — a guessed direction inverts
    the verdict, and that exact bug shipped in stage 1.
    """
    if direction not in ("lower", "higher"):
        raise ValueError(f"direction must be 'lower' or 'higher', got {direction!r}")
    if len(a) != len(b) or not a:
        raise ValueError(f"paired series must be equal-length and non-empty: {len(a)}, {len(b)}")
    deltas = [x - y for x, y in zip(a, b)]
    mean = st.fmean(deltas)
    sd = st.pstdev(deltas)
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = len(deltas) - pos - neg

    if sd == 0.0:
        se = 0.0
        t = 0.0 if mean == 0.0 else math.inf
    else:
        se = sd / (len(deltas) ** 0.5)
        t = abs(mean / se)

    if t <= CRITICAL_T:
        verdict = "NOT INTERPRETABLE"
    else:
        think_favoured = (mean < 0) if direction == "lower" else (mean > 0)
        verdict = "think better" if think_favoured else "no-think better"
    return {"mean_delta": round(mean, 4), "sd": round(sd, 4),
            "se": round(se, 4) if math.isfinite(se) else se,
            "t": round(t, 3) if math.isfinite(t) else t,
            "direction": direction, "signs_pos": pos, "signs_neg": neg, "signs_zero": zero,
            "n": len(deltas), "critical_t": CRITICAL_T, "alpha": BONFERRONI_ALPHA,
            "verdict": verdict}


def required_n(baseline: float, delta: float, *, alpha: float, power: float = 0.8) -> int:
    """Samples PER ARM needed to detect `delta` on a rate sitting at `baseline`.

    Normal approximation for two proportions:
        n = (z_{alpha/2} + z_{power})^2 * (p0*q0 + p1*q1) / delta^2

    Two things this exists to stop:

      * **Reusing a number computed at another alpha.** A reviewer sized this experiment at
        ~370 turns for a 10-point `stakes` effect using alpha = 0.05/4. This eval runs
        eleven tests, so its alpha is stricter and its requirement is larger (~447). The
        function reproduces the reviewer's 367 at alpha=0.0125, which is what makes the 447
        at alpha=0.004545 trustworthy rather than merely different.
      * **Anchoring `stakes` at 0.5.** Chance for `stakes` is the majority class, 0.738.
        Baseline variance at 0.5 is the maximum possible, so a 0.5 anchor overstates the
        sample requirement here — and, far worse, the same 0.5 anchor elsewhere would turn a
        6-point-BELOW-chance result into a "0.80 accuracy" success story.
    """
    if not 0.0 < baseline < 1.0:
        raise ValueError(f"baseline must be a rate strictly inside (0,1): {baseline}")
    if delta <= 0.0:
        raise ValueError(f"delta must be a positive effect size: {delta}")
    p1 = min(baseline + delta, 1.0 - 1e-9)
    z_alpha = critical_t_for(alpha)
    z_power = st.NormalDist().inv_cdf(power)
    var = baseline * (1 - baseline) + p1 * (1 - p1)
    return math.ceil((z_alpha + z_power) ** 2 * var / (delta ** 2))


def paired_defined(a: Sequence[Optional[float]], b: Sequence[Optional[float]]
                   ) -> Tuple[List[float], List[float]]:
    """Keep only positions where BOTH series are defined, preserving order.

    `handback_anticipation` is None on the final model turn, and a block that failed to
    parse leaves every slot undefined for that observation. Dropping only the undefined SIDE
    would leave the two arms holding equal-length lists made of DIFFERENT observations —
    arm A's turn 1 paired against arm B's turn 2 — which destroys the pairing the whole
    t-test rests on while looking perfectly healthy.
    """
    xs: List[float] = []
    ys: List[float] = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def assert_control_is_not_the_treatment(treatment: Sequence[SlotHits],
                                        control: Sequence[SlotHits], *, name: str) -> None:
    """Refuse to report a control that scored the SAME text as the treatment.

    This guard exists because a mutation test caught its absence: pointing the context-only
    control at `turns_think` instead of `turns_nothink` — a one-word edit — produces a
    control identical to the treatment, a gap of exactly zero on every slot, and a
    confident-looking "no effect" conclusion. No unit test could see it, because the wiring
    lives in the driver rather than in a function. So the check goes where the failure is:
    if every one of hundreds of paired observations agrees exactly, the two conditions are
    the same condition.
    """
    if not treatment or len(treatment) != len(control):
        raise RuntimeError(f"{name}: treatment/control length mismatch "
                           f"({len(treatment)} vs {len(control)}) — they are not paired")
    if all(t == c for t, c in zip(treatment, control)):
        raise RuntimeError(
            f"{name}: the control scored IDENTICALLY to the treatment on all "
            f"{len(treatment)} observations. That is what happens when the control is wired "
            f"to the treatment's own turns, not what happens when a real control finds no "
            f"effect. Refusing to report it.")


def shuffled_gap(real: Dict[str, Optional[float]],
                 shuffled: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Per-slot accuracy advantage of the model's OWN block over a foreign one.

    Zero means the slot values are not being used — the model produces scaffolding, not
    plans. This is the control that catches the template-inflation risk: three blocks per
    sequence triples the scaffolding tokens, and stage 1's healthy-looking loss turned out
    to be carried almost entirely by literal template text.
    """
    out: Dict[str, Optional[float]] = {}
    for slot, r in real.items():
        s = shuffled.get(slot)
        out[slot] = None if (r is None or s is None) else round(r - s, 4)
    return out


def slot_table(matched: Dict[str, Optional[float]], shuffled: Dict[str, Optional[float]],
               context_only: Optional[Dict[str, Optional[float]]] = None
               ) -> Dict[str, Dict[str, Optional[float]]]:
    """One row per slot: the rate, its floors, and the band it can move in.

    A rate without its floor is the stage-1 mistake. `stakes` at 0.80 is six points BELOW
    its 0.738 chance line's headroom midpoint, and `handback_anticipation` at 0.05 sits
    inside a band only 0.098 wide — neither reads correctly without the numbers beside it.

    TWO floors, because they rule out different things.

      `shuffled_floor` scores the block against ANOTHER skit's turn. It breaks the
      block-to-turn link, but it also breaks the shared-context link: the two skits are
      about different things, so some of the gap it shows is merely topical.

      `context_only_floor` scores the SAME think-block against the NO-THINK arm's turn for
      the SAME skit at the SAME position. Same scene, same partner turns, same everything —
      except the turn was written by a model that never saw a block. That is the control
      that separates "the model followed its plan" from "the plan and the turn both echo
      the context the model was already reading". Without it, a scorer driven purely by
      context overlap would look like plan-following, which is precisely the shape of stage
      1's mistake in a new costume.
    """
    gaps = shuffled_gap(matched, shuffled)
    ctx_gaps = shuffled_gap(matched, context_only) if context_only else {}
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    for slot in SLOT_NAMES:
        row: Dict[str, Optional[float]] = {
            "matched": matched.get(slot),
            "shuffled_floor": shuffled.get(slot),
            "gap_over_shuffled": gaps.get(slot),
            "corpus_shuffled_floor": CORPUS_FLOORS[slot],
        }
        if context_only:
            row["context_only_floor"] = context_only.get(slot)
            row["gap_over_context_only"] = ctx_gaps.get(slot)
        if slot == "stakes":
            row["headroom_above_chance"] = round(1.0 - STAKES_CHANCE, 4)
            row["note"] = ("chance is the majority class 'level' (0.738), not 0.5 — the "
                           "entire detectable band is 26.2 points")
        if slot in CORPUS_CEILINGS:
            row["corpus_ceiling"] = CORPUS_CEILINGS[slot]
            row["max_detectable_band"] = round(CORPUS_CEILINGS[slot] - CORPUS_FLOORS[slot], 4)
            row["note"] = ("ground-truth blocks themselves only hit 0.119 here, floor "
                           "0.021 — a null inside this ~0.10 band means 'within a "
                           "10-point band', NOT 'no effect'")
            # The context-only control is DEGENERATE for this slot, and saying so matters
            # more than the number does: `score_skits.score_block` scores
            # handback_anticipation against the NEXT PARTNER TURN, which is real corpus text
            # identical in both conditions, and never looks at the model's turn at all. So
            # this floor is forced to equal `matched` exactly and its gap is forced to
            # 0.0000. Read as "thinking adds nothing here" it would be flatly wrong; the
            # only informative floor for this slot is the shuffled one.
            row["context_only_floor_is_degenerate_for_this_slot"] = (
                "score_block() reads only the block's `handback` value and the NEXT PARTNER "
                "turn, never the model's own turn. The context-only control holds the "
                "partner turn fixed, so this floor is identical to `matched` BY "
                "CONSTRUCTION and `gap_over_context_only` is necessarily 0.0000. It is not "
                "evidence of no effect. Use `gap_over_shuffled` for this slot.")
        if slot in ("accept", "add"):
            row["note"] = ("on CORPUS-derived blocks this slot is ~1.000 BY CONSTRUCTION "
                           "(train/skit.py:88-93 selects it from the turn it is scored "
                           "against) — that is a wiring check. On MODEL-generated blocks, "
                           "as here, it is a real measurement: nothing forces the model's "
                           "block to agree with the turn it then writes")
        rows[slot] = row
    return rows


def manifests_block(think: Dict[str, Any], nothink: Dict[str, Any]) -> Dict[str, Any]:
    """Both training manifests, with the note that stops the obvious wrong subtraction.

    The manifests put `paired_with` immediately beside `loss_end`, which reads as an
    invitation to compare 0.504 against 1.227. They are not comparable, and the warning has
    to travel WITH them: a caveat in a section further down is a caveat the reader who
    scrolled straight to the losses never sees.
    """
    return {
        "think": think,
        "nothink": nothink,
        "loss_comparability": (
            "DO NOT compare the two arms' `loss_end` (think 0.504 vs nothink 1.227). They "
            "supervise DIFFERENT TOKEN SETS: 168.1 vs 43.4 supervised label positions per "
            "example (236.8 vs 111.8 tokens), and 74.2% of the think arm's supervised "
            "positions are the fixed five-line template, which is far more predictable than "
            "free prose. The gap is an easier token mixture, not an arm effect. `paired_with` "
            "in these manifests means the two runs share data order, seed and schedule — it "
            "does NOT mean their losses are on a common scale. The only valid arm comparison "
            "is the scorers applied to MODEL TURNS, which is what the rest of this file is."),
    }


def build_limitations(*, eval_set: str, n_skits: int) -> Dict[str, str]:
    """What this measurement can and cannot support, carried IN the artifact.

    Burying these in a report nobody re-reads is how stage 1 published a metric without its
    floor. They are emitted as part of the JSON so the caveats travel with the numbers.
    """
    return {
        "accept_and_add_are_by_construction_on_corpus_data": (
            "On CORPUS-derived blocks `accept` and `add` score ~1.000 BY CONSTRUCTION: "
            "train/skit.py:88-93 selects those slot values out of the very turn they are "
            "later scored against. On corpus data that number is a WIRING CHECK, not "
            "evidence of scorer power. Every matched rate in this file is a MODEL-generated "
            "block scored against the turn the MODEL then wrote, where the agreement is not "
            "forced — and every one of them is printed beside both this run's shuffled "
            "control and the corpus floors (accept 0.051, add 0.004, stakes 0.738, "
            "handback 0.021)."
        ),
        "handback_anticipation_band_is_only_ten_points": (
            "Ground-truth blocks hit `handback_anticipation` only 0.119 of the time on real "
            "corpus data (shuffled floor 0.021), so the maximum detectable band is ~0.098. "
            "A null on this slot means 'within a 10-point band', NOT 'no effect'. The slot "
            "also measures ANTICIPATION, never influence: the corpus partner turn was "
            "written before the model existed and cannot have heard it."
        ),
        "stakes_chance_is_the_majority_class": (
            "`stakes` chance is 0.738 ('level' is the majority class, 1023/1200 of "
            "ground-truth blocks), not 0.5. Headroom is 26.2 points. Any reading of a "
            "stakes rate against a coin flip overstates the effect badly."
        ),
        "the_two_arms_training_losses_are_not_comparable": (
            "Never compare the arms' training loss (think 0.504 vs nothink 1.227). They "
            "supervise different token sets — median example 224 vs 96 tokens, and ~74% of "
            "the think arm's supervised positions are the fixed five-line template. That "
            "gap is an easier token mixture, not an arm effect. The only valid comparison "
            "is the one in this file: the slot and failure-mode scorers applied to MODEL "
            "TURNS."
        ),
        "step_3000_is_an_inherited_budget_not_convergence": (
            "Held-out loss was still falling at step 3000 in the think arm (1.009 at step "
            "250 -> 0.639 at step 3000, monotone, no upturn). 3000 steps was an inherited "
            "budget, not a convergence point, so every number here describes an "
            "under-trained pair of arms and a longer run could move them."
        ),
        "nothink_val_curve_is_nearly_flat": (
            "The no-think arm's validation loss moved 1.8% over 3000 steps (1.580 -> 1.552) "
            "while its train loss fell 32%. Read its scores here as those of an arm that "
            "barely generalised beyond its warm start."
        ),
        "single_training_run_per_arm": (
            "One SFT run per arm, one seed. The paired tests capture ITEM variance over the "
            f"{n_skits} held-out skits of {eval_set} only. They say nothing about "
            "run-to-run variance; a retrain could land elsewhere."
        ),
        "generation_is_cpu_only": (
            "Generation runs on the CPU through transformers (torch is a +cpu build on this "
            "host and ttml is a training-only path in this repo), so this eval opens no "
            "/dev/tenstorrent device and holds no chip lease. The arms it evaluates WERE "
            "trained on Tenstorrent hardware; nothing about the measurement depends on "
            "where the forward passes ran."
        ),
    }


# ---------------------------------------------------------------------------------------
# Generation plumbing.
# ---------------------------------------------------------------------------------------


def split_generation(text: str, *, with_think: bool) -> Tuple[Optional[Any], str, str]:
    """Split one generation into `(parsed_slots_or_None, raw_block_text, turn_text)`.

    The turn is the FIRST SENTENCE AFTER the block, and the block text is excluded from it.
    That exclusion is the whole point: a think-block's `add` slot literally contains the
    word it is predicting, so scoring a "turn" that still contained the block would make
    every slot hit trivially and turn the headline measurement into a tautology. Skit turns
    are single sentences by construction (train/skit.py takes sentences[2:7]), so one
    sentence is also the right unit.

    `raw_block_text` carries a TRAILING NEWLINE because `train.improv.render_think` ends the
    think segment with one, and the turn segment follows it immediately with no separator.
    The caller re-encodes these two pieces as separate segments to extend the context (that
    is how training built every example); dropping the newline would put `</think>` and the
    turn's first token adjacent at a seam training never saw once.
    """
    body = text
    block_text = ""
    if with_think:
        m = _THINK_RE.search(text)
        if m:
            block_text = m.group(0) + "\n"
            body = text[m.end():]
    slots = parse_think(text) if with_think else None
    sents = split_sentences(body)
    return slots, block_text, (sents[0] if sents else "")


def adherence_by_turn(generations: Sequence[Sequence[str]], *, n_turns: int) -> List[float]:
    """Fraction of generations whose block parses, PER TURN POSITION.

    Pooled adherence hides the failure five turns specifically invites: a well-formed block
    for turn 1 and a collapse by turn 3. Stage 1 saw the model open a second think-block
    mid-sentence and degenerate into `cold cold cold`.
    """
    out: List[float] = []
    for t in range(n_turns):
        col = [g[t] for g in generations if len(g) > t]
        out.append((sum(1 for c in col if parse_think(c) is not None) / len(col))
                   if col else 0.0)
    return out


def four_gram_repeat_rate(turns: Sequence[str]) -> float:
    """Fraction of 4-grams that are repeats, ACROSS the model's turns concatenated.

    Across, not within. A model that writes the same turn three times is degenerate even
    though each turn alone repeats nothing — scoring turns independently and averaging
    would report 0.0 on exactly the failure this metric exists to catch.
    """
    words = [w.lower() for t in turns for w in _WORD_RE.findall(t)]
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    return round(1.0 - len(set(grams)) / len(grams), 4)


def check_tokenization_parity(tok, rec: dict) -> Dict[str, Any]:
    """Prove this eval's prompt ids ARE the training example's ids, and RAISE if not.

    `scripts/derive_skits.py:build_skit_example` encodes each of the nine segments with its
    own `tok.encode` call, and this tokenizer prepends a space per call. Training therefore
    saw `['.', 'Ġ<', 'think', '>']` at the seam. Tokenizing the assembled prompt STRING in
    one call gives `['.', '<', 'think', '>']` instead — same length, different boundary
    token, and the model is then asked to open a think-block after a token it has never seen
    in that position. It would degrade both arms identically (so pairing survives) and would
    read exactly like "the model does not follow its plan", which is the finding this whole
    eval exists to determine.

    So this is checked against a REAL training example built by the real builder, on a
    held-out skit, before a single token is generated — not asserted in a comment.
    """
    from train.improv import Slots
    from train.skit import Skit

    from scripts.derive_skits import build_skit_example

    skit = Skit(story_id=rec["story_id"], prefix=rec["prefix"], turns=tuple(rec["turns"]),
                blocks=tuple(Slots(**b) for b in rec["blocks"]))
    train_ids = build_skit_example(skit, tok, with_think=True,
                                   pad_token_id=tok.pad_token_id or 0)["input_ids"]
    prompt = tok.encode(rec["prefix"])
    block = render_think_from(rec["blocks"][0])
    seam = prompt + tok.encode(block, add_special_tokens=False)
    with_turn = seam + tok.encode(rec["turns"][0], add_special_tokens=False)
    whole_string = tok.encode(rec["prefix"] + block)
    n = len(prompt)
    out = {
        "checked_against": "scripts/derive_skits.py:build_skit_example on a held-out skit",
        "story_id": rec["story_id"],
        "prompt_is_prefix_of_training_ids": train_ids[:n] == prompt,
        "segment_wise_matches_training_at_the_think_seam": train_ids[:len(seam)] == seam,
        "segment_wise_matches_training_through_turn_0": train_ids[:len(with_turn)] == with_turn,
        "whole_string_tokenization_matches_training": (
            train_ids[:len(whole_string)] == whole_string),
        "seam_tokens_training": tok.convert_ids_to_tokens(train_ids[n - 1:n + 3]),
        "seam_tokens_segment_wise": tok.convert_ids_to_tokens(seam[n - 1:n + 3]),
        "seam_tokens_whole_string": tok.convert_ids_to_tokens(whole_string[n - 1:n + 3]),
        "note": ("This eval builds every prompt segment-wise (tok.encode per segment, ids "
                 "concatenated, add_special_tokens only on the first) — the construction "
                 "training used. Whole-string tokenization is shown here only to record "
                 "that it does NOT match."),
    }
    if not (out["prompt_is_prefix_of_training_ids"]
            and out["segment_wise_matches_training_at_the_think_seam"]
            and out["segment_wise_matches_training_through_turn_0"]):
        raise RuntimeError(
            "eval prompts do not reproduce training's tokenization: "
            f"{out['seam_tokens_segment_wise']} vs training {out['seam_tokens_training']}. "
            "Every number this run would produce would be measured at a seam the arms never "
            "saw. Refusing to generate.")
    return out


def _clip_context(ids: List[int], *, max_new: int, window: int = 512) -> List[int]:
    """Keep the RIGHTMOST tokens so context + generation fits the training window."""
    budget = window - max_new
    return ids if len(ids) <= budget else ids[-budget:]


def _score_one(block: Dict[str, str], *, turn: str, prev_turn: str,
               next_partner: Optional[str], harm: frozenset) -> SlotHits:
    return score_block(block, turn=turn, prev_turn=prev_turn, next_partner=next_partner,
                       harm=harm)


def roll_forward(tok, model, skits: List[dict], *, with_think: bool, max_new_tokens: int,
                 batch_size: int) -> Tuple[List[List[str]], List[List[Optional[Any]]],
                                           List[List[str]], int]:
    """Generate all three model turns per skit, feeding back the model's OWN turns.

    Context is assembled exactly the way training assembled it (see
    `train.skit.skit_segments` and `scripts/derive_skits.py:build_skit_example`): each
    segment is tokenized SEPARATELY and the id lists are concatenated, `add_special_tokens`
    only on the first. Tokenizing a joined string instead re-merges the seam through BPE and
    can put a token at the `<think>` boundary that training never saw once — stage 1 found
    exactly that (id 31 instead of id 19691) and it is an out-of-distribution seam at
    precisely the point being measured.

    Real partner turns are fed back verbatim; the model's own turns are fed back as it wrote
    them. Returns `(raw_generations, parsed_slots, turn_texts, n_clipped)`.
    """
    n = len(skits)
    contexts: List[List[int]] = [tok.encode(s["prefix"]) for s in skits]
    raw: List[List[str]] = [[] for _ in range(n)]
    slots: List[List[Optional[Any]]] = [[] for _ in range(n)]
    turns: List[List[str]] = [[] for _ in range(n)]
    n_clipped = 0

    for i, t_idx in enumerate(MODEL_TURNS):
        if t_idx > 0:                      # the real partner turn that precedes this one
            for j, s in enumerate(skits):
                contexts[j] = contexts[j] + tok.encode(s["turns"][t_idx - 1],
                                                       add_special_tokens=False)
        prompts = []
        for ids in contexts:
            clipped = _clip_context(ids, max_new=max_new_tokens)
            if len(clipped) != len(ids):
                n_clipped += 1
            prompts.append(clipped)
        texts = generate_batched_from_ids(tok, model, prompts,
                                          max_new_tokens=max_new_tokens, do_sample=False,
                                          batch_size=batch_size)
        for j, text in enumerate(texts):
            parsed, block_text, turn_text = split_generation(text, with_think=with_think)
            raw[j].append(text)
            slots[j].append(parsed)
            turns[j].append(turn_text)
            if block_text:
                contexts[j] = contexts[j] + tok.encode(block_text, add_special_tokens=False)
            contexts[j] = contexts[j] + tok.encode(turn_text, add_special_tokens=False)
        print(f"    turn {t_idx}: generated {len(texts)} (parsed blocks: "
              f"{sum(1 for s in slots if s[-1] is not None)}/{n})")
    return raw, slots, turns, n_clipped


# ---------------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------------


def _context_for(skit: dict, t_idx: int) -> Tuple[str, Optional[str]]:
    """`(previous turn, following partner turn or None)` for one model turn."""
    prev = skit["turns"][t_idx - 1] if t_idx > 0 else skit["prefix"]
    nxt = skit["turns"][t_idx + 1] if t_idx + 1 < len(skit["turns"]) else None
    return prev, nxt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "skits-stage2.json")
    ap.add_argument("--skits", type=Path, default=SKITS_PATH)
    ap.add_argument("--n-heldout", type=int, default=256,
                    help="tail of the skits file, matching the arms' val split")
    ap.add_argument("--n-swap", type=int, default=50)
    ap.add_argument("--max-eval-skits", type=int, default=0,
                    help="0 = use every held-out skit. A positive value TRUNCATES the "
                         "held-out set after the disjointness check — for pipeline smokes "
                         "only, and it is recorded in the artifact so a truncated run can "
                         "never be mistaken for the real one.")
    ap.add_argument("--think-max-new-tokens", type=int, default=128)
    ap.add_argument("--nothink-max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--assoc-skits", type=int, default=6000,
                    help="training skits used to build the NPMI table for groundedness")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "artifacts" / "skits" / "hf-eval")
    args = ap.parse_args()

    harm = load_harm_lexicon()
    closure = load_closure_lexicon()

    # ---------------------------------------------------------------------------------
    # [1/8] Held-out selection and the disjointness check that is CHECKED, not trusted.
    # ---------------------------------------------------------------------------------
    print("[1/8] loading skits and splitting held-out tail ...")
    manifest_think = json.loads(MANIFEST_THINK.read_text())
    manifest_nothink = json.loads(MANIFEST_NOTHINK.read_text())
    n_train = int(manifest_think["n_examples"])
    if int(manifest_nothink["n_examples"]) != n_train:
        raise RuntimeError("the two arms trained on different-sized training sets; they are "
                           "not paired and must not be compared")
    lines = args.skits.read_text().splitlines()
    all_skits = [json.loads(l) for l in lines]
    train_skits, val_skits = all_skits[:n_train], all_skits[n_train:]
    if len(val_skits) != args.n_heldout:
        raise RuntimeError(f"expected {args.n_heldout} held-out skits, found {len(val_skits)}")
    train_ids = {s["story_id"] for s in train_skits}
    held_ids = {s["story_id"] for s in val_skits}
    overlap = sorted(train_ids & held_ids)
    if overlap:
        raise RuntimeError(f"held-out skits overlap the training set on story_id(s): "
                           f"{overlap[:20]} — the eval would be measuring memorisation")
    print(f"  {len(train_skits)} train / {len(val_skits)} held-out, story_id overlap 0")
    n_heldout_available = len(val_skits)
    if args.max_eval_skits:
        val_skits = val_skits[:args.max_eval_skits]
        print(f"  TRUNCATED to {len(val_skits)} skits (--max-eval-skits) — smoke run, not "
              f"the real measurement")

    # ---------------------------------------------------------------------------------
    # [2/8] Power, recomputed at THIS eval's alpha before anything is generated.
    # ---------------------------------------------------------------------------------
    n_model_turns = len(val_skits) * len(MODEL_TURNS)
    n_handback_obs = len(val_skits) * (len(MODEL_TURNS) - 1)
    need_stakes = required_n(STAKES_CHANCE, 0.10, alpha=BONFERRONI_ALPHA)
    power = {
        "alpha": BONFERRONI_ALPHA, "critical_t": CRITICAL_T, "n_tests": N_TESTS,
        "target_power": 0.8,
        "stakes_baseline_used": STAKES_CHANCE,
        "stakes_effect_sought": 0.10,
        "required_n_per_arm_at_this_alpha": need_stakes,
        "required_n_per_arm_at_alpha_0_0125_for_reference": required_n(
            STAKES_CHANCE, 0.10, alpha=0.05 / 4),
        "available_model_turns": n_model_turns,
        "available_handback_observations": n_handback_obs,
        "powered_for_stakes": n_model_turns >= need_stakes,
        "powered_for_handback": n_handback_obs >= need_stakes,
        "note": (
            "Recomputed here rather than reusing the ~370 a reviewer derived at alpha "
            "0.05/4: eleven tests means a stricter alpha and therefore MORE samples. The "
            "same function reproduces 367 at alpha 0.0125, which is what makes the number "
            "at 0.004545 credible. Baseline is the `stakes` majority class 0.738, not 0.5. "
            "handback_anticipation has 2 observations per skit (the final model turn has no "
            "following partner turn), and its detectable band is only ~0.10 wide, so it is "
            "under-powered for a 10-point effect it could not physically show."),
    }
    print(f"  power: need {need_stakes}/arm for a 10-point stakes effect at "
          f"alpha={BONFERRONI_ALPHA:.6f}; have {n_model_turns} model turns "
          f"({n_handback_obs} handback observations)")

    # ---------------------------------------------------------------------------------
    # [3/8] Checkpoints -> HF, on the CPU. No ttml, no device, no lease needed.
    # ---------------------------------------------------------------------------------
    print("[3/8] converting SFT checkpoints -> HF model directories (CPU only) ...")
    hf_think_dir, hf_nothink_dir = args.work_dir / "think", args.work_dir / "nothink"
    cfg_think = sft_checkpoint_to_hf(CKPT_THINK, warm_start_ckpt=WARM_START_CKPT,
                                     tokenizer_dir=TOKENIZER_DIR, out_dir=hf_think_dir)
    sft_checkpoint_to_hf(CKPT_NOTHINK, warm_start_ckpt=WARM_START_CKPT,
                         tokenizer_dir=TOKENIZER_DIR, out_dir=hf_nothink_dir)
    tok_think, model_think = load_hf(hf_think_dir)
    tok_nothink, model_nothink = load_hf(hf_nothink_dir)
    tokenization_parity = check_tokenization_parity(tok_think, val_skits[0])
    print(f"  tokenization parity vs training ids: "
          f"segment-wise={tokenization_parity['segment_wise_matches_training_at_the_think_seam']} "
          f"whole-string="
          f"{tokenization_parity['whole_string_tokenization_matches_training']} "
          f"(seam {tokenization_parity['seam_tokens_training']})")

    # ---------------------------------------------------------------------------------
    # [4/8] THE SWAP TEST — first, and able to fail the stage on its own.
    #
    # Both conditions FORCE a ground-truth block into the think arm's context at model turn
    # 0 (its own, or the next skit's) and generate greedily. Any divergence is attributable
    # to the block's CONTENT, not to sampling. This measures whether block content steers
    # generation once a block is present; whether the model chooses to write one is the
    # separate `adherence` measurement below.
    # ---------------------------------------------------------------------------------
    print("[4/8] swap test (forced own vs foreign ground-truth block) ...")
    n_swap = min(args.n_swap, len(val_skits))
    swap_subset = val_skits[:n_swap]
    own_ids, swapped_ids = [], []
    for i, s in enumerate(swap_subset):
        base = tok_think.encode(s["prefix"])
        mine = render_think_from(s["blocks"][0])
        theirs = render_think_from(swap_subset[(i + 1) % n_swap]["blocks"][0])
        own_ids.append(base + tok_think.encode(mine, add_special_tokens=False))
        swapped_ids.append(base + tok_think.encode(theirs, add_special_tokens=False))
    own_cont = generate_batched_from_ids(tok_think, model_think, own_ids,
                                         max_new_tokens=args.nothink_max_new_tokens,
                                         do_sample=False, batch_size=args.batch_size)
    swap_cont = generate_batched_from_ids(tok_think, model_think, swapped_ids,
                                          max_new_tokens=args.nothink_max_new_tokens,
                                          do_sample=False, batch_size=args.batch_size)
    divergence: List[Optional[int]] = []
    for own_text, swapped_text in zip(own_cont, swap_cont):
        a = tok_think.encode(own_text, add_special_tokens=False)
        b = tok_think.encode(swapped_text, add_special_tokens=False)
        pos = None
        for k, (x, y) in enumerate(zip(a, b)):
            if x != y:
                pos = k
                break
        else:
            if len(a) != len(b):
                pos = min(len(a), len(b))
        divergence.append(pos)
    swap = swap_verdict(divergence, n_swap)
    swap["forced_think_note"] = (
        "Both conditions FORCE a ground-truth extractive block into the context. This "
        "measures whether block CONTENT changes what follows once a block is present — not "
        "whether the model organically emits one (see `adherence`).")
    print(f"  {swap['n_changed']}/{swap['n']} continuations changed "
          f"({swap['fraction_changed']:.1%}), load_bearing={swap['thinking_is_load_bearing']}")

    # ---------------------------------------------------------------------------------
    # [5/8] Roll-forward generation, both arms, three model turns each.
    # ---------------------------------------------------------------------------------
    print("[5/8] generating three model turns per held-out skit, think arm ...")
    raw_think, slots_think, turns_think, clip_think = roll_forward(
        tok_think, model_think, val_skits, with_think=True,
        max_new_tokens=args.think_max_new_tokens, batch_size=args.batch_size)
    print("      ... no-think arm ...")
    raw_nothink, slots_nothink, turns_nothink, clip_nothink = roll_forward(
        tok_nothink, model_nothink, val_skits, with_think=False,
        max_new_tokens=args.nothink_max_new_tokens, batch_size=args.batch_size)

    # ---------------------------------------------------------------------------------
    # [6/8] Scoring.
    #
    # THREE distinct slot measurements, kept apart on purpose:
    #
    #   matched   the model's OWN generated block scored against the turn the model then
    #             wrote. This is stage 2's headline: is the block a prediction it obeys?
    #   shuffled  the same generated block scored against ANOTHER skit's model turn (the
    #             task-3 control, same construction as the corpus floors). Its distance
    #             from `matched` is the entire signal.
    #   reference the skit's GROUND-TRUTH block scored against each arm's generated turn.
    #             This is the only slot measurement that is fair ACROSS arms: neither arm
    #             ever sees that block, so it asks "did this arm's turn do what the scene's
    #             real plan called for" without handing the think arm the answer words.
    # ---------------------------------------------------------------------------------
    print("[6/8] scoring slots, failure modes, adherence, degeneration ...")
    matched_hits: List[SlotHits] = []
    shuffled_hits: List[SlotHits] = []
    context_only_hits: List[SlotHits] = []
    ref_hits_think: List[Optional[SlotHits]] = []
    ref_hits_nothink: List[Optional[SlotHits]] = []
    n_scorable = 0
    for j, skit in enumerate(val_skits):
        other = val_skits[(j + 1) % len(val_skits)]
        for i, t_idx in enumerate(MODEL_TURNS):
            prev, nxt = _context_for(skit, t_idx)
            oprev, onxt = _context_for(other, t_idx)
            block = slots_think[j][i]
            if block is not None and turns_think[j][i]:
                n_scorable += 1
                matched_hits.append(_score_one(block.as_dict(), turn=turns_think[j][i],
                                               prev_turn=prev, next_partner=nxt, harm=harm))
                # The control: this same block against the NEXT skit's model turn at the
                # same position, with that skit's own surrounding turns — the exact
                # construction task 3 used to measure the corpus floors.
                foreign_turn = turns_think[(j + 1) % len(val_skits)][i]
                shuffled_hits.append(_score_one(block.as_dict(), turn=foreign_turn,
                                                prev_turn=oprev, next_partner=onxt, harm=harm))
                # The tighter control: the SAME block against the NO-THINK arm's turn for
                # THIS skit at THIS position. Same scene, same context, no block behind the
                # turn — so whatever survives here is context overlap, not plan-following.
                context_only_hits.append(_score_one(block.as_dict(),
                                                    turn=turns_nothink[j][i],
                                                    prev_turn=prev, next_partner=nxt,
                                                    harm=harm))
            ref = skit["blocks"][i]
            ref_hits_think.append(_score_one(ref, turn=turns_think[j][i], prev_turn=prev,
                                             next_partner=nxt, harm=harm))
            ref_hits_nothink.append(_score_one(ref, turn=turns_nothink[j][i], prev_turn=prev,
                                               next_partner=nxt, harm=harm))

    if matched_hits:
        assert_control_is_not_the_treatment(matched_hits, shuffled_hits,
                                            name="shuffled-slot control")
        assert_control_is_not_the_treatment(matched_hits, context_only_hits,
                                            name="context-only control")
    matched_acc = slot_accuracy(matched_hits) if matched_hits else {s: None for s in SLOT_NAMES}
    shuffled_acc = (slot_accuracy(shuffled_hits) if shuffled_hits
                    else {s: None for s in SLOT_NAMES})
    context_only_acc = (slot_accuracy(context_only_hits) if context_only_hits
                        else {s: None for s in SLOT_NAMES})
    table = slot_table(matched_acc, shuffled_acc, context_only_acc)

    # Within-arm paired contrasts: own block vs foreign block (topical + plan control), and
    # own block vs the no-think arm's turn on the SAME skit (context-only control).
    own_vs_foreign = {}
    own_vs_context_only = {}
    for slot in SLOT_NAMES:
        xs, ys = paired_defined([getattr(h, slot) for h in matched_hits],
                                [getattr(h, slot) for h in shuffled_hits])
        own_vs_foreign[slot] = (paired_verdict(xs, ys, SLOT_DIRECTIONS[slot]) if xs
                                else {"verdict": "NO DATA", "n": 0})
        xs, ys = paired_defined([getattr(h, slot) for h in matched_hits],
                                [getattr(h, slot) for h in context_only_hits])
        own_vs_context_only[slot] = (paired_verdict(xs, ys, SLOT_DIRECTIONS[slot]) if xs
                                     else {"verdict": "NO DATA", "n": 0})

    # Cross-arm paired contrast on the ground-truth reference block (tests 1-4).
    slot_arm_tests = {}
    for slot in SLOT_NAMES:
        xs, ys = paired_defined([getattr(h, slot) for h in ref_hits_think],
                                [getattr(h, slot) for h in ref_hits_nothink])
        slot_arm_tests[slot] = (paired_verdict(xs, ys, SLOT_DIRECTIONS[slot]) if xs
                                else {"verdict": "NO DATA", "n": 0})

    def _pooled(h: SlotHits) -> Optional[float]:
        vals = [v for v in (h.accept, h.add, h.stakes, h.handback_anticipation)
                if v is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    xs, ys = paired_defined([_pooled(h) for h in ref_hits_think],
                            [_pooled(h) for h in ref_hits_nothink])
    pooled_test = paired_verdict(xs, ys, "higher")                       # test 5

    # Failure-mode scorers (tests 6-9), on cross-turn intervals: each model turn is scored
    # against the text immediately preceding it (the prefix, or the real partner turn),
    # which is the same interval `stakes` uses and is identical for both arms.
    assoc_pairs: List[Tuple[str, str]] = []
    for s in train_skits[:args.assoc_skits]:
        for t_idx in MODEL_TURNS:
            prev, _ = _context_for(s, t_idx)
            assoc_pairs.append((prev, s["turns"][t_idx]))
    assoc = build_association(assoc_pairs)

    fail_series: Dict[str, Tuple[List[float], List[float]]] = {
        k: ([], []) for k in SCORER_DIRECTIONS}
    for j, skit in enumerate(val_skits):
        for i, t_idx in enumerate(MODEL_TURNS):
            prev, _ = _context_for(skit, t_idx)
            st_ = score_pair(prev, turns_think[j][i], harm=harm, assoc=assoc, closure=closure)
            sn_ = score_pair(prev, turns_nothink[j][i], harm=harm, assoc=assoc, closure=closure)
            fail_series["escalation"][0].append(st_.escalation)
            fail_series["escalation"][1].append(sn_.escalation)
            fail_series["new_harm"][0].append(float(st_.new_harm))
            fail_series["new_harm"][1].append(float(sn_.new_harm))
            fail_series["groundedness"][0].append(st_.groundedness)
            fail_series["groundedness"][1].append(sn_.groundedness)
            fail_series["affordance"][0].append(float(st_.affordance))
            fail_series["affordance"][1].append(float(sn_.affordance))
    failure_tests = {name: paired_verdict(a, b, SCORER_DIRECTIONS[name])
                     for name, (a, b) in fail_series.items()}

    # Adherence (test 10) and degeneration (test 11).
    adh_think = adherence_by_turn(raw_think, n_turns=len(MODEL_TURNS))
    adh_nothink = adherence_by_turn(raw_nothink, n_turns=len(MODEL_TURNS))
    adh_series_t = [1.0 if s is not None else 0.0 for row in slots_think for s in row]
    adh_series_n = [1.0 if parse_think(g) is not None else 0.0
                    for row in raw_nothink for g in row]
    adherence_test = paired_verdict(adh_series_t, adh_series_n, "higher")

    deg_think = [four_gram_repeat_rate(t) for t in turns_think]
    deg_nothink = [four_gram_repeat_rate(t) for t in turns_nothink]
    degeneration_test = paired_verdict(deg_think, deg_nothink, "lower")

    # ---------------------------------------------------------------------------------
    # [7/8] Verdict.
    # ---------------------------------------------------------------------------------
    print("[7/8] assembling verdict ...")
    gaps = shuffled_gap(matched_acc, shuffled_acc)
    materially_positive = sum(1 for slot in SLOT_NAMES
                              if (gaps.get(slot) or 0.0) > 0.05
                              and own_vs_foreign[slot].get("verdict") not in
                              ("NOT INTERPRETABLE", "NO DATA"))
    success = {
        "adherence_at_least_0_80_at_every_turn": all(a >= 0.80 for a in adh_think),
        "shuffled_gap_materially_positive_on_at_least_2_of_4_slots": materially_positive >= 2,
        "degeneration_no_worse_than_the_no_think_arm": (
            degeneration_test["verdict"] != "no-think better"),
        "swap_test_shows_continuations_do_change": bool(swap["thinking_is_load_bearing"]),
    }
    success["all_criteria_met"] = all(success.values())
    # Recorded AFTER all_criteria_met on purpose: the plan named three gates plus the swap
    # test, and this is a stricter supplementary one added by this task (it did not exist to
    # be passed or failed when the gate was written). It asks whether the plan-following
    # signal survives the control that holds the SCENE fixed, not merely the one that
    # swaps in an unrelated skit.
    survives_context_only = sum(
        1 for slot in SLOT_NAMES
        if own_vs_context_only[slot].get("verdict") == "think better"
        and (table[slot].get("gap_over_context_only") or 0.0) > 0.02)
    success["supplementary_plan_following_survives_the_context_only_control"] = {
        "n_slots": survives_context_only,
        "passed": survives_context_only >= 2,
        "note": ("Not part of `all_criteria_met` — the plan's gate was written before this "
                 "control existed. handback_anticipation cannot contribute here: its "
                 "context-only floor is degenerate by construction (see the slot table)."),
    }
    n_slots_favouring_think = sum(1 for v in slot_arm_tests.values()
                                  if v.get("verdict") == "think better")
    n_failure_favouring_think = sum(1 for v in failure_tests.values()
                                    if v["verdict"] == "think better")
    verdict = ("DECORATIVE" if not swap["thinking_is_load_bearing"]
               else ("STAGE 2 SUCCESS" if success["all_criteria_met"]
                     else "PARTIAL — see success_criteria"))
    if args.max_eval_skits:
        # A truncated run must never be readable as the measurement. The marker goes in the
        # verdict string itself, not only in a field further down that a reader may skip.
        verdict = f"SMOKE ({len(val_skits)} skits) — NOT THE MEASUREMENT: {verdict}"

    report = {
        "verdict": verdict,
        "limitations": build_limitations(eval_set=str(args.skits.relative_to(ROOT)),
                                         n_skits=len(val_skits)),
        "power": power,
        "swap_test": swap,
        "swap_test_detail": {"n": n_swap, "divergence_positions": divergence,
                             "note": "token index into the generated continuation; None "
                                     "means identical for the full window"},
        "slots_model_generated_blocks": {
            "n_scorable_observations": n_scorable,
            "n_model_turns_attempted": n_model_turns,
            "table": table,
            "own_block_vs_foreign_block": own_vs_foreign,
            "own_block_vs_context_only_control": own_vs_context_only,
            "note": ("`matched` is the model's OWN block scored against the turn it then "
                     "wrote. `shuffled_floor` is that same block scored against another "
                     "skit's model turn (the task-3 control). `context_only_floor` is that "
                     "same block scored against the NO-THINK arm's turn for the SAME skit "
                     "and position — same scene, same context, but a turn written without a "
                     "block — which is what separates plan-following from context echo. "
                     "These eight contrasts sit OUTSIDE the pre-declared family of 11 and "
                     "are reported at the same critical t; if they are counted as tests too "
                     "the family is 19 and the correct critical value would be 3.01 — any "
                     "|t| between 2.843 and 3.01 should be read as borderline."),
        },
        "slots_reference_block_across_arms": {
            "tests": slot_arm_tests, "pooled": pooled_test,
            "think_accuracy": slot_accuracy([h for h in ref_hits_think if h is not None]),
            "nothink_accuracy": slot_accuracy([h for h in ref_hits_nothink if h is not None]),
            "note": ("The skit's ground-truth block scored against each arm's generated "
                     "turn. Neither arm is shown that block, which is what makes this the "
                     "only slot comparison that is fair across arms — forcing it into the "
                     "think arm's context would hand it the answer words (accept/add are "
                     "lifted from the corpus turn) and measure copying."),
        },
        "failure_modes": failure_tests,
        "adherence": {"think_by_turn": adh_think, "nothink_by_turn": adh_nothink,
                      "test": adherence_test,
                      "note": "per turn POSITION, never pooled — a pooled number hides a "
                              "model that writes a good block for turn 0 and collapses by "
                              "turn 4. The no-think arm is a negative control: it never saw "
                              "a think-block in training."},
        "degeneration": {"think_mean_4gram_repeat": round(st.fmean(deg_think), 4),
                         "nothink_mean_4gram_repeat": round(st.fmean(deg_nothink), 4),
                         "test": degeneration_test,
                         "note": ("4-gram repeat rate ACROSS a skit's three model turns "
                                  "(within-turn scoring would report ~0 for a model that "
                                  "writes the same sentence three times). This is the one "
                                  "pre-declared criterion the think arm fails, and it is a "
                                  "cost of the mechanism the rest of the file shows "
                                  "working: a block commits to an `add` word and the arm "
                                  "then writes that word into its turn, block after "
                                  "block.")},
        "bonferroni": {"alpha": BONFERRONI_ALPHA, "critical_t": CRITICAL_T,
                       "n_tests": N_TESTS,
                       "family": ["slot:accept", "slot:add", "slot:stakes",
                                  "slot:handback_anticipation", "pooled_slot_accuracy",
                                  "escalation", "new_harm", "groundedness", "affordance",
                                  "adherence", "degeneration"],
                       "n_slot_tests_favouring_think": n_slots_favouring_think,
                       "n_failure_tests_favouring_think": n_failure_favouring_think,
                       "note": ("This eval's OWN alpha. Stage 1's 0.01/2.576 covers five "
                                "tests and is looser; scripts/eval_improv.paired_verdict "
                                "closes over that constant and is deliberately NOT imported "
                                "here.")},
        "success_criteria": success,
        "held_out": {
            "skits_file": str(args.skits.relative_to(ROOT)),
            "n_train": len(train_skits), "n_held_out": len(val_skits),
            "n_held_out_available": n_heldout_available,
            "truncated_for_smoke": bool(args.max_eval_skits),
            "split": "tail of the skits file in file order — identical to both arms' val "
                     "split (train_manifest.json: n_examples/n_val_examples)",
            "story_id_overlap_with_training": [],
            "verification": "held-out story_id set intersected against every training "
                            "story_id; main() raises before generating if non-empty.",
        },
        "generation_settings": {
            "think_max_new_tokens": args.think_max_new_tokens,
            "nothink_max_new_tokens": args.nothink_max_new_tokens,
            "decoding": "greedy (do_sample=False) in every condition",
            "batch_size": args.batch_size,
            "contexts_clipped_to_window": {"think": clip_think, "nothink": clip_nothink},
            "device": "CPU (torch 2.7.0+cpu); no /dev/tenstorrent access, no chip lease",
            "context_construction": "segments tokenized separately and id-concatenated, "
                                    "exactly as scripts/derive_skits.py built training "
                                    "examples",
        },
        "examples": {
            "think_first_skit": raw_think[0] if raw_think else [],
            "nothink_first_skit": raw_nothink[0] if raw_nothink else [],
        },
        # Every scored turn, both arms, in held-out order. ~200 KB, and it means any
        # re-analysis (a new scorer, a different control) is a file read rather than
        # another full regeneration — this repo has already paid for one 40-minute run
        # whose answer was sitting in the previous run's output.
        "generated_turns": {
            "order": "held-out skits in file order; three model turns each (skit turns "
                     "0, 2, 4)",
            "story_ids": [s["story_id"] for s in val_skits],
            "think": turns_think, "nothink": turns_nothink,
            "think_blocks": [[(s.as_dict() if s is not None else None) for s in row]
                             for row in slots_think],
        },
        "manifests_used": manifests_block(manifest_think, manifest_nothink),
        "tokenization_parity": tokenization_parity,
        "derivation": (json.loads(DERIVE_MANIFEST_PATH.read_text())
                       if DERIVE_MANIFEST_PATH.is_file() else None),
        "hf_conversion": {"config": cfg_think,
                          "warm_start_header_source": str(WARM_START_CKPT.relative_to(ROOT))},
    }

    print("[8/8] writing report ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    print(f"\nVERDICT: {verdict}")
    print(f"  adherence by turn (think)   : {adh_think}")
    print(f"  swap load-bearing           : {swap['thinking_is_load_bearing']} "
          f"({swap['fraction_changed']:.1%})")
    for slot in SLOT_NAMES:
        row = table[slot]
        print(f"  {slot:22} matched={row['matched']} shuffled={row['shuffled_floor']} "
              f"corpus_floor={row['corpus_shuffled_floor']} "
              f"t={own_vs_foreign[slot].get('t')} {own_vs_foreign[slot]['verdict']}")
    return 0


def render_think_from(block: Dict[str, str]) -> str:
    """Render a raw slot dict as the exact think-block text training used."""
    from train.improv import Slots
    return render_think(Slots(**{k: block[k] for k in
                                 ("offer", "accept", "add", "stakes", "handback")}))


if __name__ == "__main__":
    raise SystemExit(main())
