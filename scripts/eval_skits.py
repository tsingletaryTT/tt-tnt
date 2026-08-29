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
    load_closure_lexicon,
    load_harm_lexicon,
    score_pair,
)
from scripts.score_skits import SLOT_NAMES, SlotHits, score_block, slot_accuracy  # noqa: E402
from train.improv import content_words, parse_think, render_think, split_sentences  # noqa: E402
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
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024"

#: Corpus record separator — NOT a blank line (see scripts/derive_skits.py).
STORY_SEP = "</s>"

_THINK_RE = re.compile(r"<think>\s*.*?\s*</think>", re.S)
_WORD_RE = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------------------


#: Direction for the two auxiliary tests, declared the same way `SLOT_DIRECTIONS` is and for
#: the same reason. A code-review mutation of the literal `"lower"` at the degeneration call
#: site passed all 27 tests and flipped the artifact's top-level verdict from PARTIAL to
#: "STAGE 2 SUCCESS" — the direction was written inline, in the driver, where no test could
#: see it. Both call sites now read this table, and `score_and_assemble` (a called function
#: with a fixture test) is what exercises them.
AUX_DIRECTIONS = {"adherence": "higher", "degeneration": "lower"}

#: A slot's plan-following claim is WITHDRAWN when the arm-quality proxy explains at least
#: this share of its context-only gap. See `decompose_confound` for what the proxy is and
#: why it is a sensitivity bound rather than a correction.
CONFOUND_WITHDRAW_SHARE = 0.5


def stamp_slot_disclosures(table: Dict[str, Dict[str, Any]],
                           own_vs_foreign: Dict[str, Dict[str, object]],
                           own_vs_context_only: Dict[str, Dict[str, object]],
                           decomposition: Dict[str, Dict[str, object]],
                           *, add_handback_identity: Optional[float]) -> None:
    """Attach every slot-level caveat INSIDE the rows a reader will actually look at.

    Two review findings, one fix. `handback_anticipation` already carried its structural and
    its degenerate-control warnings in-row, while `stakes` — the slot actually WITHDRAWN from
    the plan-following claim — had rows reading "think better" with nothing attached: its
    `own_block_vs_context_only_control` entry says t = 4.158, and a reader stopping there
    would publish what the decomposition withdrew. The same standard now applies to both.

    `add_handback_identity` is the third handback degeneracy, and unlike the other two it is
    EMPIRICAL: in this run 97.24% of generated blocks put the identical word in `add` and
    `handback`, and the two slots' top-10 word lists are the same list. So handback's matched
    rate is very largely the `add` word re-tested against the next partner turn — not an
    independent anticipation plan, and a near-duplicate component inside
    `pooled_slot_accuracy`. It STRENGTHENS the `add` headline (the model has one salient word
    per block) which is exactly why disclosing it costs the result nothing.
    """
    for slot in SLOT_NAMES:
        claim = decomposition.get(slot, {}).get("claim")
        if claim in ("withdrawn", "no signal", "not measurable"):
            marker = {
                "plan_following_claim": claim,
                "do_not_read_this_row_as_plan_following": decomposition[slot].get("why"),
            }
            table[slot]["plan_following_claim"] = claim
            table[slot]["claim_why"] = decomposition[slot].get("why")
            for d in (own_vs_foreign.get(slot), own_vs_context_only.get(slot)):
                if isinstance(d, dict):
                    d.update(marker)
    if add_handback_identity is not None:
        msg = (f"THIRD DEGENERACY, EMPIRICAL: {add_handback_identity:.2%} of this run's "
               f"generated blocks put the IDENTICAL word in `add` and `handback`, and the "
               f"two slots' top-10 word lists are the same list. handback's matched rate is "
               f"therefore very largely the `add` word re-tested against the next partner "
               f"turn rather than an independent anticipation plan, and it is a "
               f"near-duplicate component inside pooled_slot_accuracy. It strengthens the "
               f"`add` headline — one salient word per block — so disclosing it costs the "
               f"result nothing.")
        table["handback_anticipation"]["add_and_handback_are_the_same_word_rate"] = round(
            add_handback_identity, 4)
        table["handback_anticipation"]["duplicates_the_add_slot"] = msg
        for d in (own_vs_foreign.get("handback_anticipation"),
                  own_vs_context_only.get("handback_anticipation")):
            if isinstance(d, dict):
                d["duplicates_the_add_slot"] = msg


def add_handback_identity_rate(slots_think: Sequence[Sequence[Any]]) -> Optional[float]:
    """Share of generated blocks whose `add` and `handback` values are the same word."""
    vals = [_block_dict(b) for row in slots_think for b in row]
    blocks = [b for b in vals if b]
    if not blocks:
        return None
    same = sum(1 for b in blocks
               if (b.get("add") or "").strip().lower() == (b.get("handback") or "").strip().lower())
    return same / len(blocks)


def dialogue_selection_bias(corpus_path: Path, skits: Sequence[dict], *, n_stories: int
                            ) -> Dict[str, Any]:
    """How much dialogue the splitter's drop rule removed from the eval population.

    The spec requires a drop rate above 50% to be reported WITH the result, because past
    that point the FILTER is choosing the behaviour rather than the model. The rate here is
    0.907, and the drop is not uniform: `train/skit.py`'s documented limitation is that
    `split_sentences` fragments dialogue-with-attribution ('"It's mine!" said Ann.' becomes
    two sentences), so a model turn following such a partner turn shares no content word
    with it and the whole skit drops. This measures the resulting bias instead of describing
    it, over the SAME story slice derivation read.
    """
    stories: List[str] = []
    buf = ""
    with corpus_path.open(errors="ignore") as fh:
        while len(stories) < n_stories:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(STORY_SEP)
            buf = parts.pop()
            for part in parts:
                text = part.strip()
                if text:
                    stories.append(text)
                if len(stories) >= n_stories:
                    break
    stories = stories[:n_stories]
    c_units = sum(1 for s in stories if '"' in s)
    c_sents = c_quoted = 0
    for s in stories:
        sents = split_sentences(s)
        c_sents += len(sents)
        c_quoted += sum(1 for x in sents if '"' in x)
    k_units = sum(1 for k in skits
                  if '"' in k["prefix"] or any('"' in t for t in k["turns"]))
    turns = [t for k in skits for t in k["turns"]]
    k_quoted = sum(1 for t in turns if '"' in t)
    cu, ku = c_units / max(len(stories), 1), k_units / max(len(skits), 1)
    cs, ks = c_quoted / max(c_sents, 1), k_quoted / max(len(turns), 1)
    return {
        "n_corpus_stories_scanned": len(stories), "n_kept_skits": len(skits),
        "units_with_dialogue_corpus": round(cu, 4),
        "units_with_dialogue_kept": round(ku, 4),
        "units_relative_change": round(ku / cu - 1, 4) if cu else None,
        "sentences_with_dialogue_corpus": round(cs, 4),
        "turns_with_dialogue_kept": round(ks, 4),
        "sentences_relative_change": round(ks / cs - 1, 4) if cs else None,
        "dialogue_marker": 'a double-quote character anywhere in the unit',
        "note": ("The eval population is the RESIDUE left by a splitter that fragments "
                 "dialogue-with-attribution. Dialogue-heavy stories are systematically "
                 "under-represented, so every number in this artifact describes model "
                 "behaviour on the flatter, more narrative 9.3% of the corpus that survived "
                 "derivation — not on the corpus."),
    }



def evaluate_success_criteria(*, adherence_think: Sequence[float],
                              gaps: Dict[str, Optional[float]],
                              contrasts: Dict[str, Dict[str, object]],
                              degeneration_test: Dict[str, object],
                              swap_is_load_bearing: bool,
                              claims: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """The plan's four gates, as a called function rather than inline driver code.

    Every one of these is a one-token edit away from inverting: `>= 2` to `>= 1`, `!=` to
    `==`, `0.80` to `0.08`. Inline in `main()` none of them was reachable by a test, and a
    mutation of exactly this kind (the degeneration direction) turned a PARTIAL verdict into
    "STAGE 2 SUCCESS" while all 27 tests stayed green.
    """
    materially_positive_slots = [slot for slot in SLOT_NAMES
                                 if (gaps.get(slot) or 0.0) > 0.05
                                 and contrasts[slot].get("verdict") not in
                                 ("NOT INTERPRETABLE", "NO DATA")]
    materially_positive = len(materially_positive_slots)
    success: Dict[str, object] = {
        "adherence_at_least_0_80_at_every_turn": all(a >= 0.80 for a in adherence_think),
        "shuffled_gap_materially_positive_on_at_least_2_of_4_slots": materially_positive >= 2,
        "degeneration_no_worse_than_the_no_think_arm": (
            degeneration_test["verdict"] != "no-think better"),
        "swap_test_shows_continuations_do_change": bool(swap_is_load_bearing),
    }
    success["all_criteria_met"] = all(bool(v) for v in success.values())
    # BLOCKER 6: which slots carried that gate, named rather than counted. The gate is the
    # PRE-DECLARED one and is deliberately left as declared — it asks a different question
    # from the confound decomposition (does the model's own block beat a FOREIGN block,
    # which no arm-identity term touches), so a slot withdrawn from the plan-following claim
    # can still legitimately answer it. But a reader must be able to see that two of the
    # four slots counted here are elsewhere marked "withdrawn" and "not measurable", so the
    # list and the claim-filtered count are emitted beside it.
    success["materially_positive_slots"] = materially_positive_slots
    if claims:
        publishable = [s for s in materially_positive_slots
                       if claims.get(s) in ("headline", "bounded")]
        success["materially_positive_slots_restricted_to_publishable_claims"] = {
            "slots": publishable, "n": len(publishable),
            "would_still_pass": len(publishable) >= 2,
            "note": ("The same gate counting only slots whose plan-following claim survives "
                     "the arm-quality decomposition. Reported for disclosure; the gate "
                     "itself is the pre-declared one above and is not silently retightened "
                     "after seeing the data."),
        }
    return success


def decompose_confound(context_only_gaps: Dict[str, Optional[float]],
                       arm_tests: Dict[str, Dict[str, object]],
                       degenerate_slots: Sequence[str] = ("handback_anticipation",)
                       ) -> Dict[str, Dict[str, object]]:
    """Split each context-only gap into an arm-quality proxy and a residual.

    THE CONTROL IS NOT "SAME EVERYTHING". The context-only floor scores the think arm's
    block against the NO-THINK arm's turn, and those are SEPARATELY TRAINED MODELS — so the
    contrast is plan-presence PLUS arm identity, not plan-presence alone. It is also only
    strictly same-context at turn position 0: `roll_forward` feeds each arm its own prior
    turns, so by position 2 the two arms are continuing different scenes of their own making.

    The cross-arm reference test (the ground-truth block scored against each arm's turn,
    which neither arm ever sees) is the best available proxy for the arm-identity part. It
    is itself NOT significant, so this is a SENSITIVITY BOUND, not a correction — the
    residual is what survives if the entire non-significant arm difference were real.

    Rules, and each is a claim about what may be published:
      * a slot whose control is degenerate by construction -> "not measurable"
      * proxy >= CONFOUND_WITHDRAW_SHARE of the gap -> "withdrawn"
      * proxy < 0 (the confound points the OTHER way) -> "headline", biased downward
      * otherwise -> "bounded": at most the gap, plausibly the residual
    """
    out: Dict[str, Dict[str, object]] = {}
    for slot in SLOT_NAMES:
        gap = context_only_gaps.get(slot)
        proxy = arm_tests.get(slot, {}).get("mean_delta")
        if slot in degenerate_slots or gap is None or proxy is None:
            out[slot] = {"context_only_gap": gap, "arm_quality_proxy": proxy,
                         "residual": None, "claim": "not measurable",
                         "why": ("the context-only control is degenerate for this slot — "
                                 "score_block reads only the block's handback value and the "
                                 "corpus next-partner turn, which is identical in both "
                                 "conditions, so the gap is forced to 0.0000")}
            continue
        residual = round(gap - float(proxy), 4)
        if gap <= 0:
            # Found by the three-skit fixture, not by the real data (where every gap is
            # positive): with gap <= 0 there is no plan-following to attribute, and the
            # `proxy < 0` branch below would have labelled it "headline" — publishing a slot
            # whose own control beat it.
            out[slot] = {"context_only_gap": gap, "arm_quality_proxy": round(float(proxy), 4),
                         "residual": residual, "claim": "no signal",
                         "why": (f"the context-only gap is {gap:+.4f} — the block's own "
                                 f"control did at least as well, so there is nothing to "
                                 f"publish regardless of the confound")}
            continue
        if float(proxy) >= CONFOUND_WITHDRAW_SHARE * gap:
            claim, why = "withdrawn", (
                f"the arm-quality proxy ({proxy:+.4f}) explains at least "
                f"{CONFOUND_WITHDRAW_SHARE:.0%} of the {gap:+.4f} gap — this slot must NOT "
                f"be published as plan-following")
        elif float(proxy) < 0:
            claim, why = "headline", (
                f"the confound points the OTHER way ({proxy:+.4f}), so {gap:+.4f} is biased "
                f"DOWNWARD — this is the safest slot to publish")
        else:
            claim, why = "bounded", (
                f"at most {gap:+.4f}, plausibly ~{residual:+.4f} once the (non-significant) "
                f"arm-quality proxy {proxy:+.4f} is subtracted")
        out[slot] = {"context_only_gap": gap, "arm_quality_proxy": round(float(proxy), 4),
                     "residual": residual, "claim": claim, "why": why}
    return out


def slot_value_visibility(val_skits: Sequence[dict], slots_think: Sequence[Sequence[Any]],
                          slot: str) -> Optional[float]:
    """How often a generated slot's value was ALREADY VISIBLE in the context.

    This is the mechanism behind the confound decomposition, measured rather than asserted:
    if a slot's value is usually already on the page, a turn written without any block hits
    it anyway, and the context-only floor is high. `accept` is mostly the protagonist's
    name; `add` names something new by definition.
    """
    seen = hits = 0
    for j, skit in enumerate(val_skits):
        for i, t_idx in enumerate(MODEL_TURNS):
            block = slots_think[j][i]
            if block is None:
                continue
            value = getattr(block, slot, "") if not isinstance(block, dict) else block.get(slot, "")
            want = set(content_words(value))
            if not want:
                continue
            context = skit["prefix"] + " " + " ".join(skit["turns"][:t_idx])
            seen += 1
            hits += bool(want & set(content_words(context)))
    return round(hits / seen, 4) if seen else None


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
        # The note comes FIRST in key order, deliberately: it is the thing a reader must see
        # before the numbers it protects, and this docstring said so while the code put it
        # last.
        "loss_comparability": (
            "DO NOT compare the two arms' `loss_end` (think 0.504 vs nothink 1.227). They "
            "supervise DIFFERENT TOKEN SETS: 168.1 vs 43.4 supervised label positions per "
            "example (236.8 vs 111.8 tokens), and 74.2% of the think arm's supervised "
            "positions are the fixed five-line template, which is far more predictable than "
            "free prose. The gap is an easier token mixture, not an arm effect. `paired_with` "
            "in these manifests means the two runs share data order, seed and schedule — it "
            "does NOT mean their losses are on a common scale. The only valid arm comparison "
            "is the scorers applied to MODEL TURNS, which is what the rest of this file is. "
            "THE SAME APPLIES TO `val_loss_first` / `val_loss_last` / the whole val curve "
            "(think 1.009 -> 0.639 vs nothink 1.580 -> 1.552): held-out loss is averaged "
            "over the same arm-specific token mixture, so cross-arm val comparisons are "
            "invalid for exactly the same reason. Within one arm, across steps, the "
            "trajectory IS meaningful — that is how we know the think arm was still "
            "improving at step 3000 and the no-think arm had nearly stopped."),
        "think": think,
        "nothink": nothink,
    }


def _drop_rate_limitation(derivation: Optional[Dict[str, Any]],
                          bias: Optional[Dict[str, Any]]) -> str:
    """The drop-rate disclosure the spec makes mandatory above 50%.

    Spec §1: "warn above 50% — past that the FILTER rather than the model is choosing the
    behaviour, and any result must be reported with that fact attached." The rate is 0.907
    and it was in the artifact only as a bare `derivation.drop_rate`, with none of the eight
    limitation entries mentioning it. It is also the strongest attack a hostile reader has,
    so it gets the numbers rather than a hedge.
    """
    if not derivation:
        return ("derivation manifest not found, so the drop rate could not be reported — "
                "which is itself disqualifying: the spec requires it WITH the result.")
    rate = derivation.get("drop_rate")
    rules = derivation.get("drops_by_rule", {})
    parts = [
        f"DROP RATE {rate:.1%} ({derivation.get('kept'):,} skits kept of "
        f"{derivation.get('stories'):,} corpus stories). The spec's own rule is that above "
        f"50% the FILTER rather than the model is choosing the behaviour, and any result "
        f"must carry that fact — so: this eval measures the model on the "
        f"{1 - rate:.1%} of the corpus that survived derivation, not on the corpus. "
        f"Drops by rule: "
        + ", ".join(f"{k} {v:,}" for k, v in sorted(rules.items(), key=lambda kv: -kv[1]))
        + ".",
        "The drop is NOT uniform, which is the part that bites: train/skit.py's documented "
        "limitation is that split_sentences fragments dialogue-with-attribution ('\"It's "
        "mine!\" said Ann.' becomes two sentences), so a model turn following such a partner "
        "turn shares no content word with it and the entire skit drops.",
    ]
    if bias:
        parts.append(
            f"MEASURED SELECTION BIAS (same story slice derivation read): units containing "
            f"dialogue {bias['units_with_dialogue_corpus']:.1%} in the corpus -> "
            f"{bias['units_with_dialogue_kept']:.1%} in the kept skits "
            f"({bias['units_relative_change']:+.0%} relative); sentences/turns containing "
            f"dialogue {bias['sentences_with_dialogue_corpus']:.1%} -> "
            f"{bias['turns_with_dialogue_kept']:.1%} ({bias['sentences_relative_change']:+.0%} "
            f"relative). The eval population is the residue left by that splitter: flatter, "
            f"more narrative prose with systematically less dialogue. Any claim about "
            f"conversational improv skill inherits that bias.")
    return " ".join(parts)


def build_limitations(*, eval_set: str, n_skits: int, derivation: Optional[Dict[str, Any]],
                     selection_bias: Optional[Dict[str, Any]]) -> Dict[str, str]:
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
        "the_eval_population_is_a_90_7_percent_filtered_residue": _drop_rate_limitation(
            derivation, selection_bias),
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


def _block_dict(block: Any) -> Optional[Dict[str, str]]:
    """A generated block as a plain dict, whether it arrived as `Slots` (live run) or as
    JSON (re-scoring a stored artifact). Both paths must score identically."""
    if block is None:
        return None
    return block if isinstance(block, dict) else block.as_dict()


def score_and_assemble(*, val_skits: Sequence[dict], turns_think: Sequence[Sequence[str]],
                       turns_nothink: Sequence[Sequence[str]],
                       slots_think: Sequence[Sequence[Any]],
                       adherence_think: Sequence[float], adherence_nothink: Sequence[float],
                       adherence_series_think: Sequence[float],
                       adherence_series_nothink: Sequence[float],
                       swap_is_load_bearing: bool, harm: frozenset, closure: frozenset,
                       assoc: Dict[str, object]) -> Dict[str, Any]:
    """Every scored number and every verdict, as ONE CALLED FUNCTION.

    This used to be inline in `main()`, and that was a hole rather than a style choice: a
    code review mutated the degeneration direction from "lower" to "higher" — one token, in
    the driver — and all 27 tests passed while the artifact's top-level verdict changed from
    PARTIAL to "STAGE 2 SUCCESS". Three sibling call sites (the adherence direction, the
    `>= 2` success gate, `_context_for`'s next-partner index) had the same property. Wiring
    that decides a published verdict has to be reachable by a fixture, so it lives here and
    `tests/test_eval_skits.py` drives it end to end on three synthetic skits.

    Pure: no model, no tokenizer, no device. Given the same stored turns it reproduces the
    same numbers, which is what makes `--rescore-from` a re-analysis rather than a rerun.
    """
    n_model_turns = len(val_skits) * len(MODEL_TURNS)
    matched_hits: List[SlotHits] = []
    shuffled_hits: List[SlotHits] = []
    context_only_hits: List[SlotHits] = []
    ref_hits_think: List[SlotHits] = []
    ref_hits_nothink: List[SlotHits] = []
    n_scorable = 0
    #: Turns where the model never closed a think-block, so `split_generation` had no block
    #: to strip and the "turn" it returned is raw template text. See `contamination` below.
    contaminated: List[Dict[str, Any]] = []

    for j, skit in enumerate(val_skits):
        other = val_skits[(j + 1) % len(val_skits)]
        for i, t_idx in enumerate(MODEL_TURNS):
            prev, nxt = _context_for(skit, t_idx)
            oprev, onxt = _context_for(other, t_idx)
            block = _block_dict(slots_think[j][i])
            if "<think>" in turns_think[j][i]:
                contaminated.append({"index": j, "story_id": skit["story_id"],
                                     "turn": t_idx, "turn_text": turns_think[j][i][:120]})
            if block is not None and turns_think[j][i]:
                n_scorable += 1
                matched_hits.append(_score_one(block, turn=turns_think[j][i],
                                               prev_turn=prev, next_partner=nxt, harm=harm))
                # The control: this same block against the NEXT skit's model turn at the
                # same position, with that skit's own surrounding turns — the exact
                # construction task 3 used to measure the corpus floors.
                foreign_turn = turns_think[(j + 1) % len(val_skits)][i]
                shuffled_hits.append(_score_one(block, turn=foreign_turn,
                                                prev_turn=oprev, next_partner=onxt, harm=harm))
                # The tighter control: the SAME block against the NO-THINK arm's turn for
                # THIS skit at THIS position. Same scene, same partner turns — but NOT "same
                # everything": the no-think arm is a separately trained model, so this
                # contrast carries plan-presence PLUS arm identity. `confound_decomposition`
                # below bounds the second part instead of pretending it is absent.
                context_only_hits.append(_score_one(block, turn=turns_nothink[j][i],
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

    own_vs_foreign: Dict[str, Dict[str, object]] = {}
    own_vs_context_only: Dict[str, Dict[str, object]] = {}
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
    slot_arm_tests: Dict[str, Dict[str, object]] = {}
    for slot in SLOT_NAMES:
        xs, ys = paired_defined([getattr(h, slot) for h in ref_hits_think],
                                [getattr(h, slot) for h in ref_hits_nothink])
        slot_arm_tests[slot] = (paired_verdict(xs, ys, SLOT_DIRECTIONS[slot]) if xs
                                else {"verdict": "NO DATA", "n": 0})
    # FIX 3: this test CANNOT return anything but t = 0.0, and a reader seeing "NOT
    # INTERPRETABLE" would conclude "no arm difference" where the truth is "no difference
    # was measurable even in principle". The warning goes in the row, not a footnote.
    slot_arm_tests["handback_anticipation"]["structurally_incapable_of_a_nonzero_result"] = (
        "score_block scores handback_anticipation against the ground-truth block's "
        "`handback` value and the CORPUS next-partner turn. Neither depends on the model's "
        "turn, so the two arms' series are identical BY CONSTRUCTION and t is forced to "
        "0.0. This is not evidence of no arm difference — no difference was measurable even "
        "in principle. It nevertheless occupies one of the eleven pre-declared family slots "
        "(that family was fixed before this was noticed, and shrinking it after seeing the "
        "data would be a worse sin than reporting it), and it dilutes "
        "`pooled_slot_accuracy`, which averages it in as a permanent tie.")

    def _pooled(h: SlotHits) -> Optional[float]:
        vals = [v for v in (h.accept, h.add, h.stakes, h.handback_anticipation)
                if v is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    xs, ys = paired_defined([_pooled(h) for h in ref_hits_think],
                            [_pooled(h) for h in ref_hits_nothink])
    pooled_test = paired_verdict(xs, ys, "higher")                       # test 5
    pooled_test["dilution_note"] = (
        "handback_anticipation contributes a permanent tie to every observation it is "
        "defined on (see slot_arm_tests), so this pooled number is shrunk toward zero by a "
        "component that could not have moved.")

    # Failure-mode scorers (tests 6-9), on cross-turn intervals: each model turn is scored
    # against the text immediately preceding it (the prefix, or the real partner turn),
    # which is the same interval `stakes` uses and is identical for both arms.
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
    # `affordance` returns t = 0.0 here, and that is NOT an identical pair of series: it is
    # a near-saturated scorer whose discordant pairs happen to balance. Say which, with the
    # numbers, or a reader will file it next to handback's structural zero.
    aff_t, aff_n = fail_series["affordance"]
    failure_tests["affordance"]["saturation_note"] = (
        f"fires on {round(100 * st.fmean(aff_t), 2)}% of think-arm and "
        f"{round(100 * st.fmean(aff_n), 2)}% of no-think-arm turns. The exact t = 0.0 is "
        f"COINCIDENTAL BALANCE, not structure — see signs_pos/signs_neg: the discordant "
        f"pairs cancel. A near-saturated scorer has little variance left to discriminate "
        f"with; it is not incapable of moving the way handback_anticipation is.")

    adherence_test = paired_verdict(adherence_series_think, adherence_series_nothink,
                                    AUX_DIRECTIONS["adherence"])

    # ---- Degeneration, and the contamination that inflates it (FIX 5) -------------------
    deg_think = [four_gram_repeat_rate(t) for t in turns_think]
    deg_nothink = [four_gram_repeat_rate(t) for t in turns_nothink]
    degeneration_test = paired_verdict(deg_think, deg_nothink, AUX_DIRECTIONS["degeneration"])
    dirty = {c["index"] for c in contaminated}
    clean = [j for j in range(len(val_skits)) if j not in dirty]
    deg_clean = ({"think_mean_4gram_repeat": round(st.fmean([deg_think[j] for j in clean]), 4),
                  "nothink_mean_4gram_repeat": round(st.fmean([deg_nothink[j] for j in clean]), 4),
                  "test": paired_verdict([deg_think[j] for j in clean],
                                         [deg_nothink[j] for j in clean],
                                         AUX_DIRECTIONS["degeneration"]),
                  "n_skits": len(clean)} if clean and dirty else None)

    # ---- Where the looping actually lives (FIX 6) --------------------------------------
    # Not "the mechanism that gives plan-following also gives looping" — that story is not
    # in the data (a model following a VARIED plan does not loop). Looping concentrates in
    # the skits where the block STOPS VARYING across the three turns.
    def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
        if len(xs) < 3:
            return None
        mx, my = st.fmean(xs), st.fmean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
        return round(num / den, 4) if den else None

    per_skit_add_hit: List[float] = []
    for j, skit in enumerate(val_skits):
        hits = []
        for i, t_idx in enumerate(MODEL_TURNS):
            block = _block_dict(slots_think[j][i])
            if block is None or not turns_think[j][i]:
                continue
            prev, nxt = _context_for(skit, t_idx)
            hits.append(_score_one(block, turn=turns_think[j][i], prev_turn=prev,
                                   next_partner=nxt, harm=harm).add)
        per_skit_add_hit.append(sum(1 for h in hits if h) / len(hits) if hits else 0.0)

    stagnant = []
    for j in range(len(val_skits)):
        adds = [(_block_dict(b) or {}).get("add") for b in slots_think[j]]
        if all(a is not None for a in adds) and len(set(adds)) == 1:
            stagnant.append(j)
    varied = [j for j in range(len(val_skits)) if j not in set(stagnant)]
    stagnation = {
        "definition": "skits whose three generated blocks all name the SAME `add` value",
        "n_stagnant": len(stagnant), "n_total": len(val_skits),
        "fraction": round(len(stagnant) / max(len(val_skits), 1), 4),
        "mean_repeat_stagnant": (round(st.fmean([deg_think[j] for j in stagnant]), 4)
                                 if stagnant else None),
        "mean_repeat_varied": (round(st.fmean([deg_think[j] for j in varied]), 4)
                               if varied else None),
        # The two correlations that decide WHICH story the degeneration finding supports.
        # "plan-following causes looping" would need the first to be large; it is ~0.
        "corr_add_hit_rate_with_repeat": _corr(per_skit_add_hit, deg_think),
        "corr_block_stagnation_with_repeat": _corr([float(j in set(stagnant))
                                                    for j in range(len(val_skits))],
                                                   deg_think),
        "test_excluding_stagnant_skits": (
            paired_verdict([deg_think[j] for j in varied], [deg_nothink[j] for j in varied],
                           AUX_DIRECTIONS["degeneration"]) if varied and stagnant else None),
        "note": ("Read the excluded-stagnant test and the two correlations before blaming "
                 "think-blocks for looping: "
                 "if it falls below the critical t, the degeneration finding is carried by "
                 "the minority of skits where the plan stopped varying — stagnation, not "
                 "plan-following. The fix that follows is to score blocks for NOVELTY "
                 "across turns, not to remove the blocks."),
    }

    # ---- The confound decomposition (FIX 2) --------------------------------------------
    ctx_gaps = {slot: table[slot].get("gap_over_context_only") for slot in SLOT_NAMES}
    decomposition = decompose_confound(ctx_gaps, slot_arm_tests)
    identity_rate = add_handback_identity_rate(slots_think)
    stamp_slot_disclosures(table, own_vs_foreign, own_vs_context_only, decomposition,
                           add_handback_identity=identity_rate)
    for slot in SLOT_NAMES:
        decomposition[slot]["slot_value_already_visible_in_context"] = slot_value_visibility(
            val_skits, slots_think, slot if slot != "handback_anticipation" else "handback")
    publishable = [s for s in SLOT_NAMES if decomposition[s]["claim"] in ("headline", "bounded")]

    success = evaluate_success_criteria(
        adherence_think=adherence_think,
        gaps={s: table[s].get("gap_over_shuffled") for s in SLOT_NAMES},
        contrasts=own_vs_foreign, degeneration_test=degeneration_test,
        swap_is_load_bearing=swap_is_load_bearing,
        claims={s: str(decomposition[s]["claim"]) for s in SLOT_NAMES})
    survives_context_only = sum(
        1 for slot in SLOT_NAMES
        if own_vs_context_only[slot].get("verdict") == "think better"
        and decomposition[slot]["claim"] in ("headline", "bounded"))
    success["supplementary_plan_following_survives_the_context_only_control"] = {
        "n_slots": survives_context_only,
        "slots": [s for s in SLOT_NAMES
                  if own_vs_context_only[s].get("verdict") == "think better"
                  and decomposition[s]["claim"] in ("headline", "bounded")],
        "passed": survives_context_only >= 2,
        "note": ("Not part of `all_criteria_met` — the plan's gate was written before this "
                 "control existed. Counts only slots whose claim SURVIVES the arm-quality "
                 "decomposition: handback_anticipation cannot contribute (degenerate "
                 "control) and stakes is withdrawn (mostly confounded)."),
    }
    verdict_core = ("DECORATIVE" if not swap_is_load_bearing
                    else ("STAGE 2 SUCCESS" if success["all_criteria_met"]
                          else "PARTIAL — see success_criteria"))

    return {
        "verdict_core": verdict_core,
        "success_criteria": success,
        "n_scorable": n_scorable,
        "n_model_turns": n_model_turns,
        "slots_model_generated_blocks": {
            "n_scorable_observations": n_scorable,
            "n_model_turns_attempted": n_model_turns,
            "table": table,
            "own_block_vs_foreign_block": own_vs_foreign,
            "own_block_vs_context_only_control": own_vs_context_only,
            "confound_decomposition": decomposition,
            "publishable_plan_following_slots": publishable,
            "add_and_handback_are_the_same_word_rate": (round(identity_rate, 4)
                                                       if identity_rate is not None else None),
            "note": (
                "`matched` is the model's OWN block scored against the turn it then wrote. "
                "`shuffled_floor` is that same block scored against another skit's model "
                "turn (the task-3 control). `context_only_floor` is that same block scored "
                "against the NO-THINK arm's turn for the same skit and position. "
                "DO NOT read the context-only contrast as plan-presence alone: the "
                "no-think arm is a SEPARATELY TRAINED MODEL, so the contrast is "
                "plan-presence PLUS arm identity, and it is strictly same-context only at "
                "turn position 0 (each arm is fed its own prior turns thereafter; the gap "
                "was checked at position 0 and holds, so this weakens the wording, not the "
                "result). `confound_decomposition` bounds the arm-identity part using the "
                "cross-arm reference test as a proxy. THE CLAIM IS NOT 'all four slots "
                "move': `add` is the headline (its confound has the opposite sign, so it is "
                "biased downward), `accept` is bounded, `stakes` is WITHDRAWN, and "
                "`handback_anticipation`'s control is degenerate by construction. "
                "These eight contrasts sit OUTSIDE the pre-declared family of 11 and are "
                "reported at the same critical t; counted as tests the family is 19 and the "
                "critical value would be 3.01 — any |t| in 2.843-3.01 is borderline."),
        },
        "slots_reference_block_across_arms": {
            "tests": slot_arm_tests, "pooled": pooled_test,
            "think_accuracy": slot_accuracy(ref_hits_think),
            "nothink_accuracy": slot_accuracy(ref_hits_nothink),
            "note": ("The skit's ground-truth block scored against each arm's generated "
                     "turn. Neither arm is shown that block, which is what makes this the "
                     "only slot comparison that is fair across arms — forcing it into the "
                     "think arm's context would hand it the answer words (accept/add are "
                     "lifted from the corpus turn) and measure copying. It doubles as the "
                     "arm-quality proxy in `confound_decomposition`; it is NOT significant "
                     "on any slot, which is why that decomposition is a bound rather than a "
                     "correction."),
        },
        "failure_modes": failure_tests,
        "adherence": {"think_by_turn": list(adherence_think),
                      "nothink_by_turn": list(adherence_nothink),
                      "test": adherence_test,
                      "direction": AUX_DIRECTIONS["adherence"],
                      "note": "per turn POSITION, never pooled — a pooled number hides a "
                              "model that writes a good block for turn 0 and collapses by "
                              "turn 4. The no-think arm is a negative control: it never saw "
                              "a think-block in training."},
        "degeneration": {"think_mean_4gram_repeat": round(st.fmean(deg_think), 4),
                         "nothink_mean_4gram_repeat": round(st.fmean(deg_nothink), 4),
                         "test": degeneration_test,
                         "direction": AUX_DIRECTIONS["degeneration"],
                         "excluding_contaminated_turns": deg_clean,
                         "stagnation_analysis": stagnation,
                         "note": ("4-gram repeat rate ACROSS a skit's three model turns "
                                  "(within-turn scoring would report ~0 for a model that "
                                  "writes the same sentence three times). The headline "
                                  "number INCLUDES the contaminated turns listed under "
                                  "`contamination`; `excluding_contaminated_turns` is the "
                                  "same test without them and is the number to quote when "
                                  "the effect size matters.")},
        "contamination": {
            "n_unparsed_blocks": n_model_turns - n_scorable,
            "n_turns_leaking_template_text": len(contaminated),
            "n_turns": len(contaminated),
            "detail": contaminated,
            "note": ("`n_unparsed_blocks` counts model turns whose block did not parse (they "
                     "are excluded from every slot measurement). A SUBSET of them "
                     "(`n_turns_leaking_template_text`) opened a think-block and never "
                     "closed it, so `split_generation` found no `</think>` to split on and "
                     "the 'turn' it returned is raw template text. These leak into the cross-arm test, "
                     "the failure scorers and degeneration. They are REPORTED rather than "
                     "silently dropped, and every affected conclusion is restated without "
                     "them — see `degeneration.excluding_contaminated_turns`."),
        },
    }


def _md5(path: Path) -> Optional[str]:
    """Streaming md5 of a file, or None if unreadable. Paired with every recorded path so a
    later reader can tell whether the file they have is the file that was measured."""
    import hashlib
    try:
        h = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _repo_relative(path: Path) -> str:
    """Repo-relative when possible, else the path as given. An absolute worktree path baked
    into a committed artifact stops resolving the moment the worktree is removed."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def measure_selection_bias(derivation: Optional[Dict[str, Any]],
                           kept_skits: Sequence[dict]) -> Optional[Dict[str, Any]]:
    """`dialogue_selection_bias` over the SAME story slice derivation actually read.

    The story count comes from the derive manifest rather than a hardcoded number, so the
    corpus side of the comparison can never silently be a different slice from the one the
    kept skits were drawn from — which would make the two percentages incomparable while
    looking fine.
    """
    if not derivation:
        return None
    corpus = ROOT / derivation["corpus"] if not Path(derivation["corpus"]).is_absolute() \
        else Path(derivation["corpus"])
    if not corpus.is_file():
        return None
    return dialogue_selection_bias(corpus, kept_skits,
                                   n_stories=int(derivation.get("stories", 0)))


def derivation_block(path: Path = DERIVE_MANIFEST_PATH) -> Optional[Dict[str, Any]]:
    """The derive manifest, with `corpus` rewritten repo-relative and md5'd.

    `_repo_relative` existed to stop absolute worktree paths leaking into a committed
    artifact and was simply not applied here, so `derivation.corpus` shipped as
    /home/ttuser/code/tt-tnt/... — a path that stops resolving the moment the worktree is
    gone, and that tells a later reader nothing about WHICH corpus file it was.
    """
    if not path.is_file():
        return None
    m = dict(json.loads(path.read_text()))
    corpus = Path(m.get("corpus", ""))
    if corpus.name:
        m["corpus"] = _repo_relative(corpus)
        m["corpus_md5"] = _md5(corpus)
        m["corpus_bytes"] = corpus.stat().st_size if corpus.is_file() else None
    m["manifest_path"] = _repo_relative(path)
    return m


def rescore_from_artifact(source: Path, out: Path, *, skits_path: Path,
                          assoc_skits: int) -> int:
    """Recompute every scored number from an artifact's STORED `generated_turns`.

    A re-analysis, not a rerun: no checkpoint, no tokenizer, no model, no device. This
    exists because the alternative — regenerating 1,536 turns to change a note or add a
    control — costs 25 minutes of CPU to reproduce numbers the previous run already wrote
    down, and this repo has paid that bill before.

    It is also a determinism check with teeth: the sections it recomputes must match the
    source artifact's on every pre-existing number, and it prints any that do not.
    """
    src = json.loads(source.read_text())
    gen = src.get("generated_turns")
    if not gen:
        raise RuntimeError(f"{source} has no `generated_turns` block — nothing to re-score. "
                           f"Only artifacts written after that block was added can be "
                           f"re-analysed without regenerating.")

    harm, closure = load_harm_lexicon(), load_closure_lexicon()
    manifest_think = json.loads(MANIFEST_THINK.read_text())
    manifest_nothink = json.loads(MANIFEST_NOTHINK.read_text())
    n_train = int(manifest_think["n_examples"])
    if int(manifest_nothink["n_examples"]) != n_train:
        raise RuntimeError("the two arms trained on different-sized training sets; they are "
                           "not paired and must not be compared")
    all_skits = [json.loads(l) for l in skits_path.read_text().splitlines()]
    train_skits, val_skits = all_skits[:n_train], all_skits[n_train:]

    # The stored turns must line up with the skits file, skit for skit, or every score
    # would be computed against the wrong scene while looking perfectly healthy.
    stored_ids = gen["story_ids"]
    if [s["story_id"] for s in val_skits] != stored_ids:
        raise RuntimeError("stored generated_turns do not correspond to this skits file's "
                           "held-out tail (story_id sequence differs) — refusing to score "
                           "turns against the wrong scenes")
    turns_think, turns_nothink = gen["think"], gen["nothink"]
    slots_think = gen["think_blocks"]

    # Adherence: recomputable exactly from the stored blocks (None == did not parse). The
    # no-think arm's is carried over, because it needs raw generations this artifact does
    # not store — and it is asserted to be the all-zero negative control it claims to be.
    adh_think = [sum(1 for row in slots_think if row[i] is not None) / len(slots_think)
                 for i in range(len(MODEL_TURNS))]
    adh_nothink = src["adherence"]["nothink_by_turn"]
    if any(a != 0.0 for a in adh_nothink):
        raise RuntimeError(f"carried-over no-think adherence is not all zero ({adh_nothink}); "
                           f"it can no longer be reconstructed from stored blocks alone")
    if [round(a, 6) for a in adh_think] != [round(a, 6)
                                            for a in src["adherence"]["think_by_turn"]]:
        raise RuntimeError(f"recomputed think adherence {adh_think} != stored "
                           f"{src['adherence']['think_by_turn']} — the stored blocks and the "
                           f"stored adherence disagree")
    adh_series_t = [1.0 if b is not None else 0.0 for row in slots_think for b in row]
    adh_series_n = [0.0] * len(adh_series_t)

    assoc_pairs: List[Tuple[str, str]] = []
    for s in train_skits[:assoc_skits]:
        for t_idx in MODEL_TURNS:
            prev, _ = _context_for(s, t_idx)
            assoc_pairs.append((prev, s["turns"][t_idx]))
    assoc = build_association(assoc_pairs)

    sections = score_and_assemble(
        val_skits=val_skits, turns_think=turns_think, turns_nothink=turns_nothink,
        slots_think=slots_think, adherence_think=adh_think, adherence_nothink=adh_nothink,
        adherence_series_think=adh_series_t, adherence_series_nothink=adh_series_n,
        swap_is_load_bearing=bool(src["swap_test"]["thinking_is_load_bearing"]),
        harm=harm, closure=closure, assoc=assoc)

    # Reproduction check against the source artifact, printed rather than assumed.
    drift = []
    for path, new in (
        ("failure_modes.new_harm.t", sections["failure_modes"]["new_harm"]["t"]),
        ("failure_modes.groundedness.t", sections["failure_modes"]["groundedness"]["t"]),
        ("degeneration.test.t", sections["degeneration"]["test"]["t"]),
        ("adherence.test.t", sections["adherence"]["test"]["t"]),
        ("slots.table.accept.matched",
         sections["slots_model_generated_blocks"]["table"]["accept"]["matched"]),
        ("slots.table.add.matched",
         sections["slots_model_generated_blocks"]["table"]["add"]["matched"]),
    ):
        node: Any = src
        for key in path.split("."):
            key = {"slots": "slots_model_generated_blocks"}.get(key, key)
            node = node[key]
        if node != new:
            drift.append(f"{path}: stored {node} -> recomputed {new}")
    print("  reproduction vs source artifact: "
          + ("IDENTICAL on every checked number" if not drift else "DRIFT " + "; ".join(drift)))

    report = dict(src)
    report.update(sections["slots_model_generated_blocks"] and {
        "slots_model_generated_blocks": sections["slots_model_generated_blocks"],
        "slots_reference_block_across_arms": sections["slots_reference_block_across_arms"],
        "failure_modes": sections["failure_modes"],
        "adherence": sections["adherence"],
        "degeneration": sections["degeneration"],
        "contamination": sections["contamination"],
        "success_criteria": sections["success_criteria"],
        "verdict": sections["verdict_core"],
    })
    derivation = derivation_block()
    selection_bias = measure_selection_bias(derivation, all_skits)
    report["derivation"] = derivation
    report["selection_bias"] = selection_bias
    # Rebuilt, not carried over: the note that protects these numbers (and now the val-loss
    # curve too) lives in `manifests_block`, and a carried-over copy would freeze whatever
    # wording the generation run happened to ship.
    report["manifests_used"] = manifests_block(manifest_think, manifest_nothink)
    report["limitations"] = build_limitations(
        eval_set=str(skits_path.relative_to(ROOT)), n_skits=len(val_skits),
        derivation=derivation, selection_bias=selection_bias)
    report["rescoring"] = {
        "source_artifact": _repo_relative(source),
        "source_artifact_md5": _md5(source),
        "skits_file": _repo_relative(skits_path), "skits_file_md5": _md5(skits_path),
        "what_was_recomputed": sorted(["slots_model_generated_blocks",
                                       "slots_reference_block_across_arms", "failure_modes",
                                       "adherence", "degeneration", "contamination",
                                       "success_criteria", "verdict", "limitations",
                                       "derivation", "selection_bias",
                                       "manifests_used"]),
        "what_was_carried_over": sorted(["swap_test", "swap_test_detail", "power",
                                         "held_out", "generation_settings", "examples",
                                         "generated_turns",
                                         "tokenization_parity",
                                         "hf_conversion", "bonferroni",
                                         "adherence.nothink_by_turn"]),
        "reproduction_check": ("every pre-existing number recomputed identically from the "
                              "stored turns" if not drift else drift),
        "note": ("No model was loaded and no token was generated: this is scoring re-run "
                 "over the turns the generation run stored. The generation itself is greedy "
                 "and was verified byte-identical across three independent runs."),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    print(f"\nVERDICT: {report['verdict']}")
    for slot in SLOT_NAMES:
        d = sections["slots_model_generated_blocks"]["confound_decomposition"][slot]
        print(f"  {slot:22} gap={d['context_only_gap']} proxy={d['arm_quality_proxy']} "
              f"residual={d['residual']} -> {d['claim']}")
    return 0


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
    ap.add_argument("--rescore-from", type=Path, default=None,
                    help="Re-score an existing artifact's stored `generated_turns` instead "
                         "of generating. No model, no tokenizer, no device — a re-analysis, "
                         "not a rerun.")
    args = ap.parse_args()

    if args.rescore_from:
        print(f"[rescore] re-analysing {args.rescore_from} — no generation, no model ...")
        return rescore_from_artifact(args.rescore_from, args.out, skits_path=args.skits,
                                     assoc_skits=args.assoc_skits)

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
    assoc_pairs: List[Tuple[str, str]] = []
    for s in train_skits[:args.assoc_skits]:
        for t_idx in MODEL_TURNS:
            prev, _ = _context_for(s, t_idx)
            assoc_pairs.append((prev, s["turns"][t_idx]))
    assoc = build_association(assoc_pairs)

    # Adherence needs the RAW generations (a block that failed to parse is still text), so
    # it is measured here and handed to the assembler, which is otherwise pure.
    adh_think = adherence_by_turn(raw_think, n_turns=len(MODEL_TURNS))
    adh_nothink = adherence_by_turn(raw_nothink, n_turns=len(MODEL_TURNS))
    adh_series_t = [1.0 if s is not None else 0.0 for row in slots_think for s in row]
    adh_series_n = [1.0 if parse_think(g) is not None else 0.0
                    for row in raw_nothink for g in row]

    sections = score_and_assemble(
        val_skits=val_skits, turns_think=turns_think, turns_nothink=turns_nothink,
        slots_think=slots_think, adherence_think=adh_think, adherence_nothink=adh_nothink,
        adherence_series_think=adh_series_t, adherence_series_nothink=adh_series_n,
        swap_is_load_bearing=bool(swap["thinking_is_load_bearing"]),
        harm=harm, closure=closure, assoc=assoc)

    # ---------------------------------------------------------------------------------
    # [7/8] Verdict.
    # ---------------------------------------------------------------------------------
    print("[7/8] assembling verdict, drop-rate disclosure and selection bias ...")
    derivation = derivation_block()
    selection_bias = measure_selection_bias(derivation, all_skits)
    success = sections["success_criteria"]
    table = sections["slots_model_generated_blocks"]["table"]
    own_vs_foreign = sections["slots_model_generated_blocks"]["own_block_vs_foreign_block"]
    n_slots_favouring_think = sum(
        1 for v in sections["slots_reference_block_across_arms"]["tests"].values()
        if v.get("verdict") == "think better")
    n_failure_favouring_think = sum(1 for v in sections["failure_modes"].values()
                                    if v["verdict"] == "think better")
    verdict = sections["verdict_core"]
    if args.max_eval_skits:
        # A truncated run must never be readable as the measurement. The marker goes in the
        # verdict string itself, not only in a field further down that a reader may skip.
        verdict = f"SMOKE ({len(val_skits)} skits) — NOT THE MEASUREMENT: {verdict}"

    report = {
        "verdict": verdict,
        "limitations": build_limitations(eval_set=str(args.skits.relative_to(ROOT)),
                                         n_skits=len(val_skits), derivation=derivation,
                                         selection_bias=selection_bias),
        "power": power,
        "swap_test": swap,
        "swap_test_detail": {"n": n_swap, "divergence_positions": divergence,
                             "note": "token index into the generated continuation; None "
                                     "means identical for the full window"},
        "slots_model_generated_blocks": sections["slots_model_generated_blocks"],
        "slots_reference_block_across_arms": sections["slots_reference_block_across_arms"],
        "failure_modes": sections["failure_modes"],
        "adherence": sections["adherence"],
        "degeneration": sections["degeneration"],
        "contamination": sections["contamination"],
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
        "derivation": derivation,
        "selection_bias": selection_bias,
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
