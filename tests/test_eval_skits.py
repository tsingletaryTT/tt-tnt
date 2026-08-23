# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Statistics and wiring tests for the stage-2 (skits) evaluation.

Two classes of test live here, and the second class is the one that matters.

  * The arithmetic tests (does `shuffled_gap` subtract, does `required_n` grow) are cheap
    and would pass against almost any implementation.
  * The WIRING tests are the ones written to fail against a plausible wrong
    implementation: importing stage 1's looser Bonferroni threshold, inverting a scorer's
    direction, scoring the think-block against text that includes the block itself, or
    dropping only one half of a paired observation and silently misaligning the two arms.

Stage 1's lesson was that a metric reported without its floor says nothing, so the floors
and the majority-class baseline are pinned here too: a `stakes` accuracy of 0.80 is BELOW
the 0.738 chance line by only 6 points, and a reader who assumed 0.5 would call it a
triumph.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_skits import (
    BONFERRONI_ALPHA,
    CORPUS_CEILINGS,
    CORPUS_FLOORS,
    CRITICAL_T,
    N_TESTS,
    SLOT_DIRECTIONS,
    adherence_by_turn,
    build_limitations,
    critical_t_for,
    four_gram_repeat_rate,
    paired_defined,
    paired_verdict,
    required_n,
    shuffled_gap,
    slot_table,
    split_generation,
)

# ---------------------------------------------------------------------------------------
# Multiple comparisons. Eleven tests, NOT stage 1's five.
# ---------------------------------------------------------------------------------------


def test_bonferroni_threshold_reflects_eleven_tests():
    """Four slots + pooled accuracy + four failure scorers + adherence + degeneration."""
    assert N_TESTS == 11
    assert abs(BONFERRONI_ALPHA - 0.05 / 11) < 1e-9
    assert CRITICAL_T > 2.8, "alpha 0.0045 two-sided needs |t| above ~2.84"


def test_critical_t_is_the_normal_quantile_of_this_modules_own_alpha():
    """The constant must be DERIVED from BONFERRONI_ALPHA, not carried over from another
    stage. Stage 1's pair (alpha 0.01, t 2.576) satisfies "t > 2.5" and would sail through
    a looser assertion while applying a threshold this eval never chose."""
    exact = critical_t_for(BONFERRONI_ALPHA)
    assert abs(CRITICAL_T - exact) < 0.01, f"{CRITICAL_T} is not this alpha's quantile {exact}"
    assert CRITICAL_T >= exact, "the threshold must never sit BELOW its own alpha's quantile"
    assert abs(critical_t_for(0.01) - 2.576) < 0.005, "the helper itself must be right"


def test_a_t_between_stage_ones_threshold_and_ours_is_not_significant_here():
    """The one test that catches `from scripts.eval_improv import paired_verdict`.

    Stage 1's module-level CRITICAL_T is 2.576. A series engineered to sit at |t| ~ 2.7
    is significant under stage 1's threshold and NOT significant under this eval's. If
    this eval ever borrows stage 1's function (which closes over stage 1's constant), the
    verdict below flips to "think better" and a null becomes a finding.
    """
    # 400 paired deltas: 227 of +1.0 and 173 of -1.0 -> mean 0.135, pstdev 0.99085,
    # se 0.049543, |t| = 2.725 — above stage 1's 2.576, below this eval's 2.843.
    a = [1.0] * 227 + [0.0] * 173
    b = [0.0] * 227 + [1.0] * 173
    v = paired_verdict(a, b, "higher")
    assert 2.576 < v["t"] < 2.843, f"fixture must sit between the thresholds, got {v['t']}"
    assert v["verdict"] == "NOT INTERPRETABLE", (
        "a |t| of %.3f clears stage 1's 2.576 but not this eval's 2.843; reporting it as "
        "significant means the wrong alpha is in force" % v["t"])


def test_direction_flips_which_arm_a_significant_verdict_names():
    """Every slot has a signed expectation. Passing the wrong one inverts the result."""
    a = [1.0] * 40
    b = [0.0] * 40
    assert paired_verdict(a, b, "higher")["verdict"] == "think better"
    assert paired_verdict(a, b, "lower")["verdict"] == "no-think better"


def test_every_slot_declares_a_direction_and_all_four_are_higher_is_better():
    """A slot's score is a HIT RATE: more hits is better for all four. Stated explicitly
    so `paired_verdict`'s required `direction` argument can never be guessed at a call
    site (that guess is exactly how stage 1 shipped an inverted label)."""
    assert SLOT_DIRECTIONS == {"accept": "higher", "add": "higher", "stakes": "higher",
                               "handback_anticipation": "higher"}


def test_zero_scatter_with_a_nonzero_mean_is_a_perfect_separation_not_noise():
    """Carried over from stage 1's CONTROLLER RULING: identical non-zero paired deltas are
    the strongest signal available, not the weakest."""
    v = paired_verdict([1.0] * 10, [0.0] * 10, "higher")
    assert v["t"] == float("inf")
    assert v["verdict"] == "think better"
    same = paired_verdict([1.0] * 10, [1.0] * 10, "higher")
    assert same["verdict"] == "NOT INTERPRETABLE"


# ---------------------------------------------------------------------------------------
# Power.
# ---------------------------------------------------------------------------------------


def test_required_n_grows_when_alpha_gets_stricter():
    """The reviewer's ~370 turns was computed at alpha 0.0125 (four tests). This eval runs
    eleven, so its alpha is stricter and its requirement MUST be larger. Reusing the
    quoted number would understate what the design needs."""
    loose = required_n(0.738, 0.10, alpha=0.05 / 4)
    strict = required_n(0.738, 0.10, alpha=BONFERRONI_ALPHA)
    assert strict > loose, f"stricter alpha must need more samples: {strict} vs {loose}"
    assert 400 < strict < 600, f"sanity: ~450 per arm expected at this alpha, got {strict}"


def test_required_n_falls_as_the_effect_being_sought_grows():
    assert required_n(0.738, 0.20, alpha=BONFERRONI_ALPHA) < required_n(
        0.738, 0.10, alpha=BONFERRONI_ALPHA)


def test_stakes_power_is_computed_against_the_majority_class_not_a_coin_flip():
    """`stakes` chance is 0.738 ("level" is the majority class), NOT 0.5. The headroom is
    26.2 points, and a power calculation anchored at 0.5 would size the experiment against
    an effect that cannot exist."""
    assert CORPUS_FLOORS["stakes"] == 0.738
    assert required_n(0.738, 0.10, alpha=BONFERRONI_ALPHA) != required_n(
        0.5, 0.10, alpha=BONFERRONI_ALPHA)


# ---------------------------------------------------------------------------------------
# Adherence, per turn position.
# ---------------------------------------------------------------------------------------


def test_adherence_is_reported_per_turn_position_not_pooled():
    """A pooled number hides a model that emits a good block for turn 1 and collapses by
    turn 3, which is the specific failure five turns invites."""
    gens = [["<think>\noffer: a\naccept: b\nadd: c\nstakes: up\nhandback: d\n</think>x"] * 1
            + ["no block here"] * 1 + ["garbage"] * 1]
    per = adherence_by_turn(gens, n_turns=3)
    assert per[0] == 1.0
    assert per[1] == 0.0
    assert per[2] == 0.0


def test_adherence_per_turn_is_not_the_pooled_rate_in_disguise():
    """Two skits, block present only at turn 0 in one of them. The pooled rate is 0.5 at
    every position; the per-position rates are 1.0, 0.5, 0.0. An implementation that
    pooled and repeated the same number three times would pass the test above (all-or-
    nothing columns) but must fail here."""
    good = "<think>\noffer: a\naccept: b\nadd: c\nstakes: up\nhandback: d\n</think>x"
    gens = [[good, good, "nope"], [good, "nope", "nope"]]
    assert adherence_by_turn(gens, n_turns=3) == [1.0, 0.5, 0.0]


# ---------------------------------------------------------------------------------------
# The shuffled-slot control.
# ---------------------------------------------------------------------------------------


def test_shuffled_gap_is_zero_when_the_control_matches_the_real_run():
    """If substituting another skit's slot values changes nothing, the model is producing
    scaffolding rather than plans — the gap is the whole signal."""
    assert shuffled_gap({"accept": 0.8}, {"accept": 0.8})["accept"] == 0.0
    assert shuffled_gap({"accept": 0.8}, {"accept": 0.3})["accept"] > 0.4


def test_shuffled_gap_handles_undefined_slots():
    assert shuffled_gap({"handback_anticipation": None},
                        {"handback_anticipation": 0.4})["handback_anticipation"] is None


def test_slot_table_prints_the_floor_beside_every_matched_rate():
    """Stage 1's mistake was a metric without its floor. Every row must carry BOTH the
    run's own shuffled control and the corpus floor, and for `handback_anticipation` the
    corpus CEILING as well — a null there means "inside a 10-point band", not "no effect".
    """
    rows = slot_table({"accept": 0.9, "add": 0.5, "stakes": 0.8,
                       "handback_anticipation": 0.05},
                      {"accept": 0.1, "add": 0.0, "stakes": 0.74,
                       "handback_anticipation": 0.02})
    for slot in ("accept", "add", "stakes", "handback_anticipation"):
        assert rows[slot]["matched"] is not None
        assert rows[slot]["shuffled_floor"] is not None
        assert rows[slot]["corpus_shuffled_floor"] == CORPUS_FLOORS[slot]
    assert rows["stakes"]["headroom_above_chance"] == round(1 - 0.738, 4)
    assert rows["handback_anticipation"]["corpus_ceiling"] == CORPUS_CEILINGS[
        "handback_anticipation"]
    assert rows["handback_anticipation"]["max_detectable_band"] == round(0.119 - 0.021, 4)


def test_limitations_name_the_by_construction_caveat_and_the_handback_band():
    lim = build_limitations(eval_set="artifacts/skits-200k/skits.jsonl tail", n_skits=256)
    blob = " ".join(str(v) for v in lim.values()).lower()
    assert "by construction" in blob
    assert "wiring check" in blob
    assert "0.119" in blob and "0.021" in blob
    assert "0.738" in blob
    assert "still falling" in blob, "step 3000 was an inherited budget, not convergence"
    assert "training loss" in blob, "the two arms' losses must be declared incomparable"


# ---------------------------------------------------------------------------------------
# Generation plumbing.
# ---------------------------------------------------------------------------------------


def test_split_generation_scores_the_turn_after_the_block_not_the_block_itself():
    """The think-block's `add` slot literally contains the word it predicts. If the turn
    handed to the scorer still contains the block text, every slot hits trivially and the
    whole measurement is a tautology.
    """
    text = ("<think>\noffer: rock friend\naccept: rock\nadd: windowsill\nstakes: level\n"
            "handback: light\n</think>\nShe put it down. Then she left.")
    slots, block_text, turn = split_generation(text, with_think=True)
    assert slots is not None and slots.add == "windowsill"
    assert "windowsill" not in turn, "the block must not leak into the scored turn"
    assert "<think>" not in turn and "</think>" not in turn
    assert turn == "She put it down.", "only the first sentence is the model's turn"
    assert block_text.startswith("<think>") and block_text.rstrip().endswith("</think>")


def test_split_generation_keeps_the_blocks_trailing_newline_for_context_feedback():
    """`train.improv.render_think` ends the think segment with a newline and the turn
    follows immediately. The eval extends the context by re-encoding these as separate
    segments, exactly as training built them, so a reconstruction that drops the newline
    puts `</think>` and the turn's first token adjacent at a seam training never saw — the
    same class of out-of-distribution seam stage 1 found at its `<think>` boundary.
    """
    text = ("<think>\noffer: a\naccept: b\nadd: c\nstakes: level\nhandback: d\n</think>\n"
            "She put it down.")
    _slots, block_text, _turn = split_generation(text, with_think=True)
    assert block_text.endswith("</think>\n")


def test_split_generation_reports_a_malformed_block_as_unparsed_but_still_yields_a_turn():
    """A missing block is an adherence miss, not a crash, and the turn it wrote is still
    real text that the failure-mode scorers must be able to see."""
    slots, block_text, turn = split_generation("She put it down. Then she left.",
                                               with_think=True)
    assert slots is None and block_text == ""
    assert turn == "She put it down."


def test_split_generation_for_the_nothink_arm_takes_the_first_sentence_only():
    slots, block_text, turn = split_generation("She put it down. Then she left.",
                                               with_think=False)
    assert slots is None and block_text == ""
    assert turn == "She put it down."


def test_four_gram_repeat_rate_separates_a_loop_from_ordinary_prose():
    """Stage 1 saw `cold cold cold`. A degeneration metric that cannot tell that apart
    from a normal turn measures nothing."""
    loop = four_gram_repeat_rate(["it was cold it was cold it was cold it was cold"])
    prose = four_gram_repeat_rate(["Mia painted a bright picture and showed it to her mom"])
    assert loop > 0.4, f"a four-fold loop must read as degenerate, got {loop}"
    assert prose == 0.0, f"non-repeating prose must read as clean, got {prose}"


def test_four_gram_repeat_rate_is_computed_across_turns_not_within_one():
    """A model that writes the SAME turn three times is degenerate even though each turn
    on its own repeats nothing. Scoring turns independently and averaging would report
    0.0 here, which is the failure the metric exists to catch."""
    turn = "She put the rock on the windowsill and smiled"
    assert four_gram_repeat_rate([turn, turn, turn]) > 0.5


# ---------------------------------------------------------------------------------------
# Pairing.
# ---------------------------------------------------------------------------------------


def test_paired_defined_drops_the_whole_pair_when_either_side_is_undefined():
    """`handback_anticipation` is None on the final model turn. Dropping only the
    undefined SIDE leaves the two arms with different lengths — or, worse, equal lengths
    made of observations from different skits, which pairs arm A's turn 1 against arm B's
    turn 2 and quietly destroys the pairing the t-test depends on.
    """
    a = [1.0, None, 0.0, 1.0]
    b = [0.0, 1.0, None, 0.0]
    xs, ys = paired_defined(a, b)
    assert xs == [1.0, 1.0] and ys == [0.0, 0.0]
    assert len(xs) == len(ys) == 2


def test_paired_defined_preserves_order_so_index_i_is_the_same_observation():
    a = [1.0, 0.0, None, 1.0, 0.0]
    b = [0.0, 0.0, 1.0, 1.0, None]
    xs, ys = paired_defined(a, b)
    assert xs == [1.0, 0.0, 1.0]
    assert ys == [0.0, 0.0, 1.0]


def test_manifests_carry_the_loss_incomparability_note_beside_the_losses():
    """The two arms' `loss_end` (0.504 vs 1.227) is the single most inviting wrong
    subtraction in this project: the manifests print `paired_with` right next to it, and
    the arms supervise different token sets (~74% of the think arm's supervised positions
    are the fixed template). The warning must ride WITH the manifests in the emitted
    artifact, not live in a section a reader who scrolled to the losses never reaches.
    """
    from scripts.eval_skits import manifests_block

    block = manifests_block({"loss_end": 0.504, "paired_with": "nothink"},
                            {"loss_end": 1.227, "paired_with": "think"})
    assert set(block) == {"think", "nothink", "loss_comparability"}
    note = block["loss_comparability"].lower()
    assert "do not compare" in note
    assert "0.504" in note and "1.227" in note
    assert "supervise" in note and "template" in note


def test_slot_table_carries_the_context_only_floor_as_a_separate_control():
    """Two floors, because they rule out different things.

    The shuffled floor scores the block against ANOTHER skit's turn, so it breaks the
    block-to-turn link AND the shared-context link at once — part of any gap it shows is
    merely topical. The context-only floor scores the same block against the NO-THINK
    arm's turn for the SAME skit and position: same scene, same partner turns, but a turn
    written by a model that never saw a block. A scorer driven purely by context overlap
    would score just as high there, which is the one way a healthy-looking `matched` rate
    could still mean nothing.
    """
    rows = slot_table({"accept": 0.9, "add": 0.5, "stakes": 0.8,
                       "handback_anticipation": 0.05},
                      {"accept": 0.1, "add": 0.0, "stakes": 0.74,
                       "handback_anticipation": 0.02},
                      {"accept": 0.6, "add": 0.05, "stakes": 0.79,
                       "handback_anticipation": 0.04})
    assert rows["accept"]["context_only_floor"] == 0.6
    assert rows["accept"]["gap_over_context_only"] == round(0.9 - 0.6, 4)
    assert rows["accept"]["shuffled_floor"] == 0.1
    # The two floors must not be the same number by construction — an implementation that
    # reused the shuffled floor for both would report a gap of 0.8 here.
    assert rows["accept"]["gap_over_context_only"] != rows["accept"]["gap_over_shuffled"]


def test_slot_table_still_works_without_a_context_only_control():
    rows = slot_table({"accept": 0.9}, {"accept": 0.1})
    assert "context_only_floor" not in rows["accept"]


def test_a_control_identical_to_the_treatment_is_refused_not_reported():
    """A mutation test found this hole: pointing the context-only control at the think
    arm's OWN turns is a one-word edit that no unit test could see, because the wiring
    lives in the driver. It produces a control identical to the treatment, a gap of exactly
    zero on every slot, and a confident "no effect". The guard goes at the layer that
    fails.
    """
    import pytest

    from scripts.eval_skits import SlotHits, assert_control_is_not_the_treatment

    treatment = [SlotHits(True, True, True, None), SlotHits(False, True, False, True)]
    with pytest.raises(RuntimeError, match="IDENTICALLY"):
        assert_control_is_not_the_treatment(treatment, list(treatment), name="ctl")
    # A genuinely different control passes, including one that differs on a single slot of
    # a single observation — the guard must not be a blunt "controls must differ a lot".
    almost = [SlotHits(True, True, True, None), SlotHits(False, True, True, True)]
    assert_control_is_not_the_treatment(treatment, almost, name="ctl")
    with pytest.raises(RuntimeError, match="length mismatch"):
        assert_control_is_not_the_treatment(treatment, almost[:1], name="ctl")
