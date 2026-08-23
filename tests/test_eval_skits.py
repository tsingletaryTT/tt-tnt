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


DERIVATION_FIXTURE = {"stories": 200000, "kept": 18610, "drop_rate": 0.907,
                      "drops_by_rule": {"turn_derivation_failed_other": 147538,
                                        "turn_derivation_failed_dialogue_pattern": 33449,
                                        "too_few_sentences": 403}}
BIAS_FIXTURE = {"units_with_dialogue_corpus": 0.535, "units_with_dialogue_kept": 0.312,
                "units_relative_change": -0.4168,
                "sentences_with_dialogue_corpus": 0.173,
                "turns_with_dialogue_kept": 0.116, "sentences_relative_change": -0.3295}


def test_limitations_name_the_by_construction_caveat_and_the_handback_band():
    lim = build_limitations(eval_set="artifacts/skits-200k/skits.jsonl tail", n_skits=256,
                            derivation=DERIVATION_FIXTURE, selection_bias=BIAS_FIXTURE)
    blob = " ".join(str(v) for v in lim.values()).lower()
    assert "by construction" in blob
    assert "wiring check" in blob
    assert "0.119" in blob and "0.021" in blob
    assert "0.738" in blob
    assert "still falling" in blob, "step 3000 was an inherited budget, not convergence"
    assert "training loss" in blob, "the two arms' losses must be declared incomparable"


def test_limitations_must_disclose_the_drop_rate_and_the_dialogue_selection_bias():
    """The spec makes this mandatory: "warn above 50% — past that the FILTER rather than the
    model is choosing the behaviour, and any result must be reported with that fact
    attached." The rate is 0.907 and eight limitation entries managed not to mention it. The
    measured bias (dialogue-bearing units 53.5% of the corpus -> 31.2% of the kept skits) is
    the specific shape of the filtering, and it is the strongest attack a hostile reader has.
    """
    lim = build_limitations(eval_set="x", n_skits=256, derivation=DERIVATION_FIXTURE,
                            selection_bias=BIAS_FIXTURE)
    blob = " ".join(str(v) for v in lim.values())
    assert "90.7%" in blob, "the drop rate itself must appear, not a hedge about it"
    assert "147,538" in blob and "33,449" in blob, "drops_by_rule counts must appear"
    assert "dialogue" in blob.lower()
    assert "53.5%" in blob and "31.2%" in blob, "the measured unit-level bias must appear"
    assert "17.3%" in blob and "11.6%" in blob, "the sentence-level bias must appear"
    assert "residue" in blob.lower(), (
        "the artifact must say plainly that the eval population is the residue left by the "
        "splitter, not the corpus")


def test_limitations_refuse_to_go_quiet_when_the_derive_manifest_is_missing():
    """A missing manifest must read as disqualifying, not as an absent caveat."""
    lim = build_limitations(eval_set="x", n_skits=256, derivation=None, selection_bias=None)
    assert "disqualifying" in " ".join(str(v) for v in lim.values()).lower()


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
    assert list(block)[0] == "loss_comparability", (
        "the warning must come BEFORE the numbers it protects in key order — this module's "
        "own docstring says so, and the code shipped it last")
    note = block["loss_comparability"].lower()
    assert "do not compare" in note
    assert "0.504" in note and "1.227" in note
    assert "supervise" in note and "template" in note
    # The val curve needs the same protection: adjacent limitation entries quote
    # 1.009 -> 0.639 against 1.580 -> 1.552 with nothing stopping the subtraction.
    assert "val_loss_first" in note and "val_loss_last" in note
    assert "within one arm" in note, (
        "the note must also say which comparison IS legitimate, or the honest use of the "
        "curve (the trajectory within an arm) reads as forbidden too")


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


# ---------------------------------------------------------------------------------------
# FIX 1 — the driver-wiring blind spot.
#
# A code review mutated ONE token in main() — the degeneration direction "lower" -> "higher"
# — and all 27 tests passed while the artifact's top-level verdict changed from PARTIAL to
# "STAGE 2 SUCCESS". Three siblings had the same property: the adherence direction, the
# `>= 2` success gate, and `_context_for`'s next-partner index. The fix is not another
# assertion about arithmetic; it is that the assembly is now a CALLED FUNCTION with a
# fixture behind it, and that both directions are declared in a table the way
# SLOT_DIRECTIONS already was.
# ---------------------------------------------------------------------------------------


def _fixture_skit(story_id, k, ref_block):
    """One synthetic skit. `ref_block` is the GROUND-TRUTH block (what the corpus turn
    would have yielded); the generated block is passed separately, so the fixture can drive
    the arm-quality proxy and the plan-following gap INDEPENDENTLY — which is what makes the
    publication layer testable at all."""
    return {"story_id": story_id,
            "prefix": f"A girl named Mia lived by the river {k}.",
            "turns": [f"Mia found a stone by the river {k}.",
                      f"Her friend laughed at the stone {k}.",
                      f"Mia threw the stone into the river {k}.",
                      f"The water splashed on her friend {k}.",
                      f"Mia laughed with her friend {k}."],
            "blocks": [dict(ref_block) for _ in range(3)]}


def _assembler_fixture():
    """Three skits engineered so that EVERY published claim is pinned by a fixture.

    The generated block and the ground-truth block are deliberately different objects, so
    the two quantities the publication layer depends on move independently:

      context-only gap  = generated block vs THINK turn  -  generated block vs NO-THINK turn
      arm-quality proxy = ground-truth block vs THINK turn - ground-truth block vs NO-THINK turn

    Engineered outcomes, one per claim branch:
      accept  gap +0.889, proxy +0.333 (< half the gap)     -> bounded
      add     gap +1.000, proxy negative (confound inverts) -> headline
      stakes  gap +1.000, proxy +1.000 (>= half the gap)    -> withdrawn
      handback                                              -> not measurable (structural)
      publishable_plan_following_slots == ["accept", "add"]

    The think arm also LOOPS (one sentence three times) while the no-think arm does not, so
    the degeneration and adherence directions stay pinned too. Between skits every turn uses
    different content words, or `assert_control_is_not_the_treatment` would refuse the whole
    fixture — as it did to my first attempt at it.
    """
    from scripts.eval_skits import score_and_assemble

    loops = ["Mia ran to the river and shouted at the cold water.",
             "Mia climbed the tall ladder and shouted at her striped hat.",
             "Mia baked a plum cake and shouted for her cheerful father."]
    gen_add = ["river", "ladder", "cake"]
    quiet = [["Mia found a smooth pebble.", "She showed the pebble to her mother.",
              "They walked home slowly."],
             ["The gate opened with a creak.", "A brown puppy trotted past the pebble.",
              "She counted the daisies."],
             ["The crayon snapped in half.", "A yellow sun dried the pebble.",
              "She started a new picture."]]
    # Ground truth: `water` (only skit 0's think turn has it), `pebble` (only the no-think
    # turns have it), stakes `up` (only the think turns escalate).
    ref_block = {"offer": "mia river", "accept": "water", "add": "pebble",
                 "stakes": "up", "handback": "river"}
    skits = [_fixture_skit(1000 + k, k, ref_block) for k in range(3)]
    gen_blocks = [[{"offer": f"mia river {k}", "accept": "mia", "add": gen_add[k],
                    "stakes": "up", "handback": gen_add[k]} for _ in range(3)]
                  for k in range(3)]
    turns_think = [[loops[k]] * 3 for k in range(3)]
    turns_nothink = quiet
    return score_and_assemble(
        val_skits=skits, turns_think=turns_think, turns_nothink=turns_nothink,
        slots_think=gen_blocks, adherence_think=[1.0, 1.0, 1.0],
        adherence_nothink=[0.0, 0.0, 0.0],
        adherence_series_think=[1.0] * 9, adherence_series_nothink=[0.0] * 9,
        swap_is_load_bearing=True,
        harm=frozenset(["shouted"]), closure=frozenset(["bed"]),
        assoc={"uni": {}, "co": {}, "n": 0})


def test_assembler_reports_a_looping_think_arm_as_worse_not_better():
    """Catches the exact one-token mutation that flipped the published verdict."""
    sections = _assembler_fixture()
    deg = sections["degeneration"]
    assert deg["think_mean_4gram_repeat"] > deg["nothink_mean_4gram_repeat"]
    assert deg["direction"] == "lower"
    assert deg["test"]["verdict"] == "no-think better", (
        "a think arm that repeats itself more must be reported as WORSE on degeneration; "
        "'think better' here means the direction was inverted")
    assert sections["success_criteria"]["degeneration_no_worse_than_the_no_think_arm"] is False
    assert sections["success_criteria"]["all_criteria_met"] is False
    assert sections["verdict_core"].startswith("PARTIAL")


def test_assembler_reports_adherence_in_the_think_arms_favour():
    """The mirror-image mutation: the think arm emits a block every time and the no-think
    arm never does, so "higher" is the only direction that can name the think arm."""
    sections = _assembler_fixture()
    assert sections["adherence"]["direction"] == "higher"
    assert sections["adherence"]["test"]["verdict"] == "think better"


def test_aux_directions_are_declared_the_way_slot_directions_are():
    from scripts.eval_skits import AUX_DIRECTIONS

    assert AUX_DIRECTIONS == {"adherence": "higher", "degeneration": "lower"}


def test_success_gate_needs_two_slots_not_one():
    """`>= 2` is a one-character edit from `>= 1`, and inline in the driver nothing could
    see it."""
    from scripts.eval_skits import evaluate_success_criteria

    good = {"verdict": "think better"}
    dull = {"verdict": "NOT INTERPRETABLE"}
    one = evaluate_success_criteria(
        adherence_think=[0.9, 0.9, 0.9],
        gaps={"accept": 0.60, "add": 0.0, "stakes": 0.0, "handback_anticipation": 0.0},
        contrasts={"accept": good, "add": dull, "stakes": dull,
                   "handback_anticipation": dull},
        degeneration_test={"verdict": "NOT INTERPRETABLE"}, swap_is_load_bearing=True)
    assert one["shuffled_gap_materially_positive_on_at_least_2_of_4_slots"] is False
    two = evaluate_success_criteria(
        adherence_think=[0.9, 0.9, 0.9],
        gaps={"accept": 0.60, "add": 0.50, "stakes": 0.0, "handback_anticipation": 0.0},
        contrasts={"accept": good, "add": good, "stakes": dull,
                   "handback_anticipation": dull},
        degeneration_test={"verdict": "NOT INTERPRETABLE"}, swap_is_load_bearing=True)
    assert two["shuffled_gap_materially_positive_on_at_least_2_of_4_slots"] is True
    assert two["all_criteria_met"] is True


def test_success_gate_fails_on_low_adherence_and_on_a_worse_degeneration_verdict():
    from scripts.eval_skits import evaluate_success_criteria

    good = {"verdict": "think better"}
    base = dict(gaps={"accept": 0.6, "add": 0.5, "stakes": 0.0, "handback_anticipation": 0.0},
                contrasts={"accept": good, "add": good, "stakes": good,
                           "handback_anticipation": good},
                swap_is_load_bearing=True)
    low = evaluate_success_criteria(adherence_think=[0.9, 0.79, 0.9],
                                    degeneration_test={"verdict": "NOT INTERPRETABLE"},
                                    **base)
    assert low["adherence_at_least_0_80_at_every_turn"] is False
    worse = evaluate_success_criteria(adherence_think=[0.9, 0.9, 0.9],
                                      degeneration_test={"verdict": "no-think better"},
                                      **base)
    assert worse["degeneration_no_worse_than_the_no_think_arm"] is False
    assert worse["all_criteria_met"] is False


def test_context_for_names_the_previous_turn_and_the_FOLLOWING_partner_turn():
    """`_context_for` decides which text every slot is scored against. An off-by-one here
    scores `stakes` against the wrong interval and `handback` against the wrong partner
    turn, and no other test in this file would notice."""
    from scripts.eval_skits import _context_for

    skit = {"prefix": "P", "turns": ["t0", "t1", "t2", "t3", "t4"]}
    assert _context_for(skit, 0) == ("P", "t1")
    assert _context_for(skit, 2) == ("t1", "t3")
    assert _context_for(skit, 4) == ("t3", None), (
        "the last model turn has NO following partner turn; anything but None here turns an "
        "undefined observation into a scored one")


# ---------------------------------------------------------------------------------------
# The PUBLICATION layer. Last round's fix pulled the VERDICT wiring into a tested function
# and left the withdrawal/publication machinery — added in the same commit — with no test at
# all: a grep of this file for `decompose_confound`, `CONFOUND_WITHDRAW_SHARE`,
# `publishable`, `slot_value_visibility`, `score_pair`, `ref_hits` or `slots_reference`
# returned zero hits. These are whole DECISION FUNCTIONS, exercised only through a driver,
# and each of the three mutations below silently rewrites a published claim.
# ---------------------------------------------------------------------------------------


def test_fixture_pins_every_published_claim_and_the_publishable_slot_list():
    """Catches swapping `ref_hits_think`/`ref_hits_nothink`: that flips every arm-quality
    proxy's sign, turns `stakes` from withdrawn into HEADLINE, demotes `add`, and makes
    `publishable_plan_following_slots` == ["accept", "add", "stakes"] — re-publishing the
    exact claim the review made us withdraw, with every other number unchanged.
    """
    sec = _assembler_fixture()["slots_model_generated_blocks"]
    claims = {k: v["claim"] for k, v in sec["confound_decomposition"].items()}
    assert claims == {"accept": "bounded", "add": "headline", "stakes": "withdrawn",
                      "handback_anticipation": "not measurable"}, (
        "each branch is engineered by the fixture: accept's proxy is under half its gap, "
        "add's proxy is NEGATIVE, stakes' proxy equals its gap, handback's control is "
        "degenerate")
    assert sec["publishable_plan_following_slots"] == ["accept", "add"], (
        "stakes must NOT appear in the publishable list; if it does, the arm-quality proxy "
        "is being computed with the arms the wrong way round")


def test_confound_withdrawal_share_is_the_declared_half_not_something_looser():
    """`CONFOUND_WITHDRAW_SHARE` 0.5 -> 0.9 moves `stakes` from withdrawn to bounded while
    the adjacent artifact note still reads "stakes is withdrawn (mostly confounded)" — a
    self-contradicting artifact, and nothing in the suite noticed."""
    from scripts.eval_skits import CONFOUND_WITHDRAW_SHARE, decompose_confound

    assert CONFOUND_WITHDRAW_SHARE == 0.5
    gaps = {"accept": 0.10, "add": 0.10, "stakes": 0.0486,
            "handback_anticipation": 0.10}
    arms = {"accept": {"mean_delta": 0.06}, "add": {"mean_delta": 0.02},
            "stakes": {"mean_delta": 0.0312},
            "handback_anticipation": {"mean_delta": 0.0}}
    out = decompose_confound(gaps, arms)
    assert out["accept"]["claim"] == "withdrawn", "0.06 is 60% of 0.10 -> withdrawn"
    assert out["add"]["claim"] == "bounded", "0.02 is 20% of 0.10 -> bounded"
    assert out["stakes"]["claim"] == "withdrawn", (
        "the REAL stakes numbers (+0.0486 gap, +0.0312 proxy = 64%) must come out withdrawn")


def test_a_nonpositive_gap_is_no_signal_and_never_a_headline():
    """Found by the fixture, not by the data: with a negative gap the `proxy < 0` branch
    would have labelled a slot HEADLINE whose own control beat it."""
    from scripts.eval_skits import decompose_confound

    out = decompose_confound({"accept": -0.20}, {"accept": {"mean_delta": -0.30}})
    assert out["accept"]["claim"] == "no signal"


def test_failure_scorers_read_the_preceding_text_then_the_turn_not_the_reverse():
    """Reversing `score_pair(prev, turn, ...)`'s two positional arguments leaves every
    scorer running and every t finite: on the real data `new_harm` falls 3.379 -> 0.895 and
    `groundedness` 3.958 -> 2.083 (both lose significance) while `affordance` goes
    0.0 -> 3.07 and MANUFACTURES a significant "no-think better". Here the think arm's turns
    introduce the only harm word, so the correct order must show harm ARRIVING in them.
    """
    fm = _assembler_fixture()["failure_modes"]
    assert fm["new_harm"]["mean_delta"] == 1.0, (
        "the think arm's turns carry the harm word and the preceding text does not, so "
        "new_harm must fire on every think observation and none of the no-think ones; a "
        "mean delta of 0.0 means prefix and continuation were passed the wrong way round")
    assert fm["new_harm"]["verdict"] == "no-think better", (
        "new_harm is lower-is-better, so the arm that introduces harm must LOSE")


def test_withdrawn_slots_carry_the_marker_inside_their_own_rows():
    """`stakes` was the only slot whose own rows read as an unqualified "think better"
    (t 4.158 context-only, t 6.28 shuffled) while the decomposition withdrew it — the same
    standard handback already got in-row had to apply to the slot we actually withdrew."""
    sec = _assembler_fixture()["slots_model_generated_blocks"]
    for row in (sec["table"]["stakes"], sec["own_block_vs_foreign_block"]["stakes"],
                sec["own_block_vs_context_only_control"]["stakes"]):
        assert row.get("plan_following_claim") == "withdrawn", row
    assert "do_not_read_this_row_as_plan_following" in sec[
        "own_block_vs_context_only_control"]["stakes"]
    # ... and a slot that survived must NOT be stamped, or the marker means nothing.
    assert "plan_following_claim" not in sec["own_block_vs_context_only_control"]["add"]


def test_handback_rows_disclose_that_it_reuses_the_add_word():
    """The third handback degeneracy, and the only empirical one: 97.24% of the real run's
    generated blocks put the identical word in `add` and `handback`. The fixture makes them
    identical in 100% of blocks, so the disclosure must appear."""
    sec = _assembler_fixture()["slots_model_generated_blocks"]
    hb = sec["table"]["handback_anticipation"]
    assert hb["add_and_handback_are_the_same_word_rate"] == 1.0
    assert "duplicates_the_add_slot" in hb
    assert "pooled" in hb["duplicates_the_add_slot"].lower(), (
        "the disclosure must say it also duplicates a component of pooled_slot_accuracy")
    assert sec["add_and_handback_are_the_same_word_rate"] == 1.0


def test_success_gate_names_the_slots_it_counted():
    """The gate counted all four slots while the artifact elsewhere marks two of them
    withdrawn / not measurable. It must at least SAY which ones carried it."""
    sec = _assembler_fixture()
    crit = sec["success_criteria"]
    assert crit["materially_positive_slots"], "the gate must name its slots, not just count"
    restricted = crit["materially_positive_slots_restricted_to_publishable_claims"]
    assert set(restricted["slots"]) <= {"accept", "add"}
    assert restricted["n"] == len(restricted["slots"])
