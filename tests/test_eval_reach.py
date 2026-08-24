# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the reach-dial EUREKA measurement (`scripts/eval_reach.py`).

This project has shipped TEN tests that passed against both correct and incorrect code, and
on the day this file was written a stale `.pyc` served a mutant through four results. So the
rules here are not stylistic:

  * every DECISION function that a published claim passes through is named, imported, called
    with a fixture, and tested against a plausible WRONG implementation -- above all
    `monotonicity_verdict`, which decides the headline;
  * two failure shapes are mandatory and have their own tests: an INVERTED comparison
    (`test_a_decreasing_series_is_not_a_monotone_dial`,
    `test_a_large_effect_in_the_wrong_direction_is_not_significant`) and the WRONG ALPHA
    (`test_a_t_between_stage_ones_threshold_and_ours_is_not_significant_here`);
  * anything that is a property of SCALE -- the `add` vocabulary's concentration, the
    tokenization seam -- is tested against the REAL artifact, because a fixture cannot have
    the property;
  * the instrument is tested as hard as the subject: `collect_targeted_association` is proved
    equal to `train.reach.build_association` on a corpus small enough to build both ways.

Run with `PYTHONPYCACHEPREFIX=...` (or clear `__pycache__`) when mutating the source.
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_reach import (  # noqa: E402
    ADHERENCE_MARGIN,
    BONFERRONI_ALPHA,
    COHERENCE_MARGIN,
    CONDITIONS,
    CRITICAL_T,
    DEFAULT_SKITS,
    DERIVE_MANIFEST,
    MEASURED_BLOCK,
    N_TESTS,
    NONSENSE_VALUE,
    TOKENIZER_DIR,
    add_word_of_value,
    adherence_guard,
    adherence_readings,
    coherence_guard,
    collect_targeted_association,
    condition_key,
    content_word_repeat_rate,
    critical_t_for,
    df_matched_pairs,
    dial_or_frequency_dial,
    encode_segments,
    eureka_verdict,
    forced_block_prefix,
    gold_reproduction,
    load_rows,
    monotonicity_verdict,
    needed_lookups,
    nonsense_control_verdict,
    ols_residuals,
    paired_contrast,
    paired_t,
    parse_forced_generation,
    particle_profile,
    prompt_segments,
    reach_distance_loo,
    training_example_ids,
)
from train.improv import content_words, render_think  # noqa: E402
from train.reach import (Association, add_word_of, build_association,  # noqa: E402
                         reach_distance, ReachSlots)

# ---------------------------------------------------------------------------------------
# Fixture series. Built so the assertions CAN fail: three fixtures in this project were
# vacuous because one value repeated through every row.
# ---------------------------------------------------------------------------------------
#: A genuinely monotone paired triple with real per-scene scatter (so pairing does work) and
#: a real between-setting step (so the contrast is not zero).
_SCENE_NOISE = [0.05, -0.04, 0.02, -0.01, 0.03, -0.03, 0.01, 0.00, 0.04, -0.02,
                0.06, -0.05, 0.02, -0.02, 0.01, 0.03, -0.01, 0.00, -0.03, 0.05]
NEAR = [0.40 + z for z in _SCENE_NOISE]
MID = [0.50 + z for z in _SCENE_NOISE]
FAR = [0.60 + z for z in _SCENE_NOISE]


def _t_of(diff: float, n: int = 20, sd: float = 1.0) -> tuple:
    """A paired pair of series whose t is (approximately) a chosen value.

    Used to place a t deliberately BETWEEN stage 1's threshold and this eval's, which is the
    band that manufactured two significant results in stage 2.
    """
    # differences with mean `diff` and stdev `sd`: take a symmetric two-point set.
    half = n // 2
    diffs = [diff + sd] * half + [diff - sd] * half
    a = [0.5] * n
    b = [0.5 + d for d in diffs]
    return a, b


# ---------------------------------------------------------------------------------------
# The thresholds. THE WRONG-ALPHA TESTS.
# ---------------------------------------------------------------------------------------
def test_the_family_size_and_alpha_are_the_spec_declared_ones():
    assert N_TESTS == 11
    assert BONFERRONI_ALPHA == pytest.approx(0.05 / 11)


def test_critical_t_is_derived_from_this_modules_own_alpha_and_never_looser():
    """Both halves matter. The first pins the value; the second catches an ANTI-CONSERVATIVE
    drift -- someone pasting stage 1's 2.576 or stage 2's constant back in -- which the first
    assertion alone would not, since 2.576 is also 'about right' for a different alpha."""
    exact = critical_t_for(BONFERRONI_ALPHA)
    assert CRITICAL_T == pytest.approx(exact, abs=0.01)
    assert CRITICAL_T >= exact, "CRITICAL_T must never be below the quantile at our alpha"
    assert CRITICAL_T > critical_t_for(0.01), "this is not stage 1's 2.576"


def test_a_t_between_stage_ones_threshold_and_ours_is_not_significant_here():
    """THE WRONG-ALPHA TEST. In stage 2 two effects landed between stage 1's 2.576 and the
    correct 2.843; importing the constant would have manufactured both. A t of ~2.70 is
    significant at stage 1's threshold and must NOT be significant here."""
    a, b = _t_of(0.60, n=20, sd=1.0)          # mean 0.60, sd ~1.026 -> t ~ 2.61..2.70
    got = paired_contrast(a, b, direction="b_greater", label="band")
    assert 2.576 < abs(got["t"]) < CRITICAL_T, f"fixture t={got['t']} is not in the band"
    assert got["significant"] is False
    assert got["in_the_declared_direction"] is True


# ---------------------------------------------------------------------------------------
# paired_t / paired_contrast. THE INVERTED-COMPARISON TESTS.
# ---------------------------------------------------------------------------------------
def test_paired_t_is_none_for_zero_scatter_rather_than_infinite():
    assert paired_t([0.1, 0.1, 0.1, 0.1]) is None
    assert paired_t([0.1]) is None
    assert paired_t([0.1, 0.2, 0.3]) is not None


def test_paired_contrast_requires_an_explicit_direction():
    with pytest.raises(ValueError):
        paired_contrast(NEAR, FAR, direction="up", label="x")


def test_paired_contrast_refuses_unpaired_input():
    with pytest.raises(ValueError):
        paired_contrast(NEAR, FAR[:-1], direction="b_greater", label="x")


def test_a_large_effect_in_the_wrong_direction_is_not_significant():
    """THE INVERTED-COMPARISON TEST, at the contrast level. |t| is enormous and points the
    wrong way: that is a significant REFUTATION, and `significant` must be False."""
    got = paired_contrast(FAR, NEAR, direction="b_greater", label="backwards")
    assert abs(got["t"]) > 10
    assert got["in_the_declared_direction"] is False
    assert got["significant"] is False


# ---------------------------------------------------------------------------------------
# monotonicity_verdict -- THE function that decides the headline.
# ---------------------------------------------------------------------------------------
def test_a_monotone_series_is_a_monotone_dial():
    v = monotonicity_verdict(NEAR, MID, FAR, label="fixture")
    assert v["monotone"] is True
    assert v["n_significant_steps"] == 3
    for step in ("near_lt_mid", "mid_lt_far", "near_lt_far"):
        assert v[step]["significant"] is True
        assert v[step]["mean_delta"] > 0


def test_a_decreasing_series_is_not_a_monotone_dial():
    """THE INVERTED-COMPARISON TEST, at the verdict level. Distance is 1 - max NPMI, so a
    farther reach is a LARGER number; an implementation that compared the wrong way round
    would call this perfectly-reversed series a working dial."""
    v = monotonicity_verdict(FAR, MID, NEAR, label="reversed")
    assert v["monotone"] is False
    assert v["n_significant_steps"] == 0


def test_a_flat_series_is_not_a_monotone_dial():
    v = monotonicity_verdict(NEAR, NEAR, NEAR, label="flat")
    assert v["monotone"] is False
    assert v["n_significant_steps"] == 0


def test_two_steps_are_not_enough_for_a_monotone_dial():
    """near<mid and mid<far can both be null while near<far is significant, and vice versa.
    All three are required, so a partial pattern must not read as a dial."""
    v = monotonicity_verdict(NEAR, NEAR, FAR, label="partial")
    assert v["near_lt_mid"]["significant"] is False
    assert v["mid_lt_far"]["significant"] is True
    assert v["monotone"] is False


def test_the_verdict_names_mid_vs_far_as_the_cleaner_contrast():
    """Spec amendment 8fb43b4: the corpus-scale frequency confound is NOT monotone (`mid` is
    the commonest bucket), so the three steps are not equally exposed and the artifact must
    say which one is cleaner."""
    v = monotonicity_verdict(NEAR, MID, FAR, label="fixture")
    assert v["cleaner_contrast"] == "mid_lt_far"
    assert "NOT monotone" in v["cleaner_contrast_why"]


# ---------------------------------------------------------------------------------------
# The guards.
# ---------------------------------------------------------------------------------------
def test_the_coherence_margin_is_the_pre_declared_one():
    assert COHERENCE_MARGIN == 0.05


def test_coherence_guard_passes_at_the_margin_and_fails_just_past_it():
    near = [0.30] * 10
    assert coherence_guard(near, [0.25] * 10)["passes"] is True          # drop exactly 0.05
    assert coherence_guard(near, [0.2499] * 10)["passes"] is False       # 0.0501
    assert coherence_guard(near, [0.40] * 10)["passes"] is True          # a RISE passes


def test_coherence_guard_reports_the_drop_it_measured():
    g = coherence_guard([0.30] * 4, [0.10] * 4)
    assert g["drop_far_below_near"] == pytest.approx(0.20)
    assert g["passes"] is False


def test_adherence_guard_catches_a_setting_that_stops_fulfilling_the_plan():
    assert adherence_guard({"near": 0.80, "mid": 0.79, "far": 0.78})["passes"] is True
    bad = adherence_guard({"near": 0.80, "mid": 0.79, "far": 0.60})
    assert bad["passes"] is False
    assert bad["worst_setting"] == "far"
    assert bad["shortfall"] == pytest.approx(0.20)
    assert ADHERENCE_MARGIN == 0.05


def test_adherence_readings_keeps_the_declared_gate_when_the_readings_disagree():
    """THE REAL CASE, as a fixture. This run's `add` hit rate is non-monotone -- near 0.477,
    mid 0.567, far 0.507 -- so the declared worst-vs-best gate FAILS while the spec's own
    narrower wording (`far` vs `near`) PASSES, because `far` is above `near`.

    The gate must stay worst-vs-best. A mutant that reads `far` vs `near` as the gate would
    flip this project's published headline from 'not met' to 'met', which is exactly why the
    gate is not chosen after seeing the data.
    """
    got = adherence_readings({"n": 0.476998, "m": 0.566586, "f": 0.507264}, "n", "m", "f")
    assert got["passes"] is False, "the DECLARED gate must still fail"
    assert got["worst_setting"] == "n"
    assert got["shortfall"] == pytest.approx(0.089588, abs=1e-6)
    assert got["reading_far_minus_near"] == pytest.approx(0.030266, abs=1e-6)
    assert got["reading_far_minus_near_passes"] is True
    assert got["reading_far_minus_best_of_near_mid"] == pytest.approx(-0.059322, abs=1e-6)
    assert got["reading_far_minus_best_passes"] is False


def test_adherence_readings_all_agree_when_far_simply_collapses():
    """The failure mode the guard was WRITTEN for: the plan gets ambitious and the model stops
    fulfilling it. All three readings must fail together there, or the extra readings would be
    a way to explain away a real collapse."""
    got = adherence_readings({"n": 0.80, "m": 0.79, "f": 0.40}, "n", "m", "f")
    assert got["passes"] is False
    assert got["reading_far_minus_near_passes"] is False
    assert got["reading_far_minus_best_passes"] is False
    assert got["worst_setting"] == "f"


def test_adherence_readings_all_pass_when_the_rate_really_holds():
    got = adherence_readings({"n": 0.80, "m": 0.79, "f": 0.78}, "n", "m", "f")
    assert got["passes"] is True
    assert got["reading_far_minus_near_passes"] is True
    assert got["reading_far_minus_best_passes"] is True


# ---------------------------------------------------------------------------------------
# dial_or_frequency_dial -- the spec's "publish it as a frequency dial" rule.
# ---------------------------------------------------------------------------------------
def _verdict(monotone: bool, n: int = 3) -> dict:
    return {"monotone": monotone, "n_significant_steps": n if monotone else 1}


def test_a_dial_that_moves_raw_but_not_controlled_distance_is_published_as_a_frequency_dial():
    got = dial_or_frequency_dial(_verdict(True), _verdict(False))
    assert got["verdict"] == "FREQUENCY DIAL"


def test_a_dial_that_survives_frequency_control_is_a_reach_dial():
    assert dial_or_frequency_dial(_verdict(True), _verdict(True))["verdict"] == "REACH DIAL"


def test_no_movement_at_all_is_no_dial():
    assert dial_or_frequency_dial(_verdict(False), _verdict(False))["verdict"] == "NO DIAL"


def _matched(near_mid: bool, mid_far: bool, near_far: bool) -> dict:
    def step(sig):
        return {"mean_delta": 0.01, "t": 3.5 if sig else 2.0, "significant": sig,
                "n_pairs": 300}
    return {"near_lt_mid": step(near_mid), "mid_lt_far": step(mid_far),
            "near_lt_far": step(near_far),
            "all_three_significant": near_mid and mid_far and near_far}


def test_the_df_matched_subsample_is_reported_but_does_not_change_the_verdict():
    """The second frequency control disagreed with the gate in this run (near<mid failed under
    matching). The disagreement must be PUBLISHED and must NOT silently redecide the verdict:
    the gate was fixed before the data existed."""
    got = dial_or_frequency_dial(_verdict(True), _verdict(True),
                                 _matched(False, True, True))
    assert got["verdict"] == "REACH DIAL", "the gate is the residualised control"
    blk = got["second_frequency_control_df_matched_subsample"]
    assert blk["steps_significant"] == 2
    assert blk["all_three_significant"] is False
    assert blk["agrees_with_the_gate"] is False
    assert "do NOT agree" in blk["READ_THIS"]


def test_the_matched_report_says_so_when_the_two_controls_agree():
    got = dial_or_frequency_dial(_verdict(True), _verdict(True), _matched(True, True, True))
    blk = got["second_frequency_control_df_matched_subsample"]
    assert blk["agrees_with_the_gate"] is True
    assert blk["steps_significant"] == 3


def test_dial_or_frequency_dial_still_works_without_the_matched_report():
    got = dial_or_frequency_dial(_verdict(True), _verdict(False))
    assert got["verdict"] == "FREQUENCY DIAL"
    assert "second_frequency_control_df_matched_subsample" not in got


def test_controlled_without_raw_is_reported_as_partial_not_as_a_dial():
    got = dial_or_frequency_dial(_verdict(False), _verdict(True))
    assert got["verdict"].startswith("PARTIAL")
    assert got["verdict"] != "REACH DIAL"


# ---------------------------------------------------------------------------------------
# nonsense_control_verdict -- THE PRIMARY CONTROL's decision function.
# ---------------------------------------------------------------------------------------
def test_a_nonsense_value_that_lands_on_far_reproduces_the_dial_and_sinks_the_headline():
    """If forcing an off-vocabulary token gets you the same distance `far` does, the effect is
    'any token in the reach line', not the dial's VALUE."""
    got = nonsense_control_verdict(NEAR, FAR, FAR)
    assert got["moved_away_from_near"] is True
    assert got["indistinguishable_from_far"] is True
    assert got["reproduces_the_dial_pattern"] is True


def test_a_nonsense_value_that_stays_at_near_does_not_reproduce_the_dial():
    got = nonsense_control_verdict(NEAR, FAR, NEAR)
    assert got["moved_away_from_near"] is False
    assert got["reproduces_the_dial_pattern"] is False


def test_a_nonsense_value_that_is_its_own_third_thing_does_not_reproduce_the_dial():
    """Away from `near` AND distinguishable from `far` is what a value-sensitive model should
    do with a word it never saw in that slot. Both halves of the rule are load-bearing."""
    got = nonsense_control_verdict(NEAR, FAR, MID)
    assert got["moved_away_from_near"] is True
    assert got["indistinguishable_from_far"] is False
    assert got["reproduces_the_dial_pattern"] is False


# ---------------------------------------------------------------------------------------
# eureka_verdict -- the headline, gate by gate.
# ---------------------------------------------------------------------------------------
def _gates(**over) -> dict:
    base = dict(dial_kind={"verdict": "REACH DIAL"},
                coherence={"passes": True, "drop_far_below_near": 0.0,
                           "declared_margin": COHERENCE_MARGIN},
                adherence={"passes": True, "shortfall": 0.0,
                           "declared_margin": ADHERENCE_MARGIN},
                nonsense={"reproduces_the_dial_pattern": False},
                nodial={"monotone": False})
    base.update(over)
    return base


def test_eureka_is_met_when_every_gate_passes():
    assert eureka_verdict(**_gates())["eureka_criterion_met"] is True


@pytest.mark.parametrize("override,expect_in_reason", [
    ({"dial_kind": {"verdict": "FREQUENCY DIAL"}}, "FREQUENCY DIAL"),
    ({"coherence": {"passes": False, "drop_far_below_near": 0.2,
                    "declared_margin": 0.05}}, "coherence"),
    ({"adherence": {"passes": False, "shortfall": 0.3,
                    "declared_margin": 0.05}}, "slot-hit"),
    ({"nonsense": {"reproduces_the_dial_pattern": True}}, "PRIMARY CONTROL"),
])
def test_every_gate_alone_can_refuse_eureka(override, expect_in_reason):
    got = eureka_verdict(**_gates(**override))
    assert got["eureka_criterion_met"] is False
    assert any(expect_in_reason in r for r in got["reasons_against"])


def test_the_nodial_arm_is_reported_but_is_not_a_gate():
    """`nodial` is off-schema in two ways at once (unknown slot name + extra line), so
    movement there cannot be attributed to the dial's VALUE. It must not be able to refuse a
    headline that every real gate passed."""
    got = eureka_verdict(**_gates(nodial={"monotone": True}))
    assert got["eureka_criterion_met"] is True
    assert got["secondary_control_nodial_monotone"] is True


# ---------------------------------------------------------------------------------------
# The frequency control.
# ---------------------------------------------------------------------------------------
def test_ols_residuals_remove_a_planted_frequency_effect():
    xs = [math.log(10 + 7 * i) for i in range(40)]
    ys = [3.0 + 2.0 * x + (0.01 if i % 2 else -0.01) for i, x in enumerate(xs)]
    res = ols_residuals(ys, xs)
    assert max(abs(r) for r in res) < 0.02, "a pure linear effect must be removed"
    assert st.fmean(res) == pytest.approx(0.0, abs=1e-9)


def test_ols_residuals_keep_what_frequency_cannot_explain():
    """The control must not be a shredder: a real effect ORTHOGONAL to log-df has to survive,
    otherwise 'the frequency-controlled effect is null' would be guaranteed."""
    xs = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
    ys = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    res = ols_residuals(ys, xs)
    assert st.stdev(res) > 0.4


def test_ols_residuals_survive_a_constant_frequency():
    res = ols_residuals([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
    assert res == pytest.approx([-1.0, 0.0, 1.0])


def test_df_matched_pairs_keeps_the_pairing_and_drops_only_mismatched_pairs():
    a_v, a_df = [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]
    b_v, b_df = [1.5, 2.5, 3.5], [0.1, 5.0, 0.2]
    av, bv = df_matched_pairs(a_v, a_df, b_v, b_df, tol=0.25)
    assert av == [1.0, 3.0] and bv == [1.5, 3.5]
    assert len(av) == len(bv)


# ---------------------------------------------------------------------------------------
# The instrument: the targeted association table.
# ---------------------------------------------------------------------------------------
_FIXTURE_STORIES = [
    "The dragon guarded the mountain cave with fire and smoke.",
    "A dragon sneezed and the cave filled with smoke again.",
    "Lily baked cookies with her mother in the warm kitchen.",
    "The kitchen smelled of cookies and Lily smiled at her mother.",
    "A rabbit hopped past the mountain and found a carrot.",
    "The carrot was orange and the rabbit ate it by the cave.",
    "Lily and the rabbit shared a cookie in the kitchen.",
    "Smoke rose from the mountain while the dragon slept.",
]


def _fixture_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text("</s>".join(_FIXTURE_STORIES) + "</s>", encoding="utf-8")
    return p


def test_the_targeted_table_equals_the_full_table_on_every_count_it_holds(tmp_path):
    """THE INSTRUMENT TEST. The real table is 31.9M pairs and ~34 minutes to build, so the
    eval counts only what it needs. That is only sound if the counts are IDENTICAL, so this
    builds BOTH ways on a corpus small enough to do it and compares every key."""
    corpus = _fixture_corpus(tmp_path)
    full = build_association(_FIXTURE_STORIES)
    obs = [{"context_words": list(content_words(_FIXTURE_STORIES[0])),
            "add_words": ["cookies", "rabbit", "smoke"]},
           {"context_words": list(content_words(_FIXTURE_STORIES[2])),
            "add_words": ["dragon", "carrot"]},
           # A MUTUAL pair: `dragon` needs `smoke` and `smoke` needs `dragon`. Without the
           # per-document dedup in `collect_targeted_association` this pair would be counted
           # TWICE per story and every distance over it would be wrong.
           {"context_words": ["dragon", "smoke", "cave"],
            "add_words": ["dragon", "smoke"]}]
    uni, by_add = needed_lookups(obs)
    targeted, own = collect_targeted_association(corpus, needed_uni=uni, by_add=by_add,
                                                 story_ids={0, 3})
    assert targeted.n_docs == full.n_docs == len(_FIXTURE_STORIES)
    assert uni, "the fixture must actually ask for something"
    for w in uni:
        assert targeted.uni.get(w, 0) == full.uni.get(w, 0), w
    asked = {(a, c) for a, cs in by_add.items() for c in cs}
    assert asked
    from train.reach import pair_key
    for a, c in asked:
        k = pair_key(a, c)
        assert targeted.co.get(k, 0) == full.co.get(k, 0), k
    assert set(own) == {0, 3}
    assert "dragon" in own[0]


def test_the_targeted_table_reproduces_reach_distance_exactly(tmp_path):
    """Equality of counts is the mechanism; equality of the DISTANCE is the claim."""
    corpus = _fixture_corpus(tmp_path)
    full = build_association(_FIXTURE_STORIES)
    ctx = list(content_words(_FIXTURE_STORIES[1]))
    obs = [{"context_words": ctx, "add_words": ["mountain", "dragon", "cookies"]}]
    uni, by_add = needed_lookups(obs)
    targeted, _ = collect_targeted_association(corpus, needed_uni=uni, by_add=by_add,
                                               story_ids=set())
    for w in ("mountain", "dragon", "cookies"):
        assert (reach_distance(w, ctx, targeted)
                == reach_distance(w, ctx, full)), w


def test_needed_lookups_never_asks_for_a_self_pair():
    """`build_association` counts pairs of DISTINCT words only, so a self-pair is a key the
    full table cannot contain. Asking for one would make the targeted table differ from the
    full one on exactly the case where a generated `add` word repeats a context word."""
    obs = [{"context_words": ["dragon", "cave", "smoke"], "add_words": ["dragon"]}]
    uni, by_add = needed_lookups(obs)
    assert "dragon" not in by_add["dragon"]
    assert by_add["dragon"] == {"cave", "smoke"}
    assert "dragon" in uni


def test_a_self_pair_is_never_counted_even_when_a_generated_add_repeats_the_context(tmp_path):
    corpus = _fixture_corpus(tmp_path)
    obs = [{"context_words": ["dragon", "cave"], "add_words": ["dragon"]}]
    uni, by_add = needed_lookups(obs)
    targeted, _ = collect_targeted_association(corpus, needed_uni=uni, by_add=by_add,
                                               story_ids=set())
    from train.reach import pair_key
    assert pair_key("dragon", "dragon") not in targeted.co
    # ... and the distance of a word to itself is therefore "no evidence", not 0.0.
    assert reach_distance("dragon", ["dragon"], targeted) is None


def test_gold_reproduction_matches_a_correct_table_and_refuses_a_wrong_one(tmp_path):
    """The decision behind the run's HARD REFUSAL, fixture-tested.

    `gold_reproduction` is what stands between "the targeted table equals the full one" and
    publishing distances measured on a different instrument than the one that defined
    near/mid/far. So it has to say `matches: False` when anything is off -- a wrong distance,
    a wrong document frequency, or a word the table cannot score.
    """
    corpus = _fixture_corpus(tmp_path)
    ctx = list(content_words(_FIXTURE_STORIES[1]))
    obs_spec = [{"context_words": ctx, "add_words": ["mountain"]}]
    uni, by_add = needed_lookups(obs_spec)
    targeted, own = collect_targeted_association(corpus, needed_uni=uni, by_add=by_add,
                                                 story_ids={1})
    truth = reach_distance_loo("mountain", ctx, targeted, own[1])
    assert truth is not None
    good = [{"story_id": 1, "context_words": ctx, "gold_add_word": "mountain",
             "gold_distance": truth, "gold_add_df": targeted.uni["mountain"]}]
    got = gold_reproduction(good, targeted, own)
    assert got["matches"] is True
    assert got["max_abs_distance_error"] == 0.0

    wrong_distance = [dict(good[0], gold_distance=truth + 1e-6)]
    assert gold_reproduction(wrong_distance, targeted, own)["matches"] is False

    wrong_df = [dict(good[0], gold_add_df=targeted.uni["mountain"] + 1)]
    bad_df = gold_reproduction(wrong_df, targeted, own)
    assert bad_df["matches"] is False and bad_df["add_df_mismatches"] == 1

    # One GOOD row beside the unscorable one, so `d_err` is non-empty and the only clause
    # that can catch this is `n_none == 0`. (With the unscorable row alone the check passed
    # for the wrong reason -- an empty `d_err` -- and a mutant that dropped `n_none == 0`
    # survived. That is the hollow-test shape this project has shipped ten times.)
    unscorable = [good[0], dict(good[0], gold_add_word="zzzznotaword")]
    bad = gold_reproduction(unscorable, targeted, own)
    assert bad["observations_rederived"] == 1
    assert bad["gold_words_with_no_evidence_in_the_targeted_table"] == 1
    assert bad["max_abs_distance_error"] == 0.0, "the surviving row must be a clean match"
    assert bad["matches"] is False


# ---------------------------------------------------------------------------------------
# The leave-one-out.
# ---------------------------------------------------------------------------------------
def _small_assoc() -> Association:
    from train.reach import pair_key
    return Association(uni={"a": 10, "b": 20, "c": 5},
                       co={pair_key("a", "b"): 4, pair_key("a", "c"): 2}, n_docs=100)


def test_loo_reduces_to_holdout_true_when_the_word_and_the_context_are_in_the_story():
    """This is what makes the gold reproduction possible: derivation passed holdout=True for
    every context word, and on gold rows every word IS in the story."""
    assoc = _small_assoc()
    own = frozenset({"a", "b", "c"})
    assert (reach_distance_loo("a", ["b", "c"], assoc, own)
            == reach_distance("a", ["b", "c"], assoc, holdout=True))


def test_loo_does_not_subtract_a_document_the_generated_word_never_entered():
    """A generated `add` word need not be in the scored story at all. Subtracting a document
    it never contributed to would bias the generated conditions against the gold ones."""
    assoc = _small_assoc()
    own = frozenset({"b", "c"})                      # "a" is NOT in the story
    got = reach_distance_loo("a", ["b", "c"], assoc, own)
    assert got == reach_distance("a", ["b", "c"], assoc, holdout=False)
    assert got != reach_distance("a", ["b", "c"], assoc, holdout=True)


def test_loo_takes_the_nearest_over_a_mixed_context():
    assoc = _small_assoc()
    own = frozenset({"a", "b"})                      # "c" came from outside the story
    got = reach_distance_loo("a", ["b", "c"], assoc, own)
    expect = min(reach_distance("a", ["b"], assoc, holdout=True),
                 reach_distance("a", ["c"], assoc, holdout=False))
    assert got == expect


def test_loo_returns_none_when_no_context_word_has_evidence():
    assoc = _small_assoc()
    assert reach_distance_loo("a", ["zzz"], assoc, frozenset({"a", "zzz"})) is None


# ---------------------------------------------------------------------------------------
# Parsing the generation.
# ---------------------------------------------------------------------------------------
def test_parse_forced_generation_excludes_the_block_from_the_turn():
    """The block literally CONTAINS the `add` word, so a 'turn' that still held the block
    would make the slot-hit rate a tautology -- which is the exact shape of failure stage 2's
    `accept` slot had."""
    text = ("add: dragon\nstakes: +0.0\nhandback: cave\n</think>\n A dragon lives in the "
            "cave. Another sentence.")
    g = parse_forced_generation(text)
    assert g.closed_block is True
    assert g.add_value == "dragon"
    assert g.handback == "cave"
    assert g.turn == "A dragon lives in the cave."
    assert "add:" not in g.turn and "</think>" not in g.turn


def test_parse_forced_generation_on_a_block_that_never_closed():
    g = parse_forced_generation("add: dragon\nstakes: +0.0\n")
    assert g.closed_block is False
    assert g.add_value == "dragon"
    assert g.turn == "", "an unclosed block has no turn to score"


def test_parse_forced_generation_takes_the_first_add_line_not_a_later_one():
    """Two separate ways a later `add` can appear, and BOTH must lose to the first.

    The first assertion is the one with teeth for the parse loop: a repeated key INSIDE the
    same block body. (An earlier version of this test only used the second case, and a mutant
    that dropped the `key not in found` guard passed it -- the `</think>` truncation was doing
    all the work, so the assertion could not fail. That is exactly the hollow-test shape this
    project has shipped ten times.)
    """
    g = parse_forced_generation("add: first\nadd: second\nstakes: +0.0\n</think>\n A turn.")
    assert g.add_value == "first"
    # ... and a whole SECOND think-block after the turn also loses: scoring that one would
    # score a plan made after the turn it is supposed to govern.
    g2 = parse_forced_generation("add: first\n</think>\n A turn. <think>\nadd: second\n")
    assert g2.add_value == "first"


def test_parse_forced_generation_when_no_add_line_was_produced():
    g = parse_forced_generation("stakes: +0.0\nhandback: cave\n</think>\n A turn.")
    assert g.add_value is None
    assert add_word_of_value(g.add_value) is None


def test_the_degeneration_floor_fires_on_a_real_loop():
    """A 4-gram version of this metric scored exactly 0.0000 on every setting -- a skit turn is
    one sentence and rarely has four content words, so it could not fire at all. This fixture
    is the actual generation that exposed that, so the replacement cannot be vacuous the same
    way."""
    assert content_word_repeat_rate("Snowmen are snowmen and snowmen.") == pytest.approx(2 / 3)
    assert content_word_repeat_rate("The snow is soft and white.") == 0.0
    assert content_word_repeat_rate("") == 0.0


def test_add_word_of_value_applies_the_same_rule_as_train_reach():
    for raw in ("Dragon", " dragon ", "dragon, cave", "DRAGON,cave"):
        slots = ReachSlots(offer="", accept="", reach="near", add=raw, stakes="+0.0",
                           handback="")
        assert add_word_of_value(raw) == add_word_of(slots)


# ---------------------------------------------------------------------------------------
# The prompt. A PROPERTY OF THE REAL ARTIFACT -- a fixture cannot have a BPE seam.
# ---------------------------------------------------------------------------------------
def test_the_forced_block_prefix_is_a_prefix_of_the_rendered_block():
    slots = ReachSlots(offer="o words", accept="a words", reach="far", add="dragon",
                       stakes="+0.0", handback="cave")
    block = render_think(slots)
    forced = forced_block_prefix({"offer": "o words", "accept": "a words"}, "far")
    assert block.startswith(forced), (forced, block)


def test_the_forced_prefix_changes_with_the_dial_and_with_nothing_else():
    b = {"offer": "o", "accept": "a"}
    near, far = forced_block_prefix(b, "near"), forced_block_prefix(b, "far")
    assert near != far
    assert near.replace("near", "far") == far


@pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").is_file(),
                    reason="tokenizer artifact not present")
@pytest.mark.skipif(not DEFAULT_SKITS.is_file(), reason="skits artifact not present")
def test_the_segment_wise_prompt_is_a_prefix_of_a_real_training_examples_ids():
    """THE SEAM TEST, on the real artifact. The arms were tokenized SEGMENT-WISE, so the
    trained sequence carries a per-segment prefix space and the think seam is
    ``['.', 'Ġ<', 'think', '>']``. A whole-string tokenization produces a bare ``'<'`` there,
    degrades every setting EQUALLY, and reads as 'the dial does nothing'."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    rows = [r for r in _some_eval_rows(6)]
    checked = 0
    for row in rows:
        gold = row["blocks"][MEASURED_BLOCK]["reach"]
        ids = encode_segments(tok, prompt_segments(row, gold))
        train_ids = training_example_ids(tok, row)
        assert train_ids[:len(ids)] == ids, row["story_id"]
        # ... and the whole-string form must NOT match, or this test has no teeth.
        whole = tok.encode("".join(prompt_segments(row, gold)))
        assert train_ids[:len(whole)] != whole, (
            f"story {row['story_id']}: whole-string tokenization matched too, so this test "
            f"cannot distinguish the two constructions")
        checked += 1
    assert checked == 6


def test_forcing_a_non_gold_dial_value_changes_the_prompt_ids():
    """If the three settings produced the same ids the whole measurement would be three
    copies of one number, and every contrast would be exactly zero."""
    pytest.importorskip("transformers")
    if not (TOKENIZER_DIR / "tokenizer.json").is_file():
        pytest.skip("tokenizer artifact not present")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    row = _some_eval_rows(1)[0]
    ids = {v: encode_segments(tok, prompt_segments(row, v))
           for v in ("near", "mid", "far", NONSENSE_VALUE)}
    assert len({tuple(v) for v in ids.values()}) == 4


_ROWS_CACHE = None


def _all_rows():
    global _ROWS_CACHE
    if _ROWS_CACHE is None:
        _ROWS_CACHE = load_rows(DEFAULT_SKITS)
    return _ROWS_CACHE


def _some_eval_rows(n: int):
    out = []
    for r in _all_rows():
        if r.get("split") == "eval":
            out.append(r)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------------------
# Properties of SCALE -- asserted against the real artifact, per the spec's rule 4.
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not DEFAULT_SKITS.is_file(), reason="skits artifact not present")
def test_the_add_vocabulary_is_dominated_by_particles_on_the_real_artifact():
    """A property of SCALE: a fixture cannot have hub structure. If the dial works, it works
    on THIS vocabulary, and a reader must be able to see that."""
    p = particle_profile(_all_rows())
    assert p["observations"] == 123042
    assert p["distinct_add_words"] == 6442
    top = {e["word"] for e in p["top_15"]}
    assert {"look", "please", "hi", "hello"} & top, top
    # measured, not guessed: 15 words out of 6,442 carry 13.3% of all observations.
    assert p["top_15_share"] == pytest.approx(0.13332, abs=1e-4)


@pytest.mark.skipif(not DERIVE_MANIFEST.is_file(), reason="derive manifest not present")
def test_the_cut_points_are_the_train_fitted_ones_and_are_not_refitted():
    """These two numbers ARE the dial. A dial whose buckets move between train and eval
    measures nothing, so the eval reads them and never fits."""
    m = json.loads(DERIVE_MANIFEST.read_text())
    cuts = m["reach"]["cut_points"]
    assert cuts["lo"] == pytest.approx(0.718431, abs=1e-6)
    assert cuts["hi"] == pytest.approx(0.824329, abs=1e-6)
    assert "training split only" in cuts["fitted_on"]
    src = (ROOT / "scripts" / "eval_reach.py").read_text()
    assert "fit_reach_terciles" not in src, "eval must never re-fit the cut points"


@pytest.mark.skipif(not DERIVE_MANIFEST.is_file(), reason="derive manifest not present")
def test_the_drop_rate_that_must_be_disclosed_is_above_the_mandatory_threshold():
    m = json.loads(DERIVE_MANIFEST.read_text())
    assert m["drop_rate"] == pytest.approx(0.9806, abs=1e-4)
    assert m["drop_rate"] > 0.5, "the spec makes disclosure mandatory above 50%"
    assert m["drops_by_rule"]


@pytest.mark.skipif(not DERIVE_MANIFEST.is_file(), reason="derive manifest not present")
def test_the_corpus_scale_frequency_confound_is_not_monotone():
    """This is WHY mid-vs-far is the cleaner contrast, and it is read from the manifest rather
    than retyped, so a re-derivation that changed it would fail here."""
    m = json.loads(DERIVE_MANIFEST.read_text())
    per = m["reach"]["frequency_confound"]["per_bucket"]
    near, mid, far = (per[b]["median_add_df"] for b in ("near", "mid", "far"))
    assert mid > far > near, (near, mid, far)
    assert m["reach"]["frequency_confound"]["spearman_df_vs_distance"] > 0


# ---------------------------------------------------------------------------------------
# Wiring: the conditions themselves.
# ---------------------------------------------------------------------------------------
def test_the_conditions_include_the_primary_and_secondary_controls():
    keys = {condition_key(a, v) for a, v in CONDITIONS}
    for v in ("near", "mid", "far"):
        assert f"dial:{v}" in keys
        assert f"nodial:{v}" in keys
    assert f"dial:{NONSENSE_VALUE}" in keys, "the PRIMARY control must be a condition"
    assert len(CONDITIONS) == 7


def test_the_nonsense_value_is_not_a_real_dial_value():
    from train.reach import REACH_VALUES
    assert NONSENSE_VALUE not in REACH_VALUES


def test_the_thresholds_are_not_imported_from_another_stage():
    """The exact hole the spec named. `scripts/eval_improv.py` carries 0.01 / 2.576 and
    `scripts/eval_skits.py` carries its own pair; importing either would apply a threshold
    this design did not choose."""
    src = (ROOT / "scripts" / "eval_reach.py").read_text()
    assert "BONFERRONI_ALPHA = 0.05 / N_TESTS" in src
    for bad in ("from scripts.eval_skits import", "from scripts.eval_improv import CRITICAL_T",
                "from scripts.eval_improv import BONFERRONI_ALPHA"):
        assert bad not in src, bad
