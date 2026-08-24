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


def test_the_cleaner_contrast_field_is_labelled_as_the_SPECS_framing_not_this_runs():
    """Spec amendment 8fb43b4 named `mid` vs `far` the cleaner contrast because the DERIVATION's
    per-bucket median add_df is non-monotone. That is a property of the gold label
    distribution, and it must not be published as a property of a run without being checked
    against the words that run produced -- an earlier version of this artifact shipped a
    hard-coded `not_monotone: true` beside it that was FALSE of the run it described."""
    v = monotonicity_verdict(NEAR, MID, FAR, label="fixture")
    # The key name is a contract -- `scripts/reach.py --about` quotes it and
    # `tests/test_reach_cli.py` pins it -- so it stays. What must not stay is shipping it as a
    # bare claim about the run.
    assert v["cleaner_contrast"] == "mid_lt_far"
    assert v["cleaner_contrast_is_THE_SPECS_DESIGNATION_not_a_finding_about_this_run"] is True
    prov = v["cleaner_contrast_provenance"]
    assert "DERIVATION" in prov
    assert "realised_frequency_profile" in prov, "must point at the computed answer"
    assert "cleaner_contrast_why" not in v, "the un-provenanced claim must be gone"


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
        return {"mean_delta": 0.004, "t": 2.4, "significant": False, "n_pairs": 300,
                "n_same_add_word": 270, "n_exact_zero_difference": 272,
                "structural_zero_share_of_kept": 0.9067, "n_informative": 28,
                "informative_only": {"mean_delta": 0.06, "t": 3.5 if sig else 2.0,
                                     "significant": sig, "n_pairs": 28}}
    return {"near_lt_mid": step(near_mid), "mid_lt_far": step(mid_far),
            "near_lt_far": step(near_far),
            "all_three_significant_over_informative_pairs": near_mid and mid_far and near_far}


def test_the_df_matched_subsample_is_reported_over_INFORMATIVE_pairs_and_does_not_gate():
    """Two things at once, both of which were wrong in an earlier draft.

    1. The matched block must be summarised over its INFORMATIVE pairs. Its kept-pool mean is
       padded with identical-word zeros and is not an effect size.
    2. It must not silently redecide the verdict; the gate is the residualised control and was
       fixed before the data existed.
    """
    got = dial_or_frequency_dial(_verdict(True), _verdict(True),
                                 _matched(False, True, True))
    assert got["verdict"] == "REACH DIAL", "the gate is the residualised control"
    blk = got["second_frequency_control_df_matched_subsample"]
    assert blk["steps_significant_over_informative_pairs"] == 2
    step = blk["per_step"]["near_lt_far"]
    assert step["n_informative"] == 28
    assert step["mean_delta_over_informative"] == 0.06
    assert step["mean_delta_over_kept_UNINTERPRETABLE"] == 0.004
    assert "structural zeros" in blk["READ_THIS"]
    assert "UNDERPOWERED" in blk["READ_THIS"]


def test_the_matched_report_counts_all_three_when_the_informative_pairs_all_reach():
    got = dial_or_frequency_dial(_verdict(True), _verdict(True), _matched(True, True, True))
    blk = got["second_frequency_control_df_matched_subsample"]
    assert blk["steps_significant_over_informative_pairs"] == 3


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
                           "declared_margin": ADHERENCE_MARGIN,
                           "best_setting": "dial:mid", "worst_setting": "dial:near",
                           "rates": {"dial:near": 0.48, "dial:mid": 0.57,
                                     "dial:far": 0.51}},
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
    ({"adherence": {"passes": False, "shortfall": 0.3, "declared_margin": 0.05,
                    "best_setting": "dial:mid", "worst_setting": "dial:near",
                    "rates": {"dial:near": 0.27, "dial:mid": 0.57,
                              "dial:far": 0.51}}}, "slot-hit"),
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


# =======================================================================================
# THE COMPOSITION LAYER.
#
# Every leaf decision above is fixture-tested. That was NOT ENOUGH, and a review proved it:
# `analyse`, `build_observations` and `per_setting_table` were imported by no test at all, and
# three one-line mutations inside them each rewrote a published claim while all 1,505 tests in
# the repo passed --
#
#   (a) the raw monotonicity contrast called with its arguments inverted -> `raw_monotone`
#       false and the verdict downgraded to PARTIAL;
#   (b) the frequency control bypassed -> a FREQUENCY DIAL publishes as a REACH DIAL;
#   (c) `add`-hit scored against the RAW generation instead of the extracted turn -> the rate
#       jumps to ~1.0 because the block text contains the `add` word, the shortfall collapses,
#       and `eureka_criterion_met` FLIPS FROM FALSE TO TRUE.
#
# That is the eleventh instance of this class on this project and the second in this file's
# own history. A 340-line composer is `main()` by another name, which the spec's Testing ¶1
# forbids. So `analyse` is driven end to end here on a synthetic corpus small enough to build
# in milliseconds, and each of (a)(b)(c) is demonstrated RED against it.
#
# The fixture is engineered to be a FREQUENCY DIAL: `add` words are chosen so that distance
# rises across near/mid/far *entirely because* document frequency rises. Raw is monotone;
# residualising on log df kills it. That makes (b) visible -- with the control bypassed the
# fixture publishes as a REACH DIAL -- and it exercises the spec's headline rule on the one
# outcome the spec says must be published as its own finding.
# =======================================================================================
import math as _math  # noqa: E402

from scripts.eval_reach import analyse, build_observations, per_setting_table  # noqa: E402

#: Context words every fixture scene shares. Three of them, so `reach_distance`'s "nearest
#: context word" has something to choose between rather than one forced answer.
_CTX = ("harbour", "lantern", "compass")

#: (word, document frequency, documents co-occurring with the whole context).
#:
#: NPMI = [log(k/N) - log(d/N) - log(cb/N)] / -log(k/N), so with `k` and the context marginal
#: fixed it falls as `d` rises -- the real NPMI frequency bias, reproduced at fixture scale, so
#: distance RISES with document frequency and with nothing else. Two `k` variants per frequency
#: level give the residual a real, NON-monotone component: without it the distance is an exact
#: linear function of log df, the residuals are float noise, and whether the residualised
#: contrast comes out significant is decided by rounding rather than by the fixture.
#:
#: The neutral filler at the end is not padding: without a large N the rarest-vs-commonest pair
#: drops below chance co-occurrence, `npmi` clamps to 0.0, and every distance saturates at
#: exactly 1.0 -- which is what the first version of this fixture did, making three of its
#: assertions unable to fail.
_ADD_WORDS = [("gull", 12, 8), ("rope", 24, 8), ("salt", 36, 8),
              ("plank", 12, 11), ("net", 24, 11), ("tide", 36, 11)]
_FILLER_CONTEXT_ONLY = 5
_FILLER_NEUTRAL = 400


def _freq_fixture_corpus(tmp_path: Path) -> Path:
    """A corpus where distance is a function of document frequency and one orthogonal term."""
    stories = []
    for word, df, k in _ADD_WORDS:
        for i in range(k):
            stories.append(f"The {word} near the {' and the '.join(_CTX)} at dusk {i}.")
        for i in range(df - k):
            stories.append(f"A {word} and a pebble number {i} rested quietly.")
    for i in range(_FILLER_CONTEXT_ONLY):
        stories.append(f"The {' and the '.join(_CTX)} waited, evening {i}.")
    for i in range(_FILLER_NEUTRAL):
        stories.append(f"A quiet unrelated sentence numbered {i} about pebbles.")
    p = tmp_path / "corpus.txt"
    p.write_text("</s>".join(stories) + "</s>", encoding="utf-8")
    return p


def _fixture_rows_and_generations(tmp_path: Path, n_scenes: int = 120):
    """Rows whose stored gold values are CORRECT for the fixture corpus, plus generations.

    The gold `reach_distances`/`add_df` are computed from the fixture corpus rather than
    invented, because `analyse` re-derives them and REFUSES to continue on a mismatch. So this
    fixture also exercises that refusal from the passing side.
    """
    corpus = _freq_fixture_corpus(tmp_path)
    prefix = f"Once the {_CTX[0]} was quiet."
    turns = [f"I see the {_CTX[1]}.", f"And the {_CTX[2]} too.",
             f"Then the {_CTX[0]} answered.", "A partner replies here.",
             "The last model turn."]

    # Build the association over everything any scene could ask for.
    probe = [{"context_words": list(_CTX), "add_words": [w for w, _, _ in _ADD_WORDS]}]
    uni, by_add = needed_lookups(probe)
    assoc, _own = collect_targeted_association(corpus, needed_uni=uni, by_add=by_add,
                                               story_ids=set())
    dist_of = {w: reach_distance(w, list(_CTX), assoc) for w, _, _ in _ADD_WORDS}
    assert all(d is not None for d in dist_of.values()), dist_of
    # A fixture that saturates cannot fail the assertions built on it. Both checks are the
    # ones the first version of this fixture would have failed.
    assert all(0.0 < d < 1.0 for d in dist_of.values()), (
        "fixture saturates: npmi clamped to 0 and every distance is 1.0", dist_of)
    for k_variant in (8, 11):
        lvl = [(df, dist_of[w]) for w, df, k in _ADD_WORDS if k == k_variant]
        lvl.sort()
        assert [d for _, d in lvl] == sorted(d for _, d in lvl), (
            "fixture is vacuous: distance does not track document frequency", lvl)

    rows = []
    for j in range(n_scenes):
        rows.append({
            "story_id": 10_000 + j, "prefix": prefix, "turns": list(turns),
            "roles": ["model", "partner", "model", "partner", "model"],
            "blocks": [{"offer": "o", "accept": "a", "reach": "mid", "add": "gull",
                        "stakes": "+0.0", "handback": "compass"} for _ in range(3)],
            "split": "eval",
            # `analyse` only reads index MEASURED_BLOCK of these two.
            "reach_distances": [0.0, dist_of["gull"], 0.0],
            "add_df": [0, assoc.uni["gull"], 0],
            "stakes_deltas": [0.0, 0.0, 0.0],
        })

    def gen(add_word, *, mention, handback="compass"):
        turn = (f"The {add_word} is here." if mention else "Nothing of the sort.")
        return (f"add: {add_word}\nstakes: +0.0\nhandback: {handback}\n</think>\n {turn}")

    # Two triples alternate -- same frequency ladder, different co-occurrence variant -- so
    # the paired differences have real scatter and the residual has a non-monotone component.
    triples = [("gull", "rope", "salt"), ("plank", "net", "tide")]
    gens = {condition_key(a, v): [] for a, v in CONDITIONS}
    for j in range(n_scenes):
        near_w, mid_w, far_w = triples[j % 2]
        # `add`-hit deliberately BELOW 1.0 and different per setting, so a mutant that scores
        # the hit against the raw generation (which always contains "add: <word>") cannot
        # reproduce these rates.
        gens[condition_key("dial", "near")].append(gen(near_w, mention=(j % 4 != 0)))
        gens[condition_key("dial", "mid")].append(gen(mid_w, mention=(j % 2 == 0)))
        gens[condition_key("dial", "far")].append(gen(far_w, mention=(j % 4 == 0)))
        gens[condition_key("dial", NONSENSE_VALUE)].append(gen(mid_w, mention=True))
        # The control conditions cycle through EVERY level. `analyse` fits the nuisance line
        # over the pooled observations of all seven conditions, so a control that always emits
        # one word puts a tall cluster at a single x and tilts the fit -- an earlier version of
        # this fixture did exactly that, and the tilt left a monotone residual that made a pure
        # frequency dial read as a reach dial. Spreading the controls keeps the fit conditioned
        # on the real geometry.
        for i, v in enumerate(("near", "mid", "far")):
            w = _ADD_WORDS[(j + 2 * i) % len(_ADD_WORDS)][0]
            gens[condition_key("nodial", v)].append(gen(w, mention=True))
    return corpus, rows, gens, assoc


def _fixture_manifests_and_derive(assoc):
    derive = {
        "drop_rate": 0.9806, "stories": 2119489, "kept": 41014,
        "drops_by_rule": {"no_dialogue": 976158},
        "gate_order": ["a"], "gate_order_note": "conditional",
        "same_speaker_filter": {"subject_reading": {"gate": "g", "risky_pair_fraction": 0.0097},
                                "what_this_filter_CANNOT_catch": "narrative-gap drift"},
        "reach": {
            "cut_points": {"lo": 0.718431, "hi": 0.824329,
                           "fitted_on": "training split only",
                           "n_fitted_on": 110739,
                           "eval_must_not_refit": "these two numbers ARE the dial"},
            "frequency_confound": {
                "spearman_df_vs_distance": 0.2078,
                "per_bucket": {"near": {"median_add_df": 16591},
                               "mid": {"median_add_df": 51269},
                               "far": {"median_add_df": 39367}}},
            "association_table": {"documents": assoc.n_docs, "vocabulary": 52302,
                                  "pairs": 31856720},
        },
    }
    arm = {
        "n_examples": 36387, "steps": 3000, "steps_note": "inherited budget",
        "val_loss_first": [250, 1.2046], "val_loss_last": [3000, 0.71245],
        "loss_comparability_WARNING": "different supervised token sets",
        "batch_order_fingerprint": {"sha256": "deadbeef" * 8, "n": 36387},
        "ruling_c_reapplied": {"rule": "union across arms",
                               "measured_before_dropping": {
                                   "dial_over_max_seq_len": 526,
                                   "nodial_over_max_seq_len": 180,
                                   "of_training_rows": 36913,
                                   "max_tokens_either_arm": 544}},
    }
    return derive, {"dial": dict(arm), "nodial": dict(arm)}


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    """`analyse` driven end to end. Module-scoped: the corpus pass is cheap but not free."""
    tmp = tmp_path_factory.mktemp("compose")
    corpus, rows, gens, assoc = _fixture_rows_and_generations(tmp)
    derive, manifests = _fixture_manifests_and_derive(assoc)
    return analyse(rows, gens, corpus=corpus, corpus_limit=None, assoc_skits=0,
                   all_rows=rows, derive=derive, manifests=manifests, gen_meta={},
                   df_match_tol=0.25, progress=0, generate_cmd="FIXTURE-CMD")


def test_analyse_wires_the_raw_contrast_in_the_declared_direction(composed):
    """MUTANT (a): the raw monotonicity call with its arguments inverted. The leaf function is
    correct and fixture-tested either way; only driving the composer catches the wiring."""
    raw = composed["effects"]["raw_distance"]
    assert raw["monotone"] is True
    for step in ("near_lt_mid", "mid_lt_far", "near_lt_far"):
        assert raw[step]["mean_delta"] > 0, step
        assert raw[step]["in_the_declared_direction"] is True
    ps = composed["per_setting"]
    assert (ps["dial:near"]["raw_distance_mean"]
            < ps["dial:mid"]["raw_distance_mean"]
            < ps["dial:far"]["raw_distance_mean"])


def test_analyse_publishes_a_frequency_dial_as_a_frequency_dial(composed):
    """MUTANT (b): the frequency control bypassed. This fixture's distance is a PURE function
    of document frequency, so the honest verdict is FREQUENCY DIAL -- the outcome the spec says
    must be published as its own finding. A bypass makes it read REACH DIAL."""
    assert composed["effects"]["raw_distance"]["monotone"] is True
    assert composed["effects"]["frequency_residualised_distance"]["monotone"] is False
    assert composed["dial_kind"]["verdict"] == "FREQUENCY DIAL"
    assert composed["dial_kind"]["frequency_controlled_monotone"] is False
    # the gate must read the RESIDUALISED verdict, not the raw one
    assert (composed["dial_kind"]["frequency_controlled_steps_significant"]
            == composed["effects"]["frequency_residualised_distance"]["n_significant_steps"])
    assert (composed["effects"]["raw_distance"]["n_significant_steps"]
            != composed["effects"]["frequency_residualised_distance"]["n_significant_steps"]), (
        "fixture is vacuous: raw and residualised agree, so a bypass would be invisible")
    assert composed["headline"]["eureka_criterion_met"] is False


def test_analyse_scores_the_add_hit_against_the_TURN_not_the_raw_generation(composed):
    """MUTANT (c): `add`-hit scored on the raw generation. The generation always contains
    `add: <word>`, so the rate jumps to ~1.0, the shortfall collapses, and on the real run
    `eureka_criterion_met` FLIPS FROM FALSE TO TRUE. The fixture's rates are deliberately
    uneven and none is 1.0."""
    ps = composed["per_setting"]
    assert ps["dial:near"]["add_slot_hit_rate"] == pytest.approx(0.75, abs=1e-6)
    assert ps["dial:mid"]["add_slot_hit_rate"] == pytest.approx(0.50, abs=1e-6)
    assert ps["dial:far"]["add_slot_hit_rate"] == pytest.approx(0.25, abs=1e-6)
    for k in ("dial:near", "dial:mid", "dial:far"):
        assert ps[k]["add_slot_hit_rate"] < 0.99, k
    assert composed["adherence_guard"]["passes"] is False
    assert composed["adherence_guard"]["shortfall"] == pytest.approx(0.50, abs=1e-6)


def test_analyse_reproduces_the_gold_distances_and_would_refuse_otherwise(composed):
    repro = composed["instrument_checks"]["gold_distance_reproduction"]
    assert repro["matches"] is True
    assert repro["max_abs_distance_error"] == 0.0
    assert repro["add_df_mismatches"] == 0
    assert repro["observations_rederived"] > 0


def test_analyse_computes_the_realised_frequency_profile_rather_than_asserting_it(composed):
    """BLOCKER 3's regression guard. `not_monotone` was a hard-coded literal and was FALSE of
    the run it described. In this fixture the realised confound IS monotone by construction."""
    prof = composed["effects"]["realised_frequency_profile"]
    means = prof["mean_log_add_df_by_setting"]
    assert means["dial:near"] < means["dial:mid"] < means["dial:far"]
    assert prof["monotone_across_the_dial"] is True
    assert prof["not_monotone"] is False
    assert "does NOT transfer" in prof["consequence_for_the_specs_cleaner_contrast_framing"]
    assert "frequency_confound_here" not in composed["effects"], (
        "the key that claimed to describe this run while quoting the derivation must be gone")


def test_analyse_reports_the_matched_control_over_informative_pairs(composed):
    """BLOCKER 1's regression guard, on real wiring: every `nodial` condition emits the SAME
    `add` word here, and the dial conditions never do, so the structural-zero accounting has to
    show up."""
    m = composed["effects"]["frequency_matched_subsample"]
    for step in ("near_lt_mid", "mid_lt_far", "near_lt_far"):
        assert "n_informative" in m[step]
        assert "structural_zero_share_of_kept" in m[step]
    agree = composed["effects"]["frequency_control_agreement"]
    assert agree["reference_contrast"] == "near_lt_far"
    assert agree["share_of_raw_surviving_frequency_control"] is not None


def test_analyse_cross_links_the_verdict_and_the_headline(composed):
    """BLOCKER 5: quoting either alone gives a materially different result."""
    assert composed["headline"]["dial_kind_verdict"] == composed["dial_kind"]["verdict"]
    assert "QUOTING_EITHER_ALONE_IS_MISLEADING" in composed["headline"]
    assert composed["dial_kind"]["eureka_criterion_met"] == (
        composed["headline"]["eureka_criterion_met"])


def test_analyse_names_the_failing_setting_in_the_reason_string(composed):
    """BLOCKER 4: 'fell 0.0896 between settings' beside a far-end narrative reads as a far-end
    collapse. The reason must name worst and best."""
    reasons = composed["headline"]["reasons_against"]
    adh = [r for r in reasons if "slot-hit" in r]
    assert adh, reasons
    reason = adh[0]
    assert composed["adherence_guard"]["worst_setting"] in reason
    assert composed["adherence_guard"]["best_setting"] in reason
    assert "NOTE THE DIRECTION" in reason
    # the fixture's worst setting IS `far`, so the string must say so rather than emitting the
    # near-side wording unconditionally
    assert "the worst setting IS `far`" in reason


def test_analyse_measures_handback_rather_than_shipping_it_unevaluated(composed):
    for k in ("dial:near", "dial:mid", "dial:far"):
        assert "handback_hit_rate" in composed["per_setting"][k]
    note = composed["per_setting"]["_disclosures"]["handback_hit_rate"]
    assert "no handback effect is claimed" in note.lower() or "NO handback effect" in note


def test_analyse_carries_the_command_that_actually_generated(composed):
    assert composed["reproduce"]["generate"] == "FIXTURE-CMD"


def test_build_observations_excludes_the_block_from_the_scored_turn():
    """The composer's own use of the parser, reached with a fixture rather than through
    `analyse`, so a failure here points at the wiring and not at the corpus."""
    row = {"story_id": 1, "prefix": f"The {_CTX[0]} waited.",
           "turns": ["a turn.", "partner one.", "model two.",
                     "partner two mentions the compass.", "model four."],
           "blocks": [{"offer": "o", "accept": "a", "reach": "mid", "add": "gull",
                       "stakes": "+0.0", "handback": "compass"}] * 3,
           "reach_distances": [0.1, 0.2, 0.3], "add_df": [1, 2, 3]}
    gens = {condition_key(a, v): ["add: kraken\nstakes: +0.0\nhandback: compass\n</think>\n"
                                  " Nothing here."]
            for a, v in CONDITIONS}
    obs = build_observations([row], gens)
    c = obs[0]["conditions"]["dial:far"]
    assert c["add_word"] == "kraken"
    assert c["add_hit"] is False, "the turn does not contain 'kraken'; only the block does"
    # scored against the REAL following partner turn (turns[MEASURED_TURN + 1]),
    # which mentions the compass -- not against the model's own turn.
    assert c["handback_hit"] is True
    assert c["handback_scorable"] is True


def test_per_setting_table_keeps_its_disclosures_out_of_the_condition_rows():
    """`_disclosures` shares the table's namespace, so every consumer must skip underscore
    keys. A consumer that does not will crash or, worse, average a string."""
    tbl = per_setting_table.__doc__
    assert "add_df" in tbl


# =======================================================================================
# THE LONGER-TRAINED RERUN: path safety, the 3000-vs-9000 diff, and the recipe's asymptote.
# =======================================================================================
from scripts.eval_reach import (DEFAULT_ARM_ROOT, DEFAULT_STEP,  # noqa: E402
                                arm_dirs_for, check_step_matches_manifests,
                                convergence_framing, default_paths, prior_step_of,
                                versus_previous)


def test_the_original_step_keeps_the_published_unsuffixed_paths():
    """`docs/measurements/reach-dial.json` is published, reviewed and quoted by
    `scripts/reach.py`. Its path must not move, or every reference to it breaks."""
    got = default_paths(DEFAULT_ARM_ROOT, DEFAULT_STEP)
    assert got["out"].name == "reach-dial.json"
    assert got["store"].name == "eval-generations.json"


def test_a_different_checkpoint_CANNOT_default_onto_the_published_artifact():
    """THE PATH-SAFETY TEST. A longer-trained rerun that reused the default `--out` would
    overwrite a published measurement in place, silently, while every test still passed. The
    default must be a different file for a different checkpoint."""
    published = default_paths(DEFAULT_ARM_ROOT, DEFAULT_STEP)["out"]
    for root, step in ((DEFAULT_ARM_ROOT, 9000),
                       (DEFAULT_ARM_ROOT.parent / "reach-conv", 9000),
                       (DEFAULT_ARM_ROOT.parent / "reach-conv", 3000)):
        got = default_paths(root, step)
        assert got["out"] != published, (root, step)
        assert str(step) in got["out"].name


def test_the_rerun_never_writes_into_a_checkpoint_directory():
    """`artifacts/reach/**` and `artifacts/reach-conv/**` are INPUTS. The store and the HF
    work-dir for a non-default checkpoint must land outside them."""
    conv = DEFAULT_ARM_ROOT.parent / "reach-conv"
    got = default_paths(conv, 9000)
    for key in ("store", "work_dir"):
        assert conv not in got[key].parents, (key, got[key])


def test_arm_dirs_for_reads_the_layout_from_one_place():
    got = arm_dirs_for(Path("/x/y"))
    assert got == {"dial": Path("/x/y/ckpt-dial"), "nodial": Path("/x/y/ckpt-nodial")}


# ---- versus_previous ------------------------------------------------------------------
def _artifact(*, step, resid_near_far, resid_t, rates, adherence_passes, shortfall,
              worst, eureka, verdict="REACH DIAL", scenes=(1, 2, 3)):
    def stepblk(d, t, sig):
        return {"mean_delta": d, "t": t, "significant": sig}
    eff = {}
    for block in ("raw_distance", "frequency_residualised_distance"):
        base = resid_near_far if block == "frequency_residualised_distance" else 0.13
        eff[block] = {"near_lt_mid": stepblk(round(base / 2, 6), resid_t, True),
                      "mid_lt_far": stepblk(round(base / 2, 6), resid_t, True),
                      "near_lt_far": stepblk(base, resid_t, True)}
    return {
        "design": {"checkpoint": {"step": step},
                   "conditions": [condition_key(a, v) for a, v in CONDITIONS]},
        "thresholds": {"critical_t": CRITICAL_T, "n_tests": N_TESTS},
        "effects": eff,
        "adherence_guard": {"rates": rates, "shortfall": shortfall, "worst_setting": worst,
                            "passes": adherence_passes},
        "coherence_guard": {"drop_far_below_near": 0.03, "passes": True},
        "dial_kind": {"verdict": verdict},
        "headline": {"eureka_criterion_met": eureka},
        "controls": {"nonsense_value_PRIMARY": {"reproduces_the_dial_pattern": False},
                     "nodial_arm_secondary": {"monotone": False}},
        "stored": {"rows": [{"story_id": i} for i in scenes]},
    }


_PRIOR = _artifact(step=3000, resid_near_far=0.060438, resid_t=12.47,
                   rates={"dial:near": 0.477, "dial:mid": 0.567, "dial:far": 0.507},
                   adherence_passes=False, shortfall=0.0896, worst="dial:near", eureka=False)


def test_versus_previous_computes_the_effect_delta_rather_than_leaving_it_to_the_reader():
    this = _artifact(step=9000, resid_near_far=0.090438, resid_t=18.0,
                     rates={"dial:near": 0.60, "dial:mid": 0.62, "dial:far": 0.61},
                     adherence_passes=True, shortfall=0.02, worst="dial:near", eureka=True)
    got = versus_previous(this, _PRIOR)
    head = got["question_1_does_the_effect_grow_with_training"]["headline"]
    assert head["prior_mean_delta"] == 0.060438
    assert head["this_mean_delta"] == 0.090438
    assert head["change"] == pytest.approx(0.03, abs=1e-6)


def test_versus_previous_computes_the_delta_in_the_right_direction():
    """A backwards subtraction reports growth as shrinkage and vice versa -- and the whole
    point of the rerun is the SIGN of this number."""
    smaller = _artifact(step=9000, resid_near_far=0.030438, resid_t=6.0,
                        rates={"dial:near": 0.477, "dial:mid": 0.567, "dial:far": 0.507},
                        adherence_passes=False, shortfall=0.0896, worst="dial:near",
                        eureka=False)
    got = versus_previous(smaller, _PRIOR)
    assert got["question_1_does_the_effect_grow_with_training"]["headline"]["change"] < 0


def test_versus_previous_calls_a_resolved_gate_an_undertraining_artefact():
    this = _artifact(step=9000, resid_near_far=0.09, resid_t=18.0,
                     rates={"dial:near": 0.60, "dial:mid": 0.62, "dial:far": 0.61},
                     adherence_passes=True, shortfall=0.02, worst="dial:near", eureka=True)
    got = versus_previous(this, _PRIOR)["question_2_does_the_adherence_gate_resolve"]
    assert got["resolved_with_more_training"] is True
    assert "RESOLVED" in got["verdict"]
    assert "undertraining artefact" in got["verdict"]


def test_versus_previous_says_undertraining_is_REFUTED_when_the_gate_still_fails():
    """THE OUTCOME THAT MUST NOT BE SOFTENED. If 3x the budget does not fix it, 'undertrained'
    stops being the explanation and the guard is saying something about the model."""
    this = _artifact(step=9000, resid_near_far=0.09, resid_t=18.0,
                     rates={"dial:near": 0.50, "dial:mid": 0.60, "dial:far": 0.55},
                     adherence_passes=False, shortfall=0.10, worst="dial:near", eureka=False)
    got = versus_previous(this, _PRIOR)["question_2_does_the_adherence_gate_resolve"]
    assert got["resolved_with_more_training"] is False
    assert "REFUTED" in got["verdict"]
    assert "STILL FAILS" in got["verdict"]


def test_versus_previous_notices_a_regression():
    prior_ok = dict(_PRIOR)
    prior_ok["adherence_guard"] = dict(_PRIOR["adherence_guard"], passes=True, shortfall=0.01)
    this = _artifact(step=9000, resid_near_far=0.09, resid_t=18.0,
                     rates={"dial:near": 0.40, "dial:mid": 0.62, "dial:far": 0.55},
                     adherence_passes=False, shortfall=0.22, worst="dial:near", eureka=False)
    assert "REGRESSED" in versus_previous(this, prior_ok)[
        "question_2_does_the_adherence_gate_resolve"]["verdict"]


def test_versus_previous_checks_comparability_rather_than_assuming_it():
    """Two runs over different held-out scenes are not two readings of one experiment. This
    project's notes carry a case where exactly that comparison produced a phantom regression."""
    other_scenes = _artifact(step=9000, resid_near_far=0.09, resid_t=18.0,
                             rates={"dial:near": 0.6, "dial:mid": 0.62, "dial:far": 0.61},
                             adherence_passes=True, shortfall=0.02, worst="dial:near",
                             eureka=True, scenes=(4, 5, 6))
    got = versus_previous(other_scenes, _PRIOR)["comparable"]
    assert got["same_held_out_scenes"] is False
    assert got["all_three"] is False

    same = _artifact(step=9000, resid_near_far=0.09, resid_t=18.0,
                     rates={"dial:near": 0.6, "dial:mid": 0.62, "dial:far": 0.61},
                     adherence_passes=True, shortfall=0.02, worst="dial:near", eureka=True)
    ok = versus_previous(same, _PRIOR)["comparable"]
    assert ok["all_three"] is True


def test_versus_previous_notices_a_changed_threshold():
    loose = _artifact(step=9000, resid_near_far=0.09, resid_t=18.0,
                      rates={"dial:near": 0.6, "dial:mid": 0.62, "dial:far": 0.61},
                      adherence_passes=True, shortfall=0.02, worst="dial:near", eureka=True)
    loose["thresholds"] = {"critical_t": 2.576, "n_tests": 4}
    got = versus_previous(loose, _PRIOR)["comparable"]
    assert got["same_alpha_and_critical_t"] is False
    assert got["all_three"] is False


# ---- convergence_framing ---------------------------------------------------------------
def _man(lr, steps=9000, first=1.2056, last=0.6217, triggered=False):
    m = {"lr": lr, "steps": steps, "val_loss_first": [250, first],
         "val_loss_last": [steps, last], "stop_reason": "hit_max_steps",
         "early_stopping": {"triggered": triggered}}
    return {"dial": m, "nodial": dict(m)}


def test_convergence_framing_names_the_constant_lr_as_the_reason_not_the_budget():
    """THE FRAMING THAT CHANGED. 'Val loss still falling' under a CONSTANT learning rate is
    what the recipe does, not evidence that more steps would help -- and an artifact that keeps
    saying 'not converged' implies the fix is a bigger budget."""
    got = convergence_framing(_man(1e-5))
    assert got["lr_is_constant_for_every_step"] is True
    assert got["lr_schedule"] == "NONE -- no decay, no warmup"
    assert "ASYMPTOTE" in got["limitation"]
    assert "CONSTANT" in got["why_it_matters"]
    assert "plateaued in practice" in got["why_it_matters"]
    assert "not converged" not in got["limitation"]


def test_convergence_framing_computes_the_relative_improvement():
    got = convergence_framing(_man(1e-5, first=1.0, last=0.9))
    assert got["relative_val_improvement"] == pytest.approx(0.1, abs=1e-9)


def test_convergence_framing_reports_that_early_stopping_never_fired():
    assert convergence_framing(_man(1e-5))["early_stopping_triggered"] is False
    assert convergence_framing(_man(1e-5, triggered=True))["early_stopping_triggered"] is True


def test_a_checkpoint_that_disagrees_with_its_manifest_is_REFUSED():
    """The provenance guard. `--step 9000` against the 3000-step directory would publish the
    3000-step model's numbers beside the 9000-step run's lr, stop_reason, val losses and
    convergence paragraph -- and nothing else in this file would notice.

    It lived in `main()` and a mutation that deleted it survived, because a driver-only guard
    is unreachable from any test. Same class as the composer mutants.
    """
    check_step_matches_manifests(_man(1e-5, steps=9000), 9000)          # no raise
    with pytest.raises(RuntimeError, match="different runs"):
        check_step_matches_manifests(_man(1e-5, steps=3000), 9000)
    with pytest.raises(RuntimeError):
        check_step_matches_manifests({"dial": {}}, 9000)                # missing entirely


def test_the_prior_step_is_recoverable_from_an_artifact_that_predates_the_field():
    """The FIRST published artifact has no `design.checkpoint`, and re-running it purely to add
    one would rewrite a reviewed measurement. The step is recovered from its limitations block
    instead, so the diff can still be keyed `versus_3000_steps`."""
    assert prior_step_of({"design": {"checkpoint": {"step": 9000}}}) == 9000
    assert prior_step_of({"limitations": [{"limitation": "x"},
                                          {"limitation": "y", "steps": 3000}]}) == 3000
    # the newer field wins when both are present
    assert prior_step_of({"design": {"checkpoint": {"step": 9000}},
                          "limitations": [{"steps": 3000}]}) == 9000
    assert prior_step_of({}) is None, "guessing a step is worse than naming it unknown"


def test_the_published_3000_step_artifact_still_yields_its_step():
    """A property of the REAL artifact: if this stops working the diff silently keys itself
    `versus_prior_steps` and the two runs stop being linkable by name."""
    art = ROOT / "docs" / "measurements" / "reach-dial.json"
    if not art.is_file():
        pytest.skip("published artifact not present")
    assert prior_step_of(json.loads(art.read_text())) == 3000


def _with_gens(art, gens):
    art = dict(art)
    art["stored"] = dict(art.get("stored", {}), generations=gens)
    return art


def test_versus_previous_measures_whether_the_model_ACTUALLY_changed():
    """The instrument check behind the headline. When two runs agree to four decimal places the
    first suspect is the same checkpoint evaluated twice -- a wrong --arm-root, a stale HF
    work-dir, a copied store. Churn answers it from the stored text, not from a provenance
    string that a misconfigured run would also print."""
    prior = _with_gens(_PRIOR, {"dial:near": ["a", "b", "c", "d"]})
    changed = _with_gens(
        _artifact(step=9000, resid_near_far=0.0604, resid_t=12.3,
                  rates={"dial:near": 0.46, "dial:mid": 0.57, "dial:far": 0.51},
                  adherence_passes=False, shortfall=0.106, worst="dial:near", eureka=False),
        {"dial:near": ["a", "B", "C", "d"]})
    got = versus_previous(changed, prior)["generation_churn"]
    assert got["generations_compared"] == 4
    assert got["generations_that_differ"] == 2
    assert got["fraction_changed"] == pytest.approx(0.5)
    assert got["the_model_really_did_change"] is True


def test_versus_previous_flags_an_IDENTICAL_rerun_rather_than_reporting_a_stable_effect():
    """The failure this exists to catch: evaluating the same checkpoint twice and reading the
    agreement as a finding about training."""
    prior = _with_gens(_PRIOR, {"dial:near": ["a", "b", "c", "d"]})
    same = _with_gens(
        _artifact(step=9000, resid_near_far=0.060438, resid_t=12.47,
                  rates=_PRIOR["adherence_guard"]["rates"], adherence_passes=False,
                  shortfall=0.0896, worst="dial:near", eureka=False),
        {"dial:near": ["a", "b", "c", "d"]})
    got = versus_previous(same, prior)["generation_churn"]
    assert got["fraction_changed"] == 0.0
    assert got["the_model_really_did_change"] is False
