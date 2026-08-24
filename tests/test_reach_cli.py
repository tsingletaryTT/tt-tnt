# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for the reach-dial CLI (`scripts/reach.py`).

NO MODEL, NO DEVICE. Everything this file touches is a pure function over strings, so the
whole suite runs in well under a second and never loads weights. The one test that needs a
real artifact (the held-out skits file, for the `offer` derivation) skips when it is absent.

This project has shipped TWELVE tests that passed against both correct and incorrect code,
and a stale `.pyc` served a mutant through four results, so run with
``PYTHONDONTWRITEBYTECODE=1``. Every substantive test below was verified RED against a
plausible wrong implementation before being kept — see the task report for the output.

THE TWO FAILURE SHAPES THIS FILE EXISTS FOR
-------------------------------------------
1. **A CLI that silently ignores `--reach`.** It would look perfect: three blocks, three
   labels, three different continuations (the model is not deterministic across prompts
   only by accident). Guarded twice, at both layers that can fail —
   `test_the_forced_dial_value_reaches_the_prompt` (the value is in the bytes sent to the
   model) and `test_the_rendered_dial_is_read_from_the_prompt_not_from_a_label` (what is
   printed is derived from those bytes, not from the flag).
2. **A tool that oversells the finding.** `test_the_honesty_footer_is_on_every_run`,
   `test_the_footer_quotes_the_residualised_effect_not_the_raw_one` and
   `test_the_quoted_figures_match_the_published_artifact` make the disclosure structural
   rather than a matter of remembering to write it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reach import (  # noqa: E402
    ACCEPT_MAX_WORDS,
    FACTS,
    FOOTER,
    MEASUREMENT,
    REACH_VALUES,
    SKITS,
    EFFECT_IN_BANDS,
    INFORMATIVE_DF_MATCHED,
    SURVIVES_FREQUENCY_CONTROL,
    TERCILE_BAND,
    Arm,
    Ink,
    about_text,
    arm_segments,
    derive_offer,
    forced_reach_in,
    parse_accept,
    render_arm,
    render_collisions,
    render_run,
    selected_values,
    terminal_width,
    want_colour,
)

SCENE = "Once upon a time there was a lonely lighthouse keeper."
PLAIN = Ink(False)


def _arm(value: str, *, forced: str = None, turn: str = "He kept the light.",
         add: str = "boat", closed: bool = True) -> Arm:
    """An `Arm` whose prompt forces `forced` (default: the same as its label).

    The two can be made to DISAGREE on purpose: that is how
    `test_the_rendered_dial_is_read_from_the_prompt_not_from_a_label` proves the rendered
    value comes from the prompt.
    """
    segs = arm_segments(SCENE, "once upon time lonely lighthouse keeper", "lighthouse",
                        forced or value)
    return Arm(value=value, segments=segs, raw="", add_value=add, turn=turn,
               closed_block=closed)


# ---------------------------------------------------------------------------------------
# 1. The dial actually reaches the model.
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", REACH_VALUES)
def test_the_forced_dial_value_reaches_the_prompt(value):
    """The bytes handed to the model must end with the forced dial line.

    Checked on the LAST segment specifically: `prompt_segments` puts the forced block prefix
    there, and a `reach:` line anywhere earlier would be a teacher-forced block from an
    earlier turn, not the dial this run is setting.
    """
    segs = arm_segments(SCENE, "once upon time lonely lighthouse keeper", "lighthouse", value)
    assert segs[-1].endswith(f"reach: {value}\n")
    assert forced_reach_in(segs) == value


def test_the_three_prompts_differ_only_in_the_dial():
    """Everything the model conditions on is byte-identical except the dial value.

    This is the property the whole design rests on: if the arms differed anywhere else, a
    difference in output would not be attributable to the dial. Stated as a byte comparison
    rather than trusted.
    """
    built = {v: arm_segments(SCENE, "once upon time lonely lighthouse keeper", "lighthouse", v)
             for v in REACH_VALUES}
    for v, segs in built.items():
        # Replacing this arm's dial line with `near`'s must reproduce `near`'s prompt exactly.
        normalised = [s.replace(f"reach: {v}\n", "reach: near\n") for s in segs]
        assert normalised == built["near"], f"{v} differs from near somewhere other than the dial"


def test_selected_values_honours_the_flag():
    assert selected_values("all") == ["near", "mid", "far"]
    assert selected_values("near") == ["near"]
    assert selected_values("mid") == ["mid"]
    assert selected_values("far") == ["far"]
    with pytest.raises(ValueError):
        selected_values("blue")


def test_the_rendered_dial_is_read_from_the_prompt_not_from_a_label():
    """An arm LABELLED far whose prompt forces near must render `reach=near`.

    The wrong implementation this catches is the natural one: printing the value the user
    typed. That version passes every other test in this file and is exactly the bug that
    would make a CLI which ignores `--reach` look flawless.
    """
    lying = _arm("far", forced="near")
    text = "\n".join(render_arm(lying, width=80, ink=PLAIN, show_think=True))
    assert "reach=near" in text
    assert "reach=far" not in text


def test_forced_reach_in_refuses_a_prompt_with_no_dial():
    with pytest.raises(ValueError):
        forced_reach_in(["Once upon a time.", "<think>\noffer: a\naccept: b\n"])


# ---------------------------------------------------------------------------------------
# 2. Honesty is structural.
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("reach", ["all", "near", "mid", "far"])
@pytest.mark.parametrize("width", [44, 80, 100])
def test_the_honesty_footer_is_on_every_run(reach, width):
    """Every rendering, at every `--reach`, every width, with and without the think-block.

    Compared on whitespace-collapsed text because the footer wraps on a narrow pane; what
    must never happen is that it is absent, not that it occupies one physical line.
    """
    arms = [_arm(v) for v in selected_values(reach)]
    for show_think in (True, False):
        lines = render_run(SCENE, arms, model_label="ckpt-dial @ step 3000", seed=1,
                           temperature=0.0, width=width, ink=PLAIN, show_think=show_think)
        flat = " ".join(" ".join(lines).split())
        assert " ".join(FOOTER.split()) in flat, \
            f"no honesty footer for --reach {reach} at width {width} (think={show_think})"


def test_the_footer_quotes_the_residualised_effect_not_the_raw_one():
    """+0.060, never +0.129 alone, with the real discount and the real cost beside it."""
    assert "+0.060" in FOOTER
    assert "0.129" not in FOOTER and "0.13" not in FOOTER
    assert "47% of raw survives frequency control" in FOOTER
    assert "88 add-words" in FOOTER and "265" in FOOTER
    assert "undertrained" in FOOTER


def test_the_retracted_figures_appear_nowhere():
    """Two figures were retracted on 2026-08-24. Neither may be printed by anything.

    * "38% / ~62% value-specific" — mis-specified: there is no zero-token baseline in the
      experiment, so the nonsense value's POSITION on the range is not a SHARE of the effect
      and understates a result that is actually stronger than it was made to sound.
    * "+0.0140, the df-matched subsample, a second control that disagrees" — the matched
      subsample is a null-enrichment filter (78% definitional zeros), not a second control,
      and restricted to informative pairs it AGREES with the residualised figure.

    Guarded as a text search over everything a user can see, because the failure mode is a
    number surviving in one surface after being fixed in another.
    """
    surfaces = {"FOOTER": FOOTER, "--about": about_text(width=88)}
    for name, text in surfaces.items():
        for banned in ("62%", "38%", "0.0140", "0.014", "value-specific",
                       "do not agree", "controls do not"):
            assert banned not in text, f"{name} still carries the retracted {banned!r}"


def test_the_two_frequency_controls_are_described_as_agreeing():
    """They agree, and --about must say so rather than manufacturing a disagreement."""
    text = about_text(width=88)
    # "agree", not `"agree" in text` -- "disagree" contains "agree", and an assertion that
    # cannot tell the two apart is exactly the hollow test this file is written against.
    assert "frequency controls agree" in " ".join(text.split())
    assert "disagree" not in text and "do not agree" not in text
    assert f"+{INFORMATIVE_DF_MATCHED:.4f}" in text
    # the two agreeing figures really are close; if they ever diverge, this text is wrong
    assert abs(INFORMATIVE_DF_MATCHED - FACTS["residualised_near_to_far"]) < 0.01


def test_the_effect_is_given_a_scale_a_person_can_feel():
    """+0.060 means nothing on its own. One tercile band is the unit the dial is cut in."""
    assert abs(TERCILE_BAND - (FACTS["tercile_hi"] - FACTS["tercile_lo"])) < 1e-12
    assert abs(EFFECT_IN_BANDS - 0.5707) < 1e-3
    text = " ".join(about_text(width=88).split())
    # BOTH halves: the band's width and the effect expressed in it. Asserting only the
    # phrase "tercile band" passes against text that gives the unit without the size.
    assert f"{TERCILE_BAND:.4f} wide" in text
    assert f"{EFFECT_IN_BANDS:.2f} of one tercile band" in text


def test_the_footer_is_one_line():
    """A paragraph would be buried and a headline would oversell. One line."""
    assert "\n" not in FOOTER
    assert len(FOOTER) <= 160


def test_about_carries_every_disclosure():
    """The things that must not be left out of `--about`, each pinned to its number."""
    text = about_text(width=88)
    assert "+0.0604" in text and "+0.1295" in text, "both the residualised and raw figures"
    assert "53%" in text and "47%" in text, "the real discount is the frequency one"
    assert "no zero-token baseline" in text, "why the nonsense value is not a share"
    assert "+0.0491" in text and "+0.0804" in text, "blue is displaced from BOTH endpoints"
    assert "88 distinct" in text and "265" in text, "the vocabulary narrowing"
    assert "please/what/now" in text, "and what it narrows TO"
    assert "3000" in text and "not converged" in text.lower(), "the model is undertrained"
    assert "FAILED" in text and "slot-hit" in text, "the gate that failed"
    assert "dial:near" in text, "and that the worst setting is near, not far"
    assert "60% of the time" in text, "the degenerate way to be near: copy a scene word"


def test_the_quoted_figures_match_the_published_artifact():
    """The transcribed constants ARE the artifact's numbers. Pinned, not remembered.

    `scripts/reach.py` hardcodes these so the footer costs nothing at startup and cannot
    fail when the artifact is absent. That is only safe if something checks them, and this
    is that something: it reads `docs/measurements/reach-dial.json` and compares each one to
    the field the source comment names.
    """
    if not MEASUREMENT.is_file():
        pytest.skip(f"{MEASUREMENT} not present")
    m = json.loads(MEASUREMENT.read_text())
    eff = m["effects"]
    assert FACTS["raw_near_to_far"] == eff["raw_distance"]["near_lt_far"]["mean_delta"]
    assert (FACTS["residualised_near_to_far"]
            == eff["frequency_residualised_distance"]["near_lt_far"]["mean_delta"])
    assert (FACTS["mid_to_far_t"]
            == eff["frequency_residualised_distance"]["mid_lt_far"]["t"])
    ctrl = m["controls"]["nonsense_value_PRIMARY"]
    assert FACTS["nonsense_vs_near_delta"] == ctrl["vs_near"]["mean_delta"]
    assert FACTS["nonsense_vs_near_t"] == ctrl["vs_near"]["t"]
    assert FACTS["far_vs_nonsense_delta"] == ctrl["vs_far"]["mean_delta"]
    assert FACTS["far_vs_nonsense_t"] == ctrl["vs_far"]["t"]
    assert ctrl["vs_near"]["significant"] is True, "blue is displaced from near"
    cuts = m["design"]["cut_points"]
    assert FACTS["tercile_lo"] == cuts["lo"] and FACTS["tercile_hi"] == cuts["hi"]
    near = m["per_setting"]["dial:near"]
    assert FACTS["near_copy_rate"] == near["add_word_already_in_the_context_rate"]
    top = tuple(w["word"] for w in m["per_setting"]["dial:far"]["top_add_words"])
    assert FACTS["far_top_add_words"] == top[:len(FACTS["far_top_add_words"])]
    for value, n in FACTS["per_setting_distinct_add_words"].items():
        assert m["per_setting"][f"dial:{value}"]["distinct_add_words"] == n
    adh = m["adherence_guard"]
    assert FACTS["adherence_worst_setting"] == adh["worst_setting"]
    assert FACTS["adherence_shortfall"] == adh["shortfall"]
    assert adh["passes"] is False, "--about says a pre-declared gate failed; it no longer does"
    assert FACTS["n_scenes"] == m["per_setting"]["dial:near"]["n"]
    assert abs(SURVIVES_FREQUENCY_CONTROL
               - eff["frequency_residualised_distance"]["near_lt_far"]["mean_delta"]
               / eff["raw_distance"]["near_lt_far"]["mean_delta"]) < 1e-12


# ---------------------------------------------------------------------------------------
# 3. Terminal output rules.
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", [44, 52, 60, 72, 80, 100])
def test_nothing_is_wider_than_the_terminal(width):
    """Wrap to the ACTUAL width. Long scenes, long offers, long turns, every width."""
    long_scene = ("Once upon a very long time ago there was an extraordinarily lonely "
                  "lighthouse keeper who counted every single wave that came in.")
    arms = [Arm(value=v,
                segments=arm_segments(long_scene, " ".join(["lighthouse"] * 12),
                                      "lighthouse keeper waves", v),
                raw="", add_value="phosphorescence",
                turn="The waves kept arriving all night long and he counted every one of "
                     "them until the sun came up over the cold grey water.",
                closed_block=(v != "far"))
            for v in REACH_VALUES]
    for show_think in (True, False):
        for line in render_run(long_scene, arms, model_label="ckpt-dial @ step 3000",
                               seed=1, temperature=0.8, width=width, ink=PLAIN,
                               show_think=show_think):
            assert len(line) <= width, f"{len(line)} > {width}: {line!r}"


def test_no_right_hand_border_characters():
    """House rule: left-side and bottom bars only.

    A right-hand border is drawn at the width the author assumed and shatters at any other,
    which on a terminal is most of the time.
    """
    arms = [_arm(v) for v in REACH_VALUES]
    lines = render_run(SCENE, arms, model_label="m", seed=1, temperature=0.0, width=80,
                       ink=PLAIN, show_think=True)
    for line in lines:
        assert not line.rstrip().endswith(("║", "╗", "╝", "│", "┐", "┘")), line
        for ch in "╗╝┐┘":
            assert ch not in line, line


def test_colour_degrades_to_plain_text():
    """`--no-color`, a pipe, and NO_COLOR all produce output with no escape byte in it."""
    arms = [_arm(v) for v in REACH_VALUES]
    plain = "\n".join(render_run(SCENE, arms, model_label="m", seed=1, temperature=0.0,
                                 width=80, ink=Ink(False), show_think=True))
    assert "\x1b" not in plain
    coloured = "\n".join(render_run(SCENE, arms, model_label="m", seed=1, temperature=0.0,
                                    width=80, ink=Ink(True), show_think=True))
    assert "\x1b" in coloured, "Ink(True) produced no colour at all"


class _Stream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_want_colour_says_no_when_piped(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert want_colour(None, _Stream(False)) is False
    assert want_colour(None, _Stream(True)) is True


def test_want_colour_honours_no_color_and_the_explicit_flags(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert want_colour(None, _Stream(True)) is False
    assert want_colour(True, _Stream(False)) is True     # --color beats the pipe
    assert want_colour(False, _Stream(True)) is False    # --no-color beats the tty


def test_a_piped_stream_gets_a_fixed_width(monkeypatch):
    """Not the launching terminal's width: a pasted run must not depend on the pane it
    happened to be produced in.

    COLUMNS is set to a value that is NOT the default first. Without that this test passes
    against a `terminal_width` that ignores the pipe entirely, because the terminal these
    tests run in is 80 columns wide -- the instrument would be reporting itself.
    """
    monkeypatch.setenv("COLUMNS", "137")
    assert terminal_width(default=80, stream=_Stream(False)) == 80
    # ... and a real terminal DOES get its own width, clamped to something readable.
    assert terminal_width(default=80, stream=_Stream(True)) == 100


def test_a_very_narrow_terminal_is_clamped_up(monkeypatch):
    """Below ~44 columns whole words stop fitting; the lower clamp keeps the output legible
    rather than shredding every line into fragments."""
    monkeypatch.setenv("COLUMNS", "20")
    assert terminal_width(default=80, stream=_Stream(True)) == 44


def test_the_think_block_can_be_switched_off():
    arm = _arm("far")
    with_think = "\n".join(render_arm(arm, width=80, ink=PLAIN, show_think=True))
    without = "\n".join(render_arm(arm, width=80, ink=PLAIN, show_think=False))
    assert "think:" in with_think and "reach=far" in with_think
    assert "think:" not in without
    assert "FAR" in without and "He kept the light." in without


def _flat(lines):
    return " ".join(" ".join(lines).split())


def test_identical_arms_are_disclosed_in_the_rendered_run():
    """Two settings landing on the same continuation must be SAID, and said in the OUTPUT.

    The arms are deliberately seeded identically -- that is what makes the dial the only
    difference between them -- so when the dial does not flip a sampling decision they
    produce the same text. A demo that hid this would be quietly implying an effect it did
    not produce.

    Asserted through `render_run`, not through `render_collisions` alone: the failure this
    guards against is the note never being WIRED IN, and a test that calls the note's own
    function directly passes happily while nothing calls it.
    """
    same = [_arm("near", turn="What is that?", add="what"),
            _arm("mid", turn="What is that?", add="what"),
            _arm("far", turn="A comet landed.", add="comet")]
    text = _flat(render_run(SCENE, same, model_label="m", seed=1, temperature=0.8,
                            width=80, ink=PLAIN, show_think=True))
    assert "near/mid came out identical" in text
    assert "near/mid/far" not in text, "far differed and must not be named in the group"


def test_all_three_identical_is_disclosed_as_one_group():
    same = [_arm(v, turn="What is that?", add="what") for v in REACH_VALUES]
    assert "near/mid/far came out identical" in _flat(
        render_run(SCENE, same, model_label="m", seed=1, temperature=0.8, width=80,
                   ink=PLAIN, show_think=True))


def test_arms_that_differ_only_in_the_add_word_are_not_called_identical():
    """The `add` word is half the visible result, so two arms that named DIFFERENT words are
    not the same outcome even when the sentence they produced is word-for-word the same.

    This is the case that separates a collision check on the whole result from one on the
    turn alone -- and the turn-only version is the natural way to write it.
    """
    arms = [_arm("near", turn="What is that?", add="what"),
            _arm("mid", turn="What is that?", add="thing"),
            _arm("far", turn="A comet landed.", add="comet")]
    assert render_collisions(arms, width=80, ink=PLAIN) == []
    assert "identical" not in _flat(render_run(SCENE, arms, model_label="m", seed=1,
                                               temperature=0.8, width=80, ink=PLAIN,
                                               show_think=True))


def test_three_different_arms_get_no_note():
    diff = [_arm("near", turn="A.", add="a"), _arm("mid", turn="B.", add="b"),
            _arm("far", turn="C.", add="c")]
    assert render_collisions(diff, width=80, ink=PLAIN) == []


def test_an_unclosed_block_is_disclosed():
    """A truncated generation must not read as a finished one."""
    text = "\n".join(render_arm(_arm("far", closed=False), width=80, ink=PLAIN,
                                show_think=True))
    assert "the block never closed" in text


# ---------------------------------------------------------------------------------------
# 4. Prompt construction against the real derivation.
# ---------------------------------------------------------------------------------------
def test_derived_offer_matches_the_gold_block_0_offer():
    """`derive_offer` reproduces `train.skit._slots_for_turn`'s block-0 offer, on real skits.

    A property of the REAL derivation, so it is tested against the real artifact: a fixture
    scene would not have the twelve-word truncation, the stopword list or the punctuation
    that the corpus does. Every eval row is checked, not a sample.
    """
    if not SKITS.is_file():
        pytest.skip(f"{SKITS} not present")
    checked = 0
    with SKITS.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "eval":
                continue
            assert derive_offer(row["prefix"]) == row["blocks"][0]["offer"], row["story_id"]
            checked += 1
    assert checked > 100, f"only {checked} eval rows checked -- not a real test of the rule"


def test_parse_accept_stops_at_the_first_newline():
    """The accept probe keeps generating past the slot; the newline is the boundary.

    Taking more would put the model's own (unforced) `reach:` line into the accept slot,
    and the forced dial would then be the SECOND reach line in the block.
    """
    assert parse_accept(" lighthouse\nreach: far\nadd: what's\n") == "lighthouse"
    assert parse_accept("  keeper boat  \n") == "keeper boat"
    assert parse_accept("") == ""


def test_parse_accept_truncates_to_the_schemas_own_cap():
    """`train.skit._slots_for_turn` joins `carried[:6]`, so no gold accept is longer.

    The model does over-run it. A seven-word accept fed back into the forced prompt is a
    shape training never saw, at exactly the seam the segment-wise tokenization exists to
    keep faithful.
    """
    assert ACCEPT_MAX_WORDS == 6
    assert parse_accept("lily lily lily lily lily lily lily\n") == "lily " * 5 + "lily"
    assert len(parse_accept("a b c d e f g h\n").split()) == 6


def test_no_gold_accept_is_longer_than_the_cap():
    """A property of the REAL derivation at scale, so it is checked against the real file."""
    if not SKITS.is_file():
        pytest.skip(f"{SKITS} not present")
    longest = 0
    with SKITS.open() as fh:
        for line in fh:
            if line.strip():
                for block in json.loads(line)["blocks"]:
                    longest = max(longest, len(block["accept"].split()))
    assert longest == ACCEPT_MAX_WORDS, f"gold accepts reach {longest} words, cap is {ACCEPT_MAX_WORDS}"


def test_the_cli_never_imports_a_device_library():
    """Importing `scripts.reach` must not pull in ttml/ttnn -- a bare import IS a device
    open, and this tool is explicitly CPU-only.

    Run in a SUBPROCESS, not against this process's `sys.modules`. In a full-suite run some
    other test has already imported ttnn, so the in-process form fails for a reason that has
    nothing to do with this module -- and, worse, in a single-file run it PASSES no matter
    what `scripts/reach.py` does, because nothing else in the session imports it either. A
    fresh interpreter is the only place the question can actually be asked.
    """
    probe = ("import sys, importlib; importlib.import_module('scripts.reach'); "
             "bad = [m for m in ('ttml', 'ttnn') if m in sys.modules]; "
             "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True,
                         text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                         "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"scripts.reach imported {out.stdout.strip()}"


def test_about_does_not_call_mid_vs_far_the_cleaner_contrast():
    """The spec designated mid<far as `cleaner` from the DERIVATION's frequency distribution,
    where mid had the highest median add_df. THIS run's realised confound is monotone
    (mean log add_df 10.79 -> 11.72 -> 12.03), so the designation does not describe this
    measurement. Printing it to a user states a false reason for a true number."""
    text = about_text().lower()
    assert "cleanest contrast" not in text, "--about revived the spec's designation as a finding"
    assert "cleaner contrast" not in text
    assert "confound is not monotone" not in text, (
        "--about states the confound is not monotone; in this run it IS monotone")
