#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Turn the reach dial by hand and watch the model reach further. CPU only.

WHAT THIS IS
------------
`docs/measurements/reach-dial.json` proves that forcing ``reach: near|mid|far`` into the
model's think-block moves the semantic distance of the word it then reaches for. That proof
is a table of t-values. This is the same thing you can FEEL: give it a scene opening, it
runs the same scene three times with only the dial changed, and prints the three plans and
the three lines side by side.

    python scripts/reach.py "Once upon a time there was a lonely lighthouse keeper."
    python scripts/reach.py                      # a held-out eval scene, picked at random
    python scripts/reach.py --reach far --seed 7 --temperature 0.8
    python scripts/reach.py --about              # what the dial is, and what it is not

THIS IS A DEMO, NOT THE MEASUREMENT
-----------------------------------
It reuses `scripts/eval_reach.py`'s prompt construction (`prompt_segments`,
`encode_segments`, `forced_block_prefix`) and its parser (`parse_forced_generation`), and
`scripts/eval_improv.py`'s CPU generation, so what you see here comes off exactly the path
the measurement generated on. Two things it deliberately does NOT do:

  * **It does not score the realised distance.** Scoring needs the corpus association table,
    and even the targeted pass `eval_reach.py` uses costs ~81 seconds of corpus streaming.
    Every number this tool quotes is read out of the published artifact, not recomputed. One
    run tells you nothing about the effect; it is n=1 against an effect of +0.060.
  * **It measures the FIRST model turn (block 0), not the measured turn (block 1).** The
    measurement forces `offer` and `accept` from a gold row so the ONLY thing differing
    between settings is four characters of dial. A typed scene has no gold row, so here
    `offer` is derived from the scene by the same rule `train.skit._slots_for_turn` uses for
    block 0 (the prefix's first twelve content words) and `accept` is GENERATED ONCE by the
    model and then frozen across all three arms. The dial value is still the only difference
    between the three prompts, which is the property that matters -- but the block index and
    the provenance of `accept` differ from the published run, so a striking output here is an
    anecdote, not a data point.

CPU ONLY, NO DEVICE
-------------------
Checkpoint -> HF dir -> `transformers`, exactly like `scripts/chat.py`. Nothing in this file
imports `ttml` or `ttnn` and nothing opens `/dev/tenstorrent/*`.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Silence TensorFlow's registration banner before `transformers` can probe for a TF backend.
# transformers only prints it because it looks for TF at import time; we are a torch-only
# tool, so the banner is pure noise on a CLI whose whole job is legible output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------------------
#: The dial arm. `artifacts/reach/ckpt-dial` holds the SFT checkpoints of the arm that was
#: trained on the six-slot schema WITH the `reach` line; `step_3000.pkl` is the 3000-step
#: checkpoint the published measurement generated from, and is what this tool loads.
DEFAULT_MODEL = ROOT / "artifacts" / "reach" / "ckpt-dial"
DIAL_STEP_PKL = "step_3000.pkl"

#: Where `scripts/eval_reach.py` already left its converted HF directories. Reused verbatim
#: when it is at least as new as the checkpoint, so the common case costs no conversion.
EVAL_HF_CACHE = ROOT / "artifacts" / "reach" / "hf-eval"
#: Our own conversion cache, used only when the eval's is missing or stale. Separate on
#: purpose: this tool must never overwrite an artifact the eval is reading.
CLI_HF_CACHE = ROOT / "artifacts" / "reach" / "hf-cli"

SKITS = ROOT / "artifacts" / "reach-skits" / "skits.jsonl"
MEASUREMENT = ROOT / "docs" / "measurements" / "reach-dial.json"

#: Needed by `sft_checkpoint_to_hf` to recover the architecture header (the SFT checkpoints
#: carry weights only). Same two values `scripts/eval_reach.py` uses.
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024"

#: Used only when no scene is given AND the held-out skits file is unavailable, so that
#: `python scripts/reach.py` with zero arguments works on a fresh clone too.
FALLBACK_SCENES: Tuple[str, ...] = (
    "Once upon a time there was a lonely lighthouse keeper.",
    "Mia found a rusty key in the garden. She wondered what it opened.",
    "Tom and Anna love snow. They want to play outside.",
)

# ---------------------------------------------------------------------------------------
# The measured facts. QUOTED, not recomputed -- and pinned to the artifact by a test.
# ---------------------------------------------------------------------------------------
# Transcribed from `docs/measurements/reach-dial.json` so that the honesty footer is
# unconditional and costs nothing at startup. They are NOT allowed to drift:
# `tests/test_reach_cli.py::test_the_quoted_figures_match_the_published_artifact` reads the
# artifact and fails if any of them stops matching.
#
# TWO FIGURES WERE RETRACTED, 2026-08-24, and must not come back:
#   * "38% of the near->far range is reproduced by a nonsense dial value, so ~62% is
#     value-specific". MIS-SPECIFIED. There is no zero-token baseline in the experiment (no
#     dial-arm, five-slot, no-`reach`-line condition was ever run), so `blue`'s position on
#     the range cannot be decomposed into a schema share and a value share at all. What the
#     data shows is stronger, not weaker: `blue` is significantly displaced from BOTH
#     endpoints in OPPOSITE directions, and an off-vocabulary token cannot itself encode
#     "near" or "far", so the near<->far ORDERING is value-specific. 38% is a position on a
#     range, never a share of an effect.
#   * "+0.0140, the df-matched subsample, a second frequency control that DISAGREES".
#     WRONG TWICE. The df-matched subsample is a null-enrichment filter, not a second
#     control: when the dial emits the SAME word in both conditions, delta-log-df is exactly
#     0 and the distance difference is exactly 0, so matching keeps 100% of the scenes where
#     the dial did nothing and ~7% of the scenes where it changed the word. The published
#     +0.0140 is a mean over a sample that is 78% definitional zeros. Restricted to the
#     INFORMATIVE pairs it is +0.063, which AGREES with the residualised +0.0604.
# `test_the_retracted_figures_appear_nowhere` fails if either number is printed again.
FACTS: Dict[str, Any] = {
    # effects.raw_distance.near_lt_far.mean_delta -- the number NOT to quote alone.
    "raw_near_to_far": 0.129473,
    # effects.frequency_residualised_distance.near_lt_far.mean_delta -- the headline.
    "residualised_near_to_far": 0.060438,
    # effects.frequency_residualised_distance.mid_lt_far.t.
    # NOT "the cleanest contrast". The spec designated mid<far as cleaner on the basis of the
    # DERIVATION's frequency distribution, where mid had the highest median add_df. In THIS run
    # the realised confound is monotone (mean log add_df 10.79 -> 11.72 -> 12.03), so that
    # designation does not describe this measurement. All three steps survive frequency control.
    "mid_to_far_t": 8.954,
    # controls.nonsense_value_PRIMARY.vs_near / .vs_far -- the nonsense dial value `blue`,
    # significantly displaced from BOTH endpoints, in opposite directions.
    "nonsense_vs_near_delta": 0.049069,
    "nonsense_vs_near_t": 11.2862,
    "far_vs_nonsense_delta": -0.080404,
    "far_vs_nonsense_t": -17.8011,
    # design.cut_points -- the two numbers that ARE the dial. Their gap is one tercile band,
    # which is the only unit in which the effect size means anything to a person.
    "tercile_lo": 0.7184308448481754,
    "tercile_hi": 0.8243294716769457,
    # per_setting.dial:*.distinct_add_words -- the dial narrows as it reaches.
    "per_setting_distinct_add_words": {"near": 265, "mid": 105, "far": 88},
    # per_setting.dial:far.top_add_words -- and what it narrows TO.
    "far_top_add_words": ("please", "what", "now", "let's", "can't"),
    # per_setting.dial:near.add_word_already_in_the_context_rate -- the degenerate way to be
    # near: copy a word already in the scene. Highest at `near`, which is also where the
    # failed gate bites.
    "near_copy_rate": 0.598063,
    "adherence_worst_setting": "dial:near",
    "adherence_shortfall": 0.089588,
    "n_scenes": 826,
    "steps_trained": 3000,
}

#: How much of the RAW near->far movement survives frequency control. This is the real
#: discount, and both frequency controls agree on it. Derived, never written as "47%", so
#: the footer and `--about` cannot disagree with the two numbers they came from.
SURVIVES_FREQUENCY_CONTROL = (FACTS["residualised_near_to_far"] / FACTS["raw_near_to_far"])

#: The effect in units a person can feel: one tercile band is the width of the `mid` bucket,
#: i.e. the gap between the two cut points that ARE the dial.
TERCILE_BAND = FACTS["tercile_hi"] - FACTS["tercile_lo"]
EFFECT_IN_BANDS = FACTS["residualised_near_to_far"] / TERCILE_BAND

#: The frequency-MATCHED near<far effect restricted to the INFORMATIVE pairs -- the scenes
#: where the dial actually changed the word. NOT A FIELD IN THE ARTIFACT and therefore not
#: pinned by the artifact test: the published `frequency_matched_subsample` figure (+0.0140)
#: is a mean over a sample that is 78% definitional zeros, because a pair whose two
#: conditions emitted the SAME word has delta-log-df exactly 0 and so is always kept by the
#: matcher. Restricted to the pairs that carry information the two frequency controls agree.
#: Source: the 2026-08-24 review of docs/measurements/reach-dial.json, not the artifact.
INFORMATIVE_DF_MATCHED = 0.063

#: One line, on every run, under the output. Not a paragraph, not buried.
FOOTER = (f"dial effect: +{FACTS['residualised_near_to_far']:.3f} residualised "
          f"(~{SURVIVES_FREQUENCY_CONTROL * 100:.0f}% of raw survives frequency control); "
          f"far narrows to {FACTS['per_setting_distinct_add_words']['far']} add-words vs "
          f"near's {FACTS['per_setting_distinct_add_words']['near']}; model undertrained")

REACH_VALUES: Tuple[str, ...] = ("near", "mid", "far")

# ---------------------------------------------------------------------------------------
# Colour, and doing without it.
# ---------------------------------------------------------------------------------------
#: near is cool, far is hot. Chosen to read as a dial even in a 16-colour terminal.
_ARM_COLOUR = {"near": "36", "mid": "32", "far": "33"}
_DIM, _BOLD, _RESET = "2", "1", "\x1b[0m"


class Ink:
    """Colour that can be switched off wholesale.

    A single object rather than a module flag so the formatting functions stay pure: a test
    renders with `Ink(False)` and compares plain strings, and the colour path cannot leak
    into the assertions.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}{_RESET}" if self.enabled else text

    def arm(self, value: str, text: str) -> str:
        return self._wrap(_ARM_COLOUR.get(value, "0") + ";1", text)

    def dim(self, text: str) -> str:
        return self._wrap(_DIM, text)

    def bold(self, text: str) -> str:
        return self._wrap(_BOLD, text)


def want_colour(flag: Optional[bool], stream=None) -> bool:
    """Whether to emit ANSI. `flag` is the explicit --color/--no-color, or None for auto.

    Auto means: a TTY, and `NO_COLOR` unset (https://no-color.org). Piping into a file must
    produce plain text, because the most likely destination of a striking run is a paste.
    """
    if flag is not None:
        return flag
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def terminal_width(default: int = 80, *, stream=None) -> int:
    """The real terminal width, clamped to something a paragraph can live in.

    Clamped ABOVE as well as below: a 300-column terminal would otherwise produce lines no
    eye can track back. Clamped below so a 30-column pane still gets whole words.
    """
    if stream is not None and not getattr(stream, "isatty", lambda: False)():
        return default
    cols = shutil.get_terminal_size((default, 24)).columns
    return max(44, min(cols, 100))


# ---------------------------------------------------------------------------------------
# Prompt construction. Pure -- no model, no tokenizer.
# ---------------------------------------------------------------------------------------
def derive_offer(scene: str) -> str:
    """The `offer` line for block 0, by the rule `train.skit._slots_for_turn` applies there.

    Block 0 has no partner turn yet, so its offer text is the WHOLE prefix, and the slot is
    its first twelve content words. Reimplemented here in one line rather than reached
    through `_slots_for_turn` because that function also needs the model's turn (which is
    the thing we are about to generate) and returns None when the turn accepts nothing.
    `tests/test_reach_cli.py::test_derived_offer_matches_the_gold_block_0_offer` pins this
    against real derived skits, so the two rules cannot drift apart silently.
    """
    from train.improv import content_words

    words = content_words(scene)
    return " ".join(words[:12]) or scene[:60]


def accept_probe_segments(scene: str, offer: str) -> List[str]:
    """Prompt segments that stop at ``accept:``, so the model chooses the accept itself.

    Segments, not a string: `scripts/derive_skits.py:build_skit_example` encoded each skit
    segment with its own `tok.encode` call and this tokenizer prepends a space per call, so
    training saw ``['.', 'Ġ<', 'think', '>']`` at the seam. Joining first and tokenizing once
    produces a bare ``'<'`` the model never saw there. See `scripts/eval_reach.py`'s module
    docstring; this is the same construction, one slot earlier.
    """
    return [scene, f"<think>\noffer: {offer}\naccept:"]


#: `train.skit._slots_for_turn` writes ``" ".join(carried[:6])``, so no gold `accept` is
#: longer than six words -- checked across all 41,014 derived skits, whose length histogram
#: tops out at exactly 6. A generated one CAN be longer (the model loops), and feeding a
#: seven-word accept back in would put the forced prompt off-schema at the seam this whole
#: design exists to keep on-schema.
ACCEPT_MAX_WORDS = 6


def parse_accept(generated: str) -> str:
    """The `accept` slot out of the probe's continuation: up to the first newline, six words.

    The model keeps going past the line (it will happily write the whole block), so the
    newline is the boundary; taking more would put its own unforced `reach:` line inside the
    accept slot and make the forced dial the SECOND reach line in the block.

    Truncated to `ACCEPT_MAX_WORDS` because the schema's own derivation truncates there. The
    model does over-run it -- greedy decoding on an unconverged 123M model produces
    ``"lily lily lily lily lily lily lily"`` -- and a seven-word accept is a shape training
    never saw.
    """
    return " ".join(generated.split("\n", 1)[0].split()[:ACCEPT_MAX_WORDS])


def arm_segments(scene: str, offer: str, accept: str, value: str) -> List[str]:
    """The prompt for ONE dial setting, built through `eval_reach.prompt_segments`.

    Routed through the eval's own function (with `block_index=0`) rather than assembled here,
    so the demo cannot drift from the measurement's prompt shape. `prompt_segments` at block 0
    touches only `prefix` and `blocks[0]`, which is why a two-key synthetic row is enough;
    the `add`/`stakes`/`handback` values are never read (everything from `add:` onward is
    what the model has to produce) and are present only to satisfy the slot dataclass.
    """
    from scripts.eval_reach import prompt_segments

    row = {"prefix": scene,
           "blocks": [{"offer": offer, "accept": accept, "reach": value,
                       "add": "", "stakes": "", "handback": ""}]}
    return prompt_segments(row, value, block_index=0)


_REACH_LINE = re.compile(r"^reach:\s*(\S+)\s*$", re.M)


def forced_reach_in(segments: Sequence[str]) -> str:
    """Read the dial value back OUT of the prompt that will actually be sent.

    THE POINT OF THIS FUNCTION. The rendered output reports the dial by parsing the prompt,
    never by echoing the flag the user typed. A CLI that built all three prompts with the
    same dial value, or ignored `--reach` entirely, would still print a perfect-looking
    ``reach=far`` if the label travelled alongside the prompt instead of being read from it.
    Raises rather than returning a default: a prompt with no forced `reach:` line is not a
    reach-dial run at all.
    """
    for seg in reversed(list(segments)):
        m = _REACH_LINE.search(seg)
        if m:
            return m.group(1)
    raise ValueError("no forced `reach:` line in the prompt segments")


def selected_values(reach: str) -> List[str]:
    """Which arms to run. `all` means the side-by-side, which is the point of the tool."""
    if reach == "all":
        return list(REACH_VALUES)
    if reach not in REACH_VALUES:
        raise ValueError(f"unknown reach {reach!r}; expected one of {REACH_VALUES} or 'all'")
    return [reach]


# ---------------------------------------------------------------------------------------
# One arm's result.
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    """One dial setting, its prompt, and what came back.

    `segments` is the authority on which dial value this arm ran: `forced_reach_in` reads it
    back for display. `value` is carried only for ordering and colour.
    """
    value: str
    segments: List[str]
    raw: str
    add_value: Optional[str]
    turn: str
    closed_block: bool


# ---------------------------------------------------------------------------------------
# Rendering. Pure functions over `Arm`s -- no model, no terminal, no colour by default.
# ---------------------------------------------------------------------------------------
def _bar(width: int) -> str:
    """House style: left-side and bottom bars only. A right-hand border breaks the moment
    the terminal is narrower than the author assumed, and it always is."""
    return "═" * max(10, width - 2)


def _wrapped(text: str, *, width: int, indent: str, hang: str) -> List[str]:
    """Wrap to the real terminal width, hanging-indented. Never returns an empty list."""
    body = textwrap.wrap(text, width=max(20, width), break_long_words=False,
                         break_on_hyphens=False) or [""]
    return [indent + body[0]] + [hang + line for line in body[1:]]


def render_header(scene: str, *, model_label: str, seed: int, temperature: float,
                  width: int, ink: Ink) -> List[str]:
    """The banner: what was asked, of which model, with which knobs.

    Every body line goes through `_wrapped` at `width - 3` (the "║  " gutter), including the
    title: at 44 columns the model label alone is wider than the pane, and an over-long
    banner is exactly the failure the no-right-border rule exists to avoid.
    """
    inner = width - 3
    decode = ("greedy — the decoding the measurement used" if temperature <= 0
              else f"sampled at T={temperature:g} — rerun with --seed {seed}")

    def body(text: str, paint) -> List[str]:
        return ["║  " + paint(ln) for ln in _wrapped(text, width=inner, indent="", hang="")]

    lines = [f"╔{_bar(width)}"]
    title, label = "the reach dial", f"  ·  {model_label}"
    if len(title) + len(label) <= inner:
        lines.append("║  " + ink.bold(title) + ink.dim(label))
    else:
        lines.append("║  " + ink.bold(title))
        lines.extend(body(model_label, ink.dim))
    # The scene gets a hanging indent under its own label, so a wrapped scene still reads as
    # one field rather than as two.
    for i, ln in enumerate(_wrapped(scene, width=inner - 7, indent="", hang="")):
        lines.append("║  " + (ink.dim("scene  ") if i == 0 else "       ") + ln)
    lines.extend(body(f"seed {seed}  ·  {decode}", ink.dim))
    # Provenance, once, rather than repeating the two shared slots per arm: `offer` is
    # derived from the scene, `accept` is the model's own and is frozen, so the dial value
    # is the ONLY thing that differs between the arms below.
    lines.extend(body("offer derived from the scene; accept generated once and frozen, so "
                      "the dial is the only difference between the arms", ink.dim))
    lines.append(f"╚{_bar(width)}")
    return lines


def render_arm(arm: Arm, *, width: int, ink: Ink, show_think: bool) -> List[str]:
    """One arm: the plan it was handed, and the line it produced.

    The `reach=` shown is read from `arm.segments` by `forced_reach_in`, i.e. from the prompt
    the model was actually given. See that function for why.
    """
    forced = forced_reach_in(arm.segments)
    label = ink.arm(arm.value, f"{arm.value.upper():<5}")
    lines: List[str] = []
    if show_think:
        offer = _slot_of(arm.segments, "offer")
        accept = _slot_of(arm.segments, "accept")
        add = arm.add_value if arm.add_value else "—"
        think = f"think: offer={offer}  accept={accept}  reach={forced}  add={add}"
        wrapped = _wrapped(think, width=width - 8, indent="", hang="")
        lines.append(f"  {label} {ink.dim(wrapped[0])}")
        lines.extend(f"        {ink.dim(w)}" for w in wrapped[1:])
        turn_indent, turn_hang = "        ", "        "
    else:
        turn_indent, turn_hang = f"  {label} ", "        "
    if arm.turn.strip():
        lines.extend(_wrapped(f"“{arm.turn.strip()}”", width=width - 8,
                              indent=turn_indent, hang=turn_hang))
    if not arm.closed_block:
        # One note, not two: an unclosed block is WHY there is no turn, so saying both
        # "no turn" and "block never closed" would report one fact twice.
        note = "(no turn — the block never closed; try --max-new-tokens)"
        lines.extend(ink.dim(ln) for ln in
                     _wrapped(note, width=width - 8, indent=turn_indent if not arm.turn.strip()
                              else "        ", hang="        "))
    return lines


def render_collisions(arms: Sequence[Arm], *, width: int, ink: Ink) -> List[str]:
    """Disclose when two settings produced the SAME continuation.

    They can, and it is not a bug: every arm is seeded identically on purpose (that is what
    makes the dial the only difference between them), so when the dial does not flip an
    actual sampling decision the three arms take the same draws and land on the same text.
    Saying so is the difference between a demo that looks broken and one that is honest —
    and it is a standing reminder that one run is n=1 against an effect of +0.060.
    """
    groups: Dict[Tuple[Optional[str], str], List[str]] = {}
    for arm in arms:
        groups.setdefault((arm.add_value, arm.turn), []).append(arm.value)
    dupes = [names for names in groups.values() if len(names) > 1]
    if not dupes:
        return []
    which = " and ".join("/".join(names) for names in dupes)
    note = (f"{which} came out identical here — the arms share a seed, so the dial only "
            f"shows when it flips a decision. One scene is n=1.")
    return [""] + [ink.dim(ln) for ln in
                   _wrapped(note, width=width - 2, indent="  ", hang="  ")]


def _slot_of(segments: Sequence[str], name: str) -> str:
    """A named slot's value out of the forced block prefix, for display."""
    m = re.search(rf"^{name}:\s*(.*)$", segments[-1], re.M)
    return m.group(1).strip() if m else ""


def render_footer(width: int, ink: Ink) -> List[str]:
    """The one honest line that goes under every run.

    Wrapped like everything else -- on a narrow pane an over-long footer would be the FIRST
    thing the terminal mangles, which is the opposite of the point.
    """
    return [""] + [ink.dim(ln) for ln in
                   _wrapped(FOOTER, width=width - 2, indent="  ", hang="  ")]


def render_run(scene: str, arms: Sequence[Arm], *, model_label: str, seed: int,
               temperature: float, width: int, ink: Ink,
               show_think: bool) -> List[str]:
    """The whole page. One function so a test can assert on the exact bytes a user sees."""
    lines = render_header(scene, model_label=model_label, seed=seed, temperature=temperature,
                          width=width, ink=ink)
    for arm in arms:
        lines.append("")
        lines.extend(render_arm(arm, width=width, ink=ink, show_think=show_think))
    lines.extend(render_collisions(arms, width=width, ink=ink))
    lines.extend(render_footer(width, ink))
    return lines


# ---------------------------------------------------------------------------------------
# --about.
# ---------------------------------------------------------------------------------------
def about_text(width: int = 80) -> str:
    """What the dial is, and — at the same volume — what it is not.

    Every number here is a `FACTS` key or derived from one, so `--about`, the footer and the
    artifact cannot disagree. Two figures that an earlier draft of this text carried were
    RETRACTED (see the comment above `FACTS`); `test_the_retracted_figures_appear_nowhere`
    fails if they reappear.
    """
    f = FACTS
    top = "/".join(f["far_top_add_words"][:4])
    paras = [
        "THE REACH DIAL",
        "",
        "A 123M-parameter model was fine-tuned on story 'skits' where every model turn is "
        "preceded by a six-slot think-block: offer, accept, reach, add, stakes, handback. "
        "`reach` is a declared intention — near, mid or far — for how semantically distant "
        "the next word it reaches for should be. This tool forces that one slot and lets "
        "everything else stay identical.",
        "",
        "WHAT WAS MEASURED",
        "",
        f"Over {f['n_scenes']} held-out scenes, scene-paired, forcing the dial moved the "
        f"realised distance of the model's `add` word monotonically: near < mid < far, all "
        f"three steps significant at a Bonferroni-corrected threshold after frequency control. "
        f"mid vs far reached t={f['mid_to_far_t']:.1f}.",
        "",
        "HOW BIG, HONESTLY",
        "",
        f"Raw near→far is +{f['raw_near_to_far']:.4f}, but NPMI rewards rare words, so a "
        f"farther-scoring word is partly just a commoner word. That is the real discount: "
        f"about {(1 - SURVIVES_FREQUENCY_CONTROL) * 100:.0f}% of the raw movement is "
        f"attributable to word frequency and about "
        f"{SURVIVES_FREQUENCY_CONTROL * 100:.0f}% survives it. Residualised on log document "
        f"frequency the effect is +{f['residualised_near_to_far']:.4f} — that is the number "
        f"to quote, never the raw one alone. A frequency-MATCHED check over the pairs where "
        f"the dial actually changed the word gives +{INFORMATIVE_DF_MATCHED:.4f}; the two "
        f"frequency controls agree.",
        "",
        f"For scale: the dial's own buckets are {TERCILE_BAND:.4f} wide (the gap between the "
        f"two fitted cut points that ARE the dial), so the effect is about "
        f"{EFFECT_IN_BANDS:.2f} of one tercile band. Real, and small.",
        "",
        "THE NONSENSE-VALUE CONTROL",
        "",
        f"A nonsense dial value (`reach: blue`) — same schema, off-vocabulary value — sits "
        f"BETWEEN the endpoints and is significantly displaced from both, in OPPOSITE "
        f"directions: blue−near = +{f['nonsense_vs_near_delta']:.4f} (t="
        f"{f['nonsense_vs_near_t']:.2f}), far−blue = "
        f"+{abs(f['far_vs_nonsense_delta']):.4f} (t={abs(f['far_vs_nonsense_t']):.2f}). An "
        f"off-vocabulary token cannot itself encode 'near' or 'far', so the near↔far "
        f"ORDERING is specific to the dial's VALUE. Where blue LANDS on the range is not a "
        f"share of the effect and cannot be read as one: the experiment has no zero-token "
        f"baseline (no dial-arm block without a `reach` line was ever run), so there is "
        f"nothing to decompose against.",
        "",
        "THE COSTS",
        "",
        f"The dial narrows — and degrades — vocabulary as it reaches. `far` used only "
        f"{f['per_setting_distinct_add_words']['far']} distinct `add` words against `near`'s "
        f"{f['per_setting_distinct_add_words']['near']}, dominated by {top}. This is the "
        f"most honest caveat available: reaching further costs range.",
        "",
        f"One pre-declared gate FAILED — the `add` slot-hit rate (does the turn actually use "
        f"the word the block named?) fell {f['adherence_shortfall']:.3f} below the best "
        f"setting. The failure's direction matters: the worst setting is "
        f"`{f['adherence_worst_setting']}`, not `far`. At `near` the model takes the "
        f"degenerate route — it names a word already in the scene "
        f"({f['near_copy_rate'] * 100:.0f}% of the time, the highest of the three) and then "
        f"does not use it.",
        "",
        f"The model is NOT converged — {f['steps_trained']} SFT steps, validation loss still "
        f"falling. Everything above is a floor.",
        "",
        "WHAT THIS TOOL DOES NOT DO",
        "",
        "It does not score the distance of what you see. Scoring needs a pass over the "
        "2.1M-story corpus; every figure above is read out of "
        "docs/measurements/reach-dial.json. A single run is an anecdote against an effect of "
        f"+{f['residualised_near_to_far']:.3f}.",
        "",
        FOOTER,
    ]
    out: List[str] = []
    for p in paras:
        out.extend(textwrap.wrap(p, width=width) if p else [""])
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
# Scene selection.
# ---------------------------------------------------------------------------------------
def load_eval_scenes(skits: Path = SKITS, *, limit: int = 4000) -> List[str]:
    """The held-out eval skits' scene openings, for the zero-argument case.

    Held-out on purpose: the scenes this tool reaches for by default are ones the arm never
    trained on, so a good-looking default run is not a memory.
    """
    import json

    if not skits.is_file():
        return []
    scenes: List[str] = []
    with skits.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "eval" and row.get("prefix"):
                scenes.append(row["prefix"])
                if len(scenes) >= limit:
                    break
    return scenes


def pick_scene(scenes: Sequence[str], seed: int) -> str:
    """A scene chosen reproducibly from the seed, so `--seed N` recovers the whole run."""
    pool = list(scenes) or list(FALLBACK_SCENES)
    return random.Random(seed).choice(pool)


# ---------------------------------------------------------------------------------------
# Model resolution and generation. Everything below here needs weights.
# ---------------------------------------------------------------------------------------
def resolve_model_dir(path: Path) -> Tuple[Path, str]:
    """An HF directory `transformers` can load, plus a short label for the banner.

    Accepts either an already-converted HF directory (has `config.json`) or an SFT checkpoint
    directory such as `artifacts/reach/ckpt-dial` (has `step_3000.pkl`). For the latter it
    prefers the conversion `scripts/eval_reach.py` already left in
    `artifacts/reach/hf-eval/<arm>` when that is at least as new as the checkpoint, and
    otherwise converts into this tool's OWN cache. It never writes into the eval's directory:
    another process may be reading it, and this tool is a demo, not a producer of artifacts.
    """
    path = Path(path)
    if (path / "config.json").is_file():
        return path, path.name
    pkl = path / DIAL_STEP_PKL
    if not pkl.is_file():
        raise FileNotFoundError(
            f"{path} is neither an HF model directory (no config.json) nor a reach SFT "
            f"checkpoint directory (no {DIAL_STEP_PKL}).")
    arm = path.name[len("ckpt-"):] if path.name.startswith("ckpt-") else path.name
    cached = EVAL_HF_CACHE / arm
    if (cached / "config.json").is_file() and \
            (cached / "config.json").stat().st_mtime >= pkl.stat().st_mtime:
        return cached, f"{path.name} @ step 3000"
    out = CLI_HF_CACHE / arm
    if not ((out / "config.json").is_file() and
            (out / "config.json").stat().st_mtime >= pkl.stat().st_mtime):
        print(f"converting {pkl.name} -> {out} (once; CPU only) ...", flush=True)
        from scripts.eval_improv import sft_checkpoint_to_hf
        sft_checkpoint_to_hf(pkl, warm_start_ckpt=WARM_START_CKPT,
                             tokenizer_dir=TOKENIZER_DIR, out_dir=out)
    return out, f"{path.name} @ step 3000"


def run_arms(scene: str, values: Sequence[str], *, model_dir: Path, seed: int,
             temperature: float, max_new_tokens: int) -> Tuple[List[Arm], str, str]:
    """Generate every requested arm. Returns the arms plus the shared `offer` and `accept`.

    The three arms are generated in SEPARATE calls, each seeded identically, so the only
    thing that differs between them is the dial value: batching them together would give
    each sequence a different draw from the shared RNG stream and quietly reintroduce the
    confound the whole design exists to remove.
    """
    from scripts.eval_improv import generate_batched_from_ids, load_hf
    from scripts.eval_reach import encode_segments, parse_forced_generation

    tok, model = load_hf(model_dir)

    offer = derive_offer(scene)
    probe_ids = encode_segments(tok, accept_probe_segments(scene, offer))
    probe = generate_batched_from_ids(tok, model, [probe_ids], max_new_tokens=12,
                                      do_sample=False, batch_size=1)[0]
    accept = parse_accept(probe) or offer.split(" ")[0]

    arms: List[Arm] = []
    for value in values:
        segs = arm_segments(scene, offer, accept, value)
        raw = generate_batched_from_ids(
            tok, model, [encode_segments(tok, segs)], max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
            batch_size=1, seed=seed)[0]
        parsed = parse_forced_generation(raw)
        arms.append(Arm(value=value, segments=segs, raw=raw, add_value=parsed.add_value,
                        turn=parsed.turn, closed_block=parsed.closed_block))
    return arms, offer, accept


# ---------------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/reach.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="?", default=None,
                   help="A scene opening (a sentence or two). Omitted: a held-out eval "
                        "skit's opening, chosen from --seed.")
    p.add_argument("--reach", choices=("near", "mid", "far", "all"), default="all",
                   help="Which dial setting(s) to force. Default `all` — the side-by-side "
                        "is the point.")
    p.add_argument("--seed", type=int, default=None,
                   help="Reproducibility. Chooses the default scene and seeds sampling. "
                        "Printed on every run so a striking output can be recovered.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 (default) is greedy, which is what the published measurement "
                        "used. Above 0 samples, and --seed then matters.")
    p.add_argument("--max-new-tokens", type=int, default=80,
                   help="Same budget the measurement generated under.")
    p.add_argument("--show-think", dest="show_think", action="store_true", default=True,
                   help="Show the think-block plan (default). The plan IS the interesting "
                        "part — the dial is a line in it.")
    p.add_argument("--no-show-think", dest="show_think", action="store_false",
                   help="Only the generated turns.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help=f"HF model dir, or a reach SFT checkpoint dir. Default "
                        f"{DEFAULT_MODEL.relative_to(ROOT)} — the dial arm; its "
                        f"{DIAL_STEP_PKL} is the 3000-step checkpoint the published "
                        f"measurement generated from.")
    p.add_argument("--color", dest="color", action="store_true", default=None,
                   help="Force ANSI colour on.")
    p.add_argument("--no-color", dest="color", action="store_false",
                   help="Plain text. Also the automatic choice when piped or under NO_COLOR.")
    p.add_argument("--about", action="store_true",
                   help="What the dial is, how big the effect really is, and what failed.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    width = terminal_width(stream=sys.stdout)

    if args.about:
        print(about_text(width))
        return 0

    ink = Ink(want_colour(args.color))
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 10_000)
    scene = args.scene.strip() if args.scene else pick_scene(load_eval_scenes(), seed)
    if not scene:
        print("ERROR: empty scene.", file=sys.stderr)
        return 2

    try:
        model_dir, label = resolve_model_dir(args.model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    arms, _offer, _accept = run_arms(scene, selected_values(args.reach), model_dir=model_dir,
                                     seed=seed, temperature=args.temperature,
                                     max_new_tokens=args.max_new_tokens)
    print("\n".join(render_run(scene, arms, model_label=label, seed=seed,
                               temperature=args.temperature,
                               width=width, ink=ink, show_think=args.show_think)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
