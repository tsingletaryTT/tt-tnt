# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for train/reach.py -- the dial, its direction, and its zero-inflation guard.

TWO OF THESE TESTS EXIST BECAUSE THE INSTRUMENT WAS CAUGHT BEING WRONG BEFORE THE CODE WAS
WRITTEN, and both failure modes would have produced a clean, significant, worthless result:

* DIRECTION. High NPMI means tightly associated, i.e. a NEAR reach. A bucketer that ranks
  association descending and calls the top tercile `far` is inverted.
  `test_a_hand_built_near_pair_and_a_hand_built_far_pair_land_in_the_expected_buckets` is the
  test the spec demands, and its fixture is built so a flipped comparison, a swapped pair of
  cut points, or a distance that forgot its ``1 -`` all change a BUCKET NAME.
* ZERO-INFLATION. `npmi` returns 0.0 both for "these never co-occurred" (ignorance) and for
  "these co-occur less than chance" (a measurement). `test_pair_has_evidence_separates_...`
  holds those apart on a fixture that contains one of each.

Fixture discipline: `DOCS` is built with per-pair frequencies chosen so the three probe pairs
land at genuinely different NPMIs (0.844 / 0.431 / 0.065) rather than at the [0,1] ends, so an
assertion about which bucket they fall in can actually fail. Two fixtures in this project were
vacuous because one word repeated through every sentence.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.derive_skits import build_skit_example  # noqa: E402
from scripts.score_improv import build_association as _pair_association  # noqa: E402
from scripts.score_improv import npmi as _pair_npmi  # noqa: E402
from train.improv import SLOT_NAMES, STAKES_EPSILON, Slots, render_think  # noqa: E402
from train.reach import (REACH_SLOT_NAMES, REACH_VALUES, ReachSlots,  # noqa: E402
                         add_word_of, block_context_words, bucket_balance,
                         build_association, fit_reach_terciles, format_stakes_delta,
                         npmi, pair_counts, pair_has_evidence, pair_key,
                         parse_reach_think,
                         parse_stakes_delta,
                         reach_bucket, reach_distance, reach_slot_names_of,
                         skit_reach_distances, skit_reach_distances_per_block,
                         skit_stakes_deltas, stakes_delta, stakes_label, with_reach)
from train.skit import MODEL_TURNS, Skit, skit_segments  # noqa: E402

# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
#: 200 documents, engineered so three probe pairs sit at three well-separated NPMIs:
#:
#:   dog/bark      88 of 200 docs together, dog in 100, bark in 88   -> npmi 0.844  NEAR
#:   dog/leash     40 together, leash in 40                          -> npmi 0.431  MID
#:   dog/monsoon   12 together, monsoon in 20                        -> npmi 0.065  FAR
#:   dog/meow      0 together                                        -> NO EVIDENCE
#:
#: The far pair is deliberately a SMALL POSITIVE rather than a clamped zero: a fixture whose
#: far pair scored exactly 0.0 could not tell "far" apart from "never seen", which is the
#: distinction the zero-evidence tests are about.
DOCS = (
    ["the dog will bark loudly"] * 48
    + ["the dog will bark and pull the leash"] * 40
    + ["the dog watched the monsoon"] * 12
    + ["the monsoon brings rain"] * 8
    + ["the cat will meow softly"] * 92
)
ASSOC = build_association(DOCS)

#: Nine distances whose terciles are unambiguous: lo = s[3] = 0.5, hi = s[6] = 0.9.
FIT_SAMPLE = [0.10, 0.20, 0.30, 0.50, 0.60, 0.70, 0.90, 0.95, 0.99]

IDF = {"bellows": 3.0, "ladle": 2.0, "apron": 1.0}


def _skit() -> Skit:
    """A five-turn skit with turn-unique vocabulary, blocks built by hand.

    Built directly rather than through `derive_skit_from_turns` so the `add` words are known
    to the test: `with_reach` and `block_context_words` are what is under test here, not the
    slot derivation (which has its own tests).
    """
    blocks = tuple(Slots(offer=f"offer{i}", accept=f"accept{i}", add=add,
                         stakes="level", handback=f"hand{i}")
                   for i, add in enumerate(("bellows", "ladle", "apron")))
    return Skit(story_id=7, prefix="Nell stood by the kiln.",
                turns=("Bring the bellows.", "The bellows are cracked.",
                       "Then bring the ladle.", "The ladle is hot.",
                       "Wrap the ladle in the apron."),
                blocks=blocks)


# --------------------------------------------------------------------------------------
# A. DIRECTION -- the test the spec demands
# --------------------------------------------------------------------------------------
def test_high_npmi_is_a_near_reach_and_the_distance_inverts_it():
    """The fact the whole dial hangs on, asserted on values before any bucketing.

    Verified against a real table in the pre-flight: npmi(dog, bark) = +0.1161,
    npmi(dog, tail) = +0.1296, npmi(cat, meow) = +0.0978 -- tightly associated words score
    HIGH. So the distance must run the other way, and `reach_distance` must not be a
    thinly-renamed association.
    """
    near, mid, far = (npmi("dog", "bark", ASSOC), npmi("dog", "leash", ASSOC),
                      npmi("dog", "monsoon", ASSOC))
    assert near > mid > far > 0.0, (near, mid, far)
    assert near == pytest.approx(0.844, abs=0.01)
    assert mid == pytest.approx(0.431, abs=0.01)
    assert far == pytest.approx(0.065, abs=0.01)

    d_near = reach_distance("bark", ["dog"], ASSOC)
    d_mid = reach_distance("leash", ["dog"], ASSOC)
    d_far = reach_distance("monsoon", ["dog"], ASSOC)
    assert d_near < d_mid < d_far, (d_near, d_mid, d_far)
    assert d_near == pytest.approx(1.0 - near)


def test_a_hand_built_near_pair_and_a_hand_built_far_pair_land_in_the_expected_buckets():
    """THE bucket test. `reach: far` must select the LOW-NPMI end.

    Dies if the comparisons in `reach_bucket` flip, if `fit_reach_terciles` returns its cut
    points the other way round (that raises), or if `reach_distance` returns the association
    instead of ``1 - association`` (the near pair then lands in `mid`).
    """
    lo, hi = fit_reach_terciles(FIT_SAMPLE)
    assert (lo, hi) == (0.50, 0.90)

    near = reach_distance("bark", ["dog"], ASSOC)
    mid = reach_distance("leash", ["dog"], ASSOC)
    far = reach_distance("monsoon", ["dog"], ASSOC)
    assert reach_bucket(near, lo, hi) == "near"
    assert reach_bucket(mid, lo, hi) == "mid"
    assert reach_bucket(far, lo, hi) == "far"
    # ...and the three really are three, so the equalities above are not all one bucket.
    assert len({reach_bucket(d, lo, hi) for d in (near, mid, far)}) == 3


def test_reach_bucket_is_monotone_and_uses_the_cut_points_as_stated():
    lo, hi = 0.25, 0.75
    probes = (0.0, 0.2499, 0.25, 0.5, 0.7499, 0.75, 1.0)
    assert [reach_bucket(d, lo, hi) for d in probes] == \
        ["near", "near", "mid", "mid", "mid", "far", "far"]
    assert REACH_VALUES == ("near", "mid", "far")


def test_reach_bucket_refuses_swapped_cut_points():
    with pytest.raises(ValueError, match="out of order"):
        reach_bucket(0.5, 0.9, 0.1)


# --------------------------------------------------------------------------------------
# B. ZERO-INFLATION -- ignorance is not distance
# --------------------------------------------------------------------------------------
def test_pair_has_evidence_separates_ignorance_from_measured_distance():
    """Both of these score 0.0-ish. Only one of them is a measurement.

    `dog`/`meow` never co-occur in DOCS: npmi is exactly 0.0 and there is no evidence.
    `dog`/`monsoon` co-occur 12 times: npmi is small but nonzero, and there IS evidence. A
    dial that bins the first as `far` is reporting table coverage as semantics.
    """
    assert npmi("dog", "meow", ASSOC) == 0.0
    assert pair_has_evidence("dog", "meow", ASSOC) is False
    assert npmi("dog", "monsoon", ASSOC) > 0.0
    assert pair_has_evidence("dog", "monsoon", ASSOC) is True
    # An unseen word is ignorance too, not distance.
    assert pair_has_evidence("dog", "gorthax", ASSOC) is False


def test_reach_distance_returns_none_rather_than_far_when_nothing_has_evidence():
    assert reach_distance("gorthax", ["dog", "bark"], ASSOC) is None
    assert reach_distance("meow", ["dog"], ASSOC) is None


def test_reach_distance_skips_evidence_free_context_words_instead_of_scoring_them_far():
    """A context word the table has never paired with must not drag the distance to 1.0.

    `meow` has no evidence with `bark`; `dog` does. The answer must be the dog distance --
    identical to the answer with `meow` absent -- and NOT 1.0.
    """
    with_junk = reach_distance("bark", ["meow", "gorthax", "dog"], ASSOC)
    without = reach_distance("bark", ["dog"], ASSOC)
    assert with_junk == without
    assert with_junk < 0.9


def test_reach_bucket_refuses_a_none_distance():
    """The exact trap: None means 'no evidence', and a silent `far` is the failure mode."""
    with pytest.raises(ValueError, match="DROP"):
        reach_bucket(None, 0.3, 0.7)


def test_an_evidenced_pair_that_scores_zero_is_far_and_is_kept():
    """The other side of the guard: a clamped 0.0 WITH evidence is a real, maximal distance.

    Built so `cat` and `bark` co-occur exactly once out of 200 documents -- far below chance,
    so NPMI clamps to 0.0 -- while the pair is still evidenced. The distance is 1.0 and it
    must be returned, not discarded, or `far` loses its far end.
    """
    docs = DOCS + ["the cat heard a bark once"]
    assoc = build_association(docs)
    assert pair_has_evidence("cat", "bark", assoc) is True
    assert npmi("cat", "bark", assoc) == 0.0
    assert reach_distance("bark", ["cat"], assoc) == 1.0


def test_holdout_subtracts_exactly_one_document():
    """LEAVE-ONE-OUT, on a fixture where the difference is a whole answer.

    `gorthax`/`vermilion` co-occur in exactly ONE document. Without a holdout that document
    makes them a PERFECT association (npmi 1.0, distance 0.0) -- and in the real derivation
    that one document is the very story being scored, so the metric would be measuring the
    scene against itself. With the holdout it is correctly no evidence at all.
    """
    assoc = build_association(["gorthax vermilion"] + ["dog bark"] * 50 + ["cat meow"] * 50)
    assert npmi("gorthax", "vermilion", assoc) == 1.0
    assert pair_has_evidence("gorthax", "vermilion", assoc) is True
    assert pair_has_evidence("gorthax", "vermilion", assoc, holdout=True) is False
    assert npmi("gorthax", "vermilion", assoc, holdout=True) == 0.0
    assert reach_distance("gorthax", ["vermilion"], assoc) == 0.0
    assert reach_distance("gorthax", ["vermilion"], assoc, holdout=True) is None
    # counts really do drop by one, on all four
    assert pair_counts("dog", "bark", assoc) == (50, 50, 50, 101)
    assert pair_counts("dog", "bark", assoc, holdout=True) == (49, 49, 49, 100)
    # ...and a pair with no co-occurrence at all is left alone rather than driven negative
    assert pair_counts("dog", "meow", assoc, holdout=True) == (50, 50, 0, 101)


def test_holdout_reaches_the_skit_level():
    """The per-skit helpers must pass it through, or the derivation silently loses it."""
    skit = _skit()
    once = build_association(["kiln bellows ladle apron nell bring wrap"])
    assert skit_reach_distances(skit, once) is not None
    assert skit_reach_distances(skit, once, holdout=True) is None
    assert skit_reach_distances_per_block(skit, once, holdout=True) == [None, None, None]


# --------------------------------------------------------------------------------------
# C. one formula, two storages
# --------------------------------------------------------------------------------------
def test_npmi_agrees_with_score_improv_on_the_same_documents():
    """`train.reach.npmi` and `scripts.score_improv.npmi` are the same formula.

    Different tables by design (whole-story documents here, (prefix, continuation) pairs
    there) and different storage (canonical ``a <= b`` here, both directions there), so the
    only way "same formula" stays true is a check. Feeding score_improv's builder one
    document per pair-slot makes the two tables carry identical counts.
    """
    other = _pair_association([(d, "") for d in DOCS])
    assert other["n"] == ASSOC.n_docs
    probes = [("dog", "bark"), ("dog", "leash"), ("dog", "monsoon"), ("cat", "meow"),
              ("dog", "meow"), ("bark", "leash"), ("rain", "monsoon")]
    for a, b in probes:
        assert npmi(a, b, ASSOC) == pytest.approx(_pair_npmi(a, b, other)), (a, b)
        # ...and symmetric, which the canonical key is what guarantees.
        assert npmi(a, b, ASSOC) == npmi(b, a, ASSOC)


def test_pair_key_is_canonical():
    assert pair_key("b", "a") == pair_key("a", "b") == ("a", "b")


def test_build_association_counts_documents_not_occurrences():
    """A word repeated inside one document counts once. Same rule as score_improv's."""
    assoc = build_association(["dog dog dog bark", "dog bark bark"])
    assert assoc.n_docs == 2
    assert assoc.uni["dog"] == 2
    assert assoc.co[pair_key("dog", "bark")] == 2


# --------------------------------------------------------------------------------------
# D. fitting the terciles
# --------------------------------------------------------------------------------------
def test_fit_reach_terciles_cuts_at_the_nearest_ranks():
    assert fit_reach_terciles([3.0, 1.0, 2.0]) == (2.0, 3.0)
    assert fit_reach_terciles(FIT_SAMPLE) == (0.50, 0.90)
    # Order of the input must not matter.
    assert fit_reach_terciles(list(reversed(FIT_SAMPLE))) == (0.50, 0.90)


def test_fit_reach_terciles_refuses_too_few_values():
    with pytest.raises(ValueError, match="need >= 3"):
        fit_reach_terciles([0.1, 0.2])


def test_the_fitted_cuts_split_a_uniform_sample_into_near_thirds():
    """Balance, on a distribution where balance is achievable. The REAL distribution's
    balance is a property of scale and is asserted against the artifact instead."""
    sample = [i / 300.0 for i in range(300)]
    lo, hi = fit_reach_terciles(sample)
    bal = bucket_balance([reach_bucket(d, lo, hi) for d in sample])
    assert bal["n"] == 300
    assert bal["max_fraction"] < 0.4, bal
    assert min(bal["counts"].values()) > 90, bal


def test_bucket_balance_reports_the_majority_class_floor():
    bal = bucket_balance(["near"] * 8 + ["mid"] + ["far"])
    assert bal["counts"] == {"near": 8, "mid": 1, "far": 1}
    assert bal["max_fraction"] == 0.8
    assert bal["unknown_values"] == 0
    assert bucket_balance([])["max_fraction"] is None


# --------------------------------------------------------------------------------------
# E. the slot, its order, and the rendered block
# --------------------------------------------------------------------------------------
def test_reach_is_declared_before_add():
    """LOAD-BEARING, not cosmetic. The block is generated left to right, so a dial declared
    after the word it governs could only relabel a choice already made."""
    assert REACH_SLOT_NAMES.index("reach") < REACH_SLOT_NAMES.index("add")
    rendered = render_think(ReachSlots("o", "a", "far", "d", "+1.0", "h"))
    assert rendered.index("reach:") < rendered.index("add:")


def test_reach_slot_names_match_the_dataclass_fields():
    """One order, read from one place. A hard-coded second list is how a reordering ships
    silently -- the rendered order comes off the dataclass."""
    assert reach_slot_names_of(ReachSlots("o", "a", "r", "d", "s", "h")) == REACH_SLOT_NAMES


def test_render_think_reads_the_dataclass_field_order():
    """`render_think` was generalised from the module-level SLOT_NAMES to the object's own
    fields so both schemas share one renderer. This pins that the five-slot schema is
    unchanged by that generalisation."""
    assert tuple(f.name for f in fields(Slots)) == SLOT_NAMES
    five = render_think(Slots("o", "a", "d", "level", "h"))
    assert five == "<think>\noffer: o\naccept: a\nadd: d\nstakes: level\nhandback: h\n</think>\n"
    six = render_think(ReachSlots("o", "a", "far", "d", "+2.5", "h"))
    assert six == ("<think>\noffer: o\naccept: a\nreach: far\nadd: d\nstakes: +2.5\n"
                   "handback: h\n</think>\n")


def test_a_v3_block_round_trips_through_render_and_parse():
    slots = ReachSlots("nell stood kiln", "kiln", "far", "bellows", "-3.4", "bellows")
    assert parse_reach_think(render_think(slots)) == slots
    # ...and prose either side of the block does not matter
    assert parse_reach_think("before\n" + render_think(slots) + "after") == slots


@pytest.mark.parametrize("bad,why", [
    ("<think>\noffer: o\naccept: a\nadd: d\nstakes: level\nhandback: h\n</think>",
     "a FIVE-slot block is a different schema and must not parse as v3"),
    ("<think>\noffer: o\naccept: a\nreach: sideways\nadd: d\nstakes: +1.0\n"
     "handback: h\n</think>", "reach outside REACH_VALUES"),
    ("<think>\noffer: o\naccept: a\nreach: far\nadd: d\nstakes: level\n"
     "handback: h\n</think>", "stakes is not a signed number"),
    ("offer: o\naccept: a\nreach: far\nadd: d\nstakes: +1.0\nhandback: h",
     "no think tags at all"),
])
def test_a_malformed_v3_block_is_none_not_a_partial(bad, why):
    """None, not a partial object: adherence is reported as a RATE and a partial parse would
    inflate it. The five-slot row is the important one -- it is what a stage-2 checkpoint
    emits, and it must NOT count as a v3 block."""
    assert parse_reach_think(bad) is None, why


def test_the_five_slot_parser_and_the_six_slot_parser_do_not_accept_each_other():
    """Two schemas, two parsers, and neither may quietly accept the other's output -- that is
    what would make two published adherence rates incomparable."""
    from train.improv import parse_think
    five = render_think(Slots("o", "a", "d", "level", "h"))
    six = render_think(ReachSlots("o", "a", "far", "d", "+1.0", "h"))
    assert parse_think(five) is not None and parse_reach_think(five) is None
    assert parse_reach_think(six) is not None and parse_think(six) is None


def test_add_word_of_takes_the_first_item_and_lowercases_it():
    assert add_word_of(Slots("o", "a", "Bellows, Ladle", "level", "h")) == "bellows"
    assert add_word_of(ReachSlots("o", "a", "far", "apron", "+0.0", "h")) == "apron"


# --------------------------------------------------------------------------------------
# F. stakes as a continuous delta (ruling D)
# --------------------------------------------------------------------------------------
def test_stakes_delta_is_the_signed_intensity_difference():
    fake = {"loud": 9.0, "quiet": 1.0}
    intensity = fake.__getitem__
    assert stakes_delta("loud", "quiet", intensity) == 8.0
    assert stakes_delta("quiet", "loud", intensity) == -8.0
    assert stakes_delta("loud", "loud", intensity) == 0.0


def test_stakes_is_tested_on_magnitude_not_a_label():
    """Ruling D: a bigger jump must produce a bigger number. A three-way label cannot say
    that, which is why stage 2's version could be 85.3% one class and still look fine."""
    fake = {"a": 0.0, "b": 1.0, "c": 20.0}
    intensity = fake.__getitem__
    small = stakes_delta("b", "a", intensity)
    large = stakes_delta("c", "a", intensity)
    assert abs(large) > abs(small) > 0


def test_format_and_parse_stakes_delta_round_trip():
    for v in (0.0, -0.0, 0.04, -0.04, 1.25, -3.44, 12.5, -100.0):
        s = format_stakes_delta(v)
        assert s[0] in "+-", s
        assert parse_stakes_delta(s) == pytest.approx(round(v, 1) or 0.0)
    assert format_stakes_delta(-0.0) == "+0.0"
    assert format_stakes_delta(-0.04) == "+0.0", "two spellings of zero is two tokens for one fact"
    assert parse_stakes_delta("level") is None
    assert parse_stakes_delta("") is None


def test_the_continuous_delta_still_recovers_the_old_label():
    """The successor keeps the predecessor's information, so the stage-2 comparison is
    still possible from the new artifact alone."""
    assert stakes_label(STAKES_EPSILON + 0.1) == "up"
    assert stakes_label(-STAKES_EPSILON - 0.1) == "down"
    assert stakes_label(0.0) == "level"
    assert stakes_label(STAKES_EPSILON) == "level"


# --------------------------------------------------------------------------------------
# G. putting the slot into a skit
# --------------------------------------------------------------------------------------
def test_block_context_words_is_the_scene_so_far_and_nothing_later():
    skit = _skit()
    ctx0 = block_context_words(skit.prefix, skit.turns, 0)
    ctx2 = block_context_words(skit.prefix, skit.turns, 2)
    assert "kiln" in ctx0 and "bellows" not in ctx0
    assert "bellows" in ctx2 and "ladle" not in ctx2
    assert "apron" not in block_context_words(skit.prefix, skit.turns, 4)
    # strictly growing, because it is the scene accumulating
    assert len(ctx0) < len(ctx2) < len(block_context_words(skit.prefix, skit.turns, 4))


def test_skit_reach_distances_are_none_if_any_single_block_lacks_evidence():
    """Whole-or-nothing, matching the derivation's own drop rule: a think-block whose
    `reach` is a guess teaches the guess."""
    skit = _skit()
    # A table that knows blocks 0 and 1's words but has never seen `apron`.
    assoc = build_association(["kiln bellows ladle nell"] * 5 + ["bring wrap"] * 5)
    per_block = skit_reach_distances_per_block(skit, assoc)
    assert per_block[0] is not None and per_block[2] is None
    assert skit_reach_distances(skit, assoc) is None
    # ...and with a table that has seen everything, all three come back.
    full = build_association(["kiln bellows ladle apron nell"] * 5)
    assert len(skit_reach_distances(skit, full) or []) == len(MODEL_TURNS)


def test_with_reach_changes_only_the_block_schema():
    skit = _skit()
    out = with_reach(skit, distances=[0.1, 0.6, 0.99], deltas=[0.0, 2.5, -3.0],
                     lo=0.5, hi=0.9)
    assert out.prefix == skit.prefix and out.turns == skit.turns
    assert out.story_id == skit.story_id
    assert [b.reach for b in out.blocks] == ["near", "mid", "far"]
    assert [b.stakes for b in out.blocks] == ["+0.0", "+2.5", "-3.0"]
    # the four carried slots are verbatim
    for old, new in zip(skit.blocks, out.blocks):
        assert (new.offer, new.accept, new.add, new.handback) == \
               (old.offer, old.accept, old.add, old.handback)


def test_with_reach_refuses_a_mismatched_number_of_readings():
    with pytest.raises(ValueError, match="one distance and one delta per block"):
        with_reach(_skit(), distances=[0.1, 0.2], deltas=[0.0, 0.0], lo=0.3, hi=0.7)


def test_skit_stakes_deltas_span_the_exchange_not_the_scene():
    """Block 0 is measured against the PREFIX, later blocks against the partner turn
    immediately before them -- the same interval `_slots_for_turn` uses."""
    seen = []

    def intensity(text: str) -> float:
        seen.append(text)
        return float(len(text))

    skit = _skit()
    got = skit_stakes_deltas(skit, intensity)
    assert len(got) == len(MODEL_TURNS)
    assert skit.prefix in seen
    assert got[0] == len(skit.turns[0]) - len(skit.prefix)
    assert got[1] == len(skit.turns[2]) - len(skit.turns[1])


# --------------------------------------------------------------------------------------
# H. the hard constraint: the label rule and the mask are untouched by the sixth slot
# --------------------------------------------------------------------------------------
class _Tok:
    """Deterministic word-level tokenizer that honours add_special_tokens."""
    pad_token_id = 0
    BOS = 1

    def __init__(self):
        self._ids: dict = {}

    def encode(self, s, add_special_tokens=True):
        ids = [self._ids.setdefault(w, len(self._ids) + 2) for w in s.split()]
        return ([self.BOS] + ids) if add_special_tokens else ids


@pytest.mark.parametrize("schema", ["five_slot", "six_slot"])
def test_the_label_rule_and_the_mask_are_the_same_under_both_schemas(schema):
    """The sixth slot makes the think segment LONGER; it must not move the boundary rule.

    The expected labels are rebuilt here from `skit_segments` with an independent position
    counter, so this checks the rule (``labels[t] = ids[t+1] if supervised[t+1] else -100``)
    rather than asking `build_skit_example` to agree with itself. A mutant using
    ``supervised[t]`` passed 1,181 tests once.
    """
    skit = _skit()
    if schema == "six_slot":
        skit = with_reach(skit, distances=[0.1, 0.6, 0.99], deltas=[0.0, 1.0, -1.0],
                          lo=0.5, hi=0.9)
    tok = _Tok()
    ex = build_skit_example(skit, tok, with_think=True, pad_token_id=tok.pad_token_id)

    ids, sup = [], []
    first = True
    for text, s in skit_segments(skit):
        seg = tok.encode(text, add_special_tokens=first)
        first = False
        ids.extend(seg)
        sup.extend([s] * len(seg))
    want = [(ids[t + 1] if sup[t + 1] else -100) for t in range(len(ids) - 1)] + [-100]
    pad = (-len(ids)) % 32
    assert ex["input_ids"][:len(ids)] == ids
    assert ex["labels"] == want + [-100] * pad
    assert len(ex["input_ids"]) % 32 == 0
    # the prefix and BOTH partner turns are never supervised
    assert sup[0] is False
    assert any(s for s in sup), "a fixture where nothing is supervised proves nothing"


def test_the_six_slot_block_is_longer_so_the_fixture_could_notice_a_shift():
    """Guards the test above from being vacuous: if the two schemas tokenised to the same
    length, the parametrisation would not be testing anything new."""
    tok5, tok6 = _Tok(), _Tok()
    five = build_skit_example(_skit(), tok5, with_think=True, pad_token_id=0)
    six = build_skit_example(with_reach(_skit(), distances=[0.1, 0.6, 0.99],
                                       deltas=[0.0, 1.0, -1.0], lo=0.5, hi=0.9),
                             tok6, with_think=True, pad_token_id=0)
    assert len(six["input_ids"]) >= len(five["input_ids"])
    assert sum(1 for v in six["labels"] if v != -100) > \
           sum(1 for v in five["labels"] if v != -100)
