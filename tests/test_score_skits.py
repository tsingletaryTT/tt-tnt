"""Prediction scorers. Each slot is a claim about text that did not exist when it was made."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_improv import load_harm_lexicon
from scripts.score_skits import score_block, slot_accuracy

HARM = load_harm_lexicon()
BLOCK = {"offer": "rock friend", "accept": "rock", "add": "windowsill",
         "stakes": "level", "handback": "light"}


def test_accept_hits_when_the_named_element_appears_in_the_turn():
    h = score_block(BLOCK, turn="She put the rock on the windowsill.",
                    prev_turn="She showed it to her friend.", next_partner=None, harm=HARM)
    assert h.accept is True


def test_accept_misses_when_it_does_not():
    h = score_block(BLOCK, turn="A bird flew past the window.",
                    prev_turn="She showed it to her friend.", next_partner=None, harm=HARM)
    assert h.accept is False


def test_add_is_scored_against_the_models_own_turn():
    hit = score_block(BLOCK, turn="She put it on the windowsill.",
                      prev_turn="x.", next_partner=None, harm=HARM)
    miss = score_block(BLOCK, turn="She put it down.",
                       prev_turn="x.", next_partner=None, harm=HARM)
    assert hit.add is True and miss.add is False


def test_stakes_compares_this_turn_against_the_previous_turn():
    up = dict(BLOCK, stakes="up")
    h = score_block(up, turn="The rock cut her hand and she cried.",
                    prev_turn="She put the rock down.", next_partner=None, harm=HARM)
    assert h.stakes is True, "harm arriving should register as up"
    h2 = score_block(up, turn="She smiled at the rock.",
                     prev_turn="She put the rock down.", next_partner=None, harm=HARM)
    assert h2.stakes is False, "no intensity change cannot satisfy stakes: up"


def test_handback_anticipation_is_scored_against_the_next_partner_turn():
    """The corpus partner CANNOT have heard the model, so a hit is anticipation, not
    influence. The metric's name carries that; this test pins the semantics."""
    h = score_block(BLOCK, turn="She put the rock on the windowsill.",
                    prev_turn="x.", next_partner='"It catches the light!" said her friend.',
                    harm=HARM)
    assert h.handback_anticipation is True
    h2 = score_block(BLOCK, turn="She put the rock on the windowsill.",
                     prev_turn="x.", next_partner="They went to bed.", harm=HARM)
    assert h2.handback_anticipation is False


def test_handback_is_undefined_without_a_following_partner_turn():
    """Block 3 has no follower. Undefined must not be scored as a miss — that would
    guarantee a 2/3 ceiling on the metric and look like a model failing."""
    h = score_block(BLOCK, turn="She put the rock on the windowsill.",
                    prev_turn="x.", next_partner=None, harm=HARM)
    assert h.handback_anticipation is None


def test_slot_accuracy_ignores_undefined_rather_than_counting_them_as_misses():
    from scripts.score_skits import SlotHits
    hits = [SlotHits(True, True, True, None), SlotHits(True, False, True, True)]
    acc = slot_accuracy(hits)
    assert acc["accept"] == 1.0
    assert acc["add"] == 0.5
    assert acc["handback_anticipation"] == 1.0, "1 hit of 1 DEFINED, not 1 of 2"


import json

import pytest

_SKITS = Path(__file__).resolve().parents[1] / "artifacts" / "skits" / "skits.jsonl"


@pytest.mark.skipif(
    not _SKITS.is_file(),
    reason=("needs artifacts/skits/skits.jsonl — derived-block accuracy is a property of "
            "the real corpus. Regenerate: python3 scripts/derive_skits.py --limit 20000"))
def test_derived_blocks_score_near_perfectly_and_shuffled_ones_do_not():
    """The scorers' sanity check, and the shuffled-slot control's calibration.

    A block DERIVED from a turn should score near 1.0 against that turn — it was read off
    it. The same block scored against a DIFFERENT skit's turn should score far lower. If
    those two are close, the scorer is not discriminating and every downstream number is
    noise.
    """
    from scripts.score_improv import load_harm_lexicon
    from train.skit import MODEL_TURNS

    harm = load_harm_lexicon()
    skits = [json.loads(l) for l in open(_SKITS)][:400]
    assert len(skits) > 50, "not enough skits to calibrate on"

    def acc(blocks_from, turns_from):
        hits = []
        for i, t_idx in enumerate(MODEL_TURNS):
            nxt = turns_from["turns"][t_idx + 1] if t_idx + 1 < 5 else None
            prev = (turns_from["turns"][t_idx - 1] if t_idx > 0 else turns_from["prefix"])
            hits.append(score_block(blocks_from["blocks"][i], turn=turns_from["turns"][t_idx],
                                    prev_turn=prev, next_partner=nxt, harm=harm))
        return slot_accuracy(hits)

    matched = [acc(s, s) for s in skits]
    shuffled = [acc(skits[i], skits[(i + 1) % len(skits)]) for i in range(len(skits))]
    for slot in ("accept", "add"):
        m = sum(a[slot] for a in matched) / len(matched)
        sh = sum(a[slot] for a in shuffled) / len(shuffled)
        assert m > 0.9, f"{slot}: derived blocks should score near 1.0, got {m:.3f}"
        assert m - sh > 0.4, (
            f"{slot}: matched {m:.3f} vs shuffled {sh:.3f} — the scorer is not "
            f"discriminating, so the shuffled-slot control cannot mean anything")
