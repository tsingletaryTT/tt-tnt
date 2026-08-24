# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Skits: five-turn improv scenes derived deterministically from single corpus stories.

WHY THIS EXISTS RATHER THAN REUSING train.improv's single-turn derivation
------------------------------------------------------------------------
Stage 1 gave the model a think-block before one continuation. Two of the five slots could
not be scored at all in that shape:

  * `handback` encodes "make your partner look good" and there was no partner, so the slot
    was decorative by construction and no scorer could have rescued it.
  * `stakes` was measured as an intensity delta INSIDE one continuation. Escalation in
    improv happens ACROSS an exchange, so that measured the wrong interval.

A skit has a later turn, so every slot becomes a falsifiable prediction about text that did
not exist when the prediction was made. That is the whole point, and it is what stage 1
structurally could not ask.

SHAPE
    prefix    sentences[0:2]   context only, never supervised
    turn 0    sentences[2]     MODEL   <- think-block 0
    turn 1    sentences[3]     PARTNER real corpus text
    turn 2    sentences[4]     MODEL   <- think-block 1
    turn 3    sentences[5]     PARTNER real corpus text
    turn 4    sentences[6]     MODEL   <- think-block 2

Measured on 2,000 stories: 99.8% have >= 7 sentences, and a five-turn skit is median 202 /
p99 257 / max 327 tokens, so 100% fit the 512 window even after tile alignment. The window
is not a constraint here.

RESOLVED 2026-08-23 (was: KNOWN LIMITATION). split_sentences used to over-split dialogue
with attribution -- '"It says!" said Person.' became two sentences, the quote and the tag --
and skits dropped whenever a model turn could not carry a word from a partner turn that
ended in dialogue. That was held deliberately, because the published stage-1 measurement
sat behind the old splitter's output. The 2026-08-23 reach-dial spec approved the fix as a
clean cutover WITH republication: `train.improv.split_sentences` now delegates to
`train.dialogue.split_sentences_dialogue`, and `docs/measurements/improv-stage1.json`
carries the new derivation numbers beside the old ones. Nothing in this module changed for
it; the drop pressure named above simply went away.

SENTENCE SLICING IS NOT THE ONLY SHAPE. `derive_skit` slices consecutive sentences, which
means its "partner" turns are the same narrator continuing -- the reason
`handback_anticipation` could not be measured in stage 2. `derive_skit_from_turns` is the
same slot derivation over turns supplied by the CALLER, which is how
`scripts/derive_dialogue_skits.py` builds skits from real two-speaker exchanges
(`train.dialogue.extract_dialogue_turns`) without forking any of this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from train.improv import (STAKES_EPSILON, Slots, content_words, render_think,
                          split_sentences)

SKIT_ROLES: Tuple[str, ...] = ("model", "partner", "model", "partner", "model")
MODEL_TURNS: Tuple[int, ...] = (0, 2, 4)
PARTNER_TURNS: Tuple[int, ...] = (1, 3)
#: prefix (2 sentences) + five turns
MIN_SENTENCES = 7


@dataclass(frozen=True)
class Skit:
    story_id: int
    prefix: str
    turns: Tuple[str, ...]      # exactly 5, roles per SKIT_ROLES
    blocks: Tuple[Slots, ...]   # exactly 3, aligned to MODEL_TURNS

    def as_dict(self) -> dict:
        return {"story_id": self.story_id, "prefix": self.prefix,
                "turns": list(self.turns), "roles": list(SKIT_ROLES),
                "blocks": [b.as_dict() for b in self.blocks]}


def _slots_for_turn(offer_text: str, established: List[str], turn: str, prev_turn: str, *,
                    idf: Dict[str, float],
                    intensity: Callable[[str], float]) -> Optional[Slots]:
    """Derive one block, or None to drop the skit.

    `offer_text` is the IMMEDIATELY PRECEDING span — the partner's turn for blocks 1 and 2,
    or, for block 0, the WHOLE two-sentence prefix (there is no partner turn yet). That is
    what makes `accept` mean accepting an offer rather than merely repeating a word from
    somewhere in the scene.

    NOTE on block 0: the design spec (2026-08-21-skits-design.md) says "the prefix's final
    sentence" here. The code uses both prefix sentences and always has; the spec sentence is
    an erratum, not a description of behaviour. The published stage-2 measurement rests on
    the two-sentence form, so the code is authoritative and the spec is corrected to match.
    `test_offer_of_block_0_is_the_whole_prefix_not_its_last_sentence` pins this.

    `established` is every content word in the scene so far, which is what `add` is measured
    against: a word already in play is not an addition even if this particular turn is the
    first to say it again.
    """
    offer_words = content_words(offer_text)
    turn_words = content_words(turn)
    if not turn_words:
        return None

    carried = [w for w in offer_words if w in set(turn_words)]
    if not carried:
        return None                      # nothing accepted -> a block -> drop the skit

    fresh = [w for w in turn_words if w not in set(established)]
    if not fresh:
        return None                      # nothing added -> also a block -> drop

    # Sort keys are TOTAL (value, then the word) so a tie cannot resolve differently between
    # processes. Python randomises string hashing, and this output becomes training data.
    fresh_ranked = sorted(set(fresh), key=lambda w: (-idf.get(w, 0.0), w))

    # stakes spans the exchange: this turn against the PREVIOUS turn, not against the scene.
    delta = intensity(turn) - intensity(prev_turn)
    stakes = ("up" if delta > STAKES_EPSILON
              else "down" if delta < -STAKES_EPSILON else "level")

    tail = content_words(split_sentences(turn)[-1]) if split_sentences(turn) else []
    introduced = [w for w in tail if w in set(fresh)]

    return Slots(
        offer=" ".join(offer_words[:12]) or offer_text[:60],
        accept=" ".join(carried[:6]),
        add=", ".join(fresh_ranked[:1]),
        stakes=stakes,
        handback=introduced[-1] if introduced else "open",
    )


def derive_skit_from_turns(prefix: str, turns: Sequence[str], *, story_id: int,
                           idf: Dict[str, float],
                           intensity: Callable[[str], float]) -> Optional[Skit]:
    """One skit from a prefix and five ALREADY-CHOSEN turns, or None to drop it.

    This is the slot derivation proper, with no opinion about where the turns came from.
    `derive_skit` supplies consecutive sentences; `scripts/derive_dialogue_skits.py`
    supplies real alternating quoted utterances. Both get identical slots, identical drop
    semantics and identical `Skit` objects, because there is exactly one implementation --
    forking it is how the two paths would silently diverge.

    DROP RULE: if ANY of the three model turns fails derivation, the WHOLE skit is dropped.
    A partial skit would silently change what the model sees, and a think-block present for
    two turns and absent for the third teaches the wrong thing.

    `offer` for block 0 is the WHOLE prefix (there is no partner turn yet); for blocks 1 and
    2 it is the immediately preceding partner turn. `established` is every content word in
    the scene so far -- prefix plus all earlier turns -- so `add` means new to the SCENE and
    not merely new to this turn. See `_slots_for_turn` for both, including the block-0
    erratum in the stage-2 spec.
    """
    turns = tuple(turns)
    if len(turns) != len(SKIT_ROLES):
        return None

    blocks: List[Slots] = []
    for i, t_idx in enumerate(MODEL_TURNS):
        prev = turns[t_idx - 1] if t_idx > 0 else prefix
        offer = prev
        established = content_words(prefix) + [w for j in range(t_idx)
                                               for w in content_words(turns[j])]
        got = _slots_for_turn(offer, established, turns[t_idx], prev,
                              idf=idf, intensity=intensity)
        if got is None:
            return None
        blocks.append(got)
    return Skit(story_id=story_id, prefix=prefix, turns=turns, blocks=tuple(blocks))


def derive_skit(story: str, *, story_id: int, idf: Dict[str, float],
                intensity: Callable[[str], float]) -> Optional[Skit]:
    """One skit from one story by CONSECUTIVE-SENTENCE slicing, or None to drop it.

    Prefix is sentences 0-1, turns are sentences 2-6. The "partner" turns this produces are
    the same narrator continuing, which is why the dialogue path exists; this function is
    kept unchanged because the published stage-2 measurement rests on it.
    """
    sents = split_sentences(story)
    if len(sents) < MIN_SENTENCES:
        return None
    return derive_skit_from_turns(" ".join(sents[0:2]), sents[2:7], story_id=story_id,
                                  idf=idf, intensity=intensity)


def skit_segments(skit: Skit) -> List[Tuple[str, bool]]:
    """The skit as ordered `(text, supervised)` segments.

    Only the model's think-blocks and its own turns are supervised. The prefix and BOTH
    partner turns are context: the model must learn to read a partner turn, not to produce
    one.
    """
    segs: List[Tuple[str, bool]] = [(skit.prefix, False)]
    for i, t_idx in enumerate(MODEL_TURNS):
        if t_idx > 0:
            segs.append((skit.turns[t_idx - 1], False))     # the partner turn before it
        segs.append((render_think(skit.blocks[i]), True))
        segs.append((skit.turns[t_idx], True))
    return segs
