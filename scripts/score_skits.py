#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Per-slot prediction scoring: does the plan predict what the model then wrote?

This is what stage 1 could not ask. There, a block was DERIVED from the continuation it
described, so it could only ever describe. Here the model generates the block and then the
turn, so each slot is a falsifiable claim about text that did not exist when the claim was
made — and a claim can be scored for accuracy.

`handback_anticipation` is named for what it measures. The corpus partner cannot have heard
the model, so a hit means the model correctly ANTICIPATED what the scene was about to need.
That is a real skill and a DIFFERENT quantity from influence. Influence becomes measurable
only in the later model-partner arm. The limitation lives in the name so no future reader
has to find it in a footnote.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_improv import intensity  # noqa: E402
from train.improv import STAKES_EPSILON, content_words  # noqa: E402

SLOT_NAMES = ("accept", "add", "stakes", "handback_anticipation")


@dataclass(frozen=True)
class SlotHits:
    accept: Optional[bool]
    add: Optional[bool]
    stakes: Optional[bool]
    handback_anticipation: Optional[bool]

    def as_dict(self) -> dict:
        return asdict(self)


def _mentions(text: str, target: str) -> bool:
    """Does `text` contain any content word of `target`? Word-level, not substring —
    substring matching once made "done" match "abandoned" in the closure lexicon."""
    if not target or target == "open":
        return False
    want = set(content_words(target))
    return bool(want & set(content_words(text))) if want else False


def score_block(block: Dict[str, str], *, turn: str, prev_turn: str,
                next_partner: Optional[str], harm: frozenset) -> SlotHits:
    delta = intensity(turn, harm) - intensity(prev_turn, harm)
    predicted = block.get("stakes", "")
    actual = ("up" if delta > STAKES_EPSILON
              else "down" if delta < -STAKES_EPSILON else "level")

    return SlotHits(
        accept=_mentions(turn, block.get("accept", "")),
        add=_mentions(turn, block.get("add", "")),
        stakes=(predicted == actual),
        # None, never False, when there is no following partner turn: scoring undefined as
        # a miss would cap the metric at 2/3 and read as the model failing.
        handback_anticipation=(None if next_partner is None
                               else _mentions(next_partner, block.get("handback", ""))),
    )


def slot_accuracy(hits: List[SlotHits]) -> Dict[str, Optional[float]]:
    """Per-slot accuracy over DEFINED predictions only."""
    out: Dict[str, Optional[float]] = {}
    for name in SLOT_NAMES:
        vals = [getattr(h, name) for h in hits]
        defined = [v for v in vals if v is not None]
        out[name] = (sum(1 for v in defined if v) / len(defined)) if defined else None
    return out
