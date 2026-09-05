# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Reads this project's own committed docs/measurements/*.json verdicts for the Gradio
demo's "Research Findings" tab. No live inference here -- the checkpoints these files
describe (reach-dial, skits stage 2, the tool-calling/LoRA ablations) are no longer
available to serve locally, so this tab quotes the real, already-published numbers
instead of trying to reproduce them.

Each measurement file was written by a different evaluation script over the course of
this project and has its own ad-hoc JSON shape -- there is no shared "verdict" field
name across them (docs/superpowers/specs/2026-09-04-gradio-demo-design.md's Tab 4
section). Rather than guess at a generic field name, FINDING_CARDS hand-picks the exact
dotted path to a real, human-readable verdict string in each specific file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
MEASUREMENTS = ROOT / "docs" / "measurements"


@dataclass(frozen=True)
class FindingCard:
    title: str
    json_path: Path
    verdict_path: Tuple[str, ...]


FINDING_CARDS: List[FindingCard] = [
    FindingCard(
        "Reach dial (controllable generation)",
        MEASUREMENTS / "reach-dial.json",
        ("headline", "dial_kind_verdict"),
    ),
    FindingCard(
        "Skits stage 2 (five-slot turn structure)",
        MEASUREMENTS / "skits-stage2.json",
        ("verdict",),
    ),
    FindingCard(
        "Tool-calling checkpoint selection",
        MEASUREMENTS / "tool-calling-s3-selection.json",
        ("why",),
    ),
    FindingCard(
        "LoRA vs full-parameter fine-tuning",
        MEASUREMENTS / "lora-vs-full-tool-calling.json",
        ("refuted_prediction",),
    ),
    FindingCard(
        "LoRA anti-forgetting prediction",
        MEASUREMENTS / "evaluation-tt-tnt-1024-vs-editor-lora.json",
        ("prediction_refuted_2026_08_31", "prediction"),
    ),
]


def _walk(data: Any, path: Tuple[str, ...]) -> Optional[str]:
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data if isinstance(data, str) else None


def load_findings(cards: Optional[List[FindingCard]] = None) -> List[Dict[str, Optional[str]]]:
    """One dict per card whose JSON file exists and parses, in card order. A card
    whose file is missing (a leaner clone) is silently omitted; a card whose file
    exists but whose verdict_path doesn't resolve to a string still appears, with
    verdict=None, so a schema drift is visible rather than silently disappearing."""
    cards = FINDING_CARDS if cards is None else cards
    result: List[Dict[str, Optional[str]]] = []
    for card in cards:
        if not card.json_path.is_file():
            continue
        try:
            data = json.loads(card.json_path.read_text())
        except json.JSONDecodeError:
            continue
        result.append({
            "title": card.title,
            "verdict": _walk(data, card.verdict_path),
            "source": str(card.json_path),
        })
    return result
