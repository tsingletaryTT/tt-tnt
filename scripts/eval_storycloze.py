#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""StoryCloze: forced-choice (log-likelihood ranking) narrative coherence eval.

See docs/superpowers/specs/2026-09-21-storycloze-benchmark-design.md for the full design and
why this replaces the literal "StoryBench" (arXiv 2506.13356) paper, which targets
instruction-tuned frontier LLMs neither model here can imitate.

Dataset: juletxara/xstory_cloze, config "en", CC BY-SA 4.0 (see README's Provenance and
licensing section). Splits are named "train" (360 rows) and "eval" (1,510 rows) -- confirmed
directly against the dataset viewer, not assumed from the more common val/test naming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate import sign_test, SignTest  # noqa: E402

STORYCLOZE_DATASET = "juletxara/xstory_cloze"
STORYCLOZE_CONFIG = "en"
# Resolved once via HfApi().dataset_info(STORYCLOZE_DATASET).sha -- see Task 1 Step 1 of the
# implementation plan. Hardcoded rather than fetched at "latest" so every run of this script
# scores the identical dataset content, the same discipline train/paths.py already documents
# for tokenizer/corpus regeneration.
STORYCLOZE_REVISION = "c4c2d88a1ec8b37fe22166d2a610f272726724b6"
STORYCLOZE_SPLITS = ("train", "eval")

DEFAULT_CACHE_DIR = ROOT / "artifacts" / "storycloze" / "hf_cache"


@dataclass(frozen=True)
class StoryClozeItem:
    """One Story Cloze item: a 4-sentence context and two candidate 5th sentences."""

    story_id: str
    context_sentences: Tuple[str, str, str, str]
    ending_1: str
    ending_2: str
    correct_ending: int  # 1 or 2, matching the dataset's own answer_right_ending encoding

    def __post_init__(self) -> None:
        if self.correct_ending not in (1, 2):
            raise ValueError(
                f"story {self.story_id}: correct_ending must be 1 or 2, got "
                f"{self.correct_ending!r}")


def load_storycloze_items(split: str, *, cache_dir: Path = DEFAULT_CACHE_DIR,
                          revision: str = STORYCLOZE_REVISION) -> List[StoryClozeItem]:
    """Load one split of the English Story Cloze mirror, cached under ``cache_dir``.

    Lazy-imports ``datasets`` inside this function rather than at module scope, matching the
    existing pattern in ``scripts/fetch_corpus.py`` -- this project declares only three runtime
    dependencies in ``pyproject.toml`` and has repeatedly declined to add a fourth.
    """
    if split not in STORYCLOZE_SPLITS:
        raise ValueError(
            f"split must be one of {STORYCLOZE_SPLITS!r} (this dataset uses train/eval, not "
            f"val/test), got {split!r}")
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(STORYCLOZE_DATASET, STORYCLOZE_CONFIG, split=split,
                        revision=revision, cache_dir=str(cache_dir))
    items: List[StoryClozeItem] = []
    for row in rows:
        items.append(StoryClozeItem(
            story_id=str(row["story_id"]),
            context_sentences=(
                row["input_sentence_1"], row["input_sentence_2"],
                row["input_sentence_3"], row["input_sentence_4"],
            ),
            ending_1=row["sentence_quiz1"],
            ending_2=row["sentence_quiz2"],
            correct_ending=int(row["answer_right_ending"]),
        ))
    if not items:
        raise ValueError(
            f"loaded zero items for split={split!r} from {STORYCLOZE_DATASET}@{revision} -- "
            f"the dataset or revision may have changed")
    return items
