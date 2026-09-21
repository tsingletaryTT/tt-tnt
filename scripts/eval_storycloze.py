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


@dataclass(frozen=True)
class EndingScore:
    """Log-likelihood of an ending's tokens, conditioned on a context, in two normalizations."""

    raw_sum_logprob: float
    mean_logprob: float
    n_tokens: int


def tokenize_sentence(tokenizer, text: str) -> List[int]:
    """Encode one sentence with no special tokens, one call per sentence.

    Matches ``train/tokenization.py::encode_batch``'s per-line encoding convention -- this
    tokenizer's ``PreTrainedTokenizerFast`` wrapper injects a leading space per encode call
    (see CLAUDE.md's tokenizer-and-corpus entry), so encoding sentence-by-sentence rather than
    concatenating raw strings before tokenizing reproduces the seam shape training data had.
    """
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def score_ending(model, context_ids: Sequence[int], ending_ids: Sequence[int]) -> EndingScore:
    """Teacher-forced log-likelihood of ``ending_ids`` conditioned on ``context_ids``.

    Computed directly from logits against next-token targets, with no ``labels=`` kwarg --
    ``LlamaForCausalLM``'s internal loss shifts labels a second time if already-aligned
    next-token labels are passed through it, which silently produces a near-uniform-ceiling
    number regardless of model quality. See ``tests/test_hf_parity.py``'s docstring for the
    concrete historical case (8.53 nats reported instead of ~3.20) this avoids, and
    ``scripts/eval_per_source.py::per_window_losses`` for the same pattern already in use here.
    """
    import torch

    full_ids = list(context_ids) + list(ending_ids)
    if len(full_ids) < 2:
        raise ValueError(
            "need at least 2 tokens total (>=1 context token + >=1 ending token) to score "
            "a next-token prediction")
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids).logits[0].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = torch.tensor(full_ids[1:], dtype=torch.long)
    token_logprobs = log_probs[torch.arange(len(targets)), targets]

    ending_start = len(context_ids) - 1  # index into targets/token_logprobs
    ending_logprobs = token_logprobs[ending_start:]
    if ending_logprobs.numel() != len(ending_ids):
        raise AssertionError(
            f"expected {len(ending_ids)} ending log-probabilities, computed "
            f"{ending_logprobs.numel()} -- context/ending token accounting is wrong")

    raw_sum = float(ending_logprobs.sum())
    n = int(ending_logprobs.numel())
    return EndingScore(raw_sum_logprob=raw_sum, mean_logprob=raw_sum / n, n_tokens=n)


@dataclass(frozen=True)
class ItemResult:
    """One scored Story Cloze item: both endings, full-context and context-blind."""

    story_id: str
    correct_ending: int
    full_1: EndingScore
    full_2: EndingScore
    blind_1: EndingScore
    blind_2: EndingScore


def score_item(model, tokenizer, item: StoryClozeItem) -> ItemResult:
    """Score both endings of ``item`` under full context and under a context-blind control.

    The blind context is exactly ``[bos_token_id]`` -- one token, so ``score_ending`` still
    has something to condition the first ending token on, but it carries no information about
    THIS item's actual story. Reusing ``score_ending`` for both conditions (rather than a
    separate blind-scoring function) is what makes
    ``test_context_blind_scores_are_identical_regardless_of_context`` a meaningful proof: the
    blind path is structurally incapable of seeing ``context_sentences``.
    """
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError(
            f"{tokenizer} has no bos_token_id -- the context-blind control needs one token "
            f"to condition the first ending token on")

    context_ids: List[int] = []
    for sentence in item.context_sentences:
        context_ids.extend(tokenize_sentence(tokenizer, sentence))
    ending_1_ids = tokenize_sentence(tokenizer, item.ending_1)
    ending_2_ids = tokenize_sentence(tokenizer, item.ending_2)

    return ItemResult(
        story_id=item.story_id,
        correct_ending=item.correct_ending,
        full_1=score_ending(model, context_ids, ending_1_ids),
        full_2=score_ending(model, context_ids, ending_2_ids),
        blind_1=score_ending(model, [bos_id], ending_1_ids),
        blind_2=score_ending(model, [bos_id], ending_2_ids),
    )


@dataclass(frozen=True)
class NormalizationStats:
    n_items: int
    accuracy: float
    context_blind_accuracy: float
    class_balance_fraction_answer_1: float
    length_bias_correlation: Optional[float]
    context_blind_length_bias_correlation: Optional[float]

    def as_json(self) -> dict:
        return {
            "n_items": self.n_items,
            "accuracy": self.accuracy,
            "context_blind_accuracy": self.context_blind_accuracy,
            "class_balance_fraction_answer_1": self.class_balance_fraction_answer_1,
            "length_bias_correlation": self.length_bias_correlation,
            "context_blind_length_bias_correlation": self.context_blind_length_bias_correlation,
        }


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or None if either series has zero variance (undefined, not 0.0)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _accuracy_and_length_bias(
    picks: Sequence[Tuple[float, float, int, int]], correct: Sequence[int],
) -> Tuple[float, Optional[float]]:
    """``picks`` is (score_1, score_2, n_tokens_1, n_tokens_2) per item.

    Returns (accuracy, length_bias_correlation), where length_bias_correlation is the Pearson
    correlation between (score_1 - score_2) and (n_tokens_1 - n_tokens_2) across items -- a
    scorer with no length bias should show this near 0.
    """
    n_correct = 0
    score_diffs: List[float] = []
    len_diffs: List[float] = []
    for (s1, s2, len1, len2), correct_ending in zip(picks, correct):
        chosen = 1 if s1 > s2 else 2
        if chosen == correct_ending:
            n_correct += 1
        score_diffs.append(s1 - s2)
        len_diffs.append(float(len1 - len2))
    accuracy = n_correct / len(picks) if picks else 0.0
    return accuracy, pearson_correlation(score_diffs, len_diffs)


def aggregate(results: Sequence[ItemResult]) -> Dict[str, NormalizationStats]:
    """Accuracy, context-blind accuracy, class balance, and length-bias per normalization."""
    if not results:
        raise ValueError("aggregate() called with zero items")

    correct = [r.correct_ending for r in results]
    n_answer_1 = sum(1 for c in correct if c == 1)
    class_balance = n_answer_1 / len(results)

    out: Dict[str, NormalizationStats] = {}
    for norm_name, attr in (("raw_sum", "raw_sum_logprob"), ("mean_per_token", "mean_logprob")):
        full_picks = [
            (getattr(r.full_1, attr), getattr(r.full_2, attr), r.full_1.n_tokens, r.full_2.n_tokens)
            for r in results
        ]
        blind_picks = [
            (getattr(r.blind_1, attr), getattr(r.blind_2, attr), r.blind_1.n_tokens, r.blind_2.n_tokens)
            for r in results
        ]
        full_acc, full_bias = _accuracy_and_length_bias(full_picks, correct)
        blind_acc, blind_bias = _accuracy_and_length_bias(blind_picks, correct)
        out[norm_name] = NormalizationStats(
            n_items=len(results),
            accuracy=full_acc,
            context_blind_accuracy=blind_acc,
            class_balance_fraction_answer_1=class_balance,
            length_bias_correlation=full_bias,
            context_blind_length_bias_correlation=blind_bias,
        )
    return out
