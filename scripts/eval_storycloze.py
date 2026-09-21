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


def choose_headline_normalization(stats: Dict[str, NormalizationStats]) -> str:
    """The normalization with the lower |length_bias_correlation| is reported as headline.

    A ``None`` bias (undefined, zero-variance case) is treated as the worst possible value
    (``math.inf``) rather than compared with ``<`` directly against a real float, so an
    undefined bias never wins by accident of Python's comparison rules.
    """

    def _key(name: str) -> float:
        bias = stats[name].length_bias_correlation
        return math.inf if bias is None else abs(bias)

    return min(stats, key=_key)


def compare_reports(a: dict, b: dict) -> dict:
    """Paired McNemar-equivalent comparison of two ``--out`` result dicts.

    McNemar's exact test over a 2x2 correct/incorrect table reduces to an exact two-sided
    binomial (sign) test over the discordant pairs -- exactly what
    ``scripts.evaluate.sign_test`` already computes, reused here rather than reimplemented.
    Refuses a pair scored on different dataset revisions or splits, the same refusal shape as
    ``evaluate.py``'s window guard and ``reach.py``'s cross-arm-set refusal -- a number
    computed from two non-comparable inputs is worse than no number.
    """
    if a["dataset_revision"] != b["dataset_revision"]:
        raise ValueError(
            f"dataset_revision mismatch: {a['dataset_revision']!r} vs {b['dataset_revision']!r} "
            f"-- refusing to compare results scored on different dataset snapshots")
    if a["split"] != b["split"]:
        raise ValueError(
            f"split mismatch: {a['split']!r} vs {b['split']!r} -- refusing to compare results "
            f"scored on different splits")

    norm = a["headline_normalization"]
    key = f"chosen_{norm}"
    by_id_a = {row["story_id"]: row for row in a["per_item"]}
    by_id_b = {row["story_id"]: row for row in b["per_item"]}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))

    deltas: List[float] = []
    for story_id in shared_ids:
        row_a, row_b = by_id_a[story_id], by_id_b[story_id]
        correct_a = row_a[key] == row_a["correct_ending"]
        correct_b = row_b[key] == row_b["correct_ending"]
        if correct_a == correct_b:
            deltas.append(0.0)
        elif correct_b and not correct_a:
            deltas.append(1.0)   # favors b
        else:
            deltas.append(-1.0)  # favors a

    result = sign_test(deltas)
    return {
        "n_items_compared": len(shared_ids),
        "headline_normalization": norm,
        "sign_test": result.as_json(),
    }


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report(model_dir: Path, items: Sequence[StoryClozeItem],
                 results: Sequence[ItemResult], *, split: str, revision: str) -> dict:
    """Assemble the full output JSON: provenance, per-normalization stats, per-item scores."""
    stats = aggregate(results)
    headline = choose_headline_normalization(stats)

    per_item = []
    for item, result in zip(items, results):
        row = {
            "story_id": result.story_id,
            "correct_ending": result.correct_ending,
            "full_1_raw_sum": result.full_1.raw_sum_logprob,
            "full_2_raw_sum": result.full_2.raw_sum_logprob,
            "full_1_mean": result.full_1.mean_logprob,
            "full_2_mean": result.full_2.mean_logprob,
            "full_1_n_tokens": result.full_1.n_tokens,
            "full_2_n_tokens": result.full_2.n_tokens,
            "blind_1_raw_sum": result.blind_1.raw_sum_logprob,
            "blind_2_raw_sum": result.blind_2.raw_sum_logprob,
            "blind_1_mean": result.blind_1.mean_logprob,
            "blind_2_mean": result.blind_2.mean_logprob,
            "chosen_raw_sum": 1 if result.full_1.raw_sum_logprob > result.full_2.raw_sum_logprob else 2,
            "chosen_mean_per_token": 1 if result.full_1.mean_logprob > result.full_2.mean_logprob else 2,
        }
        per_item.append(row)

    weights_path = model_dir / "model.safetensors"
    return {
        "schema": "tt-tnt/storycloze/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": STORYCLOZE_DATASET,
        "dataset_config": STORYCLOZE_CONFIG,
        "dataset_revision": revision,
        "split": split,
        "model_dir": str(model_dir),
        "model_weights_sha256": sha256_of_file(weights_path),
        "normalizations": {name: s.as_json() for name, s in stats.items()},
        "headline_normalization": headline,
        "headline_accuracy": stats[headline].accuracy,
        "per_item": per_item,
    }


def rescore_from_report(report: dict) -> dict:
    """Recompute every aggregate number from ``report["per_item"]`` alone -- no model needed.

    Reconstructs minimal ``EndingScore``/``ItemResult`` objects from the stored per-item
    floats rather than re-deriving anything from text, so this is a pure re-analysis of
    already-computed numbers, the same ``--rescore-from`` contract as
    ``eval_reach.py``/``eval_skits.py``.
    """
    results = []
    for row in report["per_item"]:
        results.append(ItemResult(
            story_id=row["story_id"],
            correct_ending=row["correct_ending"],
            full_1=EndingScore(row["full_1_raw_sum"], row["full_1_mean"], row["full_1_n_tokens"]),
            full_2=EndingScore(row["full_2_raw_sum"], row["full_2_mean"], row["full_2_n_tokens"]),
            blind_1=EndingScore(row["blind_1_raw_sum"], row["blind_1_mean"], row["full_1_n_tokens"]),
            blind_2=EndingScore(row["blind_2_raw_sum"], row["blind_2_mean"], row["full_2_n_tokens"]),
        ))
    stats = aggregate(results)
    headline = choose_headline_normalization(stats)
    out = dict(report)
    out["normalizations"] = {name: s.as_json() for name, s in stats.items()}
    out["headline_normalization"] = headline
    out["headline_accuracy"] = stats[headline].accuracy
    return out


def run_model(model_dir: Path, split: str, *, revision: str = STORYCLOZE_REVISION,
             cache_dir: Path = DEFAULT_CACHE_DIR) -> dict:
    """Load ``model_dir`` and score every item of ``split``. Needs torch/transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = load_storycloze_items(split, cache_dir=cache_dir, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir)).eval()

    results = [score_item(model, tokenizer, item) for item in items]
    return build_report(model_dir, items, results, split=split, revision=revision)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None,
                    help="Path to a converted HF model directory, e.g. artifacts/hf-tt-tnt-v3")
    ap.add_argument("--split", choices=STORYCLOZE_SPLITS, default="eval")
    ap.add_argument("--out", type=Path, default=None, help="Where to write the result JSON")
    ap.add_argument("--rescore-from", type=Path, default=None,
                    help="Re-derive every number from a stored result JSON -- no model, "
                         "tokenizer, or device needed")
    ap.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("A_JSON", "B_JSON"),
                    help="Paired McNemar-equivalent comparison of two result JSONs")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.compare is not None:
        a = json.loads(args.compare[0].read_text())
        b = json.loads(args.compare[1].read_text())
        result = compare_reports(a, b)
        print(json.dumps(result, indent=2))
        return 0

    if args.rescore_from is not None:
        report = json.loads(args.rescore_from.read_text())
        rescored = rescore_from_report(report)
        out_path = args.out or args.rescore_from
        out_path.write_text(json.dumps(rescored, indent=2) + "\n")
        print(f"[rescore] wrote {out_path}")
        return 0

    if args.model is None:
        raise SystemExit("--model is required unless --rescore-from or --compare is given")
    report = run_model(args.model, args.split)
    out_path = args.out or (ROOT / "docs" / "measurements" / f"storycloze-{args.model.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[eval_storycloze] wrote {out_path} -- headline accuracy "
          f"({report['headline_normalization']}): {report['headline_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
