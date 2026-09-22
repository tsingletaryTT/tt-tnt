# StoryCloze Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/eval_storycloze.py`, a CPU-only forced-choice (log-likelihood ranking)
StoryCloze evaluator, run it against `episod/tt-tnt` (v3) and `episod/tt-tnt-1024`, and publish
the results with built-in controls against Story Cloze's known context-blind-shortcut artifact.

**Architecture:** One new script, following this repo's existing eval-script conventions
(`scripts/eval_per_source.py`, `scripts/evaluate.py`, `scripts/eval_reach.py`): a data-loading
section (cached HF dataset fetch at a pinned revision), a pure scoring core (teacher-forced
log-likelihood via `transformers`, no double-shift bug), an aggregation/controls section, a
`--compare` mode reusing `scripts.evaluate.sign_test` for a McNemar-equivalent paired test, and
a CLI with `--rescore-from` (re-derive every number from stored per-item data, no model needed).

**Tech Stack:** Python 3.10+, `transformers` (already a declared dependency), `torch` (already
present — used the same way `scripts/eval_per_source.py` uses it), `datasets` (lazy-imported
inside a function, matching `scripts/fetch_corpus.py`'s existing precedent — **not** added to
`pyproject.toml`, since it is not a newly-declared dependency, just an existing lazy pattern
reused), stdlib `math`/`json`/`argparse`/`dataclasses`. No new pyproject dependency.

**Spec:** [`docs/superpowers/specs/2026-09-21-storycloze-benchmark-design.md`](../specs/2026-09-21-storycloze-benchmark-design.md)

## Global Constraints

- CPU-only. `scripts/eval_storycloze.py` must never import `ttnn` or `ttml`, and must run with
  no Tenstorrent device and no gozer lease.
- Dataset: `juletxara/xstory_cloze`, config `"en"`, CC BY-SA 4.0, fetched at a **pinned HF
  revision** (resolved in Task 1, hardcoded as a module constant — never "latest").
- Real dataset splits are named `"train"` (360 rows) and `"eval"` (1,510 rows) — **not**
  `"val"`/`"test"`.
- Every output JSON records: dataset revision, split, the model directory path, and a sha256 of
  the model's `model.safetensors` (proves which weights were actually scored, not just which
  directory name was passed).
- `--rescore-from` must re-derive every published number with **no** model, tokenizer, or
  device — verified under an import blocker (subprocess check that `torch`/`transformers` never
  get imported on that code path).
- Both raw-summed and length-normalized (mean-per-token) log-likelihood are always computed and
  reported; neither is silently dropped.
- The context-blind control and the not-hollow scorer proof are **required**, not optional
  flags — every real run computes them.
- No new dependency is added to `pyproject.toml`. `datasets` is lazy-imported inside functions,
  matching `scripts/fetch_corpus.py`'s existing pattern.
- README provenance section gets the `juletxara/xstory_cloze` CC BY-SA 4.0 entry in the same
  change that adds the script (per this project's standing licensing-discipline rule).

---

## File Structure

- **Create `scripts/eval_storycloze.py`** — the whole subsystem: data loading, scoring core,
  aggregation/controls, compare/McNemar, CLI. Single file, following this repo's convention of
  one file per eval instrument (`eval_reach.py`, `eval_skits.py`, `eval_per_source.py` are all
  single large files with clearly separated sections via `# ---` banner comments).
- **Create `tests/test_eval_storycloze.py`** — all tests for the above.
- **Modify `README.md`** — new provenance entry (near the existing `## Provenance and
  licensing` section, alongside the TinyStories/Wikipedia/dialogue/FineWeb-Edu entries) and a
  new subsection under `## External benchmarks` (alongside `### GPT-2 through the same harness`
  and `### What the benchmarks said`) reporting the real run's results.
- **No changes** to `scripts/evaluate.py` other than being imported from (its `sign_test` and
  `SignTest` are reused, not modified).

---

### Task 1: Dataset access — cached, pinned-revision StoryCloze loading

**Files:**
- Create: `scripts/eval_storycloze.py` (module docstring + imports + this section)
- Test: `tests/test_eval_storycloze.py`

**Interfaces:**
- Produces: `StoryClozeItem` (frozen dataclass: `story_id: str`, `context_sentences: Tuple[str, str, str, str]`, `ending_1: str`, `ending_2: str`, `correct_ending: int`), `STORYCLOZE_DATASET: str`, `STORYCLOZE_CONFIG: str`, `STORYCLOZE_REVISION: str`, `load_storycloze_items(split: str, *, cache_dir: Path, revision: str = STORYCLOZE_REVISION) -> List[StoryClozeItem]`, `DEFAULT_CACHE_DIR: Path` (`ROOT / "artifacts" / "storycloze" / "hf_cache"`).

- [ ] **Step 1: Resolve and hardcode the pinned HF revision**

Run this once, interactively, to resolve a concrete commit sha for `juletxara/xstory_cloze`
(never trust "latest" at run time — the same discipline `train/paths.py` already documents for
tokenizer/corpus regeneration):

```bash
python3 -c "
from huggingface_hub import HfApi
info = HfApi().dataset_info('juletxara/xstory_cloze')
print(info.sha)
"
```

Record the printed sha as the literal value of `STORYCLOZE_REVISION` below — do not leave it as
a variable resolved at import time.

- [ ] **Step 2: Write the module header and the data section**

```python
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
STORYCLOZE_REVISION = "REPLACE_WITH_RESOLVED_SHA"
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
```

- [ ] **Step 3: Write the failing test for item loading and validation**

```python
# tests/test_eval_storycloze.py
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_storycloze import StoryClozeItem  # noqa: E402


def test_storycloze_item_rejects_invalid_correct_ending():
    with pytest.raises(ValueError, match="correct_ending must be 1 or 2"):
        StoryClozeItem(
            story_id="x", context_sentences=("a", "b", "c", "d"),
            ending_1="e1", ending_2="e2", correct_ending=3,
        )


def test_storycloze_item_accepts_valid_construction():
    item = StoryClozeItem(
        story_id="x", context_sentences=("a", "b", "c", "d"),
        ending_1="e1", ending_2="e2", correct_ending=1,
    )
    assert item.correct_ending == 1
    assert item.context_sentences == ("a", "b", "c", "d")
```

- [ ] **Step 4: Run test to verify it fails before the module exists correctly, then implement**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v`
Expected first: FAIL (module or class not defined / `STORYCLOZE_REVISION` placeholder not yet
resolved is fine for this test — it does not touch the network).

Then apply Step 2's code (with the real resolved revision from Step 1) and re-run.
Expected: 2 passed.

- [ ] **Step 5: Add a live network test for `load_storycloze_items`, marked so it can be skipped offline**

```python
def test_load_storycloze_items_train_split_has_plausible_shape():
    """Requires network access to the HF Hub -- skips cleanly without it."""
    from scripts.eval_storycloze import load_storycloze_items

    try:
        items = load_storycloze_items("train")
    except Exception as exc:  # noqa: BLE001 - network/environment, not a code defect
        pytest.skip(f"could not reach the HF Hub: {exc}")
    assert len(items) > 0
    first = items[0]
    assert isinstance(first.story_id, str) and first.story_id
    assert all(isinstance(s, str) and s for s in first.context_sentences)
    assert first.correct_ending in (1, 2)
```

- [ ] **Step 6: Run it, confirm it either passes (network available) or skips cleanly**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v`
Expected: 3 passed, or 2 passed + 1 skipped with a printed reason if this sandbox has no
network access to huggingface.co.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_storycloze.py tests/test_eval_storycloze.py
git commit -m "feat(eval): StoryCloze dataset loading, pinned to a resolved HF revision"
```

---

### Task 2: Scoring core — teacher-forced log-likelihood, no double-shift bug

**Files:**
- Modify: `scripts/eval_storycloze.py` (append scoring section)
- Test: `tests/test_eval_storycloze.py` (append)

**Interfaces:**
- Consumes: nothing new from Task 1 beyond the module already existing.
- Produces: `EndingScore` (frozen dataclass: `raw_sum_logprob: float`, `mean_logprob: float`, `n_tokens: int`), `tokenize_sentence(tokenizer, text: str) -> List[int]`, `score_ending(model, context_ids: List[int], ending_ids: List[int]) -> EndingScore`.

- [ ] **Step 1: Write the failing test for the scorer's core correctness property**

This is the "not-hollow" proof from the spec: a real context must score its real continuation
higher than an obviously-garbled alternative.

```python
def _tiny_causal_lm_and_tokenizer():
    """A tiny randomly-initialized GPT2-architecture model+tokenizer for fast CPU tests.

    Not this project's own model -- the scorer is architecture-agnostic (works on any
    AutoModelForCausalLM), and using a tiny stock model keeps these tests independent of
    artifacts/hf-tt-tnt-* being present on disk.
    """
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config = GPT2Config(vocab_size=tokenizer.vocab_size, n_embd=32, n_layer=2, n_head=2,
                        n_positions=128)
    model = GPT2LMHeadModel(config).eval()
    return model, tokenizer


def test_score_ending_produces_expected_token_count():
    from scripts.eval_storycloze import score_ending, tokenize_sentence

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    context_ids = tokenize_sentence(tokenizer, "The sky is blue.")
    ending_ids = tokenize_sentence(tokenizer, "It rained.")
    result = score_ending(model, context_ids, ending_ids)
    assert result.n_tokens == len(ending_ids)
    assert isinstance(result.raw_sum_logprob, float)
    assert isinstance(result.mean_logprob, float)
    # mean is the sum averaged over exactly n_tokens, not some other count
    assert result.mean_logprob == pytest.approx(
        result.raw_sum_logprob / result.n_tokens, rel=1e-6)


def test_score_ending_requires_at_least_one_context_and_one_ending_token():
    from scripts.eval_storycloze import score_ending

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    with pytest.raises(ValueError, match="at least 2 tokens"):
        score_ending(model, context_ids=[], ending_ids=[])
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k score_ending`
Expected: FAIL (`score_ending`/`tokenize_sentence` not defined).

- [ ] **Step 3: Implement the scoring core**

```python
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
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k score_ending`
Expected: 2 passed.

- [ ] **Step 5: Add the not-hollow proof and its mutation check**

```python
def test_scorer_prefers_the_real_continuation_over_a_garbled_one():
    """The not-hollow proof: the scorer must be able to discriminate at all.

    A garbled ending (its own tokens reversed) is not a naturally-occurring completion of
    the context, so a working scorer must give it lower likelihood than the real, coherent
    continuation. This is checked BEFORE trusting the scorer on real Story Cloze data.
    """
    from scripts.eval_storycloze import score_ending, tokenize_sentence

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    context_ids = tokenize_sentence(
        tokenizer, "Sarah walked to the store. She bought some bread. She paid with cash.")
    real_ids = tokenize_sentence(tokenizer, "She walked back home.")
    garbled_ids = list(reversed(real_ids))

    real_score = score_ending(model, context_ids, real_ids)
    garbled_score = score_ending(model, context_ids, garbled_ids)

    # NOTE: a randomly-initialized tiny model has no learned preference, so this specific
    # assertion is checked against a model that has SEEN this exact continuation during a
    # few steps of fine-tuning within the test -- see the fixture below, not the bare
    # untrained model from _tiny_causal_lm_and_tokenizer().
    assert real_score.mean_logprob > garbled_score.mean_logprob
```

This test needs a model that has actually learned *something* about the real continuation, or
a random-init model's comparison is meaningless noise. Replace the fixture used in this one
test with a version fine-tuned for a handful of steps on the exact real sentence, so the
assertion has a genuine reason to hold:

```python
def _tiny_causal_lm_finetuned_on(tokenizer, text: str, steps: int = 30):
    import torch

    model, _ = _tiny_causal_lm_and_tokenizer()
    ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(ids[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
    model.eval()
    return model
```

Rewrite `test_scorer_prefers_the_real_continuation_over_a_garbled_one` to build its model with
`_tiny_causal_lm_finetuned_on(tokenizer, "Sarah walked to the store. She bought some bread. "
"She paid with cash. She walked back home.")` instead of the bare fixture, keeping everything
else the same.

- [ ] **Step 6: Run it, confirm it passes; then mutate `score_ending` to prove the test is not vacuous**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k not_hollow_placeholder_name`
(use the actual test function name from Step 5)
Expected: PASS.

Then temporarily edit `score_ending` to return `EndingScore(raw_sum_logprob=-raw_sum, ...)`
(flip the sign) and re-run the same test.
Expected: FAIL. Revert the temporary edit immediately after confirming this.

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_storycloze.py tests/test_eval_storycloze.py
git commit -m "feat(eval): teacher-forced ending scorer, not-hollow proof mutation-checked"
```

---

### Task 3: Item-level scoring, context-blind control, and aggregation

**Files:**
- Modify: `scripts/eval_storycloze.py` (append)
- Test: `tests/test_eval_storycloze.py` (append)

**Interfaces:**
- Consumes: `StoryClozeItem` (Task 1), `EndingScore`/`score_ending`/`tokenize_sentence` (Task 2).
- Produces: `ItemResult` (dataclass: `story_id: str`, `correct_ending: int`, `full_context: EndingScore`, `full_context_2: EndingScore`, `blind_1: EndingScore`, `blind_2: EndingScore` — see field naming below), `score_item(model, tokenizer, item: StoryClozeItem) -> ItemResult`, `NormalizationStats` (dataclass), `aggregate(results: List[ItemResult]) -> Dict[str, "NormalizationStats"]`, `pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]`.

- [ ] **Step 1: Write the failing test for per-item scoring shape and the context-blind purity property**

```python
def test_score_item_returns_both_endings_full_and_blind():
    from scripts.eval_storycloze import StoryClozeItem, score_item

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    item = StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="It ended well.", ending_2="It ended badly.", correct_ending=1)
    result = score_item(model, tokenizer, item)
    assert result.story_id == "s1"
    assert result.correct_ending == 1
    for field_name in ("full_1", "full_2", "blind_1", "blind_2"):
        score = getattr(result, field_name)
        assert score.n_tokens > 0


def test_context_blind_scores_are_identical_regardless_of_context():
    """The context-blind control must be provably blind, not merely named blind.

    Two DIFFERENT contexts, same ending text: if the "blind" score ever changes with the
    context, the control is leaking context and is not measuring what it claims to measure.
    """
    from scripts.eval_storycloze import StoryClozeItem, score_item

    model, tokenizer = _tiny_causal_lm_and_tokenizer()
    item_a = StoryClozeItem(
        story_id="a", context_sentences=("The sun rose.", "Birds sang.", "It was warm.", "Dew glistened."),
        ending_1="She smiled.", ending_2="unused", correct_ending=1)
    item_b = StoryClozeItem(
        story_id="b", context_sentences=("The city burned.", "Sirens wailed.", "Smoke rose.", "People fled."),
        ending_1="She smiled.", ending_2="unused", correct_ending=1)

    result_a = score_item(model, tokenizer, item_a)
    result_b = score_item(model, tokenizer, item_b)

    assert result_a.blind_1.raw_sum_logprob == pytest.approx(result_b.blind_1.raw_sum_logprob)
    assert result_a.blind_1.mean_logprob == pytest.approx(result_b.blind_1.mean_logprob)
```

- [ ] **Step 2: Run tests, confirm they fail**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k "score_item or context_blind"`
Expected: FAIL (`score_item`/`ItemResult` not defined).

- [ ] **Step 3: Implement `ItemResult` and `score_item`**

```python
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
```

- [ ] **Step 4: Run tests, confirm they pass**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k "score_item or context_blind"`
Expected: 2 passed.

- [ ] **Step 5: Mutate to prove the context-blind test is not vacuous**

Temporarily change `blind_1=score_ending(model, [bos_id], ending_1_ids)` to
`blind_1=score_ending(model, context_ids, ending_1_ids)` (leaking real context into the
"blind" path) and re-run `test_context_blind_scores_are_identical_regardless_of_context`.
Expected: FAIL (item_a and item_b now have different real contexts, so their "blind" scores
differ). Revert the temporary edit immediately after confirming this.

- [ ] **Step 6: Write the failing test for aggregation (accuracy, length-bias, class balance)**

```python
def test_aggregate_computes_accuracy_and_length_bias_per_normalization():
    from scripts.eval_storycloze import (
        EndingScore, ItemResult, aggregate,
    )

    # Item 1: correct ending (1) scores higher on both raw and mean -- a clean correct case.
    # Item 2: correct ending (2) scores LOWER on raw sum (because it's longer) but higher on
    #   mean-per-token -- this is the length-bias case the aggregation must be able to show.
    results = [
        ItemResult(
            story_id="i1", correct_ending=1,
            full_1=EndingScore(raw_sum_logprob=-2.0, mean_logprob=-1.0, n_tokens=2),
            full_2=EndingScore(raw_sum_logprob=-5.0, mean_logprob=-2.5, n_tokens=2),
            blind_1=EndingScore(raw_sum_logprob=-2.0, mean_logprob=-1.0, n_tokens=2),
            blind_2=EndingScore(raw_sum_logprob=-2.0, mean_logprob=-1.0, n_tokens=2),
        ),
        ItemResult(
            story_id="i2", correct_ending=2,
            full_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-3.0, n_tokens=1),
            full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-1.0, n_tokens=4),
            blind_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-1.0, n_tokens=1),
            blind_2=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-1.0, n_tokens=1),
        ),
    ]
    stats = aggregate(results)
    assert set(stats) == {"raw_sum", "mean_per_token"}
    # raw_sum: item 1 correct (picks ending 1, higher raw), item 2 WRONG (picks ending 1,
    # -3.0 > -4.0, but correct is ending 2) -> 1/2 accuracy
    assert stats["raw_sum"].accuracy == pytest.approx(0.5)
    # mean_per_token: item 1 correct, item 2 correct (-1.0 > -3.0, picks ending 2) -> 2/2
    assert stats["mean_per_token"].accuracy == pytest.approx(1.0)
    assert stats["raw_sum"].n_items == 2
    assert 0 <= stats["raw_sum"].class_balance_fraction_answer_1 <= 1
    assert stats["raw_sum"].length_bias_correlation is not None
```

- [ ] **Step 7: Run test, confirm it fails**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k test_aggregate`
Expected: FAIL (`aggregate`/`NormalizationStats` not defined).

- [ ] **Step 8: Implement `aggregate` and its helpers**

```python
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
```

- [ ] **Step 9: Run tests, confirm they pass**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k test_aggregate`
Expected: 1 passed.

- [ ] **Step 10: Commit**

```bash
git add scripts/eval_storycloze.py tests/test_eval_storycloze.py
git commit -m "feat(eval): item scoring, context-blind control, and aggregation"
```

---

### Task 4: Headline normalization choice and `--compare` (McNemar via `sign_test` reuse)

**Files:**
- Modify: `scripts/eval_storycloze.py` (append)
- Test: `tests/test_eval_storycloze.py` (append)

**Interfaces:**
- Consumes: `NormalizationStats`/`aggregate` (Task 3), `sign_test`/`SignTest` from `scripts.evaluate` (Task 1's import).
- Produces: `choose_headline_normalization(stats: Dict[str, NormalizationStats]) -> str`, `RunReport` (dataclass, defined fully in Task 5 but its `compare_reports` consumer is here), `compare_reports(a: dict, b: dict) -> dict` (operates on the parsed output JSON dicts, so it works identically for a freshly-scored run and a `--rescore-from`'d one).

- [ ] **Step 1: Write the failing test for headline-normalization selection**

```python
def test_choose_headline_normalization_prefers_lower_absolute_length_bias():
    from scripts.eval_storycloze import NormalizationStats, choose_headline_normalization

    stats = {
        "raw_sum": NormalizationStats(
            n_items=10, accuracy=0.6, context_blind_accuracy=0.55,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=0.7, context_blind_length_bias_correlation=0.6),
        "mean_per_token": NormalizationStats(
            n_items=10, accuracy=0.65, context_blind_accuracy=0.52,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=-0.1, context_blind_length_bias_correlation=0.05),
    }
    assert choose_headline_normalization(stats) == "mean_per_token"


def test_choose_headline_normalization_treats_none_bias_as_worst_case():
    """A normalization whose length-bias could not even be computed (zero variance) is not
    silently preferred just because None fails a naive comparison in its favor."""
    from scripts.eval_storycloze import NormalizationStats, choose_headline_normalization

    stats = {
        "raw_sum": NormalizationStats(
            n_items=10, accuracy=0.6, context_blind_accuracy=0.55,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=None, context_blind_length_bias_correlation=None),
        "mean_per_token": NormalizationStats(
            n_items=10, accuracy=0.65, context_blind_accuracy=0.52,
            class_balance_fraction_answer_1=0.5,
            length_bias_correlation=0.2, context_blind_length_bias_correlation=0.05),
    }
    assert choose_headline_normalization(stats) == "mean_per_token"
```

- [ ] **Step 2: Run, confirm failure, then implement**

```python
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
```

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k choose_headline`
Expected: 2 passed.

- [ ] **Step 3: Write the failing test for `compare_reports`' McNemar-equivalent result and its refusal**

```python
def _minimal_report(revision: str, split: str, per_item: list, headline_norm: str = "mean_per_token") -> dict:
    return {
        "schema": "tt-tnt/storycloze/1",
        "dataset_revision": revision,
        "split": split,
        "headline_normalization": headline_norm,
        "per_item": per_item,
    }


def test_compare_reports_runs_a_paired_sign_test_over_headline_correctness():
    from scripts.eval_storycloze import compare_reports

    # Both models score item 1 correctly (concordant), item 2: model A wrong, model B right
    # (discordant, favors B), item 3: model A right, model B wrong (discordant, favors A).
    per_item_a = [
        {"story_id": "s1", "correct_ending": 1, "chosen_mean_per_token": 1},
        {"story_id": "s2", "correct_ending": 1, "chosen_mean_per_token": 2},
        {"story_id": "s3", "correct_ending": 1, "chosen_mean_per_token": 1},
    ]
    per_item_b = [
        {"story_id": "s1", "correct_ending": 1, "chosen_mean_per_token": 1},
        {"story_id": "s2", "correct_ending": 1, "chosen_mean_per_token": 1},
        {"story_id": "s3", "correct_ending": 1, "chosen_mean_per_token": 2},
    ]
    a = _minimal_report("rev1", "eval", per_item_a)
    b = _minimal_report("rev1", "eval", per_item_b)

    result = compare_reports(a, b)
    assert result["n_items_compared"] == 3
    assert result["sign_test"]["n"] == 2  # two discordant pairs
    assert result["sign_test"]["n_negative"] == 1  # favors A (item s3)
    assert result["sign_test"]["n_positive"] == 1  # favors B (item s2)


def test_compare_reports_refuses_mismatched_dataset_revision():
    from scripts.eval_storycloze import compare_reports

    a = _minimal_report("rev1", "eval", [])
    b = _minimal_report("rev2", "eval", [])
    with pytest.raises(ValueError, match="dataset_revision"):
        compare_reports(a, b)


def test_compare_reports_refuses_mismatched_split():
    from scripts.eval_storycloze import compare_reports

    a = _minimal_report("rev1", "eval", [])
    b = _minimal_report("rev1", "train", [])
    with pytest.raises(ValueError, match="split"):
        compare_reports(a, b)
```

- [ ] **Step 4: Run, confirm failure, then implement `compare_reports`**

```python
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
```

- [ ] **Step 5: Run tests, confirm they pass**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k compare_reports`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_storycloze.py tests/test_eval_storycloze.py
git commit -m "feat(eval): headline normalization choice and paired McNemar-equivalent compare"
```

---

### Task 5: CLI, output JSON, `--rescore-from`, import purity

**Files:**
- Modify: `scripts/eval_storycloze.py` (append CLI/main)
- Test: `tests/test_eval_storycloze.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `sha256_of_file(path: Path) -> str`, `build_report(model_dir: Path, items: List[StoryClozeItem], results: List[ItemResult], *, split: str, revision: str) -> dict`, `run_model(model_dir: Path, split: str) -> dict` (loads model+tokenizer, scores every item, calls `build_report`), `rescore_from_report(report: dict) -> dict` (recomputes `aggregate`/`headline_normalization`/etc. purely from `report["per_item"]`, no model), `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test for the output JSON schema and `--rescore-from` round-trip**

```python
def test_build_report_records_provenance_and_per_item_scores(tmp_path):
    from scripts.eval_storycloze import StoryClozeItem, ItemResult, EndingScore, build_report

    item = StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="ok", ending_2="bad", correct_ending=1)
    result = ItemResult(
        story_id="s1", correct_ending=1,
        full_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-0.5, n_tokens=2),
        full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-2.0, n_tokens=2),
        blind_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
        blind_2=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
    )
    model_dir = tmp_path / "hf-fake"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")

    report = build_report(model_dir, [item], [result], split="eval", revision="rev1")

    assert report["schema"] == "tt-tnt/storycloze/1"
    assert report["dataset_revision"] == "rev1"
    assert report["split"] == "eval"
    assert report["model_dir"] == str(model_dir)
    assert len(report["model_weights_sha256"]) == 64  # hex sha256 digest length
    assert report["headline_normalization"] in ("raw_sum", "mean_per_token")
    assert len(report["per_item"]) == 1
    row = report["per_item"][0]
    assert row["story_id"] == "s1"
    assert row["correct_ending"] == 1
    assert "chosen_raw_sum" in row and "chosen_mean_per_token" in row


def test_rescore_from_report_reproduces_the_original_aggregate_stats(tmp_path):
    from scripts.eval_storycloze import (
        StoryClozeItem, ItemResult, EndingScore, build_report, rescore_from_report,
    )

    item = StoryClozeItem(
        story_id="s1", context_sentences=("A.", "B.", "C.", "D."),
        ending_1="ok", ending_2="bad", correct_ending=1)
    result = ItemResult(
        story_id="s1", correct_ending=1,
        full_1=EndingScore(raw_sum_logprob=-1.0, mean_logprob=-0.5, n_tokens=2),
        full_2=EndingScore(raw_sum_logprob=-4.0, mean_logprob=-2.0, n_tokens=2),
        blind_1=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
        blind_2=EndingScore(raw_sum_logprob=-3.0, mean_logprob=-1.5, n_tokens=2),
    )
    model_dir = tmp_path / "hf-fake"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"fake weights")

    original = build_report(model_dir, [item], [result], split="eval", revision="rev1")
    rescored = rescore_from_report(original)

    assert rescored["normalizations"] == original["normalizations"]
    assert rescored["headline_normalization"] == original["headline_normalization"]
```

- [ ] **Step 2: Run, confirm failure, then implement**

```python
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
```

- [ ] **Step 3: Run tests, confirm they pass**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k "build_report or rescore_from_report"`
Expected: 2 passed.

- [ ] **Step 4: Implement `run_model` and `main`, wiring the CLI**

```python
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
```

- [ ] **Step 5: Write and run a CLI-level `--rescore-from` round-trip test using a real subprocess**

```python
def test_rescore_from_cli_needs_no_torch_or_transformers_import(tmp_path):
    """Verified in a subprocess, matching tests/test_ttml_forward.py's import-purity pattern --
    this test session may have already imported torch/transformers elsewhere."""
    import subprocess

    report = {
        "schema": "tt-tnt/storycloze/1", "dataset_revision": "rev1", "split": "eval",
        "dataset": "juletxara/xstory_cloze", "dataset_config": "en",
        "model_dir": "fake", "model_weights_sha256": "0" * 64,
        "normalizations": {}, "headline_normalization": "mean_per_token",
        "headline_accuracy": 0.0,
        "per_item": [{
            "story_id": "s1", "correct_ending": 1,
            "full_1_raw_sum": -1.0, "full_2_raw_sum": -4.0,
            "full_1_mean": -0.5, "full_2_mean": -2.0,
            "full_1_n_tokens": 2, "full_2_n_tokens": 2,
            "blind_1_raw_sum": -3.0, "blind_2_raw_sum": -3.0,
            "blind_1_mean": -1.5, "blind_2_mean": -1.5,
            "chosen_raw_sum": 1, "chosen_mean_per_token": 1,
        }],
    }
    in_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    in_path.write_text(json.dumps(report))

    probe = (
        "import sys; "
        "from scripts.eval_storycloze import main; "
        f"main(['--rescore-from', {str(in_path)!r}, '--out', {str(out_path)!r}]); "
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                           check=True, cwd=str(ROOT))
    assert result.stdout.strip().splitlines()[-1] == "", (
        f"--rescore-from pulled in: {result.stdout.strip()}")
    assert out_path.is_file()
```

- [ ] **Step 6: Run it, confirm it passes**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k rescore_from_cli`
Expected: 1 passed.

- [ ] **Step 7: Add the import-purity test for `ttnn`/`ttml`**

```python
def test_eval_storycloze_module_imports_no_tenstorrent():
    """Matches tests/test_ttml_forward.py's pattern: checked in a subprocess since this test
    session may already have imported plenty of things transitively."""
    import subprocess

    probe = (
        "import sys; import scripts.eval_storycloze; "
        "bad=[m for m in ('ttnn','ttml') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                        check=True, cwd=str(ROOT))
    assert out.stdout.strip() == "", f"scripts.eval_storycloze pulled in: {out.stdout.strip()}"
```

- [ ] **Step 8: Run the full test file**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v`
Expected: all pass (network-dependent test from Task 1 Step 5 passes or skips cleanly).

- [ ] **Step 9: Commit**

```bash
git add scripts/eval_storycloze.py tests/test_eval_storycloze.py
git commit -m "feat(eval): CLI, output JSON, --rescore-from, --compare, import-purity test"
```

---

### Task 6: README provenance + real run against both models

**Files:**
- Modify: `README.md`
- Create: `docs/measurements/storycloze-tt-tnt.json`, `docs/measurements/storycloze-tt-tnt-1024.json`, `docs/measurements/storycloze-tt-tnt-vs-tt-tnt-1024.json`
- Test: `tests/test_eval_storycloze.py` (append README provenance test)

**Interfaces:**
- Consumes: `main`/`run_model` (Task 5).
- Produces: no new code interfaces — this task is running the finished tool for real and documenting the result.

- [ ] **Step 1: Write the failing test for the README provenance entry**

```python
def test_readme_documents_storycloze_license():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "juletxara/xstory_cloze" in readme
    assert "CC BY-SA 4.0" in readme
```

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k readme_documents_storycloze`
Expected: FAIL.

- [ ] **Step 2: Add the README provenance entry**

In `README.md`'s `## Provenance and licensing` section, after the existing FineWeb-Edu/mission
entries, add:

```markdown
Evaluation data — Story Cloze. `scripts/eval_storycloze.py` scores both published models
against the Story Cloze Test (Mostafazadeh et al. 2016) via
[`juletxara/xstory_cloze`](https://huggingface.co/datasets/juletxara/xstory_cloze)'s English
config, under **CC BY-SA 4.0** (the license the multilingual XStoryCloze project applied to its
whole release, inherited here rather than re-derived). This is evaluation data only — never
mixed into any training corpus — and is fetched from the Hugging Face Hub at a pinned revision,
not redistributed. Worth stating plainly: the original English Story Cloze Test is nominally
gated behind a research-use request form at the University of Rochester; this project is
relying on the upstream XStoryCloze project's own CC BY-SA 4.0 licensing decision for the
mirror it uses, not re-requesting the original release.
```

- [ ] **Step 3: Run the test, confirm it passes**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v -k readme_documents_storycloze`
Expected: 1 passed.

- [ ] **Step 4: Run the real evaluation against both published models**

```bash
python3 scripts/eval_storycloze.py --model artifacts/hf-tt-tnt-v3 --split eval \
  --out docs/measurements/storycloze-tt-tnt.json
python3 scripts/eval_storycloze.py --model artifacts/hf-tt-tnt-1024 --split eval \
  --out docs/measurements/storycloze-tt-tnt-1024.json
python3 scripts/eval_storycloze.py \
  --compare docs/measurements/storycloze-tt-tnt.json docs/measurements/storycloze-tt-tnt-1024.json \
  > docs/measurements/storycloze-tt-tnt-vs-tt-tnt-1024.json
```

If either model directory is missing or the HF Hub is unreachable from this environment, stop
here and report that explicitly rather than fabricating numbers — the next section assumes
these three files exist and are real.

- [ ] **Step 5: Read the results and write the README's "What the benchmarks said" companion section**

Under `## External benchmarks`, after `### What the benchmarks said`, add a new subsection
`### StoryCloze (forced-choice narrative coherence)` reporting, from the three JSON files just
produced: each model's headline accuracy and which normalization was headline, the
context-blind accuracy for each (stated plainly as a ceiling on what the full-context number
can claim, per the spec's Controls section), the class balance, and the `--compare` sign-test
result (n, n_negative, n_positive, p_two_sided) with an explicit reading of which model it
favors or whether it's inconclusive. Do not round a `p_two_sided` above the usual 0.05
threshold into "significant," and do not describe the context-blind number as subtracted out
of the headline — state both and let the gap speak for itself, matching this section's
existing prose style (see `### What the benchmarks said`'s `NOT SIGNIFICANT`/`AT CHANCE`-style
plain statements above it).

- [ ] **Step 6: Run the full test suite once more**

Run: `python3 -m pytest tests/test_eval_storycloze.py -v`
Expected: all pass (or the one network test skips cleanly, which is fine since Step 4 already
proved live network+model access worked in this environment).

- [ ] **Step 7: Commit**

```bash
git add README.md docs/measurements/storycloze-tt-tnt.json \
  docs/measurements/storycloze-tt-tnt-1024.json \
  docs/measurements/storycloze-tt-tnt-vs-tt-tnt-1024.json tests/test_eval_storycloze.py
git commit -m "docs: StoryCloze results for tt-tnt and tt-tnt-1024, provenance entry"
```

---

## Self-Review

**Spec coverage:**
- Data source/licensing (corrected to `juletxara/xstory_cloze`) → Task 1, Task 6 Step 2.
- Scoring methodology (both normalizations, no double-shift bug) → Task 2, Task 3.
- Controls (context-blind, class balance, not-hollow proof, McNemar paired comparison) → Task 2
  Step 5, Task 3 Steps 1-9, Task 4.
- CLI/reproducibility (`--model`, `--split`, `--rescore-from`, `--compare`, provenance fields in
  every output JSON) → Task 5.
- Testing (not-hollow mutation check, context-blind purity mutation check, license test,
  import-purity test, compare's mismatch refusal, McNemar hand-checked fixture) → Tasks 2-6,
  each with an explicit mutation or hand-computed-fixture step.
- Deliberately-out-of-scope items (long-memory variant, literal StoryBench, LLM-judge eval) are
  spec-only; no task attempts them, matching the spec.

**Placeholder scan:** `STORYCLOZE_REVISION = "REPLACE_WITH_RESOLVED_SHA"` in Task 1 Step 2 is
intentional and explicitly resolved in Task 1 Step 1 before that code is committed — the
placeholder text itself never reaches a commit. No other TBD/TODO remains.

**Type consistency:** `EndingScore`, `ItemResult`, `NormalizationStats`, `StoryClozeItem` are
defined once (Tasks 1-3) and referenced by the same names/fields in every later task;
`aggregate`'s return type (`Dict[str, NormalizationStats]`) matches what
`choose_headline_normalization` and `build_report` consume in Tasks 4-5.
