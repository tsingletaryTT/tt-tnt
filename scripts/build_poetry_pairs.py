#!/usr/bin/env python3
# scripts/build_poetry_pairs.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Poetry-instructions, in the shape of isaacrehg/poetry-instructions (HF), built from
this project's own CC0-1.0 poetry corpus slice instead of that dataset -- see this task's
entry in docs/superpowers/plans/2026-08-27-editor-training.md for why.

Two instruction shapes only (the two that don't need per-poem author attribution, which
this project's corpus pipeline does not carry):

  - continuation: "Continue this poem: <first part>" -> "<rest of the poem>"
  - keywords:     "Write a poem about: <keywords>"    -> "<the whole poem>"

Keywords are the top-N highest-IDF content words actually present in the poem -- a plain,
auditable substitute for isaacrehg's "keyphrase model", consistent with this project's
general preference for simple, explainable detectors over an added ML dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_WORD_RE = re.compile(r"[a-z']+")
_STOPWORDS = frozenset(
    "a an of in on at to for with as that this and but or so is was were be been "
    "i you he she it we they my your his her its our their".split()
)

#: The exact delimiter phrases the SFT prompt templates use (Task 4's build_poetry_example) --
#: checked for collisions the same way Task 2 checks its own delimiters.
DELIMITER_STRINGS = ["Continue this poem:", "Continuation:", "Write a poem about:", "Poem:"]

#: Task 4's dry-run against the real corpus found 99.86% of poetry-instructions examples
#: exceeding MAX_SEQ_LEN=512 tokens -- Gutenberg poems in this corpus run long (median
#: combined example length 655 tokens, max 1796), and build_pairs never bounded poem length
#: against the training window. This module has no tokenizer available, so word count is
#: only a first-pass reduction, not a guarantee: a follow-up measurement with the real
#: tokenizer found this corpus's tokens/word ratio has a heavy tail (median 1.71, p90 1.95,
#: max 3.13) that no flat word cap bounds reliably -- at this cap alone, ~55% of poetry
#: examples still exceeded 512 tokens. The actual guarantee lives in
#: scripts/train_editor.py's `_truncate_to_max_seq_len`, which truncates `input_ids`/
#: `labels` to MAX_SEQ_LEN explicitly, with the real tokenizer, at construction time. This
#: cap is kept anyway because it still cuts extreme cases cheaply before that.
MAX_POEM_WORDS = 300


def _cap_lines_to_word_budget(lines: List[str], max_words: int = MAX_POEM_WORDS) -> List[str]:
    """Keep as many whole lines from the START as fit within `max_words` (plain whitespace
    word count), dropping the rest from the END. Never mid-line. A poem already under
    budget is returned unchanged (same list contents, no truncation applied). The first
    line is always kept even if it alone exceeds the budget, so a single very long line
    doesn't collapse the poem to nothing."""
    kept: List[str] = []
    total = 0
    for line in lines:
        n = len(line.split())
        if kept and total + n > max_words:
            break
        kept.append(line)
        total += n
    return kept


def check_delimiter_collisions(poetry_txt: Path) -> List[str]:
    text = poetry_txt.read_text(encoding="utf-8", errors="replace")
    return [
        line.strip()[:100] for line in text.splitlines()
        if any(d in line for d in DELIMITER_STRINGS)
    ]


def split_poems(poetry_txt: Path) -> List[str]:
    """Split on the corpus's own document-separator convention (a line that is exactly
    `</s>`), dropping empty segments (e.g. two consecutive separators)."""
    text = poetry_txt.read_text(encoding="utf-8", errors="replace")
    segments = text.split("</s>")
    return [s.strip("\n") for s in segments if s.strip()]


def _words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def build_idf(poems: List[str]) -> Dict[str, float]:
    """Standard IDF: log(N / doc_freq(word)), documents = poems."""
    n = len(poems)
    doc_freq: Counter = Counter()
    for poem in poems:
        doc_freq.update(set(_words(poem)))
    return {word: math.log(n / freq) for word, freq in doc_freq.items()}


def top_keywords(poem: str, idf: Dict[str, float], n: int = 4) -> List[str]:
    """The `n` distinct words in `poem` with the highest IDF score, in descending order."""
    seen: Dict[str, float] = {}
    for w in _words(poem):
        if w in idf and w not in seen:
            seen[w] = idf[w]
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:n]]


def build_pairs(poems: List[str], *, seed: int) -> List[dict]:
    """One pair per poem, kind chosen deterministically per-poem by `seed`."""
    idf = build_idf(poems)
    rng = random.Random(seed)
    pairs = []
    for poem in poems:
        lines = poem.split("\n")
        if rng.random() < 0.5 and len(lines) >= 4:
            # Cap the poem's line budget BEFORE computing the 2/3 cut point, so the
            # input+target combined (not just the target) fits under MAX_POEM_WORDS.
            capped = _cap_lines_to_word_budget(lines)
            cut = max(1, len(capped) * 2 // 3)
            pairs.append({
                "kind": "continuation",
                "input": "\n".join(capped[:cut]),
                "arg": None,
                "target": "\n".join(capped[cut:]),
            })
        else:
            kw = top_keywords(poem, idf, n=4)
            capped = _cap_lines_to_word_budget(lines)
            pairs.append({"kind": "keywords", "input": None, "arg": kw, "target": "\n".join(capped)})
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--poetry-txt", type=Path, default=ROOT / "artifacts" / "corpus" / "poetry.txt")
    p.add_argument("--seed", type=int, default=5489)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "poetry-pairs" / "pairs.jsonl")
    args = p.parse_args()

    collisions = check_delimiter_collisions(args.poetry_txt)
    if collisions:
        print(f"WARNING: {len(collisions)} line(s) contain a literal delimiter string. "
              f"First 5:")
        for line in collisions[:5]:
            print(f"  {line}")
    else:
        print("No delimiter-string collisions found in poetry.txt.")

    poems = split_poems(args.poetry_txt)
    print(f"{len(poems)} poems split from {args.poetry_txt}")
    pairs = build_pairs(poems, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    kind_counts = Counter(p["kind"] for p in pairs)
    print(f"wrote {len(pairs)} pairs ({dict(kind_counts)}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
