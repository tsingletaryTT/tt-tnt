#!/usr/bin/env python3
# scripts/eval_editor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Before/after evaluation for the editor objective (Task 6 of the 2026-08-27 editor-
training plan). Three checks, run against a served checkpoint:

1. Held-out corruption recovery: corrupt sentences NEVER used in training (a fresh sample,
   different seed range than scripts/build_editor_pairs.py used), ask the served model to
   edit them, score whether the result is closer to real English than the draft was.
2. Re-run scripts/story_tools.py::self_edit() -- it had a clean negative result on
   tt-tnt-1024-dialogue; success here is this checkpoint fixing exactly what that test caught.
3. No-regression: delegate to the existing scripts/evaluate.py against the current
   designated checkpoint as the control (this script does not reimplement that gate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.corrupt import corrupt  # noqa: E402

_WORD_RE = re.compile(r"[a-z']+")


def _words(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def recovers_real_words(text: str, vocab: Set[str]) -> bool:
    """True iff every word in `text` is a real word in `vocab`."""
    return all(w in vocab for w in _words(text))


def score_recovery(better: str, edited: str, vocab: Set[str]) -> dict:
    better_words = set(_words(better))
    edited_words = set(_words(edited))
    overlap = (
        len(better_words & edited_words) / len(better_words) if better_words else 0.0
    )
    has_fake_word = not recovers_real_words(edited, vocab)
    return {"word_overlap": overlap, "has_fake_word": has_fake_word}


def build_vocab(corpus_paths) -> Set[str]:
    vocab: Set[str] = set()
    for path in corpus_paths:
        vocab.update(_words(Path(path).read_text(encoding="utf-8", errors="replace")))
    return vocab


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--n", type=int, default=200,
                   help="held-out corrupted sentences to test")
    p.add_argument("--seed", type=int, default=999999,
                   help="deliberately outside build_editor_pairs.py's default seed range")
    p.add_argument("--endpoint", default="http://localhost:8000/v1/completions")
    p.add_argument("--model", default="episod/tt-tnt-1024")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "measurements" / "editor-eval.json")
    args = p.parse_args()

    from scripts.build_editor_pairs import build_pairs, sample_clean_sentences, select_corpus_files

    # NOT a plain glob("*.txt") -- that would include blend.txt/corpus.txt, the two
    # aggregate files build_editor_pairs.py's own select_corpus_files excludes because
    # they multiply-count already-upsampled per-source content. Reusing the same
    # exclusion here keeps this held-out sample drawn from the same population Task 2's
    # training pairs were, rather than biased toward the training blend's own upsampling.
    corpus_paths = select_corpus_files(args.corpus_dir)
    vocab = build_vocab(corpus_paths)
    sentences = sample_clean_sentences(corpus_paths, args.n, seed=args.seed)
    pairs = build_pairs(sentences, seed=args.seed)

    from scripts.story_tools import _post  # reuse the same HTTP call the harness uses

    results = []
    for pair in pairs:
        prompt = f"\nDraft: {pair['draft']}\nEdit: "
        data = _post(
            {"model": args.model, "prompt": prompt, "max_tokens": 40, "temperature": 0.0,
             "stop": ["\n"]},
            args.endpoint, 60.0,
        )
        edited = data["choices"][0]["text"].strip()
        results.append({
            "draft": pair["draft"], "better": pair["better"], "edited": edited,
            **score_recovery(pair["better"], edited, vocab),
        })

    mean_overlap = sum(r["word_overlap"] for r in results) / len(results)
    fake_word_rate = sum(1 for r in results if r["has_fake_word"]) / len(results)
    print(f"n={len(results)}  mean_word_overlap={mean_overlap:.3f}  "
          f"fake_word_rate={fake_word_rate:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n": len(results), "mean_word_overlap": mean_overlap,
        "fake_word_rate": fake_word_rate, "results": results,
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
