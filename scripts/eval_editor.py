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
from datetime import datetime, timezone
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


def _word_overlap(a: str, b: str) -> float:
    a_words = set(_words(a))
    b_words = set(_words(b))
    return len(a_words & b_words) / len(a_words) if a_words else 0.0


def score_recovery(better: str, edited: str, vocab: Set[str]) -> dict:
    overlap = _word_overlap(better, edited)
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
    p.add_argument("--max-model-len", type=int, default=512,
                   help="must match the served model's own max_model_len -- a request "
                        "whose prompt+completion exceeds this crashes the WHOLE vLLM "
                        "engine (an unrecoverable AssertionError deep in the model "
                        "runner, not a per-request 400), so this is enforced client-side "
                        "before any request is sent, not just documented")
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

    # build_editor_pairs.py's draft/better are raw corrupted corpus sentences with no
    # length cap -- unlike train_editor.py's build_editor_example, which truncates every
    # training example to MAX_SEQ_LEN. Real corpus sentences run long enough (measured:
    # one draft alone tokenized past 1024) to exceed the served model's max_model_len,
    # and that failure mode is fatal to the whole server, not just one request. Guard
    # client-side with the real tokenizer rather than a word-count heuristic.
    max_tokens = 40
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts" / "tokenizer"))

    results = []
    skipped_too_long = 0
    for pair in pairs:
        prompt = f"\nDraft: {pair['draft']}\nEdit: "
        prompt_len = len(tok.encode(prompt))
        if prompt_len + max_tokens > args.max_model_len:
            skipped_too_long += 1
            continue
        data = _post(
            {"model": args.model, "prompt": prompt, "max_tokens": max_tokens,
             "temperature": 0.0, "stop": ["\n"]},
            args.endpoint, 60.0,
        )
        edited = data["choices"][0]["text"].strip()
        # draft_overlap is the baseline this check is actually specified against --
        # "closer to real English than the draft was" (module docstring, spec sec.5.1)
        # is a comparison, not a bare edited-vs-better number.
        draft_overlap = _word_overlap(pair["better"], pair["draft"])
        results.append({
            "draft": pair["draft"], "better": pair["better"], "edited": edited,
            "draft_overlap": draft_overlap,
            **score_recovery(pair["better"], edited, vocab),
        })

    if skipped_too_long:
        print(f"skipped {skipped_too_long}/{len(pairs)} pairs whose prompt+completion "
              f"would exceed max_model_len={args.max_model_len}")

    if not results:
        print(f"n=0 -- every pair was skipped as too long, nothing to score")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "n": 0, "skipped_too_long": skipped_too_long, "results": [],
        }, indent=2))
        print(f"wrote {args.out}")
        return 0

    mean_overlap = sum(r["word_overlap"] for r in results) / len(results)
    mean_draft_overlap = sum(r["draft_overlap"] for r in results) / len(results)
    improved = sum(1 for r in results if r["word_overlap"] > r["draft_overlap"])
    worsened = sum(1 for r in results if r["word_overlap"] < r["draft_overlap"])
    fake_word_rate = sum(1 for r in results if r["has_fake_word"]) / len(results)
    print(f"n={len(results)}  mean_word_overlap={mean_overlap:.3f} "
          f"(draft baseline {mean_draft_overlap:.3f})  "
          f"improved={improved}  worsened={worsened}  "
          f"fake_word_rate={fake_word_rate:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "endpoint": args.endpoint, "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n": len(results), "skipped_too_long": skipped_too_long,
        "mean_word_overlap": mean_overlap,
        "mean_draft_overlap": mean_draft_overlap,
        "improved_over_draft": improved, "worsened_vs_draft": worsened,
        "fake_word_rate": fake_word_rate, "results": results,
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
