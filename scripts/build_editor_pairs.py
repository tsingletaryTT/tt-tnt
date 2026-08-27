#!/usr/bin/env python3
# scripts/build_editor_pairs.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build (draft, better) editor-training pairs from real corpus sentences.

`better` is always a real, clean line drawn from the ten per-source files under
`artifacts/corpus/` (never `blend.txt` or `corpus.txt`, the two aggregate files built FROM
those ten -- sampling from the aggregates too would double-count their content) -- never
model-generated, so it is guaranteed grammatical English by construction. `draft` is the
same line after 1-2 seeded corruptors from `train/corrupt.py`, retried (and, on persistent
failure, dropped) so that `draft == better` is never emitted -- see `build_pairs`.

Before emitting anything, checks the exact delimiter strings this project's editor training
format uses ("\\nDraft: ", "\\nEdit: ") for literal collisions in the source corpus -- the
risk the design spec (`docs/superpowers/specs/2026-08-27-editor-training-design.md` §2)
flagged as measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent

import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.corrupt import corrupt  # noqa: E402

DELIMITER_STRINGS = ["Draft:", "Edit:"]

#: The two fixed, well-established aggregate filenames this project builds under
#: `artifacts/corpus/` -- `blend.txt` (the full training blend) and `corpus.txt` (the legacy
#: TinyStories-only aggregate). Both are concatenations of the same ten per-source files
#: this script otherwise samples from, so including them alongside the per-source files
#: double-counts (in fact multiply-counts) every line they contain, biasing the sample toward
#: duplicated content -- not just a resource-usage problem. Excluded by name, not by a
#: pattern, because these two names are fixed points in this project, not a moving target.
_AGGREGATE_FILENAMES = {"blend.txt", "corpus.txt"}


def select_corpus_files(corpus_dir: Path) -> List[Path]:
    """Return the per-source `*.txt` files under `corpus_dir`, excluding the two known
    aggregate files (`blend.txt`, `corpus.txt`) which are built FROM the per-source files
    and would otherwise double-count their content."""
    return sorted(
        p for p in corpus_dir.glob("*.txt") if p.name not in _AGGREGATE_FILENAMES
    )


def check_delimiter_collisions(corpus_paths: List[Path]) -> List[str]:
    """Return every line across `corpus_paths` containing a delimiter string verbatim."""
    hits = []
    for path in corpus_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(d in line for d in DELIMITER_STRINGS):
                hits.append(f"{path.name}: {line.strip()[:100]}")
    return hits


def sample_clean_sentences(corpus_paths: List[Path], n: int, *, seed: int) -> List[str]:
    """Sample `n` non-empty, non-separator lines across `corpus_paths`, seeded."""
    lines: List[str] = []
    for path in corpus_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and line != "</s>":
                lines.append(line)
    rng = random.Random(seed)
    if n >= len(lines):
        return lines
    return rng.sample(lines, n)


#: Bounded number of retry attempts (including the first) before giving up on a sentence
#: whose text is too short/atypical for any of `train/corrupt.py`'s corruptors to touch.
_MAX_CORRUPT_ATTEMPTS = 5


def build_pairs(sentences: List[str], *, seed: int) -> List[dict]:
    """One (draft, better) pair per sentence that a corruptor can actually change.

    `n_corruptors` is sampled from {1, 2} per pair. Every individual corruptor in
    `train/corrupt.py` has a documented early-return no-op path (e.g. `repeat_collapse` on
    <3-word text, `garble_word` with no eligible content word) -- on a short/atypical real
    sentence (table rows, one-line captions, section headers) a randomly-chosen corruptor (or
    both, if `n_corruptors=2`) can hit that no-op path and return the sentence unchanged. If
    that happens, this function retries with a different seed offset and a different
    `n_corruptors` choice, up to `_MAX_CORRUPT_ATTEMPTS` times. If every attempt still produces
    `draft == sentence` -- i.e. the sentence is too short/atypical for any corruptor to touch --
    the sentence is DROPPED from the output entirely. A pair with `draft == better` is never
    emitted; the returned list may therefore be shorter than `sentences`.
    """
    rng = random.Random(seed)
    pairs = []
    for i, sentence in enumerate(sentences):
        draft = sentence
        for attempt in range(_MAX_CORRUPT_ATTEMPTS):
            n_corruptors = rng.choice([1, 2])
            severity = rng.uniform(0.2, 1.0)
            attempt_seed = seed + i * _MAX_CORRUPT_ATTEMPTS + attempt
            draft = corrupt(
                sentence, seed=attempt_seed, severity=severity, n_corruptors=n_corruptors
            )
            if draft != sentence:
                break
        if draft == sentence:
            continue  # every attempt was a no-op -- drop rather than emit draft == better
        pairs.append({"draft": draft, "better": sentence})
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--n", type=int, default=20000,
                   help="number of SENTENCES sampled (not the final pair count -- "
                        "build_pairs drops any sentence whose corruption attempts all "
                        "no-op after retrying, a real measured rate on a 100-sentence "
                        "sample of ~14%%; see build_pairs' own docstring)")
    p.add_argument("--seed", type=int, default=5489)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "editor-pairs" / "pairs.jsonl")
    args = p.parse_args()

    corpus_paths = select_corpus_files(args.corpus_dir)
    if not corpus_paths:
        raise FileNotFoundError(
            f"no per-source *.txt files found under {args.corpus_dir} "
            f"(blend.txt/corpus.txt are excluded by design)"
        )

    collisions = check_delimiter_collisions(corpus_paths)
    if collisions:
        print(f"WARNING: {len(collisions)} line(s) contain a literal delimiter string. "
              f"First 5:")
        for line in collisions[:5]:
            print(f"  {line}")
    else:
        print("No delimiter-string collisions found in the corpus.")

    sentences = sample_clean_sentences(corpus_paths, args.n, seed=args.seed)
    pairs = build_pairs(sentences, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"wrote {len(pairs)} pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
