#!/usr/bin/env python3
# scripts/build_editor_pairs.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Build (draft, better) editor-training pairs from real corpus sentences.

`better` is always a real, clean line drawn from `artifacts/corpus/*.txt` -- never
model-generated, so it is guaranteed grammatical English by construction. `draft` is the
same line after 1-2 seeded corruptors from `train/corrupt.py`.

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


def build_pairs(sentences: List[str], *, seed: int) -> List[dict]:
    """One (draft, better) pair per sentence. `n_corruptors` sampled from {1, 2} per pair."""
    rng = random.Random(seed)
    pairs = []
    for i, sentence in enumerate(sentences):
        n_corruptors = rng.choice([1, 2])
        severity = rng.uniform(0.2, 1.0)
        draft = corrupt(sentence, seed=seed + i, severity=severity, n_corruptors=n_corruptors)
        pairs.append({"draft": draft, "better": sentence})
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-dir", type=Path, default=ROOT / "artifacts" / "corpus")
    p.add_argument("--n", type=int, default=20000,
                   help="number of pairs to build (default matches the skits-200k order "
                        "of magnitude before its own drop rate, not after it -- there is no "
                        "drop rate here, so this is the real final count)")
    p.add_argument("--seed", type=int, default=5489)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "editor-pairs" / "pairs.jsonl")
    args = p.parse_args()

    corpus_paths = sorted(args.corpus_dir.glob("*.txt"))
    if not corpus_paths:
        raise FileNotFoundError(f"no *.txt files found under {args.corpus_dir}")

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
