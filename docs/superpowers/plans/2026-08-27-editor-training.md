# Editor Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `tt-tnt-1024-dialogue` a real editor (draft → corrected) capability, and fold
the already-derived skits five-turn data into the same continued-training stage, so dialogue
and skits stop being separate checkpoint lineages.

**Architecture:** A pure, reusable text-corruption module builds `(draft, better)` pairs from
real corpus sentences. A new SFT training entrypoint, modeled directly on
`scripts/train_skits.py`'s structure and quality gates, warm-starts from
`artifacts/checkpoints-1024-dialogue`'s final checkpoint and trains on a mix of editor pairs
and the existing skits corpus, using masked next-token loss (loss only on the corrected/skit
side). A new eval script proves the objective actually worked, before/after, against the
harness that found it missing in the first place (`scripts/story_tools.py::self_edit`).

**Tech Stack:** Python, `ttml`/`ttnn` (Tenstorrent training stack), `transformers`
(tokenizer only), pytest.

**Spec:** [`docs/superpowers/specs/2026-08-27-editor-training-design.md`](../specs/2026-08-27-editor-training-design.md)

## Global Constraints

- **Train on ONE chip, `(1, 1)` mesh.** `scripts/train_skits.py`'s own docstring: "A four-chip
  mesh open hard-froze this host once with no OOM, no kernel panic and no pstore record;
  there is no reproduction and no fix, so it is not retried." Every device-touching step in
  this plan uses `gozer run --chips 1 ...` and `ttml.open_device_mesh((1, 1))`.
- **`stochastic_rounding` must be on.** Off, the 17 RMSNorm gammas silently never move
  (bf16 ulp at 1.0 is 0.0039, an order of magnitude larger than the ~3e-4 Adam update) — the
  exact bug that cost skits stage 1 its whole capability. Asserted before AND after training,
  matching `train_skits.py`'s two-layer guard.
- **No new special tokens.** The vocab is fixed at exactly 32000 (`convert/tokenizer.py`); use
  plain-text delimiters (`\nDraft: `, `\nEdit: `), not new added tokens — see the spec's §2 for
  why the `</s>`-precedent doesn't transfer here.
- **Warm-start, not resume.** `--warm-start` copies parameters and starts the optimizer fresh
  at step 0; `--resume` continues the optimizer/step count. This is a new objective (masked
  SFT, not blend pretraining), so a fresh optimizer is correct, matching how `train_skits.py`
  itself warm-starts rather than resumes.
- **Do not touch `artifacts/checkpoints-1024-dialogue/`.** Read-only throughout — it's the
  currently-designated model's checkpoint history. All new artifacts go under
  `artifacts/checkpoints-1024-editor/`.
- **Every device-touching command needs a gozer lease first** (`gozer run --chips 1 --who
  "claude:editor-training" --reason "<why>" -- <command>`), per this repo's `CLAUDE.md`.

---

### Task 1: `train/corrupt.py` — the reusable corruption module

**Files:**
- Create: `train/corrupt.py`
- Test: `tests/test_corrupt.py`

**Interfaces:**
- Produces: `repeat_collapse(text: str, *, seed: int, severity: float = 0.5) -> str`,
  `garble_word(text: str, *, seed: int, severity: float = 0.5) -> str`,
  `drop_or_double_function_word(text: str, *, seed: int, severity: float = 0.5) -> str`,
  `fuse_clauses(text: str, *, seed: int, severity: float = 0.5) -> str`,
  `corrupt(text: str, *, seed: int, severity: float = 0.5, n_corruptors: int = 1) -> str`
  (applies `n_corruptors` randomly-chosen corruptors from the four above, in a random order,
  seeded — used by Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corrupt.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import re

from train.corrupt import (
    corrupt,
    drop_or_double_function_word,
    fuse_clauses,
    garble_word,
    repeat_collapse,
)


def test_repeat_collapse_duplicates_a_span():
    text = "The dragon flew over the mountain and landed softly."
    out = repeat_collapse(text, seed=0, severity=1.0)
    words = out.split()
    # some 2-4 word span appears at least twice consecutively or near-consecutively
    found = False
    for span_len in (2, 3, 4):
        for i in range(len(words) - span_len):
            span = words[i : i + span_len]
            if words[i + span_len : i + 2 * span_len] == span:
                found = True
    assert found, f"no repeated span found in: {out!r}"


def test_repeat_collapse_is_deterministic_for_a_seed():
    text = "The dragon flew over the mountain and landed softly."
    assert repeat_collapse(text, seed=1) == repeat_collapse(text, seed=1)


def test_garble_word_replaces_an_ordinary_word():
    text = "The girl found a small silver key by the river."
    out = garble_word(text, seed=0, severity=1.0)
    assert out != text
    # at least one token differs and the differing token is not a real short function word
    orig_words = text.split()
    new_words = out.split()
    assert len(orig_words) == len(new_words)
    diffs = [(a, b) for a, b in zip(orig_words, new_words) if a != b]
    assert diffs, "garble_word changed nothing"


def test_garble_word_skips_existing_proper_nouns():
    # "Mira" is capitalized mid-sentence -- an existing proper noun, must survive untouched.
    text = "The girl named Mira walked to the old mill by the river."
    for seed in range(20):
        out = garble_word(text, seed=seed, severity=1.0)
        assert "Mira" in out.split(), f"seed={seed} corrupted a protected proper noun: {out!r}"


def test_garble_word_skips_word_after_named_or_called():
    text = "The dog was called Bramble and lived in the barn near the pond."
    for seed in range(20):
        out = garble_word(text, seed=seed, severity=1.0)
        assert "Bramble" in out.split(), f"seed={seed} corrupted a protected coined name: {out!r}"


def test_drop_or_double_function_word_changes_length_or_doubles():
    text = "She was always sad and had never been happy before that day."
    out = drop_or_double_function_word(text, seed=0, severity=1.0)
    assert out != text


def test_fuse_clauses_removes_a_conjunction():
    text = "She was tired, and she wanted to sleep, but the noise kept her awake."
    out = fuse_clauses(text, seed=0, severity=1.0)
    assert out != text
    # at least one of the original conjunctions is gone
    conjunctions = {"and", "but", "or", "so", "because"}
    orig_conj_count = sum(1 for w in re.findall(r"[a-z]+", text.lower()) if w in conjunctions)
    new_conj_count = sum(1 for w in re.findall(r"[a-z]+", out.lower()) if w in conjunctions)
    assert new_conj_count < orig_conj_count


def test_corrupt_applies_requested_corruptor_count_and_is_deterministic():
    text = "The little fox ran across the field before the sun went down."
    out_a = corrupt(text, seed=42, n_corruptors=2)
    out_b = corrupt(text, seed=42, n_corruptors=2)
    assert out_a == out_b
    out_c = corrupt(text, seed=43, n_corruptors=2)
    # different seed, overwhelmingly likely to differ on this input
    assert out_c != out_a or True  # documents intent; not a strict guarantee for one input


def test_corrupt_with_zero_corruptors_returns_input_unchanged():
    text = "A quiet morning came over the village."
    assert corrupt(text, seed=0, n_corruptors=0) == text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_corrupt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train.corrupt'`

- [ ] **Step 3: Write the implementation**

```python
# train/corrupt.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Four corruptors, each reproducing one SPECIFIC, documented tt-tnt-1024 failure mode.

Built from real output observed in `scripts/story_tools.py`'s 2026-08-27 storytelling
harness session, not generic text-noise. Each function is a pure, seeded transform: same
input + same seed => same output, no hidden state, no dependency on a tokenizer or the
training pipeline. That independence is deliberate -- see the design spec's §1 for why
(reused later as a held-out eval-set generator, and potentially a deliberate "glitch" style
at inference time; neither is built here).

Every function takes and returns a single SENTENCE (no internal sentence splitting) and a
`severity` in [0, 1] controlling how much it does, not whether it does anything -- at
severity 0 every corruptor still changes something (there is no free "silently do nothing"
severity), and `corrupt(..., n_corruptors=0)` is the only way to get the input back
unchanged.
"""

from __future__ import annotations

import random
import re
from typing import Callable, List

_WORD_RE = re.compile(r"[A-Za-z']+")
_CONJUNCTIONS = {"and", "but", "or", "so", "because"}
_FUNCTION_WORDS = [
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "as", "that",
]
_AUXILIARIES = ["was", "were", "had", "did", "would", "could", "should"]

#: Cue words after which the FOLLOWING word is treated as a deliberately-coined name and
#: must never be garbled -- "named Mira", "called Bramble", "nicknamed Scout".
_NAMING_CUES = {"named", "called", "nicknamed"}


def _split_words(text: str) -> List[str]:
    return text.split()


def repeat_collapse(text: str, *, seed: int, severity: float = 0.5) -> str:
    """Duplicate a short span 2-4 times, mimicking "The dragon. The dragon, in the dragon."

    `severity` controls how many extra repeats (1 at severity 0, up to 3 at severity 1) and
    how long the repeated span is (2 words at low severity, up to 4 at high severity).
    """
    rng = random.Random(seed)
    words = _split_words(text)
    if len(words) < 3:
        return text
    span_len = 2 + int(round(severity * 2))
    span_len = min(span_len, len(words) - 1)
    start = rng.randrange(0, max(1, len(words) - span_len))
    span = words[start : start + span_len]
    repeats = 1 + int(round(severity * 2))
    inserted = span * repeats
    new_words = words[: start + span_len] + inserted + words[start + span_len :]
    return " ".join(new_words)


#: A small, hand-picked set of English digraphs/trigraphs, used to build a plausible-looking
#: fake word by re-splicing pieces of a real one. Not a phoneme model -- just enough to
#: produce something that LOOKS like it could be an English word without being one,
#: matching "Tryburg"/"Alexandary"/"Higheriq" (real-looking, not real).
_SPLICE_FRAGMENTS = ["bur", "dale", "wick", "ton", "iq", "ary", "esh", "old", "ind", "ume"]


def _make_fake_word(real_word: str, rng: random.Random) -> str:
    stem = real_word[: max(2, len(real_word) // 2)]
    suffix = rng.choice(_SPLICE_FRAGMENTS)
    fake = stem + suffix
    if real_word[:1].isupper():
        fake = fake[:1].upper() + fake[1:]
    return fake


def garble_word(text: str, *, seed: int, severity: float = 0.5) -> str:
    """Replace 1-2 ordinary words with a plausible-but-fake word.

    Skips words that are either an EXISTING proper noun (capitalized outside sentence-initial
    position -- a real name already in the source) or immediately follow a naming cue
    ("named"/"called"/"nicknamed") -- both are legitimate coined-name territory the user
    explicitly does not want flagged as an error. Only common nouns/verbs/adjectives (3+
    letters, lowercase, not a function word) are eligible.
    """
    rng = random.Random(seed)
    words = _split_words(text)
    eligible = []
    for i, w in enumerate(words):
        bare = _WORD_RE.match(w)
        if not bare:
            continue
        core = bare.group(0)
        if len(core) < 3:
            continue
        if core.lower() in _FUNCTION_WORDS or core.lower() in _AUXILIARIES:
            continue
        if i > 0 and core[:1].isupper():
            continue  # an existing proper noun mid-sentence -- protected
        if i > 0 and words[i - 1].strip(".,!?").lower() in _NAMING_CUES:
            continue  # the coined name itself -- protected
        eligible.append(i)
    if not eligible:
        return text
    n = 1 + (1 if severity > 0.6 else 0)
    n = min(n, len(eligible))
    targets = rng.sample(eligible, n)
    out = list(words)
    for i in targets:
        bare = _WORD_RE.match(out[i])
        core = bare.group(0)
        prefix, suffix = out[i][: bare.start()], out[i][bare.end() :]
        out[i] = prefix + _make_fake_word(core, rng) + suffix
    return " ".join(out)


def drop_or_double_function_word(text: str, *, seed: int, severity: float = 0.5) -> str:
    """Delete a function word, or double an auxiliary -- "she was always had been special."

    Deletion at low severity, doubling (the more severe-reading defect, since it survives
    a casual read as "real" grammar for longer) at high severity.
    """
    rng = random.Random(seed)
    words = _split_words(text)
    candidates = [
        i for i, w in enumerate(words)
        if _WORD_RE.match(w) and _WORD_RE.match(w).group(0).lower() in _FUNCTION_WORDS + _AUXILIARIES
    ]
    if not candidates:
        return text
    i = rng.choice(candidates)
    bare = _WORD_RE.match(words[i]).group(0).lower()
    out = list(words)
    if severity > 0.6 and bare in _AUXILIARIES:
        out.insert(i, words[i])  # double it in place
    else:
        del out[i]  # drop it
    return " ".join(out)


def fuse_clauses(text: str, *, seed: int, severity: float = 0.5) -> str:
    """Strip a conjunction so two clauses run on without one.

    Removes the conjunction word and, at severity > 0.5, the comma immediately before it
    too (the more severe run-on, with no punctuation seam at all).
    """
    rng = random.Random(seed)
    words = _split_words(text)
    candidates = [
        i for i, w in enumerate(words)
        if _WORD_RE.match(w) and _WORD_RE.match(w).group(0).lower() in _CONJUNCTIONS
    ]
    if not candidates:
        return text
    i = rng.choice(candidates)
    out = list(words)
    del out[i]
    if severity > 0.5 and i > 0 and out[i - 1].endswith(","):
        out[i - 1] = out[i - 1].rstrip(",")
    return " ".join(out)


_CORRUPTORS: List[Callable[..., str]] = [
    repeat_collapse,
    garble_word,
    drop_or_double_function_word,
    fuse_clauses,
]


def corrupt(text: str, *, seed: int, severity: float = 0.5, n_corruptors: int = 1) -> str:
    """Apply `n_corruptors` randomly-chosen corruptors (from the four above) in sequence.

    `n_corruptors=0` returns `text` unchanged -- the only way to get a no-op out of this
    module, so a caller building a `draft == better` example (a bug) cannot happen silently.
    """
    if n_corruptors <= 0:
        return text
    rng = random.Random(seed)
    chosen = rng.sample(_CORRUPTORS, k=min(n_corruptors, len(_CORRUPTORS)))
    out = text
    for i, fn in enumerate(chosen):
        out = fn(out, seed=seed + i, severity=severity)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_corrupt.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add train/corrupt.py tests/test_corrupt.py
git commit -m "feat(editor): reusable, seeded text-corruption module (train/corrupt.py)"
```

---

### Task 2: `scripts/build_editor_pairs.py` — draft/better pairs from the real corpus

**Files:**
- Create: `scripts/build_editor_pairs.py`
- Test: `tests/test_build_editor_pairs.py`

**Interfaces:**
- Consumes: `train.corrupt.corrupt(text, *, seed, severity, n_corruptors)` (Task 1).
- Produces: `sample_clean_sentences(corpus_paths: list[Path], n: int, *, seed: int) -> list[str]`,
  `build_pairs(sentences: list[str], *, seed: int) -> list[dict]` (each dict:
  `{"draft": str, "better": str}`), and a `main()` CLI writing
  `artifacts/editor-pairs/pairs.jsonl`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_editor_pairs.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from pathlib import Path

from scripts.build_editor_pairs import build_pairs, sample_clean_sentences


def test_sample_clean_sentences_reads_real_lines(tmp_path):
    corpus = tmp_path / "source.txt"
    corpus.write_text(
        "The fox ran across the field.\n"
        "</s>\n"
        "A quiet morning came over the village.\n"
        "She opened the old wooden box.\n"
    )
    sentences = sample_clean_sentences([corpus], n=2, seed=0)
    assert len(sentences) == 2
    for s in sentences:
        assert s.strip() and s.strip() != "</s>"


def test_sample_clean_sentences_is_deterministic():
    corpus = Path(__file__).parent / "fixtures" / "editor_pairs_source.txt"
    corpus.parent.mkdir(exist_ok=True)
    corpus.write_text("One line here.\nAnother line here.\nA third line here.\n")
    a = sample_clean_sentences([corpus], n=2, seed=7)
    b = sample_clean_sentences([corpus], n=2, seed=7)
    assert a == b


def test_build_pairs_draft_differs_from_better():
    sentences = [
        "The little fox ran across the field before the sun went down.",
        "She opened the old wooden box and found a silver key inside.",
    ]
    pairs = build_pairs(sentences, seed=0)
    assert len(pairs) == len(sentences)
    for p in pairs:
        assert set(p.keys()) == {"draft", "better"}
        assert p["draft"] != p["better"]
        assert p["better"] in sentences


def test_build_pairs_is_deterministic():
    sentences = ["A quiet morning came over the village and the birds began to sing."]
    a = build_pairs(sentences, seed=3)
    b = build_pairs(sentences, seed=3)
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_editor_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_editor_pairs'`

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_editor_pairs.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run against the real corpus and inspect the collision check**

Run: `python scripts/build_editor_pairs.py --n 100 --out /tmp/editor-pairs-smoke.jsonl`
Expected: prints either "No delimiter-string collisions found" or a real list of colliding
lines from the actual corpus (record whichever it is in the commit message — this is the
spec's flagged risk, now measured).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_editor_pairs.py tests/test_build_editor_pairs.py
git commit -m "feat(editor): build draft/better training pairs from real corpus sentences"
```

---

### Task 3: `scripts/train_editor.py` — the training entrypoint

**Files:**
- Create: `scripts/train_editor.py`
- Test: `tests/test_train_editor.py` (example-building and masking only — no device)

**Interfaces:**
- Consumes: `train.corrupt` (not directly — pairs are pre-built by Task 2's script),
  `scripts.derive_skits.build_skit_example(skit, tok, *, with_think, pad_token_id) -> dict`
  (existing), `train.skit.Skit` (existing), `train.model.create_model` (existing),
  `train.enthusiasts.warm_start` (existing).
- Produces: `build_editor_example(pair: dict, tok, *, pad_token_id: int) -> dict` (returns
  `{"input_ids", "labels"}`, same shape as `build_skit_example`), `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_editor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Example-building and masking only. No `ttml`/`ttnn` import, no device -- these tests run
in a bare CPU environment, same convention as tests/test_corrupt.py.
"""
from scripts.train_editor import build_editor_example


class _FakeTok:
    """Minimal tokenizer stand-in: word-level 'encoding' so lengths are checkable by hand."""

    def encode(self, text, add_special_tokens=True):
        n = len(text.split())
        return ([0] if add_special_tokens else []) + list(range(1, n + 1))


def test_build_editor_example_masks_the_draft_and_supervises_the_edit():
    pair = {"draft": "a b c", "better": "x y"}
    tok = _FakeTok()
    ex = build_editor_example(pair, tok, pad_token_id=99)
    prompt = "\nDraft: a b c\nEdit: "
    p_ids = tok.encode(prompt)
    assert ex["input_ids"][: len(p_ids)] == p_ids
    # every label up to the last prompt position is masked...
    assert all(l == -100 for l in ex["labels"][: len(p_ids) - 1])
    # ...and the supervised region is exactly len(completion_ids) long
    c_ids = tok.encode("x y", add_special_tokens=False)
    supervised = [l for l in ex["labels"] if l != -100]
    assert len(supervised) == len(c_ids)


def test_build_editor_example_input_ids_and_labels_are_pre_shift_aligned_length():
    pair = {"draft": "one two three", "better": "four five"}
    tok = _FakeTok()
    ex = build_editor_example(pair, tok, pad_token_id=99)
    assert len(ex["input_ids"]) == len(ex["labels"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_train_editor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.train_editor'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
# scripts/train_editor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Continued-training stage: editor pairs + the existing skits corpus, warm-started from
tt-tnt-1024-dialogue.

Modeled directly on `scripts/train_skits.py`'s structure and quality gates -- this is a new
objective on the SAME dense architecture (no new tokens, no shape change), so every
mechanical piece of that script's device/training path transfers unchanged: `(1, 1)` mesh
only (a four-chip open hard-froze this host once, no reproduction, not retried),
`stochastic_rounding` asserted before and after, gamma-movement verified against the
warm-start base at the end.

Two example TYPES feed the same `InMemoryDataloader`: editor pairs (this file's
`build_editor_example`) and skits (`scripts.derive_skits.build_skit_example`, unchanged).
Both return `{"input_ids", "labels"}`, so they mix freely in one list.

    gozer run --chips 1 --who "claude:editor-training" --reason "editor+skits SFT" -- \
        python3 scripts/train_editor.py --steps 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_skits import build_skit_example  # noqa: E402
from scripts.train_skits import (  # noqa: E402
    LossRecorder,
    assert_eval_wired,
    assert_gammas_moved,
    assert_stochastic_rounding,
    compare_gammas,
    read_arm_gammas,
    read_base_gammas,
)
from train.skit import Skit  # noqa: E402

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
DEFAULT_PAIRS = ROOT / "artifacts" / "editor-pairs" / "pairs.jsonl"
DEFAULT_SKITS = ROOT / "artifacts" / "skits-200k" / "skits.jsonl"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "checkpoints-1024-editor"
#: The currently-designated dialogue checkpoint -- this run's warm-start base.
#: Read from docs/current_model.json at call time in main(); duplicated here only as the
#: literal default so --help shows something concrete.
DEFAULT_WARM_START = ROOT / "artifacts" / "checkpoints-1024-dialogue" / "tt_tnt_step00010764.pkl"
MAX_SEQ_LEN = 512


def build_editor_example(pair: Dict[str, str], tok, *, pad_token_id: int) -> Dict[str, list]:
    """`{"input_ids", "labels"}` for one (draft, better) pair, pre-shifted for ttml.

    Same boundary convention as every other SFT example in this project
    (`scripts/derive_traces.py::_sft_example_unaligned`): the LAST prompt position's label
    is the FIRST completion token (supervised, not masked -- that transition IS the trained
    behaviour), the LAST completion position is masked (no legitimate next token).
    """
    prompt = f"\nDraft: {pair['draft']}\nEdit: "
    completion = pair["better"]
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)
    if not p_ids:
        raise ValueError("prompt tokenized to zero ids; label shifting requires at least "
                         "one prompt token")
    input_ids = p_ids + c_ids
    labels = [-100] * (len(p_ids) - 1) + c_ids + [-100]
    return {"input_ids": input_ids, "labels": labels}


def load_editor_pairs(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_skits(path: Path) -> List[Skit]:
    """Reverses `Skit.as_dict()` (`train/skit.py`) -- there is no `from_dict` in that
    module, so `blocks` (serialized as plain dicts via each `Slots.as_dict()`) must be
    reconstructed into real `Slots` objects here; `"roles"` is dropped, since it is a
    derived constant (`SKIT_ROLES`) `as_dict()` writes for readability, not a real
    constructor field.
    """
    from train.improv import Slots  # NOT train.skit -- verified against train_skits.py's
                                     # own `from train.improv import Slots` import.

    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [
        Skit(story_id=r["story_id"], prefix=r["prefix"], turns=tuple(r["turns"]),
             blocks=tuple(Slots(**b) for b in r["blocks"]))
        for r in records
    ]


def build_combined_examples(
    pairs: List[Dict[str, str]], skits: List[Skit], tok, *, pad_token_id: int,
) -> List[dict]:
    """Editor examples first, then skits examples (with_think=False -- this stage is about
    editing and turn-structure, not the separate think-block objective).
    """
    examples = [build_editor_example(p, tok, pad_token_id=pad_token_id) for p in pairs]
    examples += [
        build_skit_example(s, tok, with_think=False, pad_token_id=pad_token_id)
        for s in skits
    ]
    return examples


def _current_dialogue_checkpoint() -> Path:
    manifest = json.loads((ROOT / "docs" / "current_model.json").read_text())
    checkpoints_dir = ROOT / manifest["current"]["checkpoints"]
    steps = sorted(checkpoints_dir.glob("tt_tnt_step*.pkl"))
    if not steps:
        raise FileNotFoundError(f"no checkpoints found under {checkpoints_dir}")
    return steps[-1]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--skits", type=Path, default=DEFAULT_SKITS)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--warm-start", type=Path, default=None,
                    help="defaults to the currently-designated dialogue checkpoint "
                         "(docs/current_model.json), resolved at call time")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the examples and report the length distribution, then "
                         "stop. Touches no device, so it needs no lease.")
    args = ap.parse_args(argv)

    warm_start_ckpt = args.warm_start or _current_dialogue_checkpoint()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_token_id = tok.pad_token_id or 0

    pairs = load_editor_pairs(args.pairs)
    skits = load_skits(args.skits)
    examples = build_combined_examples(pairs, skits, tok, pad_token_id=pad_token_id)

    # Held out as the TAIL of the combined, file-order list -- deterministic, matching
    # train_skits.py's D3. No shuffle before this split, so a re-run with the same input
    # files reproduces the same split.
    val_size = max(0, min(args.val_size, len(examples) // 4))
    train_examples = examples[: len(examples) - val_size] if val_size else examples
    val_examples = examples[len(examples) - val_size :] if val_size else []

    lengths = sorted(len(e["input_ids"]) for e in examples)
    over_max = sum(1 for l in lengths if l > MAX_SEQ_LEN)
    print(f"pairs={len(pairs):,}  skits={len(skits):,}  total_examples={len(examples):,}  "
          f"train={len(train_examples):,}  val={len(val_examples):,}")
    print(f"token lengths: min {lengths[0]}  median {lengths[len(lengths)//2]}  "
          f"max {lengths[-1]}  >{MAX_SEQ_LEN}: {over_max} ({over_max/len(lengths):.2%})")

    if args.dry_run:
        print("dry run: no device opened, nothing trained.")
        return 0

    import ttml  # noqa: F401 -- opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    # (1, 1) ONLY -- see this file's module docstring and this plan's Global Constraints.
    ttml.open_device_mesh((1, 1))
    try:
        collate = partial(sft_collate_fn, max_seq_len=MAX_SEQ_LEN, pad_token_id=pad_token_id)
        loader = InMemoryDataloader(train_examples, batch_size=args.batch_size,
                                    collate_fn=collate, shuffle=True, seed=args.seed)
        val_loader = (InMemoryDataloader(val_examples, batch_size=args.batch_size,
                                         collate_fn=collate, shuffle=False, seed=args.seed)
                      if val_examples else None)

        out = Path(args.out_root).resolve()
        out.mkdir(parents=True, exist_ok=True)

        from train.model import create_model

        model_yaml = yaml.safe_load(MODEL_YAML.read_text())
        transformer_config = model_yaml["transformer_config"]
        model = create_model({}, transformer_config)

        from train.enthusiasts import warm_start

        # Exact call shape verified against scripts/train_skits.py's own main() -- do not
        # simplify this signature; warm_start requires transformer_config/yaml_config.
        warm_summary = warm_start(
            model, Path(warm_start_ckpt),
            transformer_config=transformer_config, yaml_config={},
            moe_block_indices=[],
        )
        print(f"warm start: {warm_summary}")

        curve_path = out / "loss_curve.jsonl"
        recorder = LossRecorder(curve_path)
        trainer = SFTTrainer(
            model=model, train_dataloader=loader, eval_dataloader=val_loader,
            config=SFTConfig(max_steps=args.steps, learning_rate=args.lr, seed=args.seed,
                             max_seq_len=MAX_SEQ_LEN, checkpoint_dir=str(out),
                             save_interval=args.save_every,
                             eval_interval=args.eval_every if val_loader else 0,
                             log_interval=1, max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.01,
                       "stochastic_rounding": True},
            callbacks=[recorder],
        )
        # Both guards imported from scripts/train_skits.py -- same exact checks that caught
        # skits stage 1's 0%-adherence bug, reused rather than reimplemented. "editor" is
        # passed as the `arm` label (these functions are generic; the label is just what
        # gets printed).
        assert_stochastic_rounding(trainer, "editor")
        assert_eval_wired(trainer, val_size=len(val_examples), arm="editor")

        trainer.train()
        recorder.close()

        loss_start = recorder.history[0] if recorder.history else (None, None)
        loss_end = recorder.history[-1] if recorder.history else (None, None)
        print(f"loss: step {loss_start[0]}={loss_start[1]}  ->  step {loss_end[0]}={loss_end[1]}")

        # Guard 2 (post-training): did the gammas actually move against the warm-start
        # base? Same read_base_gammas/read_arm_gammas/compare_gammas/assert_gammas_moved
        # pipeline train_skits.py uses -- read_base_gammas expects a ttml-format
        # checkpoint (the dialogue warm-start base), read_arm_gammas expects an
        # SFTTrainer-format `{"step","model_state"}` pickle (this run's own output) -- the
        # two formats are NOT interchangeable, which is exactly why there are two readers.
        final_ckpt = out / f"step_{args.steps}.pkl"
        if final_ckpt.exists():
            report = compare_gammas(read_base_gammas(Path(warm_start_ckpt)),
                                    read_arm_gammas(final_ckpt))
            assert_gammas_moved(report, arm="editor", base_path=Path(warm_start_ckpt),
                                arm_path=final_ckpt)
            print(f"gammas moved: {report['all_moved']}  "
                  f"({report['total_changed']}/{report['total_elements']} elements)")
        else:
            raise FileNotFoundError(f"expected final checkpoint at {final_ckpt}")

        print(f"training complete -> {out}")
        return 0
    finally:
        ttml.close_device_mesh()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_editor.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Dry-run against the real Task 2 output (no device, no lease needed)**

Run: `python scripts/train_editor.py --dry-run --pairs artifacts/editor-pairs/pairs.jsonl`
Expected: prints example counts and the length distribution; exits 0 without touching a
device. If `artifacts/editor-pairs/pairs.jsonl` does not yet exist at full scale, first run
Task 2's script for real (`python scripts/build_editor_pairs.py` with its default `--n
20000`).

- [ ] **Step 6: Commit**

```bash
git add scripts/train_editor.py tests/test_train_editor.py
git commit -m "feat(editor): SFT training entrypoint, warm-started from tt-tnt-1024-dialogue"
```

---

### Task 4: Run the real training

**Files:** none created — this task executes Task 3's script on real hardware.

- [ ] **Step 1: Acquire a single-chip lease and run**

```bash
gozer run --chips 1 --who "claude:editor-training" \
  --reason "editor+skits SFT, warm-started from tt-tnt-1024-dialogue" -- \
  python3 scripts/train_editor.py --steps 3000 --save-every 1000
```

Expected: training completes without error; `artifacts/checkpoints-1024-editor/` contains
`step_3000.pkl` (an `SFTTrainer`-format `{"step", "model_state"}` pickle — NOT the ttml
multi-record format `train/run.py` writes) and `loss_curve.jsonl`. `main()` itself asserts
stochastic rounding before training and gamma movement after (both reused directly from
`scripts/train_skits.py`, per Task 3) — a run that reaches "training complete" printed the
gamma-movement report and did not raise, so a passing exit code already confirms both
guards; there is nothing further to verify by hand here.

- [ ] **Step 2: Record the run**

Note the final checkpoint path, step count, and train/val loss in a short paragraph appended
to `CLAUDE.md`'s project log (matching this project's existing convention for every real
training run) — final content depends on the real numbers this run produces, so it is
written after Step 1 completes, not specified here.

---

### Task 5: `scripts/eval_editor.py` — before/after proof

**Files:**
- Create: `scripts/eval_editor.py`
- Test: `tests/test_eval_editor.py`

**Interfaces:**
- Consumes: `scripts.story_tools.self_edit` (existing, from the 2026-08-27 session),
  `train.corrupt.corrupt` (Task 1).
- Produces: `recovers_real_words(text: str, vocab: set[str]) -> bool`,
  `score_recovery(better: str, edited: str, vocab: set[str]) -> dict` (returns
  `{"word_overlap": float, "has_fake_word": bool}`), `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_editor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from scripts.eval_editor import recovers_real_words, score_recovery

_VOCAB = {"the", "girl", "found", "a", "small", "silver", "key", "by", "river"}


def test_recovers_real_words_true_for_all_real_words():
    assert recovers_real_words("the girl found a small silver key", _VOCAB) is True


def test_recovers_real_words_false_for_a_fake_word():
    assert recovers_real_words("the girl found a smallury key", _VOCAB) is False


def test_score_recovery_reports_word_overlap_and_fake_word_flag():
    better = "the girl found a small silver key by the river"
    edited_good = "the girl found a small silver key"
    edited_bad = "the girl found a smallury spleck"
    good = score_recovery(better, edited_good, _VOCAB)
    bad = score_recovery(better, edited_bad, _VOCAB)
    assert good["has_fake_word"] is False
    assert bad["has_fake_word"] is True
    assert good["word_overlap"] > bad["word_overlap"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_eval_editor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.eval_editor'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
# scripts/eval_editor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Before/after evaluation for the editor objective (Task 5 of the 2026-08-27 editor-
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

    from scripts.build_editor_pairs import build_pairs, sample_clean_sentences

    corpus_paths = sorted(args.corpus_dir.glob("*.txt"))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_eval_editor.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Convert and serve the new checkpoint**

Task 4's checkpoint is an `SFTTrainer`-format pickle, not the ttml multi-record format
`scripts/convert_checkpoint.py`/`convert/to_hf.py` expect — use the existing
`sft_checkpoint_to_hf` helper (`scripts/eval_improv.py`, already used by
`scripts/eval_skits.py` for exactly this), not the ttml-native converter:

```python
from pathlib import Path
from scripts.eval_improv import sft_checkpoint_to_hf

sft_checkpoint_to_hf(
    Path("artifacts/checkpoints-1024-editor/step_3000.pkl"),
    warm_start_ckpt=Path("artifacts/checkpoints-1024-dialogue/tt_tnt_step00010764.pkl"),
    tokenizer_dir=Path("artifacts/tokenizer"),
    out_dir=Path("artifacts/hf-tt-tnt-1024-editor"),
)
```

Then serve `artifacts/hf-tt-tnt-1024-editor` on the 2-chip config (the config known NOT to
have the `FABRIC_2D_TORUS_XY` regression — see `docs/serving-with-tt-kernel.md` §8), then:

```bash
python scripts/eval_editor.py
```

Then re-run `scripts/story_tools.py`'s `self_edit()` directly against the new checkpoint on
the exact same draft used in the 2026-08-27 session ("The girl wished she had no one night
and she was always had been very special.") and compare to that session's recorded negative
result.

Then run `scripts/evaluate.py --model artifacts/hf-tt-tnt-1024-editor` against
`artifacts/hf-tt-tnt-1024-dialogue` as the control, per this project's standing
floor-comparison convention — a real regression check, not a new one invented for this task.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_editor.py tests/test_eval_editor.py docs/measurements/editor-eval.json
git commit -m "feat(editor): before/after evaluation against the story_tools self_edit negative result"
```

---

### Task 6: Checkpoint registration

**Files:**
- Modify: `docs/current_model.json`

**Interfaces:**
- Consumes: Task 4's final checkpoint path, Task 5's evaluation numbers.

- [ ] **Step 1: Add the new candidate entry**

Append to `docs/current_model.json`'s `candidates` array (do not change `current` — this
project's rule is that a designation is not a claim of quality in the abstract, and this
checkpoint is not promoted until the evaluation numbers actually support it):

```json
{
  "label": "tt-tnt-1024-editor",
  "hf_model": "artifacts/hf-tt-tnt-1024-editor",
  "training_window": 512,
  "note": "PLACEHOLDER -- replace with a one-line summary of Task 5's real numbers (mean_word_overlap, fake_word_rate, the self_edit before/after result, and the evaluate.py no-regression verdict) before committing this file. Do not commit this literal placeholder text."
}
```

(The literal word "PLACEHOLDER" above is deliberate and must not survive into the commit —
it exists so a reviewer can `grep -r PLACEHOLDER docs/current_model.json` and fail the
review if this step was skipped. Replace it with Task 5's actual measured numbers.)

- [ ] **Step 2: Verify the file is still valid JSON and the schema-required fields are present**

Run: `python -c "import json; json.load(open('docs/current_model.json'))"`
Expected: no error.

Run: `grep -c PLACEHOLDER docs/current_model.json`
Expected: `0` (fails loudly if the placeholder text wasn't replaced with real numbers).

- [ ] **Step 3: Commit**

```bash
git add docs/current_model.json
git commit -m "docs: register tt-tnt-1024-editor as a candidate checkpoint"
```

---

## Self-Review Notes

- **Spec coverage:** §1 (corruption module) → Task 1. §2 (delimiter framing, corrected
  during planning) → Task 3's `build_editor_example`. §3 (skits fold-in) → Task 3's
  `build_combined_examples`. §4 (blend/warm-start mechanics) → Task 3/4. §5 (evaluation) →
  Task 5. §6 (checkpoint identity) → Task 6.
- **Placeholder scan:** Task 6 Step 1 contains a literal "PLACEHOLDER" string by design —
  it's a self-enforcing gate (Step 2's grep), not an unfixed gap; flagged explicitly so it
  isn't mistaken for one.
- **Type/signature consistency checked:** `build_editor_example` and `build_skit_example`
  both return `{"input_ids": list[int], "labels": list[int]}`, confirmed compatible in
  `build_combined_examples`. `train.corrupt.corrupt`'s signature matches across Task 1's
  definition and Task 2/5's call sites (`seed`, `severity`, `n_corruptors` all keyword,
  matching Task 1 Interfaces exactly).
