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
`severity` in [0, 1] controlling how much it does, not whether it does anything on a TYPICAL
input -- but each corruptor has its own early no-op case (too short, no eligible word,
nothing to change) and can leave short or atypical sentences untouched even at severity 1.0.
`scripts/build_editor_pairs.py::build_pairs` exists specifically to handle that: it retries
with a different seed/corruptor and drops the sentence if every attempt still no-ops (a real
measured rate on this corpus: 14/100).
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

    `n_corruptors=0` returns `text` unchanged by design -- but it is not the ONLY way to
    get a no-op: each individual corruptor can also no-op on a short or atypical sentence
    even with `n_corruptors>=1` (see the module docstring). A caller building `draft`/
    `better` pairs must check for and handle that case itself --
    `scripts/build_editor_pairs.py::build_pairs` does, via a bounded retry-then-drop loop.
    """
    if n_corruptors <= 0:
        return text
    rng = random.Random(seed)
    chosen = rng.sample(_CORRUPTORS, k=min(n_corruptors, len(_CORRUPTORS)))
    out = text
    for i, fn in enumerate(chosen):
        out = fn(out, seed=seed + i, severity=severity)
    return out
