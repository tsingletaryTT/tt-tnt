#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tools for driving tt-tnt-1024 through a served endpoint, one short turn at a time.

Why this exists: tt-tnt-1024 is a small base completion model. Asked to write a whole
story in one unattended generation, it collapses into repetition within a few dozen
tokens (`docs/serving-with-tt-kernel.md` §7 lineage; the same pattern is visible in any
long, unguided `/v1/completions` call against it). It needs help the way a human
first-drafter needs an outline: short asks, one at a time, judged and kept or discarded
before the next one is written.

The turn structure here is deliberately the SAME five-slot schema this project trained
into the (separate) skits checkpoints -- `offer / accept / add / stakes / handback`,
see `train/skit.py` -- reused here as a PROMPTING AND JUDGING discipline rather than
something the served model itself knows. tt-tnt-1024-dialogue was never trained on this
schema; nothing here assumes it was. The schema is doing the same job it did in that
training data: forcing each turn to carry something from what came before (`accept`),
introduce one genuinely new thing (`add`), move the stakes rather than stall
(`stakes`), and leave a real opening for the next turn (`handback`) -- now applied by
an external judge (a human, or an agent) at generation time instead of baked into
weights at training time.

This module is intentionally just the TOOLS: propose candidates, and score them
cheaply enough to filter out the obviously-degenerate ones before a human or agent
reads them. The actual per-turn judgment -- which candidate best satisfies `accept`,
whether the `add` word is a real addition, whether stakes moved -- is deliberately
left to the caller. Automating that judgment away would just reintroduce the
one-shot-generation failure this module exists to avoid, one turn at a time instead
of all at once.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

DEFAULT_ENDPOINT = "http://localhost:8000/v1/completions"
DEFAULT_MODEL = "episod/tt-tnt-1024"

# The same five roles `train/skit.py::Slots` derives from real corpus turns. Here they are
# prompt-construction hints, not measurements -- see module docstring.
SLOTS = ("offer", "accept", "add", "stakes", "handback")

_SLOT_HINT = {
    # No preceding turn to accept from; this slot starts the scene.
    "offer": "",
    # Ask for a continuation that keeps whatever the previous turn introduced in view,
    # rather than changing the subject.
    "accept": "",
    # The one slot worth a real prompt nudge: this project's own measurements
    # (`docs/measurements/reach-dial.json` and the task-7 particle-filter work) found that
    # without a push, this model's "fresh word" choices collapse onto the same ~25 common
    # verbs/particles. Asking explicitly for something new is a cheap, honest nudge -- it
    # does not guarantee a good word, it only widens what gets proposed for the caller to
    # judge.
    "add": " Something new should appear now:",
    # Stakes egress: no separate prompt hint (a coherent continuation is the whole ask);
    # whether stakes moved is a judgment call for the caller, not a probe-time distinction.
    "stakes": "",
    "handback": "",
}

# Sentence-ending punctuation vLLM should stop generation at, so a turn is exactly one
# clean sentence -- never a run of degenerate fragments the caller would have to trim.
_STOP = [". ", ".\n", "! ", "!\n", "? ", "?\n"]


@dataclass
class Candidate:
    text: str
    finish_reason: str
    repetition_flag: Optional[str] = None  # set by score_candidates; None = not flagged


@dataclass
class ProposeResult:
    slot: str
    prompt: str
    candidates: List[Candidate] = field(default_factory=list)


def _post(payload: dict, endpoint: str, timeout: float) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body[:500]}") from exc


def propose(
    story_so_far: str,
    slot: str,
    *,
    n: int = 3,
    max_tokens: int = 40,
    temperature: float = 0.8,
    top_p: float = 0.95,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
) -> ProposeResult:
    """Ask the served model for `n` independent one-sentence continuations for one slot.

    Each candidate is generated from the SAME prompt (`story_so_far` plus the slot's
    hint, if any) with independent sampling, and stopped at the first sentence-ending
    punctuation. Returns raw candidates -- call `score_candidates` before trusting any
    of them, and read them yourself before picking one; this function does not judge.
    """
    if slot not in SLOTS:
        raise ValueError(f"Unknown slot {slot!r}; expected one of {SLOTS}")
    prompt = story_so_far + _SLOT_HINT[slot]

    result = ProposeResult(slot=slot, prompt=prompt)
    for _ in range(n):
        data = _post(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": _STOP,
            },
            endpoint,
            timeout,
        )
        choice = data["choices"][0]
        text = choice["text"].strip()
        if text and text[-1] not in ".!?":
            text += "."
        result.candidates.append(Candidate(text=text, finish_reason=choice["finish_reason"]))
    return result


_WORD_RE = re.compile(r"[A-Za-z']+")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def score_candidates(story_so_far: str, candidates: List[Candidate]) -> List[Candidate]:
    """Flag (not filter -- the caller decides) obviously-degenerate candidates in place.

    Two cheap checks, both measuring things this model is DOCUMENTED to fail at
    (`docs/current_model.json`'s "worse 4-gram repeat rate" qualification, and every
    long unguided generation this project has recorded):

    - `empty`: nothing generated (stop sequence fired on token 0).
    - `internal_repeat`: the candidate repeats a 3-word span against itself, or against
      the immediately preceding ~40 words of the story -- the collapse signature this
      whole module exists to route around ("The dragon. The dragon, in the dragon...").

    Returns the same list, annotated. Does not raise, drop, or reorder -- a flagged
    candidate may still be the best available one if every candidate is flagged, and
    that is a fact the caller needs to see, not one this function should hide.
    """
    context_words = _words(story_so_far)[-40:]

    def trigrams(words: List[str]) -> set:
        return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}

    context_tri = trigrams(context_words)

    for cand in candidates:
        if not cand.text:
            cand.repetition_flag = "empty"
            continue
        cw = _words(cand.text)
        if len(cw) < 3:
            continue
        cand_tri = trigrams(cw)
        if len(cw) != len(set(cw)) and max(cw.count(w) for w in set(cw)) >= 3:
            cand.repetition_flag = "internal_repeat"
        elif cand_tri & context_tri:
            cand.repetition_flag = "repeats_context"
    return candidates


def self_edit(
    story_so_far: str,
    draft_turn: str,
    *,
    n: int = 3,
    max_tokens: int = 30,
    temperature: float = 0.6,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
) -> List[Candidate]:
    """Ask the model to rewrite ITS OWN draft turn, framed as an editing task.

    Prototype only, prompt-level, no training involved. tt-tnt-1024-dialogue was never
    trained to edit or critique text -- this tests whether framing the SAME base model as
    "fix this sentence" does anything useful at all, as a cheap, immediate probe before
    investing in an actual editor-training run (see the note this session added to
    CLAUDE.md's project log: "train the model to be an editor and review its own work").
    A clean negative result here (no better than a fresh `propose` call) is itself the
    argument FOR that training investment, not a reason to abandon the idea.
    """
    prompt = (
        f"{story_so_far}\n\n"
        f"Draft: {draft_turn}\n"
        f"Better version:"
    )
    candidates: List[Candidate] = []
    for _ in range(n):
        data = _post(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "stop": _STOP,
            },
            endpoint,
            timeout,
        )
        choice = data["choices"][0]
        text = choice["text"].strip()
        if text and text[-1] not in ".!?":
            text += "."
        candidates.append(Candidate(text=text, finish_reason=choice["finish_reason"]))
    return score_candidates(story_so_far, candidates)


def render_result(result: ProposeResult) -> str:
    lines = [f"[{result.slot}] prompt tail: ...{result.prompt[-80:]!r}"]
    for i, cand in enumerate(result.candidates):
        flag = f"  <- {cand.repetition_flag}" if cand.repetition_flag else ""
        lines.append(f"  ({i}) [{cand.finish_reason}] {cand.text!r}{flag}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--story", required=True, help="Story so far (plain text).")
    p.add_argument("--slot", required=True, choices=SLOTS)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead.")
    args = p.parse_args()

    result = propose(
        args.story,
        args.slot,
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        endpoint=args.endpoint,
        model=args.model,
    )
    score_candidates(args.story, result.candidates)

    if args.json:
        print(
            json.dumps(
                {
                    "slot": result.slot,
                    "prompt": result.prompt,
                    "candidates": [
                        {"text": c.text, "finish_reason": c.finish_reason, "flag": c.repetition_flag}
                        for c in result.candidates
                    ],
                }
            )
        )
    else:
        print(render_result(result))


if __name__ == "__main__":
    main()
