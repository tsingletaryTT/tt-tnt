#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Precision and recall for `train.content_add`, against labels drawn BEFORE it existed.

WHY A SCRIPT AND NOT A DOCSTRING CLAIM
======================================
The filter in `train.content_add` decides the vocabulary of the `add` slot, which is the one
slot the reach dial is built on. A filter that cannot be checked is a filter that gets
believed, and the specific way this one could be wrong was known in advance: the
part-of-speech signal that separates ``comet`` from ``hello`` on this corpus does it through
CAPITALISATION, not grammar (see `train.content_add`'s docstring). So the numbers this script
produces go into the derivation manifest, beside the yield, unconditionally.

TWO INDEPENDENT VALIDATION SETS, NEITHER OF THEM THE THING THE FILTER WAS TUNED ON
---------------------------------------------------------------------------------
1. **`LABELS_PATH`** -- 250 real `add` observations sampled uniformly (seed 20260823) from
   `artifacts/reach-skits/skits.jsonl`'s 123,042, each hand-labelled IN ITS OWN TURN before
   the classifier was run over them. See `docs/measurements/reach-content-add-labels.md` for
   the labelling rule and its judgement calls.
2. **`TOP25_GROUND_TRUTH`** -- the 25 commonest `add` values in the same artifact, with the
   task's own labels: particles, except ``look love come want give doing mean``, which are
   genuine verbs and must be KEPT. This is the set `NARRATION_FLOOR` was chosen against, so
   it is reported as a fit, not as a test -- and it is exactly the set a crude stoplist would
   get wrong, which is why it is worth publishing beside the sample.

Scoring convention: the POSITIVE class is CONTENT. Precision is "of what the filter keeps,
how much really names a thing or an action"; recall is "of what really does, how much the
filter keeps". Both matter and they trade against each other -- precision is what makes
`add=comet` rather than `add=what's`, recall is what pays for it in yield.

    python3 scripts/validate_content_add.py --corpus artifacts/corpus/tinystories.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.content_add import (GATE_NAMES, NARRATION_FLOOR,  # noqa: E402
                               SpeechProfile, content_add_reasons, narration_rate)

LABELS_PATH = ROOT / "docs" / "measurements" / "reach-content-add-labels.jsonl"

#: The measured top 25 `add` values of `artifacts/reach-skits/`, with the task's labels.
#: True = must be KEPT. The seven Trues are the hard cases: genuine verbs that read as
#: particles, and the reason a frequency stoplist is the wrong instrument here.
TOP25_GROUND_TRUTH: Tuple[Tuple[str, bool], ...] = (
    ("look", True), ("please", False), ("hi", False), ("hello", False), ("love", True),
    ("what", False), ("wow", False), ("why", False), ("come", True), ("yes", False),
    ("thank", False), ("ok", False), ("okay", False), ("want", True), ("mine", False),
    ("let's", False), ("doing", True), ("where", False), ("our", False), ("mean", True),
    ("can't", False), ("what's", False), ("give", True), ("course", False), ("hey", False),
)

#: One turn per top-25 word, lifted verbatim from `artifacts/reach-skits/skits.jsonl`, so the
#: per-instance POS gate has the context it needs. Hand-picked to be the word's TYPICAL use,
#: not its most favourable one: `mean` is given its verb reading because that is the reading
#: the ground truth labels True, and the adjective reading is tested separately in
#: `tests/test_content_add.py`.
TOP25_TURNS: Dict[str, str] = {
    "look": "Look, mom, a monkey!",
    "please": "Can I have a lemon, please?",
    "hi": "Hi Billy!",
    "hello": "Hello, little frog! Are you okay?",
    "love": "Thank you, Mama. The broken pasta is good. We love you.",
    "what": "Mom, what is that?",
    "wow": "Wow, Lily, you have an amazing heart.",
    "why": "Why are there so many people here?",
    "come": "Come on, Ben, let's go up the hill!",
    "yes": "Yes, it is big. Do you want to play it?",
    "thank": "Thank you, Mr. Lee. Have a nice day too.",
    "ok": "OK, mom, I will imagine that I have a crane.",
    "okay": "Okay, I will clean my room.",
    "want": "No, Mommy! I want music now!",
    "mine": "That is mine, not yours!",
    "let's": "Come on, let's go to the park.",
    "doing": "Sam, what are you doing?",
    "where": "Where is my ball?",
    "our": "Where is our king? What have you done to him?",
    "mean": "What does melt mean?",
    "can't": "Now you can't play your violin anymore!",
    "what's": "What's wrong Jack?",
    "give": "Give me the car, Ben!",
    "course": "Of course, I can help you.",
    "hey": "Hey, that is my toy!",
}


def load_labels(path: Path = LABELS_PATH) -> List[dict]:
    """The hand-labelled sample, as dicts with `add`, `turn` and `label`."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def confusion(pairs: Sequence[Tuple[bool, bool]]) -> dict:
    """`(kept, is_content)` pairs -> the confusion counts, precision, recall and F1.

    A pure function of the pairs so a fixture can call it with hand-built counts and the
    arithmetic is checked without a corpus. Precision and recall are None, not 0.0, when their
    denominator is empty -- a rate over nothing is not zero, and printing 0.0000 there is how
    an empty run reads as a catastrophic one.
    """
    tp = sum(1 for k, g in pairs if k and g)
    fp = sum(1 for k, g in pairs if k and not g)
    fn = sum(1 for k, g in pairs if not k and g)
    tn = sum(1 for k, g in pairs if not k and not g)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    return {"n": len(pairs), "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "positive_class": "content (the word names a thing or an action)"}


def validate_sample(profile: SpeechProfile, *, floor: float = NARRATION_FLOOR,
                    labels: Optional[List[dict]] = None) -> dict:
    """The hand-labelled sample's confusion, plus every disagreement, named.

    The error lists are not decoration. `false_positive_examples` is what a reader sees in the
    artifact's top-25 (``kind``, ``let``), and `false_negative_examples` is what the yield
    paid for (``cry``, ``unicorn``, ``teddy``) -- both belong in the manifest where the number
    is, not in a report that the artifact will outlive.
    """
    rows = labels if labels is not None else load_labels()
    pairs: List[Tuple[bool, bool]] = []
    fps: List[dict] = []
    fns: List[dict] = []
    for row in rows:
        why = content_add_reasons(row["add"], row["turn"], profile, floor=floor)
        kept = not why
        gold = row["label"] == "content"
        pairs.append((kept, gold))
        if kept and not gold:
            fps.append({"add": row["add"], "turn": row["turn"][:80]})
        elif not kept and gold:
            fns.append({"add": row["add"], "rejected_by": list(why),
                        "turn": row["turn"][:80]})
    out = confusion(pairs)
    out["labels"] = str(LABELS_PATH.relative_to(ROOT))
    out["narration_floor"] = floor
    out["false_positive_examples"] = fps
    out["false_negative_examples"] = fns
    out["labelled_before_the_classifier_ran"] = True
    return out


def validate_top25(profile: SpeechProfile, *, floor: float = NARRATION_FLOOR) -> dict:
    """The measured top-25 `add` vocabulary against the task's labels, word by word.

    Reported as a FIT and not as a held-out test: `NARRATION_FLOOR` was chosen so that
    ``love`` (0.293, the lowest-rate must-keep) clears it and ``thank`` (0.182, the
    highest-rate must-go) does not. Publishing it anyway because it is the set that shows what
    a crude stoplist would have destroyed -- seven of these 25 are verbs.
    """
    pairs: List[Tuple[bool, bool]] = []
    per_word: List[dict] = []
    for word, gold in TOP25_GROUND_TRUTH:
        turn = TOP25_TURNS[word]
        why = content_add_reasons(word, turn, profile, floor=floor)
        kept = not why
        pairs.append((kept, gold))
        per_word.append({"word": word, "should_keep": gold, "kept": kept,
                         "narration_rate": round(narration_rate(word, profile), 4),
                         "rejected_by": list(why)})
    out = confusion(pairs)
    out["per_word"] = per_word
    out["disagreements"] = [r["word"] for r in per_word if r["kept"] != r["should_keep"]]
    out["status"] = ("FIT, not a held-out test: NARRATION_FLOOR was chosen against this set. "
                     "Published because it is the set a frequency stoplist gets wrong.")
    return out


def floor_sensitivity(profile: SpeechProfile,
                      floors: Sequence[float] = (0.10, 0.15, 0.20, 0.25, 0.30),
                      labels: Optional[List[dict]] = None) -> dict:
    """Sample precision/recall across candidate floors, so the choice is visible as a choice.

    A threshold reported without its neighbours is indistinguishable from a threshold fitted
    to the number it produced.
    """
    rows = labels if labels is not None else load_labels()
    return {str(f): {k: v for k, v in validate_sample(profile, floor=f, labels=rows).items()
                     if k in ("precision", "recall", "f1", "true_positive",
                              "false_positive", "false_negative", "true_negative")}
            for f in floors}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "tinystories.txt")
    ap.add_argument("--limit", type=int, default=None,
                    help="stories for the speech profile; default None = the whole corpus.")
    ap.add_argument("--floor", type=float, default=NARRATION_FLOOR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from scripts.derive_dialogue_skits import iter_stories
    from train.content_add import build_speech_profile, profile_meta
    print(f"building the speech profile over {args.corpus} ...", flush=True)
    profile = build_speech_profile(iter_stories(args.corpus, args.limit))
    print(f"  {profile.stories:,} stories, "
          f"{len(profile.narration):,} narration words", flush=True)

    report = {"profile": profile_meta(profile),
              "narration_floor": args.floor,
              "hand_labelled_sample": validate_sample(profile, floor=args.floor),
              "measured_top_25": validate_top25(profile, floor=args.floor),
              "floor_sensitivity": floor_sensitivity(profile)}
    s = report["hand_labelled_sample"]
    print(f"hand-labelled sample (n={s['n']}): precision {s['precision']} "
          f"recall {s['recall']} f1 {s['f1']}  "
          f"TP={s['true_positive']} FP={s['false_positive']} "
          f"FN={s['false_negative']} TN={s['true_negative']}")
    t = report["measured_top_25"]
    print(f"measured top-25: precision {t['precision']} recall {t['recall']}  "
          f"disagreements {t['disagreements']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
