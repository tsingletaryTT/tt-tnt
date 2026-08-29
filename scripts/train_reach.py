#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Task 3: two paired SFT arms over reach-dial skits -- `dial` vs `nodial`.

WHAT IS BEING TRAINED, AND WHY THERE ARE TWO ARMS
=================================================
We are building a DIAL. The v3 think-block carries a `reach` slot (`near`/`mid`/`far`)
declaring how semantically distant the `add` word is from what is already in play. At
inference the dial becomes an INPUT: force `reach: far`, get a more distant beat. `add` is
the one slot stage 2 proved works (+0.326 over a context-matched floor), and it works
precisely because its value is absent from the visible context -- so `reach` turns that
mechanism into something controllable.

  * ARM `dial`   -- trained on the skits WITH the `reach` slot present in the think-block.
  * ARM `nodial` -- trained on the IDENTICAL skits with the `reach` slot ABSENT and
                    everything else the same (`train.reach.drop_reach`).

`nodial` is a NEGATIVE CONTROL and nothing else. The primary experiment (task 4) is
WITHIN-model: force near/mid/far on the same held-out scenes and measure realised distance.
`nodial` answers the one question that design cannot answer by itself -- "does injecting ANY
token into the block perturb generation?" -- because a forced `reach` token should do
NOTHING to a model that never learned one. That makes it a valid control only if it saw the
same material, which is why the pairing discipline below is inherited from stage 2 intact.

THE INDEPENDENT VARIABLE IS THE BLOCK'S SCHEMA, NOT THE BLOCK'S PRESENCE
-----------------------------------------------------------------------
Stage 2's arms were `think` vs `nothink`: one had a think-block, the other had none, and
`build_skit_example(with_think=False)` dropped the block's segments entirely. THAT IS NOT
WHAT HAPPENS HERE. Both arms of this task call `build_skit_example` with
``with_think=True``; both therefore get all nine segments, the same positional supervision
mask, and a think-block on every model turn. They differ in ONE line inside that block.
Passing ``with_think=False`` for `nodial` would build stage 2's experiment under this
task's name and the control would answer a different question than the one asked.

PAIRED BY CONSTRUCTION (inherited from scripts/train_skits.py, unchanged)
------------------------------------------------------------------------
Both arms read the SAME skits file, build one example per skit in FILE ORDER, and hand that
list to an ``InMemoryDataloader`` with the same ``seed`` and ``batch_size``. That loader's
shuffle is a pure function of ``(seed, epoch)`` and the dataset length, and both arms have
identical lengths, so both arms visit the same skits in the same order at every step. The
manifest records a fingerprint of the first epoch's index order, read off the CONSTRUCTED
loader, so the pairing is checkable after the fact rather than merely asserted here.

    gozer run --chips 1 --who "claude:reach" --reason "reach SFT arm dial" -- \
        python3 scripts/train_reach.py --arm dial --steps 3000
    gozer run --chips 1 --who "claude:reach" --reason "reach SFT arm nodial" -- \
        python3 scripts/train_reach.py --arm nodial --steps 3000

Run the arms SEQUENTIALLY, on ONE BOARD, with a ``(1, 1)`` mesh. A four-chip mesh open
hard-froze this host once with no OOM, no kernel panic and no pstore record; there is no
reproduction and no fix, so it is not retried.

THE FILE'S OWN SPLIT IS AUTHORITATIVE -- THIS IS THE DELTA FROM train_skits.py
=============================================================================
`scripts/train_skits.py` holds out its own tail of ``--val-size`` (default 256) examples.
For THIS artifact that would be wrong, and wrong in a way that destroys the experiment
rather than merely blurring it. `artifacts/reach-skits/derive_manifest.json` carries
``split.TRAINER_MUST_RESPECT_THIS``: 36,913 train / 4,101 eval, every row carrying its own
``split`` label, and **the tercile cut points that DEFINE the three dial buckets were fitted
on the training split only**. Train on a 256-row tail and ~3,845 rows the derivation marked
`eval` land in training; the eval population was then inside the fit's population after all,
and the dial's buckets stop meaning anything on the very scenes task 4 measures.

So this trainer splits on the ``split`` FIELD, never on a size:

  * `partition_by_split` filters by label, preserving file order.
  * `assert_split_respected` RAISES -- before any device is opened -- if a single
    eval-labelled row reached the training list, if any row carries an unknown label, or if
    the resulting counts disagree with the derive manifest's ``n_train``/``n_eval``. The
    count cross-check is the one that catches the interesting failure: a silent read of the
    wrong field, or an off-by-one, produces a plausible-looking split that no
    label-by-label check inside the same wrong read would notice.

``--val-cap`` caps how much of the held-out split feeds the IN-TRAINING validation curve
(default 256, the head of the eval rows). It does not move the split: capping only shortens
the curve's eval set, and every eval-labelled row stays out of training either way. The full
4,101 remain on disk for task 4.

RULING C IS RE-APPLIED HERE, BECAUSE THE DERIVATION APPLIED IT TO THE WRONG BLOCK
================================================================================
`derive_manifest.json` reports ``token_lengths.over_max_seq_len: 0`` and ``max: 512`` under
ruling C ("Excluded, NOT truncated"). That does not describe what trains here. The gate that
enforced it (`screen_candidate`) runs BEFORE `train.reach.with_reach`, because `with_reach`
needs the tercile cut points and those are not fitted until every candidate is collected --
so it measured the FIVE-slot, label-`stakes` block, while the rows that shipped carry a
six-slot block with a numeric `stakes`. Measured on the shipped artifact: **526 dial (1.43%)
and 180 nodial (0.49%) training examples exceed 512 tokens**, max 544, and `sft_collate_fn`
would truncate every one of them silently -- costing each skit its final supervised model
turn, and costing the DIAL arm three times more often than the control. See
`over_length_indices`. The union across arms is dropped from both, so the arms stay the same
length and therefore the same permutation. `--keep-over-length` restores the truncating
behaviour, as a recorded choice rather than a hidden default.

THE GUARD THAT MATTERS
======================
`stochastic_rounding` defaults **off** in ttml's optimizer registry. In bfloat16 the ulp at
1.0 is 0.0039, an order of magnitude larger than the ~3e-4 Adam updates the 17 RMSNorm gamma
tensors receive, so with deterministic rounding every one of those updates rounds straight
back to 1.0 and is discarded -- silently, for the entire run. An earlier stage shipped both
arms that way and measured 0% schema adherence; the same arms measured 98% once the flag was
on. Two guards, at the two layers that can fail:

  * before training -- read the flag back out of `trainer.optimizer.get_state_dict()`, the
    optimizer the C++ factory actually built, and RAISE if it is off. Checking the dict we
    passed in would prove intent only, and a warning is what let the bug through the first
    time;
  * after training -- compare the final checkpoint's 17 gammas against the warm-start base
    and RAISE if any is bit-identical. The pre-training guard proves the flag reached the
    optimizer; only this one proves the parameters moved.

Both are imported from `scripts.train_skits` rather than copied. They are reviewed, they are
tested, and a second copy is how one of them would get fixed and the other would not.

NEVER COMPARE THE TWO ARMS' TRAINING LOSSES
-------------------------------------------
They supervise DIFFERENT TOKEN SETS: the `dial` arm's block carries an extra ``reach: ...``
line, so its sequences are longer and its per-token loss is averaged over a different
population. The number is recorded per arm for monitoring the run and for nothing else. The
manifest says so beside the loss fields, because a manifest that ships two numbers side by
side invites exactly the subtraction that is meaningless here.

Also available without a device, for checking an existing run::

    python3 scripts/train_reach.py --verify-gammas artifacts/reach/ckpt-dial/step_3000.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_skits import build_skit_example  # noqa: E402
# The guards, the curve recorder and the gamma comparison are IMPORTED, not re-implemented:
# they are the reviewed, tested versions from stage 2 and a fork would let one copy drift.
from scripts.train_skits import (EXPECTED_GAMMAS, LossRecorder,  # noqa: E402
                                 assert_eval_wired, assert_gammas_moved,
                                 assert_stochastic_rounding, compare_gammas,
                                 length_stats, read_arm_gammas, read_base_gammas,
                                 verify_gammas_cli)
from train.reach import (NODIAL_SLOT_NAMES, REACH_SLOT_NAMES, REACH_VALUES,  # noqa: E402
                         ReachSlots, drop_reach, reach_slot_names_of)
from train.skit import SKIT_ROLES, Skit  # noqa: E402

MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024"
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")

#: The task-2 derivation: 41,014 skits / 123,042 observations over the full 2,119,489-story
#: corpus, with a `split` label and an `add_df` on every row. NOT artifacts/dialogue-skits
#: (task 1, five-slot blocks, no dial) and NOT artifacts/skits-200k (stage 2).
DEFAULT_SKITS = ROOT / "artifacts" / "reach-skits" / "skits.jsonl"

#: Written beside the skits by `scripts/derive_dialogue_skits.py`. Read, not assumed: its
#: `split.n_train`/`n_eval` are what `assert_split_respected` cross-checks the partition
#: against, so a wrong-field read cannot produce a plausible split unnoticed.
DEFAULT_DERIVE_MANIFEST = DEFAULT_SKITS.parent / "derive_manifest.json"

DEFAULT_OUT_ROOT = ROOT / "artifacts" / "reach"

#: 8 blocks x {attention_norm, mlp_norm} plus the final llama/ln_fc = 17. Re-exported from
#: stage 2 so the two trainers cannot disagree about how many gammas this architecture has.
EXPECTED_GAMMAS = EXPECTED_GAMMAS

MAX_SEQ_LEN = 512

#: The two arm names. `dial` is the treatment, `nodial` the negative control.
ARMS: Tuple[str, ...] = ("dial", "nodial")

#: Row labels this trainer understands. Anything else is a file it cannot split correctly,
#: and it says so rather than guessing.
SPLIT_LABELS: Tuple[str, ...] = ("train", "eval")

#: Recorded in the manifest so task 4 builds its prompts the same way. Examples are
#: tokenized SEGMENT-WISE (`build_skit_example` encodes each `skit_segments` entry
#: separately), so the trained sequence carries a per-segment leading space: the block
#: boundary is `['.', 'Ġ<', 'think', '>']`, not `['.', '<', 'think', '>']`. An eval prompt
#: assembled by concatenating strings and tokenizing once asks the model to emit `<think>`
#: after a token it never saw there.
TOKENIZATION_NOTE = (
    "Examples are tokenized SEGMENT-WISE: build_skit_example encodes each skit_segments "
    "entry separately, with add_special_tokens=True on the prefix segment only. This "
    "tokenizer prefixes a space to the start of each encode call, so the trained sequence "
    "carries a PREFIX SPACE at every segment boundary. Measured, not assumed: the "
    "think-block boundary is the token sequence ['.', '\u0120<', 'think', '>'] -- a "
    "SPACE-PREFIXED '<'. Task 4 must build prompts the same way (encode segment by segment "
    "and concatenate ids), NOT by joining the strings and tokenizing once: that produces a "
    "bare '<' and asks the model to emit <think> after a token it never saw there."
)


# --------------------------------------------------------------------------------------
# arm -> block schema.  The independent variable of the whole experiment.
# --------------------------------------------------------------------------------------
def with_dial_for_arm(arm: str) -> bool:
    """`--arm` -> "does this arm's think-block carry the `reach` line?", in one place.

    Named and tested separately rather than inlined as ``arm == "dial"``: this single
    boolean IS the independent variable, and an inverted or constant value produces two runs
    that look perfectly healthy and mean nothing. Stage 2 learned this the same way, with
    `with_think_for_arm`.

    Note what this boolean does NOT control: whether a think-block is present at all. Both
    arms have one. See the module docstring.
    """
    if arm == "dial":
        return True
    if arm == "nodial":
        return False
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def block_for_arm(slots: ReachSlots, *, with_dial: bool):
    """One block, rendered for one arm: the v3 six-slot block, or it minus `reach`.

    A function rather than a conditional at the call site so the control's definition --
    "identical except the dial is absent" -- has a name a test can hold.
    `drop_reach` copies the other five values by field name, so this cannot silently
    re-label a slot.
    """
    return slots if with_dial else drop_reach(slots)


def skit_for_arm(skit: Skit, *, with_dial: bool) -> Skit:
    """The same skit with every block rendered for one arm. Text is untouched.

    Prefix, turns and roles are carried across verbatim -- only the block dataclass changes,
    so `train.skit.skit_segments` and `scripts.derive_skits.build_skit_example` keep
    producing the same nine segments and the same positional supervision mask for both arms.
    """
    return Skit(story_id=skit.story_id, prefix=skit.prefix, turns=skit.turns,
                blocks=tuple(block_for_arm(b, with_dial=with_dial) for b in skit.blocks))


# --------------------------------------------------------------------------------------
# reach-skits.jsonl -> (Skit, split label) rows
# --------------------------------------------------------------------------------------
def load_reach_skits(path: Path) -> List[Tuple[Skit, str]]:
    """Rebuild `(Skit-with-ReachSlots, split_label)` rows from a reach `skits.jsonl`.

    ORDER IS LOAD-BEARING and this function must never sort, filter or deduplicate: the arms
    are paired by consuming the same examples in the same order, and the dataloader's shuffle
    permutes POSITIONS, not story ids. Reordering here would silently unpair the two arms
    while leaving every count identical.

    Four things are VALIDATED rather than trusted, each because failing to would be
    invisible downstream:

      * `roles` must equal `SKIT_ROLES` -- a file whose roles disagree would get the
        partner/model supervision split wrong with no other symptom;
      * the block keys must be exactly `REACH_SLOT_NAMES` -- a five-slot (task 1) file fed
        to this trainer would train a `nodial`-shaped arm under the name `dial`;
      * `reach` must be one of `REACH_VALUES` -- the dial is a three-way categorical and a
        fourth value is a token task 4 can never force;
      * `split` must be one of `SPLIT_LABELS` -- a missing or misspelled label is the exact
        failure `assert_split_respected` exists to catch, and it is cheaper to catch here,
        on the row, than in an aggregate count.
    """
    rows: List[Tuple[Skit, str]] = []
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            roles = tuple(rec.get("roles", ()))
            if roles != SKIT_ROLES:
                raise ValueError(
                    f"{path}:{lineno}: roles {roles} != {SKIT_ROLES}; this is not a skits "
                    f"file this trainer can supervise correctly")
            turns = tuple(rec["turns"])
            if len(turns) != len(SKIT_ROLES):
                raise ValueError(f"{path}:{lineno}: expected {len(SKIT_ROLES)} turns, "
                                 f"got {len(turns)}")

            blocks: List[ReachSlots] = []
            for b in rec["blocks"]:
                if tuple(b) != REACH_SLOT_NAMES:
                    raise ValueError(
                        f"{path}:{lineno}: block keys {tuple(b)} != {REACH_SLOT_NAMES}; "
                        f"this is not a v3 six-slot reach file (a five-slot file trained "
                        f"here would produce a `nodial`-shaped arm labelled `dial`)")
                if b["reach"] not in REACH_VALUES:
                    raise ValueError(f"{path}:{lineno}: reach {b['reach']!r} not in "
                                     f"{REACH_VALUES}")
                blocks.append(ReachSlots(**b))

            split = rec.get("split")
            if split not in SPLIT_LABELS:
                raise ValueError(
                    f"{path}:{lineno}: split {split!r} not in {SPLIT_LABELS}. Every row of "
                    f"a reach skits file carries its own split label and this trainer "
                    f"splits on that field, never on a size -- see "
                    f"derive_manifest.json:split.TRAINER_MUST_RESPECT_THIS")

            rows.append((Skit(story_id=rec["story_id"], prefix=rec["prefix"], turns=turns,
                              blocks=tuple(blocks)), split))
    return rows


def partition_by_split(rows: Sequence[Tuple[Skit, str]]
                       ) -> Tuple[List[Skit], List[Skit]]:
    """`(train_skits, eval_skits)` by each row's OWN label, preserving file order.

    A FILTER, not a slice. The derive manifest says the eval split is the tail of the file,
    and it is -- but a slice would encode that fact in this trainer as well as in the
    derivation, and the two would then have to be kept in agreement by hand. Filtering on
    the label is correct whatever the layout, and `split_layout` below reports the layout
    separately so a change in it is visible rather than load-bearing.
    """
    train = [s for s, lab in rows if lab == "train"]
    ev = [s for s, lab in rows if lab == "eval"]
    return train, ev


def split_layout(rows: Sequence[Tuple[Skit, str]]) -> Dict[str, Any]:
    """Where the eval-labelled rows SIT in the file. Reported, deliberately not enforced.

    The derive manifest claims the eval split is a contiguous TAIL. That claim is worth
    recording next to the run -- if it ever stops being true, the number in the manifest is
    how a reader finds out -- but it is not worth raising on, because `partition_by_split`
    does not depend on it. Enforcing it would turn a harmless change in the derivation's
    layout into a failed 3000-step run.
    """
    labels = [lab for _, lab in rows]
    first_eval = labels.index("eval") if "eval" in labels else None
    contiguous_tail = (first_eval is not None
                       and all(lab == "eval" for lab in labels[first_eval:]))
    return {
        "n_rows": len(labels),
        "first_eval_index": first_eval,
        "eval_is_contiguous_tail": bool(contiguous_tail),
        "note": "the derive manifest claims the eval split is the file's tail; recorded "
                "here, not enforced -- partition_by_split filters on the label and does "
                "not depend on the layout",
    }


def read_manifest_split(path: Path) -> Dict[str, Any]:
    """The derive manifest's `split` block, or `{}` if there is no manifest beside the file.

    Returns a dict rather than raising when absent so a hand-made smoke file (a fixture, a
    20-row slice) can be trained on without inventing a manifest for it.
    `assert_split_respected` treats an absent manifest as "no cross-check available" and
    says so in its printout, rather than silently skipping a check the caller believes ran.
    """
    p = Path(path)
    if not p.exists():
        return {}
    blob = json.loads(p.read_text())
    got = blob.get("split")
    return dict(got) if isinstance(got, dict) else {}


def assert_split_respected(rows: Sequence[Tuple[Skit, str]], train: Sequence[Skit],
                           ev: Sequence[Skit], *, manifest_split: Dict[str, Any],
                           arm: str) -> Dict[str, Any]:
    """RAISE unless the partition is the file's own split, exactly. Returns the evidence.

    THE ASSERTION THIS WHOLE SCRIPT EXISTS FOR. The tercile cut points that define
    `near`/`mid`/`far` were fitted on the training split ONLY. If one eval-labelled row
    trains, the eval population was inside the fit's population, and task 4's dial measures
    a bucket boundary it helped choose. That is not a bias to be noted; it is the
    measurement returning its own input.

    Three independent checks, because each catches a failure the others cannot:

      1. IDENTITY. Every skit in `train` carries the label `train`, matched by identity
         (`id()`), not by story_id. Story ids are not unique across a derivation that can
         emit more than one skit per story, so a story_id-keyed check could pass while an
         eval row rode in under a duplicate id. Identity is exact and free.
      2. COUNTS vs THE MANIFEST. `n_train`/`n_eval` must match the derivation's own figures.
         This is the check that catches reading the WRONG FIELD: a `rec.get("split")` that
         found nothing, defaulted, and produced a tidy-looking 41,014/0 passes check (1)
         perfectly, because every row really does carry the label it was given.
      3. DISJOINTNESS AND COVERAGE. `len(train) + len(eval) == len(rows)`, so no row was
         dropped or counted twice on the way through.

    Raises. Never warns: a warning for exactly this class of bug is what cost an earlier
    stage 3000 steps of two frozen-gamma arms in a long log.
    """
    labels_by_id = {id(s): lab for s, lab in rows}

    leaked = [s for s in train if labels_by_id.get(id(s)) != "train"]
    if leaked:
        ids = [s.story_id for s in leaked[:10]]
        raise RuntimeError(
            f"[{arm}] {len(leaked)} row(s) NOT labelled `train` reached the training list "
            f"(story_ids {ids}{'...' if len(leaked) > 10 else ''}). The reach terciles were "
            f"fitted on the training split only, so an eval-labelled row in training makes "
            f"the dial's buckets partly a function of the population task 4 measures. "
            f"Refusing to start.")
    misfiled = [s for s in ev if labels_by_id.get(id(s)) != "eval"]
    if misfiled:
        raise RuntimeError(
            f"[{arm}] {len(misfiled)} row(s) NOT labelled `eval` reached the held-out list; "
            f"the partition is not the file's split. Refusing to start.")

    if len(train) + len(ev) != len(rows):
        raise RuntimeError(
            f"[{arm}] partition does not cover the file: {len(train)} train + {len(ev)} "
            f"eval != {len(rows)} rows. Refusing to start.")

    checked_against_manifest = False
    if manifest_split:
        want_train = manifest_split.get("n_train")
        want_eval = manifest_split.get("n_eval")
        if want_train is not None and want_eval is not None:
            checked_against_manifest = True
            if (len(train), len(ev)) != (int(want_train), int(want_eval)):
                raise RuntimeError(
                    f"[{arm}] split counts disagree with the derivation: got "
                    f"{len(train)} train / {len(ev)} eval, manifest says "
                    f"{want_train} / {want_eval}. Either the wrong field was read or the "
                    f"skits file is not the one the manifest describes. Refusing to start.")

    print(f"[{arm}] split respected: {len(train):,} train / {len(ev):,} eval, read from "
          f"each row's own `split` field"
          + (f"; counts cross-checked against the derive manifest "
             f"({manifest_split.get('n_train')}/{manifest_split.get('n_eval')})"
             if checked_against_manifest
             else "; NO derive manifest beside the skits file, so the counts were NOT "
                  "cross-checked"))
    return {
        "source": "each row's own `split` field (NOT a --val-size tail)",
        "n_train": len(train),
        "n_eval": len(ev),
        "eval_labelled_rows_in_training": 0,
        "checked_against_derive_manifest": checked_against_manifest,
        "derive_manifest_n_train": manifest_split.get("n_train"),
        "derive_manifest_n_eval": manifest_split.get("n_eval"),
        "why": "the reach terciles were fitted on the training split only; an eval row in "
               "training makes the dial's bucket boundaries partly a function of the "
               "population task 4 measures",
    }


def build_arm_examples(skits: Sequence[Skit], tok, *, arm: str,
                       pad_token_id: int) -> List[dict]:
    """One `{"input_ids", "labels", "story_id"}` per skit, in the given order.

    ``with_think=True`` FOR BOTH ARMS -- see the module docstring. The arm changes the block
    SCHEMA (via `skit_for_arm`), never the block's presence, so both arms get all nine
    segments and the same positional supervision mask.

    `story_id` rides along for provenance; `sft_collate_fn` reads only `input_ids` and
    `labels` and ignores anything else. It is what makes "both arms consumed the same skits
    in the same order" checkable from the two runs' artifacts instead of assumed.
    """
    with_dial = with_dial_for_arm(arm)
    out: List[dict] = []
    for skit in skits:
        ex = build_skit_example(skit_for_arm(skit, with_dial=with_dial), tok,
                               with_think=True, pad_token_id=pad_token_id)
        ex["story_id"] = skit.story_id
        out.append(ex)
    return out


def over_length_indices(skits: Sequence[Skit], tok, *, pad_token_id: int,
                        max_seq_len: int = MAX_SEQ_LEN) -> List[int]:
    """Positions whose built example exceeds `max_seq_len` IN EITHER ARM. The UNION.

    WHY THIS EXISTS: THE DERIVATION'S LENGTH GATE MEASURED THE WRONG BLOCK
    ---------------------------------------------------------------------
    `scripts/derive_dialogue_skits.py::screen_candidate` excludes a skit whose built example
    exceeds the window -- ruling C, "Excluded, NOT truncated" -- and the derive manifest
    accordingly reports ``token_lengths.over_max_seq_len: 0`` and ``max: 512``. But that gate
    runs BEFORE `train.reach.with_reach`, because `with_reach` needs the tercile cut points,
    which are not fitted until every candidate has been collected. So the length it measured
    is the FIVE-slot, label-`stakes` block, and the rows that shipped carry a six-slot block
    with a numeric `stakes`. Both changes add tokens after the gate had already passed them.

    Measured on the shipped artifact, tile-padded, at max_seq_len=512:

        dial   arm: 526 of 36,913 examples over (1.43%), max 544
        nodial arm: 180 of 36,913 examples over (0.49%), max 544

    `sft_collate_fn` truncates those silently, and what a skit loses off its tail is its
    FINAL SUPERVISED MODEL TURN. Two reasons that is not acceptable here rather than merely
    noted:

      * it is partial data loss the artifact believes it already excluded, so nothing
        downstream would think to look for it;
      * it falls THREE TIMES harder on the dial arm, because the dial arm's block is the
        longer one. That is a systematic difference between the arms which is NOT the dial --
        precisely the confound the pairing discipline exists to prevent.

    So ruling C is applied here, at the layer where it failed, and applied to the UNION
    across both arms rather than per-arm. The union is what keeps the arms paired: dropping
    each arm's own over-length rows would leave the two arms with different lengths, hence
    different dataloader permutations, hence no pairing at all. Both runs compute the same
    union from the same file and therefore drop the same positions.

    Building both arms' examples to get the union costs one extra CPU tokenization pass and
    no device time.
    """
    over: List[int] = []
    for i, skit in enumerate(skits):
        for with_dial in (True, False):
            ex = build_skit_example(skit_for_arm(skit, with_dial=with_dial), tok,
                                    with_think=True, pad_token_id=pad_token_id)
            if len(ex["input_ids"]) > max_seq_len:
                over.append(i)
                break
    return over


def drop_positions(skits: Sequence[Skit], positions: Sequence[int]) -> List[Skit]:
    """`skits` without `positions`, order otherwise preserved.

    A separate named function so "the two arms dropped the same rows" is a property of a
    tested transformation rather than of two copies of a list comprehension.
    """
    drop = set(positions)
    return [s for i, s in enumerate(skits) if i not in drop]


def block_schema_report(skit: Skit) -> Dict[str, Any]:
    """The two arms' rendered slot orders, read off the objects the trainer will render.

    Goes in the manifest so "the arms differ in the `reach` line and nothing else" is a
    recorded fact about this run rather than a claim in a docstring. Read off
    `skit_for_arm`'s output, not off the module constants, so a bug in `skit_for_arm` shows
    up here.
    """
    dial = skit_for_arm(skit, with_dial=True).blocks[0]
    nodial = skit_for_arm(skit, with_dial=False).blocks[0]
    # Pin both against the schema module's constants. Without this the report would
    # faithfully describe whatever `skit_for_arm` produced, including a wrong thing.
    if reach_slot_names_of(dial) != REACH_SLOT_NAMES:
        raise RuntimeError(f"dial block renders {reach_slot_names_of(dial)}, expected "
                           f"{REACH_SLOT_NAMES}")
    if reach_slot_names_of(nodial) != NODIAL_SLOT_NAMES:
        raise RuntimeError(f"nodial block renders {reach_slot_names_of(nodial)}, expected "
                           f"{NODIAL_SLOT_NAMES}")
    return {
        "dial_slots": list(reach_slot_names_of(dial)),
        "nodial_slots": list(reach_slot_names_of(nodial)),
        "difference": sorted(set(reach_slot_names_of(dial))
                             - set(reach_slot_names_of(nodial))),
        "reach_before_add": (list(reach_slot_names_of(dial)).index("reach")
                             < list(reach_slot_names_of(dial)).index("add")),
        "reach_before_add_note": "load-bearing: the block generates left to right, so a "
                                 "dial declared after the word it governs cannot govern it",
    }


# --------------------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Repo-relative path when the path is under the repo, absolute otherwise."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


class _EarlyStop(Exception):
    """Raised from inside `EarlyStopOnPlateau.on_eval_end` to break `SFTTrainer.train()`.

    `SFTTrainer.train()` iterates `for _ in tqdm(range(self.step, cfg.max_steps))` and
    offers no `should_stop` flag, no early-exit return, and no way for a callback to
    influence the loop -- the hooks are fire-and-forget. An exception is therefore the
    only mechanism available, and it is used deliberately rather than as a shortcut:
    control returns to `main`, which records WHY it stopped and saves a checkpoint at the
    stopping step. Callers MUST catch it; an uncaught `_EarlyStop` would unwind past
    `close_device_mesh` teardown as an ordinary failure.
    """

    def __init__(self, step: int, reason: str) -> None:
        super().__init__(f"early stop at step {step}: {reason}")
        self.step = step
        self.reason = reason


class EarlyStopOnPlateau:
    """Stop training when validation loss has not improved for `patience` consecutive evals.

    WHY THIS EXISTS
    ---------------
    The first pair of arms trained a flat 3000 steps -- an INHERITED budget matched to
    stage 2, not a convergence criterion -- and both were still falling monotonically at
    the end (dial 1.2046 -> 0.7124, nodial 1.2795 -> 0.7514). Every effect measured on
    those arms is therefore a FLOOR, and one pre-declared adherence gate that failed may
    be an undertraining artefact rather than a property of the dial. A budget that ends
    when the val curve stops improving replaces "we ran out of steps" with a statement
    about the model.

    WHAT COUNTS AS AN IMPROVEMENT
    -----------------------------
    Strictly lower than the best validation loss seen so far (`min_delta` = 0.0 by
    default). The counter resets on every improvement, so `patience` counts CONSECUTIVE
    non-improving evals, not cumulative ones -- a curve that ratchets down slowly, one
    improvement every few evals, keeps training. That is the intended behaviour: the
    question is whether the curve has flattened, and a curve still finding new minima has
    not.

    Note that the run therefore ends `patience * eval_interval` steps AFTER the best
    checkpoint. `best_step` / `best_loss` are recorded so the distinction between "the
    step we stopped at" and "the step that was best" stays visible in the manifest rather
    than being collapsed into one number.

    Duck-typed on `TrainerCallback` for the same reason `LossRecorder` is: the trainer
    calls exactly the hooks it fires and nothing enforces a base class.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_loss: float | None = None
        self.best_step: int | None = None
        #: consecutive evals since the last improvement
        self.since_best = 0
        self.stopped_at: int | None = None

    # -- hooks the trainer fires; all no-ops but on_eval_end ------------------------
    def on_train_begin(self, trainer) -> None:
        pass

    def on_after_forward(self, trainer, batch, loss) -> None:
        pass

    def on_after_backward(self, trainer, batch) -> None:
        pass

    def on_step_end(self, trainer, step, *args, **kwargs) -> None:
        pass

    def on_before_optimizer_step(self, trainer) -> None:
        pass

    def on_save(self, trainer, step, path) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_eval_end(self, trainer, step, eval_loss) -> None:
        loss = float(eval_loss)
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_step = int(step)
            self.since_best = 0
            return
        self.since_best += 1
        # patience == 0 disables the stopper entirely: nothing to compare against yet on
        # the first eval, and a zero budget for non-improvement would end every run at its
        # second eval point.
        if self.patience > 0 and self.since_best >= self.patience:
            self.stopped_at = int(step)
            raise _EarlyStop(
                int(step),
                f"validation loss has not improved for {self.since_best} consecutive "
                f"evals (best {self.best_loss} at step {self.best_step})",
            )

    def report(self) -> Dict[str, Any]:
        """The stopper's state, for the manifest."""
        return {
            "patience_evals": self.patience,
            "min_delta": self.min_delta,
            "best_val_loss": self.best_loss,
            "best_val_step": self.best_step,
            "evals_since_best_at_end": self.since_best,
            "triggered": self.stopped_at is not None,
            "triggered_at_step": self.stopped_at,
            "rule": "stop when validation loss has not been beaten for `patience` "
                    "CONSECUTIVE evals; the counter resets on every new minimum",
        }


def _order_fingerprint(loader, epoch: int = 0) -> Dict[str, Any]:
    """A checkable fingerprint of the order this loader will visit examples in.

    Read off the CONSTRUCTED loader (`_epoch_indices`), not recomputed from a copy of its
    permutation formula: a copy would agree with itself forever, including after ttml
    changed how it shuffles. Both arms record this; identical fingerprints across the two
    manifests is the evidence that the comparison is genuinely paired. Private attribute, so
    its absence is reported rather than raised -- a missing fingerprint must not kill a
    3000-step run, it just leaves the pairing unfingerprinted.
    """
    if not hasattr(loader, "_epoch_indices"):
        return {"source": "unavailable (loader has no _epoch_indices)", "sha256": None}
    idx = np.asarray(loader._epoch_indices(epoch))
    return {
        "source": "loader._epoch_indices(0)",
        "n": int(idx.size),
        "first_8": [int(i) for i in idx[:8]],
        "sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=list(ARMS),
                    help="which paired arm to train (required unless --verify-gammas)")
    ap.add_argument("--skits", type=Path, default=DEFAULT_SKITS,
                    help="reach skits.jsonl to train on (default: the full-corpus task-2 "
                         "derivation)")
    ap.add_argument("--derive-manifest", type=Path, default=None,
                    help="derive_manifest.json to cross-check the split counts against "
                         "(default: beside --skits)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--keep-over-length", action="store_true",
                    help="do NOT re-apply ruling C -- keep the rows whose example exceeds "
                         "max_seq_len and let sft_collate_fn truncate them silently. "
                         "Present so the behaviour is a recorded choice rather than a "
                         "hidden default; see over_length_indices for why the default is "
                         "to drop them.")
    ap.add_argument("--val-cap", type=int, default=256,
                    help="how many of the HELD-OUT rows feed the in-training validation "
                         "curve (head of the eval split; 0 = no curve). This does NOT move "
                         "the split -- every eval-labelled row stays out of training "
                         "either way.")
    ap.add_argument("--eval-every", type=int, default=250,
                    help="SFTConfig.eval_interval -- steps between validation passes. "
                         "0 disables evaluation entirely, which is what silently cost "
                         "stage 1 its whole validation curve.")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop when validation loss has not improved for this many "
                         "CONSECUTIVE evals (0 = never stop early, run the full --steps). "
                         "At --eval-every 250 a patience of 6 is 1500 steps of no new "
                         "minimum. Requires a validation curve (--val-cap > 0); with no "
                         "eval dataloader no eval ever fires and this can never trigger.")
    ap.add_argument("--warm-start", type=Path, default=WARM_START_CKPT)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the examples, run the split guard and report the length "
                         "distribution, then stop. Touches no device, needs no lease.")
    ap.add_argument("--verify-gammas", type=Path, default=None,
                    help="compare an existing step_*.pkl's gammas against the warm-start "
                         "base and exit. Touches no device.")
    args = ap.parse_args(argv)

    if args.verify_gammas is not None:
        return verify_gammas_cli(args.verify_gammas, args.warm_start)
    if args.arm is None:
        ap.error("--arm is required unless --verify-gammas is given")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_token_id = tok.pad_token_id or 0

    rows = load_reach_skits(args.skits)
    train_skits, eval_skits = partition_by_split(rows)
    manifest_path = (args.derive_manifest if args.derive_manifest is not None
                     else Path(args.skits).parent / "derive_manifest.json")
    split_report = assert_split_respected(rows, train_skits, eval_skits,
                                          manifest_split=read_manifest_split(manifest_path),
                                          arm=args.arm)
    layout = split_layout(rows)

    # Ruling C, applied at the layer where the derivation's version of it failed. See
    # `over_length_indices`: the derivation's gate measured the pre-`with_reach` block, so
    # 526 dial / 180 nodial examples would in fact be silently truncated, unequally between
    # the arms. The UNION is dropped from both arms so they stay the same length and
    # therefore the same permutation.
    over_positions = ([] if args.keep_over_length
                      else over_length_indices(train_skits, tok,
                                               pad_token_id=pad_token_id))
    kept_skits = drop_positions(train_skits, over_positions)
    print(f"[{args.arm}] ruling C (re-applied): dropped {len(over_positions):,} of "
          f"{len(train_skits):,} training rows whose example exceeds {MAX_SEQ_LEN} tokens "
          f"IN EITHER ARM (the union, so both arms drop the same rows); "
          f"{len(kept_skits):,} remain")

    train_examples = build_arm_examples(kept_skits, tok, arm=args.arm,
                                        pad_token_id=pad_token_id)
    # The curve's eval set is the HEAD of the held-out rows. Capping shortens the curve, not
    # the split: `eval_skits` is already entirely out of `train_examples` above.
    val_cap = max(0, min(args.val_cap, len(eval_skits)))
    # The held-out rows get the same treatment, for the same reason: a truncated validation
    # example measures a scene that stops mid-turn. Screened over the capped head only.
    val_head = eval_skits[:val_cap]
    val_over = ([] if args.keep_over_length
                else over_length_indices(val_head, tok, pad_token_id=pad_token_id))
    val_examples = build_arm_examples(drop_positions(val_head, val_over), tok,
                                      arm=args.arm, pad_token_id=pad_token_id)

    stats = length_stats(train_examples)
    schema = block_schema_report(train_skits[0]) if train_skits else {}
    print(f"arm={args.arm}  skits={args.skits}  rows={len(rows):,}  "
          f"train={len(train_examples):,}  val={len(val_examples):,} "
          f"(of {len(eval_skits):,} held out)  seed={args.seed}")
    print(f"block schema: dial={schema.get('dial_slots')}  "
          f"nodial={schema.get('nodial_slots')}  difference={schema.get('difference')}")
    print(f"token lengths: min {stats['min']}  median {stats['median']}  "
          f"p99 {stats['p99']}  max {stats['max']}  "
          f">{MAX_SEQ_LEN}: {stats['over_max_seq_len']} "
          f"({stats['over_max_seq_len_frac']:.2%})")
    if stats["over_max_seq_len"]:
        print(f"NOTE: {stats['over_max_seq_len']} example(s) exceed max_seq_len="
              f"{MAX_SEQ_LEN} and WILL be truncated by sft_collate_fn, losing their final "
              f"supervised turn. The dial arm's examples are longer by construction (one "
              f"extra slot line), so this does not fall equally on the two arms. Recorded "
              f"in the manifest.")

    if args.dry_run:
        print("dry run: no device opened, nothing trained.")
        return 0

    import ttml  # noqa: F401 -- opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    # SFTTrainer.__init__ -> _build_loss_fn() calls ttml.mesh(), a module-global that only
    # open_device_mesh populates. This repo's initialize_device()/AutoContext.open_device()
    # opens a device but leaves that global empty ("Device mesh is not initialized").
    # (1, 1) ONLY: one board, one chip's worth of mesh. A 4-chip open hard-froze this host.
    yaml_config: dict = {}
    ttml.open_device_mesh((1, 1))
    try:
        collate = partial(sft_collate_fn, max_seq_len=MAX_SEQ_LEN,
                          pad_token_id=pad_token_id)
        loader = InMemoryDataloader(train_examples, batch_size=args.batch_size,
                                    collate_fn=collate, shuffle=True, seed=args.seed)
        val_loader = (InMemoryDataloader(val_examples, batch_size=args.batch_size,
                                        collate_fn=collate, shuffle=False, seed=args.seed)
                      if val_examples else None)
        order = _order_fingerprint(loader)
        print(f"batch order fingerprint (must match the other arm): {order}")

        out = Path(args.out_root).resolve() / f"ckpt-{args.arm}"
        out.mkdir(parents=True, exist_ok=True)

        from train.model import create_model

        model_yaml = yaml.safe_load(MODEL_YAML.read_text())
        transformer_config = model_yaml["transformer_config"]
        model = create_model(yaml_config, transformer_config)

        # Dense model, dense checkpoint, identical shape: every parameter name is shared,
        # so moe_block_indices=[] and the whole checkpoint transfers with nothing fresh.
        from train.enthusiasts import warm_start

        warm_summary = warm_start(
            model, Path(args.warm_start),
            transformer_config=transformer_config, yaml_config=yaml_config,
            moe_block_indices=[],
        )
        print(f"warm start: {warm_summary}")

        curve_path = out / "loss_curve.jsonl"
        recorder = LossRecorder(curve_path)
        stopper = (EarlyStopOnPlateau(args.early_stop_patience)
                   if args.early_stop_patience > 0 and val_loader is not None else None)
        if args.early_stop_patience > 0 and val_loader is None:
            raise SystemExit(
                "--early-stop-patience was given but there is no validation dataloader "
                "(--val-cap 0, or no held-out rows survived). Early stopping would then "
                "silently never trigger and the run would look like a full-budget run "
                "that simply never converged.")
        trainer = SFTTrainer(
            model=model, train_dataloader=loader, eval_dataloader=val_loader,
            config=SFTConfig(max_steps=args.steps, learning_rate=args.lr, seed=args.seed,
                             max_seq_len=MAX_SEQ_LEN, checkpoint_dir=str(out),
                             save_interval=args.save_every,
                             eval_interval=args.eval_every if val_loader else 0,
                             log_interval=1,  # per-step curve: fires on_step_end every step
                             max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.01,
                       "stochastic_rounding": True},
            callbacks=[recorder, stopper] if stopper is not None else [recorder],
        )
        assert_stochastic_rounding(trainer, args.arm)
        assert_eval_wired(trainer, val_size=len(val_examples), arm=args.arm)

        # `SFTTrainer.train()` has no early-exit path, so the stopper breaks out by
        # raising. Catching it here -- and nowhere wider -- keeps a genuine failure
        # distinguishable from a deliberate stop, and leaves the `finally:` device
        # teardown below in charge either way.
        stop_reason = "hit_max_steps"
        try:
            trainer.train()
        except _EarlyStop as stop:
            stop_reason = "converged"
            print(f"[{args.arm}] EARLY STOP: {stop}")
        recorder.close()
        stopping_step = int(trainer.step)
        print(f"[{args.arm}] stopped at step {stopping_step} of a {args.steps}-step "
              f"budget; reason={stop_reason}")
        if stop_reason == "hit_max_steps" and stopper is not None:
            print(f"[{args.arm}] NOTE: the budget was the binding constraint, not "
                  f"convergence -- validation loss last improved at step "
                  f"{stopper.best_step} ({stopper.since_best} eval(s) ago).")

        # `save_interval` only fires on multiples of --save-every, so an early stop at, say,
        # step 4250 would leave no checkpoint AT the stopping step and the gamma guard below
        # nothing to check. Save one explicitly; `_save_checkpoint` writes to
        # `step_{trainer.step}.pkl`, which is exactly the path the guard then reads.
        final_ckpt = out / f"step_{stopping_step}.pkl"
        if not final_ckpt.exists():
            trainer._save_checkpoint()
            print(f"[{args.arm}] saved stopping-step checkpoint {final_ckpt}")

        loss_start = recorder.history[0] if recorder.history else (None, None)
        loss_end = recorder.history[-1] if recorder.history else (None, None)
        print(f"loss: step {loss_start[0]}={loss_start[1]}  ->  "
              f"step {loss_end[0]}={loss_end[1]}")
        if recorder.eval_history:
            print(f"val loss: step {recorder.eval_history[0][0]}="
                  f"{recorder.eval_history[0][1]}  ->  step "
                  f"{recorder.eval_history[-1][0]}={recorder.eval_history[-1][1]}")

        # --- guard 2: did the gammas actually move? -------------------------------------
        # `final_ckpt` is the STOPPING-step checkpoint, saved above -- not step_{--steps},
        # which an early-stopped run never reaches.
        gamma_report: Dict[str, Any]
        if final_ckpt.exists():
            gamma_report = compare_gammas(read_base_gammas(Path(args.warm_start)),
                                          read_arm_gammas(final_ckpt))
        else:
            gamma_report = {"error": f"no final checkpoint at {final_ckpt}",
                            "all_moved": False, "frozen": ["<checkpoint missing>"],
                            "n_tensors": 0}

        manifest = {
            "task": "reach dial, task 3 -- two paired SFT arms over reach-dial skits",
            "arm": args.arm,
            "arm_meaning": ("the think-block carries the `reach` dial"
                            if args.arm == "dial" else
                            "NEGATIVE CONTROL: the identical think-block with the `reach` "
                            "line removed and nothing else changed. Forced `reach` tokens "
                            "should do NOTHING to this arm; that is the guard against "
                            "'any injected token perturbs generation'."),
            "paired_with": "nodial" if args.arm == "dial" else "dial",
            "seed": args.seed,
            "skits": _rel(Path(args.skits)),
            "derive_manifest": _rel(manifest_path),
            "rows_read": len(rows),
            "n_examples": len(train_examples),
            "ruling_c_reapplied": {
                "applied": not args.keep_over_length,
                "max_seq_len": MAX_SEQ_LEN,
                "training_rows_before": len(train_skits),
                "training_rows_dropped": len(over_positions),
                "training_rows_after": len(train_examples),
                "val_rows_dropped": len(val_over),
                "rule": "drop a row whose built example exceeds max_seq_len IN EITHER ARM "
                        "(the union), so both arms drop the same rows and stay paired",
                "why": "the DERIVATION's length gate (screen_candidate, ruling C) ran "
                       "BEFORE train.reach.with_reach, because with_reach needs the tercile "
                       "cut points and those are not fitted until every candidate is "
                       "collected. It therefore measured the FIVE-slot, label-stakes block, "
                       "while the rows that shipped carry a six-slot block with a numeric "
                       "stakes. Both changes add tokens after the gate passed them, so "
                       "derive_manifest.json's token_lengths.over_max_seq_len: 0 and "
                       "max: 512 do NOT describe what trains here.",
                "measured_before_dropping": {
                    "dial_over_max_seq_len": 526,
                    "nodial_over_max_seq_len": 180,
                    "of_training_rows": 36913,
                    "max_tokens_either_arm": 544,
                    "note": "526 vs 180 is why this is dropped rather than noted: silent "
                            "truncation costs a skit its FINAL SUPERVISED MODEL TURN and "
                            "falls three times harder on the dial arm, which is a "
                            "systematic difference between the arms that is not the dial.",
                },
            },
            "n_val_examples": len(val_examples),
            "n_held_out": len(eval_skits),
            "split": split_report,
            "split_layout": layout,
            "val_curve_note": (f"the in-training validation curve uses the FIRST "
                               f"{len(val_examples)} of the {len(eval_skits)} held-out "
                               f"rows (--val-cap). Capping shortens the curve, not the "
                               f"split: every eval-labelled row is out of training "
                               f"regardless."),
            "block_schema": schema,
            "steps": args.steps,
            "steps_note": ("--steps is a CEILING, not a schedule. The first pair of arms "
                           "ran a flat 3000 steps -- an inherited budget matched to stage "
                           "2 for comparability, not a convergence criterion -- and both "
                           "were still falling monotonically at the end. This run stops on "
                           "the validation curve instead; see stop_reason."
                           if stopper is not None else
                           "no early-stopping criterion was in force: this run trained the "
                           "full --steps budget by construction, so its endpoint says "
                           "nothing about convergence."),
            "max_steps_budget": args.steps,
            "stopping_step": stopping_step,
            "stop_reason": stop_reason,
            "stop_reason_meaning": {
                "converged": "validation loss went --early-stop-patience consecutive "
                             "evals without a new minimum, and training was cut short of "
                             "--steps",
                "hit_max_steps": "the --steps ceiling was reached first. If an early-stop "
                                 "criterion was in force and did NOT fire, the BUDGET is "
                                 "still the binding constraint and this run, like the "
                                 "3000-step arms before it, demonstrates no convergence.",
            }[stop_reason],
            "early_stopping": (stopper.report() if stopper is not None
                               else {"enabled": False}),
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_interval": args.eval_every if val_loader else 0,
            "log_interval": 1,
            # Declared here AND verified before training via
            # trainer.optimizer.get_state_dict(), then verified again after training by
            # comparing the 17 gammas against the warm-start base (gamma_check below).
            "stochastic_rounding": True,
            "max_seq_len": MAX_SEQ_LEN,
            "mesh": "(1,1)",
            "tokenization": TOKENIZATION_NOTE,
            "example_token_lengths": stats,
            "batch_order_fingerprint": order,
            "warm_start": _rel(Path(args.warm_start)),
            "warm_start_summary": warm_summary,
            "checkpoint_dir": _rel(out),
            "loss_curve": _rel(curve_path),
            "loss_curve_note": "every step logged (not sampled); split=train|val",
            "loss_comparability_WARNING":
                "DO NOT COMPARE THIS ARM'S LOSS WITH THE OTHER ARM'S. The two arms "
                "supervise DIFFERENT TOKEN SETS -- the `dial` arm's block carries an extra "
                "`reach: ...` line, so its sequences are longer and its per-token loss is "
                "averaged over a different population. These numbers monitor this run and "
                "support no cross-arm inference whatsoever.",
            "loss_step_start": loss_start[0],
            "loss_start": loss_start[1],
            "loss_step_end": loss_end[0],
            "loss_end": loss_end[1],
            "val_loss_first": (recorder.eval_history[0] if recorder.eval_history
                               else None),
            "val_loss_last": (recorder.eval_history[-1] if recorder.eval_history
                              else None),
            #: The whole validation trajectory, so "did it flatten?" is answerable from
            #: the manifest alone rather than only from the interleaved per-step curve.
            "val_loss_curve": [[int(st), float(lo)] for st, lo in recorder.eval_history],
            "final_checkpoint": _rel(final_ckpt),
            "gamma_check": gamma_report,
        }
        (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {out / 'train_manifest.json'}")

        # Manifest first, then raise: a void run must still leave its evidence on disk.
        assert_gammas_moved(gamma_report, arm=args.arm, base_path=Path(args.warm_start),
                            arm_path=final_ckpt)
    finally:
        # Mirror of open_device_mesh; also clears ttml's global mesh state. Skipping the
        # teardown aborts in MetalContext::destroy_all_instances.
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
