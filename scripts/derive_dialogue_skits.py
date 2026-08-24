#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Corpus -> REAL two-speaker dialogue skits -> SFT examples.

WHAT IS DIFFERENT FROM scripts/derive_skits.py, AND WHAT IS DELIBERATELY THE SAME
================================================================================
Different: where the five turns come from. `derive_skits.py` slices consecutive sentences,
so its "partner" turn is the same narrator's next sentence -- which is why
`handback_anticipation` topped out at **0.119 on ground truth** in stage 2. There was no
second voice to anticipate. Here the turns are real quoted utterances, alternating by
POSITION (`train.dialogue.extract_dialogue_turns`), and the prefix is the story text before
the first one.

The same, by IMPORT rather than by copy -- because "identical" asserted in a docstring is
how two pipelines drift:

  * `build_skit_example` (the pre-shifted label rule, the positional nine-segment
    supervision mask, and `TILE` alignment) is imported from `scripts.derive_skits`.
  * `Skit`, `skit_segments`, `MODEL_TURNS`, `PARTNER_TURNS` and the slot derivation itself
    (`derive_skit_from_turns`) come from `train.skit`.

So the label rule is `labels[t] = ids[t+1] if supervised[t+1] else -100` with
`labels[-1] = -100`; the prefix and BOTH partner turns are never supervised; every example
is a multiple of 32 tokens. Those are one implementation, shared, and
`tests/test_derive_dialogue_skits.py` re-derives the spans independently to check them here
too rather than trusting that sharing.

SELECTION BIAS IS PART OF THE OUTPUT, NOT A FOOTNOTE
----------------------------------------------------
Requiring five quoted utterances keeps a MINORITY of stories -- the dialogue-heavy tail.
Stage 2's artifact omitted its drop rate and its selection bias until a final review caught
both, so this script measures them and writes them into the manifest unconditionally:
`selection_bias` carries the corpus's own dialogue rate beside the kept population's, and
`drop_rate_warning` prints the >50% notice to stdout exactly as `derive_skits.py` does.

WHAT TASK 2 (THE REACH DIAL) ADDED
----------------------------------
The block gained a sixth slot, and three gates gained a reason to exist. In order:

* **`reach`**, the dial: the tercile of how far the `add` word is from what is already in
  play (`train.reach`). Rendered BEFORE `add`, because the block is generated left to right
  and a dial declared after the word it governs could only relabel a choice already made.
  The cut points are fitted on the TRAINING SPLIT ONLY and written into the manifest, so eval
  cannot silently re-fit them -- a dial whose buckets move between train and eval measures
  nothing. That is also why derivation is now TWO PASSES: nothing can be bucketed until the
  scan is finished and the split is known.
* **`stakes` as a continuous signed delta** (ruling D), replacing the up/level/down label
  stage 2 withdrew as mostly confounded. Not a headline slot; carried so the withdrawal has a
  successor rather than a silent disappearance.
* **`same_voice_pair`** (ruling A): drop the skit when any adjacent pair shows no evidence
  that the voice changed. The premise of this whole path is that the partner turn is a
  DIFFERENT voice, so it is worth yield to protect. The filter never decides WHO speaks --
  see `train.dialogue.same_voice_risk`, including what it provably cannot catch.
* **`over_max_seq_len`** (ruling C): exclude, do not truncate. `sft_collate_fn` truncates
  silently and a truncated example teaches a scene that stops mid-turn.
* **`reach_no_evidence`**: drop the skit when a block's `add` word has no co-occurrence with
  anything in play, because `npmi` returns 0.0 both for "distant" and for "never seen" and
  binning the second as `far` would make the dial partly a dial on our own ignorance.

Output goes to a NEW path (`artifacts/reach-skits/`). Task 1's `artifacts/dialogue-skits/` is
left exactly as it was.

    # the shipped run: whole corpus (ruling B), ruling C armed by its tokenizer
    python3 scripts/derive_dialogue_skits.py \
        --tokenizer artifacts/hf-tt-tnt-1024-dialogue
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_skits import TILE, build_idf, build_skit_example  # noqa: E402,F401
from scripts.score_improv import intensity, load_harm_lexicon  # noqa: E402
from train.dialogue import (MIN_UTTERANCES, adjacent_gaps,  # noqa: E402
                            dialogue_prefix, extract_dialogue_turns, is_tag_only_gap,
                            quoted_utterances, same_voice_risk,
                            split_sentences_dialogue, voice_changes_throughout)
from train.improv import content_words  # noqa: E402
from train.reach import (REACH_SLOT_NAMES, REACH_VALUES, Association,  # noqa: E402
                         bucket_balance, build_association, fit_reach_terciles,
                         skit_reach_distances, skit_reach_distances_per_block,
                         skit_stakes_deltas, with_reach)
from train.skit import (MODEL_TURNS, PARTNER_TURNS, SKIT_ROLES,  # noqa: E402,F401
                        Skit, derive_skit_from_turns, skit_segments)

STORY_SEP = "</s>"

#: Chunk size for `iter_stories`. The corpus is 1.9 GB and a run at `--limit 400000` needs
#: only its first ~18%, so the file is streamed rather than read whole.
_READ_CHUNK = 1 << 22


def iter_stories(path: Path, limit: Optional[int] = None) -> Iterator[str]:
    """Stories from a corpus file, streamed, stripped, empties skipped.

    Separator is `</s>`, NOT a blank line -- TinyStories paragraphs are blank-line separated
    WITHIN a story, so splitting on blank lines silently triples the story count and every
    per-story rate computed from it.

    Streaming is not premature: reading 1.9 GB with `read_text().split()` costs ~4 GB of
    peak RSS for a run that needs the first 18% of it.
    """
    seen = 0
    buf = ""
    with path.open("r", errors="ignore") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(STORY_SEP)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                yield p
                seen += 1
                if limit is not None and seen >= limit:
                    return
    if buf.strip() and (limit is None or seen < limit):
        yield buf.strip()


def md5_of(path: Path) -> str:
    """md5 of the corpus file, so a manifest names a specific bytes-on-disk, not a path."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repo_relative(path: Path) -> str:
    """`path` relative to the repo root when it is inside it, else its absolute form.

    Manifests are read on other machines and get diffed against each other; an absolute
    `/home/<someone>/code/...` makes two identical derivations look different.
    """
    p = path.resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------------------
# decision functions -- each named, each with its own fixture test
# --------------------------------------------------------------------------------------
def dialogue_unit_counts(story: str) -> Tuple[int, int]:
    """`(units, units containing a quoted utterance)` for one story.

    "Unit" is a sentence as `split_sentences_dialogue` sees it, which is the same unit the
    stage-2 selection-bias numbers were quoted in (54.6% of corpus units carried dialogue;
    the eval population had 31.0%). This is the CORPUS BASELINE half of the retention
    measurement -- it is computed over every story scanned, kept or dropped.
    """
    units = split_sentences_dialogue(story)
    return len(units), sum(1 for u in units if '"' in u)


def classify_turn_failure(prefix: str, turns: Sequence[str]) -> str:
    """Which gate a would-be skit failed, for the drop table. READ-ONLY DIAGNOSTICS.

    Nothing gates on this. `derive_skit_from_turns` has already decided (whole-or-nothing,
    by design), and this re-walks its two per-turn conditions -- `accept` (a content word
    carried from the offer) and `add` (a content word new to the scene) -- only to say WHICH
    one failed and at WHICH model turn. Same pattern, and same justification, as
    `derive_skits.py`'s `dialogue_pattern` recompute: a drop table that says only
    "derivation failed" cannot tell you whether to fix the gate or accept the yield.

    Returns e.g. `"no_accept_at_turn_2"`, `"no_add_at_turn_0"`, `"no_content_at_turn_4"`, or
    `"unclassified"` if every turn passes both re-walked conditions (which would mean this
    diagnostic and the real derivation disagree -- worth seeing in the table).
    """
    for t_idx in MODEL_TURNS:
        offer = turns[t_idx - 1] if t_idx > 0 else prefix
        turn_words = content_words(turns[t_idx])
        if not turn_words:
            return f"no_content_at_turn_{t_idx}"
        if not [w for w in content_words(offer) if w in set(turn_words)]:
            return f"no_accept_at_turn_{t_idx}"
        established = set(content_words(prefix))
        for j in range(t_idx):
            established |= set(content_words(turns[j]))
        if not [w for w in turn_words if w not in established]:
            return f"no_add_at_turn_{t_idx}"
    return "unclassified"


def tag_only_gap_count(story: str, n_turns: int = MIN_UTTERANCES) -> Tuple[int, int]:
    """`(adjacent pairs, pairs separated by nothing but an attribution tag)`.

    An UPPER BOUND on how often two adjacent turns are the SAME voice -- the one thing
    alternation-by-position cannot verify without the speaker attribution the spec forbids.
    See `train.dialogue.is_tag_only_gap` for why it is an upper bound and not an estimate.
    Reported, never gated on.
    """
    utts = quoted_utterances(story)[:n_turns]
    pairs = max(len(utts) - 1, 0)
    tag_only = sum(1 for k in range(pairs)
                   if is_tag_only_gap(story[utts[k].end:utts[k + 1].start]))
    return pairs, tag_only


def token_length_report(lengths: List[int], max_seq_len: int) -> dict:
    """Length percentiles for the built examples, and how many the trainer would TRUNCATE.

    This exists because of a stage-1 finding: `sft_collate_fn` silently truncates anything
    past `max_seq_len` -- no raise, no skip, no log -- and 159 stage-1 examples were being
    cut with that fact appearing in no drop table. The dialogue path makes the risk LARGER,
    not smaller: its prefix is everything before the first quote, which can be three
    paragraphs where `derive_skit`'s prefix was always two sentences. So the number is
    measured here, at derivation time, rather than discovered later.

    `lengths` are tile-aligned `input_ids` lengths from `build_skit_example` with
    `with_think=True` (the longer arm). Returns `None`-free JSON-able values, or an empty
    report when no lengths were collected (the flag is optional).
    """
    if not lengths:
        return {"measured": False,
                "note": "pass --tokenizer to measure; unmeasured means the 512-token "
                        "truncation risk is UNKNOWN, not absent."}
    s = sorted(lengths)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(p * (len(s) - 1)))]

    over = sum(1 for v in s if v > max_seq_len)
    return {"measured": True, "n": len(s), "arm": "with_think (the longer arm)",
            "max_seq_len": max_seq_len,
            "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99), "max": s[-1],
            "over_max_seq_len": over,
            "over_max_seq_len_fraction": round(over / len(s), 4),
            "note": "sft_collate_fn TRUNCATES anything longer than max_seq_len silently. "
                    "A non-zero over_max_seq_len is a real, if partial, data loss."}


def drop_rate_warning(rate: float) -> Optional[str]:
    """The >50% notice, as a value rather than a `print` buried in `main`.

    `derive_skits.py:164` prints this inline; here it is a function so a test can assert
    that a 90% drop rate DOES warn and a 10% one does not, without running a derivation.
    Above this line the filter, not the model, is choosing the behaviour -- and for this
    script the drop rate is expected to be ~80%, so the notice is the normal case and must
    reach both stdout and the manifest.
    """
    if rate > 0.5:
        return (f"WARNING: drop rate {rate:.1%} is above 50% — the FILTER is choosing the "
                f"behaviour, not the model. Report this with any result.")
    return None


def selection_bias(counts: Dict[str, int]) -> dict:
    """The manifest's `selection_bias` block: what requiring 5 utterances did to the sample.

    Pure function of the accumulated counters, so the published numbers are computed by
    something a fixture can call. Every rate carries its own numerator and denominator in
    the same block; a rate that is degenerate by construction says so in its own row rather
    than in a footnote nobody reads.

    Keys, and what each one answers:
      * `stories_with_any_dialogue_fraction` -- how much of the corpus has ANY dialogue.
      * `stories_with_min_utterances_fraction` -- how much of it clears this script's gate.
      * `corpus_dialogue_unit_fraction` -- the baseline the stage-2 population fell 43%
        below: fraction of all sentence-units, over every story scanned, carrying a quote.
      * `kept_turn_dialogue_fraction` -- 1.0, DEGENERATE BY CONSTRUCTION (every kept turn is
        a quoted utterance; that is the point of the script, not a finding).
      * `dialogue_utterance_retention` -- of every quoted utterance in the scanned corpus,
        the fraction that lives in a story we KEPT. This is the honest retention number:
        it is < 1 because dialogue in 1-4-utterance stories is discarded wholesale.
      * `mean_utterances_*` -- how much more dialogue-heavy a kept story is than an average
        one. This is the bias, stated as a ratio rather than implied.
    """
    scanned = max(counts["stories_scanned"], 1)
    units = max(counts["corpus_units"], 1)
    all_utts = max(counts["corpus_utterances"], 1)
    kept = counts["stories_kept"]
    kept_turns = kept * len(PARTNER_TURNS + MODEL_TURNS)
    return {
        "stories_scanned": counts["stories_scanned"],
        "stories_with_any_dialogue": counts["stories_with_any_dialogue"],
        "stories_with_any_dialogue_fraction":
            round(counts["stories_with_any_dialogue"] / scanned, 4),
        "stories_with_min_utterances": counts["stories_with_min_utterances"],
        "stories_with_min_utterances_fraction":
            round(counts["stories_with_min_utterances"] / scanned, 4),
        "min_utterances": MIN_UTTERANCES,
        "corpus_units": counts["corpus_units"],
        "corpus_dialogue_units": counts["corpus_dialogue_units"],
        "corpus_dialogue_unit_fraction":
            round(counts["corpus_dialogue_units"] / units, 4),
        "kept_turns": kept_turns,
        "kept_turn_dialogue_fraction": 1.0 if kept_turns else None,
        "kept_turn_dialogue_fraction_note":
            "DEGENERATE BY CONSTRUCTION: every kept turn is a quoted utterance because "
            "that is the selection rule. Reported so the 1.0 is not mistaken for a "
            "measurement. The comparable corpus figure is corpus_dialogue_unit_fraction.",
        "corpus_utterances": counts["corpus_utterances"],
        "utterances_in_kept_stories": counts["utterances_in_kept_stories"],
        "dialogue_utterance_retention":
            round(counts["utterances_in_kept_stories"] / all_utts, 4),
        "mean_utterances_per_scanned_story":
            round(counts["corpus_utterances"] / scanned, 4),
        "mean_utterances_per_kept_story":
            round(counts["utterances_in_kept_stories"] / kept, 4) if kept else None,
        "bias_statement":
            "Kept stories are the dialogue-heavy tail of the corpus. Any behaviour measured "
            "on skits derived here is measured on that tail, not on TinyStories.",
    }


# --------------------------------------------------------------------------------------
# reach-dial decision functions (task 2) -- each named, each with its own fixture test
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ReachCandidate:
    """A skit that cleared every gate, with the numbers the dial needs, before bucketing.

    `distances` and `deltas` are one per model turn. They are computed in pass 1 and held
    because the tercile cut points can only be fitted once the TRAINING SPLIT is known, and
    the split is the tail of the file in derivation order -- so nothing can be bucketed until
    the scan is finished. `n_tokens` is None when no tokenizer was supplied, which is the same
    thing as ruling C not having been applied; see `ruling_c_report`.
    """
    skit: Skit
    distances: Tuple[float, ...]
    deltas: Tuple[float, ...]
    n_tokens: Optional[int]


def voice_change_audit(story: str, n_turns: int = MIN_UTTERANCES) -> Dict[str, int]:
    """Per-story gap audit: how many adjacent gaps trip each same-voice indicator.

    Three numbers over the same `adjacent_gaps`, so the diagnostic and the gate cannot end up
    looking at different spans:

      * `tag_only`      -- task 1's published UPPER BOUND indicator (39.6% of pairs).
      * `risky_subject` -- the gate this script actually applies: a tag-only gap whose tag
        identifies NOBODY (``he said.``), or no gap at all.
      * `risky_strict`  -- the same gate reading "no NEW CAPITALISED token" literally, which
        also fires on ``The fox replied,``.

    All three are accumulated over the kept population so the manifest can state what the
    filter cost AND what the alternative would have cost, rather than asserting that the
    chosen reading was the cheap one. See `train.dialogue.same_voice_risk`.
    """
    gaps = adjacent_gaps(story, n_turns)
    return {"pairs": len(gaps),
            "tag_only": sum(1 for g in gaps if is_tag_only_gap(g)),
            "risky_subject": sum(1 for g in gaps if same_voice_risk(g)),
            "risky_strict": sum(1 for g in gaps
                                if same_voice_risk(g, strict_names=True))}


def screen_candidate(skit: Skit, story: str, *, assoc: Association, intensity_fn,
                     tok, pad_token_id: int, max_seq_len: int, strict_names: bool,
                     story_in_table: bool) -> Tuple[Optional[ReachCandidate],
                                                    Optional[str]]:
    """The three reach-era gates, IN ORDER, or `(None, drop_rule)`.

    `derive_dialogue_skit` has already applied task 1's gates. These are the ones the
    controller's rulings added, and the ORDER IS PART OF THE CONTRACT because every count
    after the first is conditional on the ones before it:

      1. `same_voice_pair`   (ruling A) -- some adjacent pair shows no evidence that the
         voice changed, so alternation-by-position may have put `model` on both sides of it.
         Cheapest, and it is the one protecting the premise of the whole dialogue path, so it
         runs first.
      2. `over_max_seq_len`  (ruling C) -- the built example exceeds the window and
         `sft_collate_fn` would truncate it SILENTLY, teaching a scene that stops mid-turn.
         Excluded, not truncated. Requires a tokenizer; without one this gate cannot run at
         all and is recorded as not applied rather than skipped quietly.
      3. `reach_no_evidence` -- no context word has any co-occurrence with the `add` word at
         some block, so the dial reading would be our own ignorance rather than a distance.
         Last because it is the most expensive.

    `story_in_table` says whether THIS story is one of the association table's documents; when
    it is, the distance is computed leave-one-out. Without that, every pair is guaranteed a
    co-occurrence -- its own story -- and the metric partly measures the scene against itself.
    See `train.reach.pair_counts`.

    A fixture story that fails MORE THAN ONE of these must report the FIRST -- that is what
    `test_screen_candidate_reports_the_first_gate_not_the_worst` pins, and it is the exact
    hole that let a gate-order mutation survive task 1's `classify_turn_failure` table.
    """
    if not voice_changes_throughout(story, strict_names=strict_names):
        return None, "same_voice_pair"

    n_tokens: Optional[int] = None
    if tok is not None:
        n_tokens = len(build_skit_example(skit, tok, with_think=True,
                                          pad_token_id=pad_token_id)["input_ids"])
        if n_tokens > max_seq_len:
            return None, "over_max_seq_len"

    distances = skit_reach_distances(skit, assoc, holdout=story_in_table)
    if distances is None:
        return None, "reach_no_evidence"

    return ReachCandidate(skit=skit, distances=tuple(distances),
                          deltas=tuple(skit_stakes_deltas(skit, intensity_fn)),
                          n_tokens=n_tokens), None


def fit_split_sizes(n: int, eval_fraction: float) -> Tuple[int, int]:
    """`(n_train, n_eval)` with the EVAL SPLIT AS THE TAIL of the file, no RNG.

    Same convention as `scripts/train_skits.py`'s hold-out ("the TAIL of the file in file
    order ... no RNG is involved, deliberately"), for the same reason: a seeded shuffle is one
    more thing that has to match between the arms, and between derivation and eval, for the
    pairing to hold.

    The tercile cut points are fitted on the HEAD only, so `n_train` must be able to support
    a fit; below three training values the caller cannot proceed and the whole run should say
    so rather than fit on everything and pretend.
    """
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
    n_eval = int(n * eval_fraction)
    return n - n_eval, n_eval


def ruling_c_report(lengths_all: List[int], lengths_kept: List[int], excluded: int,
                    max_seq_len: int, applied: bool) -> dict:
    """What ruling C did: excluded over-length skits, or could not run at all.

    `token_length_report` describes the distribution; this says whether the EXCLUSION
    happened. `applied` is False exactly when no tokenizer was supplied, and in that case the
    output may contain examples the trainer would truncate -- an unmeasured risk, which is
    not the same as no risk, so it carries a warning string of its own.
    """
    if not applied:
        return {"applied": False, "excluded": 0, "max_seq_len": max_seq_len,
                "warning": "RULING C NOT APPLIED: no --tokenizer, so over-length skits "
                           "could not be identified and were NOT excluded. "
                           "sft_collate_fn would truncate them silently."}
    n = len(lengths_all)
    return {"applied": True, "excluded": excluded, "max_seq_len": max_seq_len,
            "candidates_measured": n,
            "excluded_fraction": round(excluded / n, 4) if n else None,
            "kept_max": max(lengths_kept) if lengths_kept else None,
            "note": "Excluded, NOT truncated. A truncated example teaches a scene that "
                    "stops mid-turn; task 1 measured 2.82% over the window and there is "
                    "yield headroom, so they are dropped under their own rule."}


def same_speaker_filter_report(counts: Dict[str, int], *, mode: str) -> dict:
    """What ruling A's filter cost, and what the stricter reading would have cost.

    Denominators are stated per row because they differ: the PAIR rates are over the pairs of
    every skit that reached the filter, and the SKIT rates are over those skits. The `applied`
    row is the only one that changed the output; the other is measured on the same population
    for comparison and is inert.
    """
    reached = max(counts["reached_filter"], 1)
    pairs = max(counts["filter_pairs"], 1)
    return {
        "applied": mode,
        "skits_reaching_the_filter": counts["reached_filter"],
        "adjacent_pairs_examined": counts["filter_pairs"],
        "subject_reading": {
            "gate": "train.dialogue.same_voice_risk (tag-only gap whose tag identifies no "
                    "subject, or no gap at all)",
            "risky_pairs": counts["risky_pairs_subject"],
            "risky_pair_fraction": round(counts["risky_pairs_subject"] / pairs, 4),
            "skits_dropped": counts["risky_skits_subject"],
            "skit_drop_fraction": round(counts["risky_skits_subject"] / reached, 4),
        },
        "strict_names_reading": {
            "gate": "train.dialogue.same_voice_risk(strict_names=True) -- the literal 'no "
                    "NEW CAPITALISED token' wording from the task-1 proposal",
            "risky_pairs": counts["risky_pairs_strict"],
            "risky_pair_fraction": round(counts["risky_pairs_strict"] / pairs, 4),
            "skits_dropped": counts["risky_skits_strict"],
            "skit_drop_fraction": round(counts["risky_skits_strict"] / reached, 4),
        },
        "tag_only_pairs": counts["filter_tag_only"],
        "tag_only_pair_fraction": round(counts["filter_tag_only"] / pairs, 4),
        "why_subject_and_not_strict_names":
            "The strict reading also fires on gaps that name the speaker with a COMMON noun "
            "-- 'The rabbit looked up and said,' / 'The fox replied,' -- so it discards "
            "scenes that alternate perfectly and is closer to a genre filter (it removes "
            "animal dialogue) than to a voice-safety one. Both costs are recorded above so "
            "the choice is auditable; --strict-speaker-names switches it.",
        "what_this_filter_CANNOT_catch":
            "A gap that carries narrative and then re-introduces the SAME speaker is "
            "invisible to any attribution-free filter. Real kept example (story 685): turn 1 "
            "and turn 2 are both Lily, separated by 'They went to the airport ... Lily got "
            "bored and said,', so parity puts `model` on the mother at turn 0 and on the "
            "daughter at turn 2. Separating that from a genuine change requires comparing "
            "the tag's subject with the PREVIOUS tag's subject, i.e. exactly the speaker "
            "attribution that got an earlier probe's speakers backwards. Published as a "
            "limitation, not resolved.",
    }


def reach_report(distances_train: List[float], distances_all: List[float],
                 buckets_train: List[str], buckets_eval: List[str],
                 lo: float, hi: float, zero_evidence_skits: int,
                 zero_evidence_observations: int, observations_examined: int,
                 assoc_meta: dict) -> dict:
    """The manifest's `reach` block: the table, the zeros, the cut points, the balance.

    Everything an eval needs in order NOT to re-fit anything, and everything a reader needs
    in order to tell a dial from a dial on our own coverage:

      * `association_table` -- how the table was built, because the zero rate is a property
        of the table and not of the corpus. Task 1's pre-flight measured 6.7% exact zeros on
        a (prefix_word, turn_word) table; this one is whole-story and symmetric.
      * `zero_evidence_*` -- re-measured HERE, on the table actually built. Inherited numbers
        would describe a different instrument.
      * `cut_points` -- fitted on the TRAINING SPLIT ONLY, in DISTANCE space, with their NPMI
        equivalents beside them so the direction is unmistakable.
      * `bucket_balance` -- train and eval separately: the eval split is bucketed with the
        TRAIN cut points, so its balance drifting is a real finding about the tail of the
        file, not a re-fit.
    """
    s = sorted(distances_all)

    def pct(p: float) -> Optional[float]:
        return round(s[min(len(s) - 1, int(p * (len(s) - 1)))], 4) if s else None

    return {
        "slot_order": list(REACH_SLOT_NAMES),
        "slot_order_note": "`reach` is rendered BEFORE `add` on purpose: the block is "
                           "generated left to right, so a dial declared after the word it "
                           "governs could only relabel a choice already made.",
        "values": list(REACH_VALUES),
        "metric": "distance = 1 - max NPMI(add_word, c) over the content words already in "
                  "play (prefix + all earlier turns). 0 = on top of the scene, 1 = as far "
                  "as the table can say. HIGH NPMI IS A NEAR REACH, so `far` is the HIGH "
                  "DISTANCE tercile; nothing in train/reach.py sorts an association.",
        "association_table": assoc_meta,
        "holdout": "LEAVE-ONE-OUT. Every distance is computed with the scored story's own "
                   "document subtracted from all four counts, because that story IS a "
                   "document of this table: without it every pair is guaranteed one "
                   "co-occurrence (its own), the no-evidence rate is structurally 0.0000, "
                   "and a rare add word paired with a rare context word scores a PERFECT "
                   "association produced entirely by the scene being scored. Measured on "
                   "20,000 stories: 0.45% of observations rest on exactly one own-story "
                   "co-occurrence. See train.reach.pair_counts.",
        "zero_evidence": {
            "observations_examined": observations_examined,
            "observations_without_evidence": zero_evidence_observations,
            "observation_fraction": (round(zero_evidence_observations
                                           / observations_examined, 6)
                                     if observations_examined else None),
            "skits_dropped": zero_evidence_skits,
            "note": "npmi() returns exactly 0.0 for BOTH 'no association' and 'never "
                    "co-occurred', so these are EXCLUDED rather than binned as `far`. "
                    "Re-measured on the table above, not inherited: the pre-flight's 6.7% "
                    "came from a sparse (prefix_word, turn_word) table.",
        },
        "distance_distribution": {
            "n": len(s), "p5": pct(0.05), "p25": pct(0.25), "p50": pct(0.50),
            "p75": pct(0.75), "p95": pct(0.95), "min": round(s[0], 4) if s else None,
            "max": round(s[-1], 4) if s else None,
            "distinct": len(set(s)),
        },
        "cut_points": {
            "lo": lo, "hi": hi,
            "fitted_on": "training split only (the head of the file)",
            "n_fitted_on": len(distances_train),
            "rule": "distance < lo -> near; distance < hi -> mid; else far",
            "npmi_equivalents": {"lo": round(1.0 - lo, 6), "hi": round(1.0 - hi, 6)},
            "eval_must_not_refit":
                "These two numbers ARE the dial. Eval reads them from here; a dial whose "
                "buckets move between train and eval measures nothing.",
        },
        "bucket_balance_train": bucket_balance(buckets_train),
        "bucket_balance_eval": bucket_balance(buckets_eval),
    }


# --------------------------------------------------------------------------------------
def derive_dialogue_skit(story: str, *, story_id: int, idf: Dict[str, float],
                         intensity_fn) -> Tuple[Optional[Skit], Optional[str]]:
    """One dialogue skit from one story, or `(None, drop_rule)`.

    The gates, in the order they fire:
      1. `no_dialogue`         -- no quoted utterance at all.
      2. `too_few_utterances`  -- 1..4 utterances; a skit needs five turns.
      3. `empty_prefix`        -- the story opens on dialogue, so block 0 has no offer to
         accept and the skit could only ever drop at the first turn. Reported separately
         because it is a property of the STORY's shape, not of the slots.
      4. `turn_derivation_failed_*` -- accept/add failed at some model turn;
         `classify_turn_failure` says which.
    """
    turns = extract_dialogue_turns(story)
    if turns is None:
        n = len(quoted_utterances(story))
        return None, "no_dialogue" if n == 0 else "too_few_utterances"
    if len(turns) != len(SKIT_ROLES):
        # Unreachable while extract_dialogue_turns returns exactly DIALOGUE_TURNS == 5, and
        # kept as a named drop rather than an IndexError in classify_turn_failure if that
        # ever changes. (A mutation that loosened MIN_UTTERANCES to 3 crashed here.)
        return None, "wrong_turn_count"
    prefix = dialogue_prefix(story)
    if not prefix:
        return None, "empty_prefix"
    skit = derive_skit_from_turns(prefix, turns, story_id=story_id, idf=idf,
                                  intensity=intensity_fn)
    if skit is None:
        return None, f"turn_derivation_failed_{classify_turn_failure(prefix, turns)}"
    return skit, None


def association_meta(assoc: Association, *, population: str,
                     corpus_md5: str) -> dict:
    """How the association table was built, plus a fingerprint, so eval can rebuild it.

    The table is NOT written to disk: at corpus scale it is tens of millions of pairs, and it
    is a pure function of (corpus bytes, document population, `content_words`). What goes in
    the manifest is everything needed to rebuild it and a fingerprint to prove the rebuild
    matched -- the same reasoning as `corpus_md5` naming bytes-on-disk rather than a path.
    """
    top = sorted(assoc.uni.items(), key=lambda kv: (-kv[1], kv[0]))[:200]
    h = hashlib.md5()
    h.update(f"{assoc.n_docs}|{len(assoc.uni)}|{len(assoc.co)}|".encode())
    h.update("|".join(f"{w}:{c}" for w, c in top).encode())
    return {
        "builder": "train.reach.build_association",
        "document": "ONE WHOLE STORY (all within-story content-word co-occurrence, "
                    "symmetric). NOT the sparse (prefix_word, turn_word) pairs the "
                    "pre-flight diagnosed the zero-inflation on, and not "
                    "scripts.score_improv.build_association's (prefix, continuation) "
                    "pair-documents either.",
        "population": population,
        "documents": assoc.n_docs,
        "vocabulary": len(assoc.uni),
        "pairs": len(assoc.co),
        "pair_key": "canonical, a <= b (half the memory of the both-directions form)",
        "counting": "document frequency: each unordered pair of distinct content words is "
                    "counted once per document, however often either word repeats",
        "corpus_md5": corpus_md5,
        "fingerprint_md5": h.hexdigest(),
        "fingerprint_inputs": "documents|vocabulary|pairs|top-200 unigrams by (-df, word)",
        "self_inclusion_note":
            "Every scanned story is a document of this table, INCLUDING the stories the "
            "skits come from, so a rare add/context pair can owe its only co-occurrence to "
            "the very story being scored. That bias is uniform across skits -- which is what "
            "matters for a tercile comparison -- precisely because the document population "
            "is the whole scan and not a subsample; a subsample would make it apply to some "
            "skits and not others.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "tinystories.txt")
    ap.add_argument("--limit", type=int, default=None,
                    help="stories to scan; default None = the WHOLE corpus (ruling B).")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "reach-skits" / "skits.jsonl",
                    help="NEW path. artifacts/dialogue-skits/ is task 1's output and is "
                         "left alone.")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="HF tokenizer dir. WITHOUT it ruling C cannot be applied: "
                         "over-length skits are neither identified nor excluded, and the "
                         "manifest records that as a warning rather than a measurement.")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--assoc-stories", type=int, default=0,
                    help="documents for the association table; 0 = every story scanned "
                         "(the default, and the only setting under which the table's "
                         "self-inclusion bias is uniform across skits).")
    ap.add_argument("--eval-fraction", type=float, default=0.1,
                    help="tail of the file held out. Tercile cut points are fitted on the "
                         "HEAD only, and recorded, so eval cannot re-fit them.")
    ap.add_argument("--strict-speaker-names", action="store_true",
                    help="ruling A's filter under the literal 'no NEW CAPITALISED token' "
                         "reading. Measured either way; see same_speaker_filter.")
    args = ap.parse_args(argv)

    tok = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer      # local import: CPU-only, no ttml/ttnn
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    pad_token_id = (tok.pad_token_id or 0) if tok is not None else 0

    stories = list(iter_stories(args.corpus, args.limit))
    print(f"stories read: {len(stories):,} (separator {STORY_SEP!r}, NOT a blank line)")
    idf = build_idf(stories)
    harm = load_harm_lexicon()

    def intensity_fn(text: str) -> float:
        return intensity(text, harm)

    n_assoc = args.assoc_stories or len(stories)
    print(f"building the association table over {n_assoc:,} whole-story documents "
          f"(this is the slow part) ...", flush=True)
    assoc = build_association(stories[:n_assoc])
    print(f"  vocabulary {len(assoc.uni):,}  pairs {len(assoc.co):,}", flush=True)

    counts: Counter = Counter()
    drops: Counter = Counter()
    counts["stories_scanned"] = len(stories)

    # ---------------- pass 1: scan, gate, and collect the numbers the dial needs ---------
    candidates: List[ReachCandidate] = []
    lengths_all: List[int] = []
    for i, story in enumerate(stories):
        # Corpus baseline, over EVERY story, kept or dropped -- so what the gates discard is
        # measurable rather than inferred. See `selection_bias`.
        units, dialogue_units = dialogue_unit_counts(story)
        n_utts = len(quoted_utterances(story))
        counts["corpus_units"] += units
        counts["corpus_dialogue_units"] += dialogue_units
        counts["corpus_utterances"] += n_utts
        counts["stories_with_any_dialogue"] += 1 if n_utts else 0
        counts["stories_with_min_utterances"] += 1 if n_utts >= MIN_UTTERANCES else 0

        skit, rule = derive_dialogue_skit(story, story_id=i, idf=idf,
                                          intensity_fn=intensity_fn)
        if skit is None:
            drops[rule] += 1
            continue

        # Ruling A's filter cost, and the stricter reading's, on the SAME population.
        audit = voice_change_audit(story)
        counts["reached_filter"] += 1
        counts["filter_pairs"] += audit["pairs"]
        counts["filter_tag_only"] += audit["tag_only"]
        counts["risky_pairs_subject"] += audit["risky_subject"]
        counts["risky_pairs_strict"] += audit["risky_strict"]
        counts["risky_skits_subject"] += 1 if audit["risky_subject"] else 0
        counts["risky_skits_strict"] += 1 if audit["risky_strict"] else 0

        cand, rule = screen_candidate(skit, story, assoc=assoc, intensity_fn=intensity_fn,
                                      tok=tok, pad_token_id=pad_token_id,
                                      max_seq_len=args.max_seq_len,
                                      strict_names=args.strict_speaker_names,
                                      story_in_table=i < n_assoc)
        if rule == "over_max_seq_len":
            # Recomputed for the length report only: the gate already decided.
            lengths_all.append(len(build_skit_example(
                skit, tok, with_think=True, pad_token_id=pad_token_id)["input_ids"]))
        if rule == "reach_no_evidence":
            # The denominator of the zero-evidence RATE is the observations whose reach was
            # actually attempted -- i.e. the skits that reached gate 3. A skit dropped at the
            # length gate never had a distance computed, so counting its three blocks here
            # would dilute the rate with observations nobody measured.
            counts["reach_observations"] += len(MODEL_TURNS)
            # Read-only diagnostics, same pattern as classify_turn_failure: the gate has
            # already dropped this skit; this only says how many of its three observations
            # carried no evidence, which is what makes the zero rate an OBSERVATION rate.
            per_block = skit_reach_distances_per_block(skit, assoc,
                                                       holdout=i < n_assoc)
            counts["reach_observations_no_evidence"] += sum(1 for d in per_block
                                                            if d is None)
        if cand is None:
            drops[rule] += 1
            continue

        counts["reach_observations"] += len(MODEL_TURNS)
        if cand.n_tokens is not None:
            lengths_all.append(cand.n_tokens)
        counts["utterances_in_kept_stories"] += n_utts
        pairs, tag_only = tag_only_gap_count(story)
        counts["kept_adjacent_pairs"] += pairs
        counts["kept_tag_only_gaps"] += tag_only
        counts["kept_prefix_words"] += len(content_words(skit.prefix))
        counts["kept_prefix_units"] += len(split_sentences_dialogue(skit.prefix))
        candidates.append(cand)

    kept = len(candidates)
    counts["stories_kept"] = kept          # selection_bias' denominator for the kept side

    # ---------------- fit: cut points come from the TRAINING SPLIT ONLY ------------------
    # The floor is on TRAINING OBSERVATIONS, not on skits: each skit contributes one distance
    # per model turn, so two skits already carry six observations and can be fitted. Getting
    # this wrong the first time refused a two-skit fixture that was perfectly fittable.
    n_train, n_eval = fit_split_sizes(kept, args.eval_fraction)
    distances_train = [d for c in candidates[:n_train] for d in c.distances]
    if len(distances_train) < 3:
        print(f"FATAL: {kept} candidate skit(s) -> {len(distances_train)} training "
              f"observation(s): terciles cannot be fitted. Nothing written. Re-run with a "
              f"larger --limit or inspect the drop table.")
        for rule, n in sorted(drops.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {rule:44} {n:,}")
        return 1
    lo, hi = fit_reach_terciles(distances_train)
    print(f"terciles fitted on {len(distances_train):,} training observations: "
          f"lo={lo:.6f} hi={hi:.6f}  (distance = 1 - max NPMI; far is the HIGH end)")

    # ---------------- pass 2: bucket every skit with those cut points, and write ---------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    buckets_train: List[str] = []
    buckets_eval: List[str] = []
    distances_all: List[float] = []
    lengths_kept: List[int] = []
    with args.out.open("w") as fh:
        for k, cand in enumerate(candidates):
            skit = with_reach(cand.skit, distances=cand.distances, deltas=cand.deltas,
                              lo=lo, hi=hi)
            row = skit.as_dict()
            row["split"] = "train" if k < n_train else "eval"
            # FULL precision, deliberately. Rounding the distance beside the bucket makes
            # the artifact self-inconsistent at a cut point: a boundary observation rounds to
            # the other side of `lo` and a scorer re-deriving the bucket from the row
            # disagrees with the row. Caught by
            # test_bucket_balance_on_the_real_artifact on story 717.
            row["reach_distances"] = list(cand.distances)
            row["stakes_deltas"] = list(cand.deltas)
            fh.write(json.dumps(row) + "\n")
            got = [b.reach for b in skit.blocks]
            (buckets_train if k < n_train else buckets_eval).extend(got)
            distances_all.extend(cand.distances)
            if cand.n_tokens is not None:
                lengths_kept.append(cand.n_tokens)

    total = len(stories)
    rate = 1 - kept / max(total, 1)
    warning = drop_rate_warning(rate)
    manifest = {
        "schema": "reach-skits v3 (six-slot think-block; see reach.slot_order)",
        "supersedes": "artifacts/dialogue-skits/ (task 1, five-slot blocks). That artifact "
                      "is left on disk untouched; this one is a NEW path.",
        "corpus": repo_relative(args.corpus),
        "corpus_md5": md5_of(args.corpus),
        "separator": STORY_SEP,
        "tile": TILE,
        "limit": args.limit,
        "turn_source": "train.dialogue.extract_dialogue_turns (alternation by position over "
                       "quoted utterances; no speaker attribution)",
        "splitter": "train.dialogue.split_sentences_dialogue",
        "stories": total,
        "kept": kept,
        "drop_rate": round(rate, 4),
        "drop_rate_warning": warning,
        "drops_by_rule": dict(sorted(drops.items(), key=lambda kv: (-kv[1], kv[0]))),
        "gate_order": ["no_dialogue / too_few_utterances / wrong_turn_count",
                       "empty_prefix",
                       "turn_derivation_failed_* (accept/add, task 1)",
                       "same_voice_pair (ruling A)",
                       "over_max_seq_len (ruling C)",
                       "reach_no_evidence"],
        "gate_order_note": "Every count after the first is CONDITIONAL on the gates before "
                           "it. screen_candidate reports the FIRST gate a skit fails, not "
                           "the worst.",
        "split": {
            "n_train": n_train, "n_eval": n_eval, "eval_fraction": args.eval_fraction,
            "rule": "the eval split is the TAIL of the file in derivation order; no RNG "
                    "(same convention as scripts/train_skits.py's hold-out)",
            "each_row_carries_its_split": True,
        },
        "reach": reach_report(
            distances_train, distances_all, buckets_train, buckets_eval, lo, hi,
            zero_evidence_skits=drops.get("reach_no_evidence", 0),
            zero_evidence_observations=counts["reach_observations_no_evidence"],
            observations_examined=counts["reach_observations"],
            assoc_meta=association_meta(
                assoc,
                population=(f"the first {n_assoc:,} stories of the scan"
                            if args.assoc_stories else
                            f"EVERY story scanned ({n_assoc:,})"),
                corpus_md5=md5_of(args.corpus))),
        "stakes": {
            "form": "continuous signed delta, rendered by train.reach.format_stakes_delta",
            "definition": "intensity(this model turn) - intensity(the turn before it), in "
                          "harm-lexicon hits per 100 content words",
            "why": "RULING D. Stage 2 withdrew the up/level/down label as mostly confounded "
                   "(85.3% one class); carried continuously so the withdrawal has a "
                   "successor rather than a silent disappearance. NOT a headline slot.",
            "per_row": "stakes_deltas carries the FULL-PRECISION values beside the "
                       "rendered (1 d.p.) strings, so a scorer reads magnitude from the "
                       "row and never has to re-derive it from the text.",
        },
        "same_speaker_filter": same_speaker_filter_report(
            counts, mode=("strict_names" if args.strict_speaker_names else "subject")),
        "ruling_c": ruling_c_report(lengths_all, lengths_kept,
                                    drops.get("over_max_seq_len", 0), args.max_seq_len,
                                    applied=tok is not None),
        "selection_bias": selection_bias(counts),
        "same_speaker_risk": {
            "adjacent_pairs": counts["kept_adjacent_pairs"],
            "tag_only_gaps": counts["kept_tag_only_gaps"],
            "tag_only_gap_fraction": (round(counts["kept_tag_only_gaps"]
                                            / counts["kept_adjacent_pairs"], 4)
                                      if counts["kept_adjacent_pairs"] else None),
            "note": "Task 1's UPPER BOUND indicator, recomputed on the POST-FILTER "
                    "population so it can be compared with task 1's 0.3959. It is an upper "
                    "bound because it fires on genuine two-speaker exchanges too "
                    "('\"Hello,\" said Amy. \"Goodbye,\" said Ben.'). What acted on it is "
                    "same_speaker_filter above. See train.dialogue.is_tag_only_gap.",
        },
        "prefix_shape": {
            "mean_content_words": (round(counts["kept_prefix_words"] / kept, 2)
                                   if kept else None),
            "mean_units": round(counts["kept_prefix_units"] / kept, 2) if kept else None,
            "note": "The prefix is EVERYTHING before the first quoted utterance, so it is "
                    "typically longer than derive_skit's fixed two sentences. That makes "
                    "block 0's accept gate easier (more offer words) and its add gate "
                    "harder (more established words) than in stage 2, and it is why "
                    "token_lengths is worth measuring.",
        },
        "token_lengths": token_length_report(lengths_kept, args.max_seq_len),
        "token_lengths_before_exclusion": token_length_report(lengths_all,
                                                              args.max_seq_len),
    }
    (args.out.parent / "derive_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"kept {kept:,}/{total:,}  drop rate {rate:.1%}  "
          f"(train {n_train:,} / eval {n_eval:,})")
    for rule, n in sorted(drops.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {rule:44} {n:,}")
    bb = manifest["reach"]["bucket_balance_train"]
    print(f"reach buckets (train): {bb['counts']}  max_fraction {bb['max_fraction']}")
    ze = manifest["reach"]["zero_evidence"]
    print(f"zero-evidence observations: {ze['observations_without_evidence']:,}/"
          f"{ze['observations_examined']:,} = {ze['observation_fraction']}")
    sf = manifest["same_speaker_filter"]
    print(f"ruling A ({sf['applied']}): dropped "
          f"{sf['subject_reading']['skits_dropped'] if sf['applied'] == 'subject' else sf['strict_names_reading']['skits_dropped']:,}"
          f" of {sf['skits_reaching_the_filter']:,} skits reaching the filter")
    rc = manifest["ruling_c"]
    print(f"ruling C: applied={rc['applied']} excluded={rc['excluded']:,}")
    sb = manifest["selection_bias"]
    print(f"dialogue: {sb['stories_with_any_dialogue_fraction']:.1%} of stories have any; "
          f"{sb['stories_with_min_utterances_fraction']:.1%} clear >={MIN_UTTERANCES}; "
          f"corpus units carrying dialogue {sb['corpus_dialogue_unit_fraction']:.1%}; "
          f"utterance retention {sb['dialogue_utterance_retention']:.1%}")
    if warning:
        print(warning)
    if not rc["applied"]:
        print(rc["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
