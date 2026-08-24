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

    python3 scripts/derive_dialogue_skits.py --limit 400000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_skits import TILE, build_idf, build_skit_example  # noqa: E402,F401
from scripts.score_improv import intensity, load_harm_lexicon  # noqa: E402
from train.dialogue import (MIN_UTTERANCES, dialogue_prefix,  # noqa: E402
                            extract_dialogue_turns, is_tag_only_gap,
                            quoted_utterances, split_sentences_dialogue)
from train.improv import content_words  # noqa: E402
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "tinystories.txt")
    ap.add_argument("--limit", type=int, default=400000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts" / "dialogue-skits" / "skits.jsonl")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="HF tokenizer dir. Optional, but WITHOUT it the manifest's "
                         "token_lengths block records that the truncation risk is unknown "
                         "rather than measured.")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args(argv)

    tok = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer      # local import: CPU-only, no ttml/ttnn
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer))

    stories = list(iter_stories(args.corpus, args.limit))
    print(f"stories read: {len(stories):,} (separator {STORY_SEP!r}, NOT a blank line)")
    idf = build_idf(stories)
    harm = load_harm_lexicon()

    counts: Counter = Counter()
    drops: Counter = Counter()
    counts["stories_scanned"] = len(stories)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    lengths: List[int] = []
    with args.out.open("w") as fh:
        for i, story in enumerate(stories):
            # Corpus baseline, over EVERY story: what fraction of the corpus's sentence
            # units and utterances exist at all, so what the gate discards is measurable
            # rather than inferred. quoted_utterances is recomputed here and inside
            # derive_dialogue_skit; it is a cheap regex scan and keeping the gate as the
            # single decision point is worth more than the saved microseconds.
            units, dialogue_units = dialogue_unit_counts(story)
            n_utts = len(quoted_utterances(story))
            counts["corpus_units"] += units
            counts["corpus_dialogue_units"] += dialogue_units
            counts["corpus_utterances"] += n_utts
            counts["stories_with_any_dialogue"] += 1 if n_utts else 0
            counts["stories_with_min_utterances"] += 1 if n_utts >= MIN_UTTERANCES else 0

            skit, rule = derive_dialogue_skit(story, story_id=i, idf=idf,
                                              intensity_fn=lambda t: intensity(t, harm))
            if skit is None:
                drops[rule] += 1
                continue
            counts["utterances_in_kept_stories"] += n_utts
            pairs, tag_only = tag_only_gap_count(story)
            counts["kept_adjacent_pairs"] += pairs
            counts["kept_tag_only_gaps"] += tag_only
            counts["kept_prefix_words"] += len(content_words(skit.prefix))
            counts["kept_prefix_units"] += len(split_sentences_dialogue(skit.prefix))
            if tok is not None:
                ex = build_skit_example(skit, tok, with_think=True,
                                        pad_token_id=tok.pad_token_id or 0)
                lengths.append(len(ex["input_ids"]))
            fh.write(json.dumps(skit.as_dict()) + "\n")
            kept += 1

    counts["stories_kept"] = kept
    total = len(stories)
    rate = 1 - kept / max(total, 1)
    warning = drop_rate_warning(rate)
    manifest = {
        "corpus": repo_relative(args.corpus),
        "corpus_md5": md5_of(args.corpus),
        "separator": STORY_SEP,
        "tile": TILE,
        "turn_source": "train.dialogue.extract_dialogue_turns (alternation by position over "
                       "quoted utterances; no speaker attribution)",
        "splitter": "train.dialogue.split_sentences_dialogue",
        "stories": total,
        "kept": kept,
        "drop_rate": round(rate, 4),
        "drop_rate_warning": warning,
        "drops_by_rule": dict(sorted(drops.items(), key=lambda kv: (-kv[1], kv[0]))),
        "selection_bias": selection_bias(counts),
        "same_speaker_risk": {
            "adjacent_pairs": counts["kept_adjacent_pairs"],
            "tag_only_gaps": counts["kept_tag_only_gaps"],
            "tag_only_gap_fraction": (round(counts["kept_tag_only_gaps"]
                                            / counts["kept_adjacent_pairs"], 4)
                                      if counts["kept_adjacent_pairs"] else None),
            "note": "UPPER BOUND on the fraction of adjacent turn pairs that are the SAME "
                    "voice, which alternation-by-position would then mislabel. It is an "
                    "upper bound because the indicator fires on genuine two-speaker "
                    "exchanges too ('\"Hello,\" said Amy. \"Goodbye,\" said Ben.'); "
                    "separating those needs the speaker attribution that got the probe's "
                    "speakers backwards, so this is published rather than resolved. See "
                    "train.dialogue.is_tag_only_gap.",
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
        "token_lengths": token_length_report(lengths, args.max_seq_len),
    }
    (args.out.parent / "derive_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"kept {kept:,}/{total:,}  drop rate {rate:.1%}")
    for rule, n in sorted(drops.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"    {rule:44} {n:,}")
    sb = manifest["selection_bias"]
    print(f"dialogue: {sb['stories_with_any_dialogue_fraction']:.1%} of stories have any; "
          f"{sb['stories_with_min_utterances_fraction']:.1%} clear >={MIN_UTTERANCES}; "
          f"corpus units carrying dialogue {sb['corpus_dialogue_unit_fraction']:.1%}; "
          f"utterance retention {sb['dialogue_utterance_retention']:.1%}")
    if warning:
        print(warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
