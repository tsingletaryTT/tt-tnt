#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The reach-dial EUREKA measurement: does forcing `reach` move the word the model reaches for?

ONE QUESTION
------------
Does forcing ``reach: near|mid|far`` change the semantic distance of the `add` word the model
then produces, WITHOUT destroying coherence? Everything in this file serves that.

THE DESIGN — WITHIN-MODEL, SCENE-PAIRED
---------------------------------------
Not a between-arm contrast. The SAME model generates over the SAME held-out scene once per
forced dial setting, so per-scene variance cancels the way stage 2's step-pairing cancelled
per-step variance -- and no arm-identity confound has to be bounded, which is exactly what
cost stage 2 its `accept` headline (+0.134 -> ~+0.083).

Per scene we measure ONE model turn (turn 2, the middle one): it is the only model turn with
a real partner turn both before it (so `offer` has something to accept) and after it (so
`handback` is scorable), and its context is prefix + turn 0 + turn 1. The two earlier
segments are TEACHER-FORCED from the row, and so are the block's `offer` and `accept` lines,
so the only thing that differs between the three settings is the four characters of the dial
value. That is the whole point: everything the model conditions on is byte-identical across
settings except the dial.

THREE CONTROLS, IN ORDER OF STRENGTH
------------------------------------
1. **Nonsense value (PRIMARY).** Same arm, same well-formed six-slot block, `reach: blue` --
   an off-vocabulary value of the same shape. If the monotone pattern is specific to
   near/mid/far, a meaningless value must not reproduce it. This is the primary control
   because it holds the SCHEMA fixed and varies only the VALUE.
2. **`nodial` arm (secondary, and weaker than it looks).** That arm learned a FIVE-slot
   schema, so a forced `reach:` line is off-schema two ways at once -- an unknown slot name
   AND an extra line. Movement there could be generic malformed-block sensitivity rather than
   sensitivity to the dial's value, which is why it is not the primary control.
3. **Frequency control (mandatory, spec amendment 8fb43b4).** NPMI is not frequency-neutral:
   two rare words that co-occur at all score high, a common word's NPMI is capped low by its
   own marginal. So `far` partly selects COMMONER words. Raw distance, distance residualised
   on log document frequency, a document-frequency-matched paired subsample, AND the realised
   `add_df` per setting are all reported. **A dial that moves raw distance but not
   frequency-controlled distance is a FREQUENCY DIAL** and is published as that. Because the
   corpus-scale confound is NOT monotone (`mid` has the highest median `add_df`),
   **`mid` vs `far` is the cleaner contrast** and is labelled as such.

WHY THE PROMPTS ARE BUILT SEGMENT-WISE
--------------------------------------
`scripts/derive_skits.py:build_skit_example` encodes each `train.skit.skit_segments` entry
with its own `tok.encode` call, and this tokenizer prepends a space per call. Training
therefore saw ``['.', 'Ġ<', 'think', '>']`` at the seam. Tokenizing the assembled prompt
string in one call gives a bare ``'<'`` instead and asks the model to open a think-block
after a token it never saw there. It would degrade every setting EQUALLY -- pairing would
survive -- and would read exactly as "the dial does nothing", which is the finding this file
exists to determine. `check_prompt_parity` proves the ids are a PREFIX of a real training
example's ids and RAISES otherwise, before a token is generated.

WHY THE ASSOCIATION TABLE IS COUNTED TARGET-FIRST
-------------------------------------------------
The distance metric needs the corpus association table (2,119,489 whole-story documents,
52,302 words, 31.9M pairs). Building it in full costs ~34 minutes and it is not persisted.
But `train.reach.reach_distance` only ever LOOKS UP the pairs it is asked about, so
`collect_targeted_association` makes one streaming pass over the same corpus and counts
only the unigrams and pairs this eval needs. The counts are IDENTICAL to the full table's by
construction (same corpus, same `content_words`, same document = one story, same canonical
pair key), and that is not asserted: `tests/test_eval_reach.py` proves targeted == full on a
fixture corpus, and this run RE-DERIVES every gold row's stored `reach_distances` from the
targeted table and refuses to continue if they do not match to 1e-12.

MULTIPLE COMPARISONS — THE CONSTANTS BELOW ARE THIS EVAL'S OWN
--------------------------------------------------------------
`BONFERRONI_ALPHA` and `CRITICAL_T` are DEFINED HERE and deliberately not imported from
`scripts/eval_improv.py` (stage 1: 0.01 / 2.576) or `scripts/eval_skits.py` (stage 2). In
stage 2 two effects landed between stage 1's 2.576 and the correct 2.843; an import would
have manufactured both. `tests/test_eval_reach.py` carries a test that fails if a t in that
band is ever called significant here.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.derive_dialogue_skits import iter_stories  # noqa: E402
from scripts.derive_skits import build_skit_example  # noqa: E402
from scripts.score_improv import build_association as build_groundedness_assoc  # noqa: E402
from scripts.score_improv import (load_closure_lexicon,  # noqa: E402
                                  load_harm_lexicon, score_pair)
from scripts.score_skits import _mentions  # noqa: E402
from train.improv import content_words, render_think, split_sentences  # noqa: E402
from train.reach import (REACH_VALUES, Association, add_word_of,  # noqa: E402
                         build_association, pair_key, reach_bucket, reach_distance,
                         ReachSlots, spearman)
from train.skit import MODEL_TURNS, Skit  # noqa: E402

# ---------------------------------------------------------------------------------------
# Pre-declared thresholds. THIS MODULE'S OWN — see the module docstring.
# ---------------------------------------------------------------------------------------
#: 3 dial contrasts (near<mid, mid<far, near<far) + 3 coherence guards + 3 adherence checks
#: + 2 handback checks. Stated as the spec pre-declared it, BEFORE any data existed. The two
#: handback checks are not run by this file (see `n_tests_note` in the artifact); keeping them
#: in the family can only make alpha STRICTER, never looser, so it is the conservative choice
#: and is disclosed rather than quietly recounted.
N_TESTS = 11
BONFERRONI_ALPHA = 0.05 / N_TESTS


def critical_t_for(alpha: float) -> float:
    """Two-sided normal critical value at `alpha`.

    A function, not a magic number, so `CRITICAL_T` is DERIVED from this module's own alpha
    and a pasted-in constant from another stage fails a test instead of looking plausible.
    """
    return st.NormalDist().inv_cdf(1.0 - alpha / 2.0)


#: Two-sided critical value at BONFERRONI_ALPHA (~0.0045455). The exact normal quantile is
#: 2.83760; the spec pins 2.843, which is 0.005 STRICTER, so keeping the spec's number can
#: only ever refuse a marginal result and never manufacture one.
CRITICAL_T = 2.843

#: Groundedness at `far` may fall no more than this below its `near` value. Pre-declared in
#: the spec before any data existed. DO NOT MOVE IT.
COHERENCE_MARGIN = 0.05

#: The `add` slot-hit rate must not fall more than this below the best setting's. A model
#: that stops FULFILLING an ambitious plan has a decorative dial.
ADHERENCE_MARGIN = 0.05

#: Off-vocabulary dial value for the PRIMARY control: same shape, same schema, and no meaning
#: AS A DIAL SETTING.
#:
#: "Nonsense" is a claim about the SLOT, not about the word. `blue` is a perfectly ordinary
#: high-frequency content word with its own NPMI profile against any scene, and this corpus is
#: full of blue things. What makes it a control is that it is off-vocabulary for `reach`: the
#: arms never saw it in that position, and it cannot encode "near" or "far". A truly
#: meaning-free token (a random string) would also be off-DISTRIBUTION for the tokenizer and
#: would confound "unknown value" with "unknown token", so a real word is the right trade --
#: but the artifact must not call it meaningless, and does not.
NONSENSE_VALUE = "blue"

#: The model turn this eval measures. See the module docstring.
MEASURED_BLOCK = 1
MEASURED_TURN = MODEL_TURNS[MEASURED_BLOCK]

DEFAULT_SKITS = ROOT / "artifacts" / "reach-skits" / "skits.jsonl"
DERIVE_MANIFEST = ROOT / "artifacts" / "reach-skits" / "derive_manifest.json"
CORPUS = ROOT / "artifacts" / "corpus" / "tinystories.txt"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024-dialogue"
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")
#: The 3000-step arms this eval was written against. Kept as the DEFAULT so
#: `docs/measurements/reach-dial.json` -- published, reviewed, and quoted by
#: `scripts/reach.py` -- still reproduces from a bare `--rescore-from` with no extra flags.
DEFAULT_ARM_ROOT = ROOT / "artifacts" / "reach"
DEFAULT_STEP = 3000
ARM_DIRS = {"dial": DEFAULT_ARM_ROOT / "ckpt-dial",
            "nodial": DEFAULT_ARM_ROOT / "ckpt-nodial"}


def _repo_relative(path: Path) -> str:
    """`path` relative to the repo root when it is inside it, else its absolute form.

    Paths go into the artifact as provenance, so an absolute path from this machine is noise
    and a `relative_to` that raises on an outside path is a crash. Both handled in one place.
    """
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def arm_dirs_for(arm_root: Path) -> Dict[str, Path]:
    """`{arm: checkpoint directory}` under `arm_root`. One place the layout is known."""
    return {"dial": arm_root / "ckpt-dial", "nodial": arm_root / "ckpt-nodial"}


def default_paths(arm_root: Path, step: int) -> Dict[str, Path]:
    """Output, store and work-dir defaults for one (arm_root, step) pair.

    A NAMED FUNCTION because the alternative is a default that silently overwrites a published
    measurement. `docs/measurements/reach-dial.json` is the 3000-step result: it is reviewed,
    it is referenced by `scripts/reach.py`, and a longer-trained rerun that reused the default
    `--out` would destroy it in place while every test still passed. The 3000-step pair keeps
    the historical unsuffixed paths; every other pair gets its own suffix, so two runs cannot
    collide by default and a reader can tell which checkpoint an artifact came from by its
    name alone.

    The work-dir and the generation store are also suffixed, and deliberately NOT placed under
    the checkpoint directory: `artifacts/reach/**` and `artifacts/reach-conv/**` are inputs and
    are written to by nothing here.
    """
    is_original = (arm_root.resolve() == DEFAULT_ARM_ROOT.resolve() and step == DEFAULT_STEP)
    suffix = "" if is_original else f"-{step}"
    return {
        "out": ROOT / "docs" / "measurements" / f"reach-dial{suffix}.json",
        "store": (ROOT / "artifacts" / "reach" / "eval-generations.json" if is_original
                  else ROOT / "artifacts" / f"reach-eval{suffix}" / "eval-generations.json"),
        "work_dir": (ROOT / "artifacts" / "reach" / "hf-eval" if is_original
                     else ROOT / "artifacts" / f"reach-eval{suffix}" / "hf-eval"),
    }

#: (arm, forced dial value). `dial`/near|mid|far is the treatment; `dial`/blue is the PRIMARY
#: control; `nodial`/near|mid|far is the secondary control.
CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("dial", "near"), ("dial", "mid"), ("dial", "far"), ("dial", NONSENSE_VALUE),
    ("nodial", "near"), ("nodial", "mid"), ("nodial", "far"),
)


def condition_key(arm: str, value: str) -> str:
    """The one place a `(arm, value)` pair becomes a string key, so the stored generations,
    the per-setting table and the verdicts cannot drift onto different spellings."""
    return f"{arm}:{value}"


# ---------------------------------------------------------------------------------------
# Prompt construction, and the parity proof.
# ---------------------------------------------------------------------------------------
def forced_block_prefix(block: Dict[str, str], value: str) -> str:
    """The think-block TEXT up to and including the forced `reach:` line.

    Rendered from the same field order `train.improv.render_think` uses, truncated after the
    dial. `offer` and `accept` are the row's own (teacher-forced) so that the ONLY difference
    between two settings of the same scene is the dial value; everything from `add:` onward is
    what the model has to produce.
    """
    return (f"<think>\noffer: {block['offer']}\n"
            f"accept: {block['accept']}\n"
            f"reach: {value}\n")


def prompt_segments(row: Dict[str, Any], value: str, *, block_index: int = MEASURED_BLOCK
                    ) -> List[str]:
    """The prompt as ORDERED SEGMENTS, exactly as `train.skit.skit_segments` orders them.

    Segments, not a string: the caller encodes each one with its own `tok.encode` call and
    concatenates the ids, which is how `scripts/derive_skits.py:build_skit_example` built
    every training example. See the module docstring for what a whole-string tokenization
    would do to this measurement.
    """
    t_idx = MODEL_TURNS[block_index]
    segs: List[str] = [row["prefix"]]
    for i, ti in enumerate(MODEL_TURNS[:block_index]):
        if ti > 0:
            segs.append(row["turns"][ti - 1])
        segs.append(render_think(ReachSlots(**row["blocks"][i])))
        segs.append(row["turns"][ti])
    if t_idx > 0:
        segs.append(row["turns"][t_idx - 1])
    segs.append(forced_block_prefix(row["blocks"][block_index], value))
    return segs


def encode_segments(tok, segments: Sequence[str]) -> List[int]:
    """Concatenated ids, `add_special_tokens` on the FIRST segment only.

    The construction training used. A later segment is a continuation of the same sequence,
    not a fresh one, so it must not pick up its own BOS.
    """
    ids = tok.encode(segments[0])
    for s in segments[1:]:
        ids += tok.encode(s, add_special_tokens=False)
    return ids


def training_example_ids(tok, row: Dict[str, Any]) -> List[int]:
    """The real training example's ids for `row`, built by the real builder."""
    skit = Skit(story_id=row["story_id"], prefix=row["prefix"], turns=tuple(row["turns"]),
                blocks=tuple(ReachSlots(**b) for b in row["blocks"]))
    return build_skit_example(skit, tok, with_think=True,
                             pad_token_id=tok.pad_token_id or 0)["input_ids"]


def check_prompt_parity(tok, rows: Sequence[Dict[str, Any]], *, n_check: int = 32
                        ) -> Dict[str, Any]:
    """Prove this eval's prompt ids ARE a prefix of the dial arm's training ids, and RAISE if not.

    Checked on real held-out rows against the real builder, with the dial forced to the row's
    OWN gold value -- the one case where the forced prompt must be byte-identical to what
    training saw. If it is not, every number this run would produce is measured at a seam the
    arms never saw, so this refuses to generate.

    Also RECORDS (does not require) two negatives, so a reader can see the check has teeth:
      * the whole-string tokenization does NOT match -- the failure mode this guards against;
      * on the `nodial` arm the forced prompt is NOT a prefix of anything, BY CONSTRUCTION,
        because that arm's schema has no `reach` line at all. That is the point of the
        secondary control and the reason it is weaker than the nonsense-value one.
    """
    from train.reach import drop_reach

    checked = 0
    bad: List[int] = []
    whole_string_matches = 0
    nodial_prefix_matches = 0
    seam_tokens: List[str] = []
    seam_tokens_whole: List[str] = []
    for row in rows[:n_check]:
        gold = row["blocks"][MEASURED_BLOCK]["reach"]
        segs = prompt_segments(row, gold)
        ids = encode_segments(tok, segs)
        train_ids = training_example_ids(tok, row)
        if train_ids[:len(ids)] != ids:
            bad.append(row["story_id"])
        # the whole-string form, recorded as a NEGATIVE
        whole = tok.encode("".join(segs))
        if train_ids[:len(whole)] == whole:
            whole_string_matches += 1
        # the nodial arm's example has no `reach` line, so the forced prompt cannot prefix it
        nod = Skit(story_id=row["story_id"], prefix=row["prefix"], turns=tuple(row["turns"]),
                   blocks=tuple(drop_reach(ReachSlots(**b)) for b in row["blocks"]))
        nod_ids = build_skit_example(nod, tok, with_think=True,
                                    pad_token_id=tok.pad_token_id or 0)["input_ids"]
        if nod_ids[:len(ids)] == ids:
            nodial_prefix_matches += 1
        if not seam_tokens:
            n_prefix = len(tok.encode(segs[0]))
            seam_tokens = tok.convert_ids_to_tokens(train_ids[n_prefix - 1:n_prefix + 3])
            seam_tokens_whole = tok.convert_ids_to_tokens(whole[n_prefix - 1:n_prefix + 3])
        checked += 1
    out = {
        "checked_against": "scripts/derive_skits.py:build_skit_example on held-out rows",
        "rows_checked": checked,
        "segment_wise_prompt_is_a_prefix_of_the_dial_training_ids": not bad,
        "rows_that_failed": bad[:10],
        "whole_string_tokenization_also_matches": whole_string_matches,
        "whole_string_note": ("0 is the expected value. The whole-string form re-merges the "
                             "segment seams through BPE and produces a bare '<' where "
                             "training saw a space-prefixed 'Ġ<'; it is recorded here only "
                             "to show this eval is not doing that."),
        "seam_tokens_training_and_segment_wise": seam_tokens,
        "seam_tokens_whole_string": seam_tokens_whole,
        "forced_prompt_is_a_prefix_of_a_NODIAL_training_example": nodial_prefix_matches,
        "nodial_note": ("0 is expected AND is the whole reason `nodial` is only the SECONDARY "
                        "control: that arm's five-slot schema has no `reach` line, so a forced "
                        "one is off-schema in two ways at once (unknown slot name + extra "
                        "line). The nonsense-value condition on the dial arm keeps the schema "
                        "well-formed and varies only the value, which is why it is primary."),
    }
    if bad:
        raise RuntimeError(
            f"eval prompts do not reproduce training's tokenization on {len(bad)} row(s) "
            f"(story_id {bad[:5]}). Every number this run would produce would be measured "
            f"at a seam the arms never saw. Refusing to generate.")
    return out


# ---------------------------------------------------------------------------------------
# Parsing what the model produced.
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ForcedGeneration:
    """One generation's parse. `add_value is None` means the block never named an `add`."""
    closed_block: bool
    add_value: Optional[str]
    stakes: Optional[str]
    handback: Optional[str]
    turn: str


def parse_forced_generation(text: str) -> ForcedGeneration:
    """Parse a continuation that STARTS MID-BLOCK, just after the forced `reach:` line.

    `scripts.eval_skits.split_generation` cannot be reused: it looks for a whole
    ``<think>...</think>`` span and this generation has no opening tag, because the opening
    tag is in the PROMPT. So the block body here is everything before the first
    ``</think>`` and the turn is the FIRST SENTENCE AFTER it.

    The turn must EXCLUDE the block: the block literally contains the `add` word, so scoring
    a "turn" that still held the block would make the slot-hit rate a tautology and the
    groundedness of the turn partly a groundedness of our own prompt.
    """
    idx = text.find("</think>")
    closed = idx >= 0
    body = text[:idx] if closed else text
    after = text[idx + len("</think>"):] if closed else ""
    found: Dict[str, str] = {}
    for line in body.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key in ("add", "stakes", "handback") and value and key not in found:
            found[key] = value
    sents = split_sentences(after)
    return ForcedGeneration(closed_block=closed, add_value=found.get("add"),
                            stakes=found.get("stakes"), handback=found.get("handback"),
                            turn=(sents[0] if sents else ""))


def content_word_repeat_rate(turn: str) -> float:
    """Fraction of the turn's content words that are repeats. A degeneration floor per setting.

    NOT `scripts.eval_skits.four_gram_repeat_rate`, for two reasons. The first is the import:
    that module carries stage 2's threshold constants and this eval's must not be able to
    reach them by any path. The second is that it would be VACUOUS here -- a skit turn is a
    single sentence, so it usually has fewer than four content words and produces no 4-grams
    at all. Measured: every setting scored exactly 0.0000 while generations like "Snowmen are
    snowmen and snowmen" sat in the sample. A metric that cannot fire is worse than no metric.

    Type/token over content words fires on exactly that case (1 - 1/3 = 0.667).
    """
    words = list(content_words(turn))
    if not words:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def add_word_of_value(value: Optional[str]) -> Optional[str]:
    """The single word an `add` slot VALUE names, by the same rule `train.reach.add_word_of`
    applies to a slots object -- one rule, reached from both a dataclass and a raw string, so
    the generated word and the derived word are normalised identically."""
    if not value:
        return None
    w = add_word_of(ReachSlots(offer="", accept="", reach="near", add=value, stakes="+0.0",
                               handback=""))
    return w or None


# ---------------------------------------------------------------------------------------
# The association table, counted target-first.
# ---------------------------------------------------------------------------------------
def needed_lookups(observations: Sequence[Dict[str, Any]]
                   ) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """`(every unigram we will ask for, {add_word: the context words it is paired with})`.

    `a == c` is EXCLUDED: `train.reach.build_association` counts pairs of DISTINCT words
    only, so a self-pair would be a key the full table cannot contain and counting one here
    would make the targeted table differ from the full one on exactly the case where a
    generated `add` word repeats a context word.
    """
    uni: Set[str] = set()
    by_add: Dict[str, Set[str]] = {}
    for obs in observations:
        ctx = set(obs["context_words"])
        uni |= ctx
        for a in obs["add_words"]:
            uni.add(a)
            by_add.setdefault(a, set()).update(c for c in ctx if c != a)
    return uni, by_add


def collect_targeted_association(corpus: Path, *, needed_uni: Set[str],
                                 by_add: Dict[str, Set[str]], story_ids: Set[int],
                                 limit: Optional[int] = None, progress: int = 0
                                 ) -> Tuple[Association, Dict[int, FrozenSet[str]]]:
    """One streaming corpus pass: the counts this eval needs, and nothing else.

    Equivalent to `train.reach.build_association` restricted to `needed_uni` and the pairs in
    `by_add` -- same corpus, same `content_words`, same "document = one whole story", same
    canonical `pair_key`, same `n_docs`. Costs ~2 minutes instead of the full table's ~34,
    and the full table is not persisted so there is nothing to reuse.

    THE EQUIVALENCE IS NOT ASSERTED. `tests/test_eval_reach.py` proves targeted == full on a
    fixture corpus, and `main` re-derives every gold row's stored `reach_distances` from this
    table and refuses to continue on any mismatch.

    Also returns each requested story's OWN content words, which is what makes the
    leave-one-out holdout exact (see `reach_distance_loo`).
    """
    add_words = set(by_add)
    uni: Dict[str, int] = {}
    co: Dict[Tuple[str, str], int] = {}
    own: Dict[int, FrozenSet[str]] = {}
    n = 0
    for i, story in enumerate(iter_stories(corpus, limit)):
        n += 1
        words = set(content_words(story))
        if i in story_ids:
            own[i] = frozenset(words)
        for w in words & needed_uni:
            uni[w] = uni.get(w, 0) + 1
        touched: Set[Tuple[str, str]] = set()
        for a in words & add_words:
            for c in by_add[a] & words:
                touched.add(pair_key(a, c))
        for k in touched:
            co[k] = co.get(k, 0) + 1
        if progress and n % progress == 0:
            print(f"    ... {n:,} stories", flush=True)
    return Association(uni=uni, co=co, n_docs=n), own


def reach_distance_loo(add_word: str, context_words: Sequence[str], assoc: Association,
                       own_words: Optional[FrozenSet[str]]) -> Optional[float]:
    """`train.reach.reach_distance` with an EXACT leave-one-out, over two calls.

    Derivation passed `holdout=True` for every context word because the `add` word and every
    context word are spans of the scored story, so the story always contributes 1 to that
    pair's count. A GENERATED `add` word need not be in the story at all, and subtracting a
    document it never contributed to would bias the generated conditions relative to the gold
    ones. So the context is split: pairs the story really does contribute to are scored
    leave-one-out, the rest are scored whole, and the nearest of the two answers wins.

    On gold rows every context word IS in the story and so is the `add` word, so this reduces
    EXACTLY to `reach_distance(..., holdout=True)` -- which is what lets `main` reproduce the
    stored `reach_distances` bit for bit and thereby prove the targeted table.
    """
    if own_words is None or add_word not in own_words:
        in_own: List[str] = []
        rest = list(context_words)
    else:
        in_own = [c for c in context_words if c in own_words]
        rest = [c for c in context_words if c not in own_words]
    a = reach_distance(add_word, in_own, assoc, holdout=True) if in_own else None
    b = reach_distance(add_word, rest, assoc, holdout=False) if rest else None
    cands = [d for d in (a, b) if d is not None]
    return min(cands) if cands else None


# ---------------------------------------------------------------------------------------
# Statistics. Every decision this eval publishes is one of these functions.
# ---------------------------------------------------------------------------------------
def paired_t(diffs: Sequence[float]) -> Optional[float]:
    """One-sample t on the paired differences, or None when it is undefined.

    Zero scatter is NOT infinite significance: a constant difference means the estimator has
    no sampling variance to divide by and t is undefined, not enormous. Returns None so the
    caller reports "undefined" instead of publishing an artefact of a zero denominator.
    """
    n = len(diffs)
    if n < 2:
        return None
    sd = st.stdev(diffs)
    if sd == 0.0:
        return None
    return st.fmean(diffs) / (sd / math.sqrt(n))


def paired_contrast(a: Sequence[float], b: Sequence[float], *, direction: str,
                    label: str) -> Dict[str, Any]:
    """Is `b` greater than (`direction="b_greater"`) / less than `a`, paired and significant?

    `direction` is REQUIRED and has no default. A guessed direction inverts the verdict, and
    an inverted verdict here rewrites the headline of the whole spec -- so the caller must
    name it and `tests/test_eval_reach.py::test_a_decreasing_series_is_not_a_monotone_dial`
    fails if the comparison is ever flipped.

    `significant` is the CONJUNCTION of the threshold and the direction. A large |t| pointing
    the wrong way is a significant refutation, never a pass.
    """
    if direction not in ("b_greater", "b_less"):
        raise ValueError(f"direction must be 'b_greater' or 'b_less', got {direction!r}")
    if len(a) != len(b):
        raise ValueError(f"unpaired input: {len(a)} vs {len(b)} -- the whole design is paired")
    diffs = [y - x for x, y in zip(a, b)]
    t = paired_t(diffs)
    mean = st.fmean(diffs) if diffs else None
    right_way = (mean is not None and (mean > 0 if direction == "b_greater" else mean < 0))
    return {
        "label": label, "n_pairs": len(diffs), "direction": direction,
        "mean_a": round(st.fmean(a), 6) if a else None,
        "mean_b": round(st.fmean(b), 6) if b else None,
        "mean_delta": round(mean, 6) if mean is not None else None,
        "t": round(t, 4) if t is not None else None,
        "critical_t": CRITICAL_T, "alpha": BONFERRONI_ALPHA,
        "in_the_declared_direction": right_way,
        "significant": bool(t is not None and abs(t) >= CRITICAL_T and right_way),
        "zero_scatter": t is None and len(diffs) >= 2,
    }


def monotonicity_verdict(near: Sequence[float], mid: Sequence[float], far: Sequence[float],
                         *, label: str) -> Dict[str, Any]:
    """THE DECISION FUNCTION OF THIS SPEC: is `near < mid < far` on paired distances?

    Distance is 1 - max NPMI, so a FARTHER reach is a LARGER number and every step must be
    `b_greater`. Getting that backwards inverts the whole result, so the direction is passed
    explicitly per contrast and a decreasing series is tested to come out NOT monotone.

    `monotone` requires all three contrasts significant AND in the declared direction --
    including `near < far`, which cannot be inferred from the two adjacent steps once either
    of them is null.
    """
    steps = [paired_contrast(near, mid, direction="b_greater", label=f"{label}: near<mid"),
             paired_contrast(mid, far, direction="b_greater", label=f"{label}: mid<far"),
             paired_contrast(near, far, direction="b_greater", label=f"{label}: near<far")]
    return {
        "label": label,
        "near_lt_mid": steps[0], "mid_lt_far": steps[1], "near_lt_far": steps[2],
        "monotone": all(s["significant"] for s in steps),
        "n_significant_steps": sum(1 for s in steps if s["significant"]),
        # KEY NAME IS A CONTRACT: `scripts/reach.py --about` quotes this field and
        # `tests/test_reach_cli.py` pins it. The name stays; what changed is that it no longer
        # ships as a bare claim about the run.
        "cleaner_contrast": "mid_lt_far",
        "cleaner_contrast_is_THE_SPECS_DESIGNATION_not_a_finding_about_this_run": True,
        "cleaner_contrast_provenance": (
            "THE SPEC'S framing, and a property of the DERIVATION's gold label distribution, "
            "not of this run: the corpus-scale per-bucket median add_df is non-monotone (near "
            "16,591 / mid 51,269 / far 39,367 -- `mid` commonest), which would make "
            "near-vs-far the most frequency-exposed step. Whether that transfers has to be "
            "CHECKED against the add words this run actually produced -- see "
            "effects.realised_frequency_profile, which computes it. Do not quote this field "
            "as a finding about this run."),
    }


def coherence_guard(g_near: Sequence[float], g_far: Sequence[float], *,
                    margin: float = COHERENCE_MARGIN) -> Dict[str, Any]:
    """Did groundedness survive at `far`? The "nobody can follow" test.

    Pre-declared in the spec before any data existed: groundedness at `far` may fall no more
    than `margin` below its `near` value. A dial that only produces noise at `far` has FAILED,
    not succeeded, so this is a hard gate on the EUREKA verdict and not a footnote.

    A RISE passes trivially, which is correct -- the guard is one-sided by design.
    """
    near_mean = st.fmean(g_near) if g_near else None
    far_mean = st.fmean(g_far) if g_far else None
    drop = (near_mean - far_mean) if (near_mean is not None and far_mean is not None) else None
    return {
        "groundedness_near": round(near_mean, 6) if near_mean is not None else None,
        "groundedness_far": round(far_mean, 6) if far_mean is not None else None,
        "drop_far_below_near": round(drop, 6) if drop is not None else None,
        "declared_margin": margin,
        "passes": bool(drop is not None and drop <= margin),
        "note": ("pre-declared in docs/superpowers/specs/2026-08-23-reach-dial-design.md "
                 "before any data existed; not moved."),
        "the_guard_is_NOT_independent_of_the_effect": (
            "READ THIS BEFORE READING THE VERDICT. `groundedness` is the mean over the turn's "
            "FRESH words of the strongest NPMI to the context, and the dial's realised "
            "distance is 1 - the strongest NPMI of the `add` word to the context. They are "
            "the same quantity with opposite signs over overlapping word sets, so asking for "
            "a farther reach mechanically pushes groundedness DOWN. That cuts both ways: a "
            "failed guard is not independent evidence that the model became incoherent, and a "
            "passed guard is not independent evidence that it stayed coherent. The guard is "
            "applied exactly as pre-declared -- moving it after seeing the data is worse -- "
            "but it should be read as 'how much NPMI mass did the turn as a whole lose', not "
            "as a semantically independent coherence judgement."),
        "context_spans_differ": (
            "groundedness scores the generated turn against the PRECEDING PARTNER TURN only "
            "(the stage-2 convention, kept for comparability), while the reach distance scores "
            "the `add` word against prefix + turn 0 + turn 1. Different spans."),
    }


def adherence_readings(rates: Dict[str, float], near_key: str, mid_key: str, far_key: str,
                       *, margin: float = ADHERENCE_MARGIN) -> Dict[str, Any]:
    """The `add` slot-hit rate read THREE ways, with the declared gate named as the gate.

    `adherence_guard` implements the brief's wording -- "the rate must HOLD AT EVERY SETTING"
    -- as worst-vs-best, and that is the gate; it was written before any data existed and is
    not moved. But the spec's stated WORRY was narrower ("if the model stops FULFILLING its
    plan when the plan gets AMBITIOUS"), i.e. a fall at `far`. Those two readings can disagree,
    and when they do a reader must be able to see which one failed and in which direction
    rather than being handed one boolean. So all three are reported and only one is the gate.
    """
    declared = adherence_guard({near_key: rates[near_key], mid_key: rates[mid_key],
                                far_key: rates[far_key]}, margin=margin)
    far_vs_near = rates[far_key] - rates[near_key]
    far_vs_best = rates[far_key] - max(rates[near_key], rates[mid_key])
    return {
        **declared,
        "THE_GATE": "worst_vs_best (`hold at every setting`), reported above as `passes`",
        "reading_far_minus_near": round(far_vs_near, 6),
        "reading_far_minus_near_passes": bool(far_vs_near >= -margin),
        "reading_far_minus_best_of_near_mid": round(far_vs_best, 6),
        "reading_far_minus_best_passes": bool(far_vs_best >= -margin),
        "which_setting_is_worst": declared["worst_setting"],
        "is_this_undertraining": (
            "The rate sits around 0.5 at EVERY setting: the model fulfils its own declared "
            "plan barely half the time regardless of the dial, which looks far more like a "
            "weak plan-follower than a dial-specific defect. Whether 'weak' means "
            "'undertrained' is answerable rather than arguable -- run the identical eval on a "
            "longer-trained checkpoint of the SAME run and look. If a `versus_*_steps` block "
            "is present in this artifact, it has the answer; and note that the learning rate "
            "is CONSTANT in this recipe, so 'val loss still falling' was never evidence that "
            "more steps would fix it (see limitations). Either way the gate fails here and "
            "the verdict stands: this is an explanation to test, not an excuse."),
        "READ_THIS": (
            "The three readings do not agree, and the direction of the failure matters. The "
            "declared gate fails, but the WORST setting is `near`, not `far`: the rate is "
            "non-monotone (mid highest), so this is not the 'ambitious plan goes unfulfilled' "
            "failure the guard was written for. Under `far` vs `near` -- the spec's own "
            "wording -- `far` is ABOVE `near` and the reading passes. Under `far` vs the best "
            "setting it fails, narrowly. The declared gate is the one that decides "
            "`eureka_criterion_met`; it is not moved after the fact, and the disagreement is "
            "published rather than resolved in our favour."),
    }


def adherence_guard(rates: Dict[str, float], *, margin: float = ADHERENCE_MARGIN
                    ) -> Dict[str, Any]:
    """Does the `add` slot-hit rate HOLD at every setting?

    If the model stops FULFILLING its plan when the plan gets ambitious, the dial is
    decorative -- it moves the declaration without moving the behaviour. Measured as the
    worst setting's shortfall against the best setting's rate.
    """
    if not rates:
        return {"passes": False, "rates": {}, "note": "no rates"}
    best = max(rates.values())
    worst_key = min(rates, key=lambda k: rates[k])
    shortfall = best - rates[worst_key]
    return {
        "rates": {k: round(v, 6) for k, v in rates.items()},
        "best_setting": max(rates, key=lambda k: rates[k]),
        "worst_setting": worst_key,
        "shortfall": round(shortfall, 6),
        "declared_margin": margin,
        "passes": bool(shortfall <= margin),
    }


def nonsense_control_verdict(near: Sequence[float], far: Sequence[float],
                             nonsense: Sequence[float]) -> Dict[str, Any]:
    """Does a MEANINGLESS dial value reproduce the dial's effect? THE PRIMARY CONTROL.

    A single off-vocabulary value cannot produce a monotone triple, so "does the nonsense
    value reproduce the pattern" has to be asked about the thing that would actually sink the
    headline: **forcing any token at all pushes the model to the `far` end.** That is true
    exactly when the nonsense condition

      * differs from `near` in the same direction `far` does, AND
      * is NOT distinguishable from `far`.

    Both halves are required. A nonsense value that sits away from `near` but also away from
    `far` is not reproducing the dial -- it is its own third thing, which is what a
    value-sensitive model should do with a word it never saw in that slot.

    Returns `reproduces_the_dial_pattern`, and THAT is the gate `eureka_verdict` reads.
    """
    vs_near = paired_contrast(near, nonsense, direction="b_greater",
                              label="nonsense vs near")
    vs_far = paired_contrast(far, nonsense, direction="b_greater", label="nonsense vs far")
    far_vs_near = paired_contrast(near, far, direction="b_greater", label="far vs near")
    away_from_near = bool(vs_near["significant"])
    indistinguishable_from_far = not bool(
        vs_far["t"] is not None and abs(vs_far["t"]) >= CRITICAL_T)
    rng = far_vs_near["mean_delta"]
    position = None
    if vs_near["mean_delta"] is not None and rng not in (None, 0):
        position = round(vs_near["mean_delta"] / rng, 4)
    # The equivalence bound the gate does NOT have. `indistinguishable_from_far` is a
    # failure-to-reject, so on its own it says "we could not tell them apart", not "they are
    # the same". Reporting the smallest difference this n could have detected turns that into
    # a statement a reader can check: the gate can only fire when `blue` lands inside this
    # band around `far`, and the band is a small slice of the dial's range.
    sdd = None
    band = None
    if vs_far["t"] not in (None, 0) and vs_far["mean_delta"] is not None:
        se = abs(vs_far["mean_delta"] / vs_far["t"])
        sdd = round(CRITICAL_T * se, 6)
        if rng:
            band = round(sdd / abs(rng), 4)
    return {
        "role": ("PRIMARY CONTROL -- same arm, well-formed on-schema six-slot block, "
                 "off-vocabulary value. It isolates the dial's VALUE from its SCHEMA, which "
                 "the `nodial` arm cannot."),
        "vs_near": vs_near, "vs_far": vs_far, "far_vs_near_for_reference": far_vs_near,
        "position_on_the_near_far_range": position,
        "position_is_NOT_a_share_of_the_effect": (
            "This number is where the off-vocabulary value LANDS between `near` and `far`. It "
            "is NOT the fraction of the dial that is schema rather than value, and an earlier "
            "draft published it as if it were, which UNDERSTATED the result. There is no "
            "zero-token baseline in this design -- every condition has some token in the "
            "reach line -- so `blue`'s position cannot be decomposed into a schema share and "
            "a value share. What the data support is the ORDERING: `blue` is significantly "
            "displaced from `near` and from `far` in OPPOSITE directions, and an "
            "off-vocabulary token cannot encode `near` or `far`, so both displacements are "
            "value-specific. Publish the ordering, not the percentage."),
        "moved_away_from_near": away_from_near,
        "distinguishable_from_far": not indistinguishable_from_far,
        "indistinguishable_from_far": indistinguishable_from_far,
        "smallest_detectable_difference_from_far": sdd,
        "smallest_detectable_difference_as_share_of_range": band,
        "THE_GATE_IS_WEAK_AND_HERE_IS_HOW_WEAK": (
            "`indistinguishable_from_far` is a FAILURE TO REJECT, not an equivalence test, so "
            "this gate can only fire when the nonsense value lands within "
            f"{sdd} of `far` -- about {band} of the near-to-far range. It passes over the "
            "other ~90%. On this run it passes on POSITIVE evidence rather than on absence of "
            "it (the difference from `far` is many standard errors), which is the only reason "
            "it is worth anything here; a future run where `blue` drifted toward `far` "
            "without crossing that band would pass this gate while deserving to fail. Read "
            "`vs_far` and the bound, not the boolean."),
        "reproduces_the_dial_pattern": bool(away_from_near and indistinguishable_from_far),
        "rule": ("fails (i.e. reproduces the pattern) only when the nonsense value BOTH moves "
                 "significantly away from `near` in the dial's direction AND cannot be "
                 "distinguished from `far`. Either one alone is not 'any token gives you "
                 "far'."),
    }


def ols_residuals(ys: Sequence[float], xs: Sequence[float]) -> List[float]:
    """`ys` with the best straight-line fit on `xs` removed. Frequency control, method 2.

    A zero-variance `xs` cannot explain anything, so the residuals are the centred `ys` --
    NOT a division by zero and not a silent pass-through of the raw values.
    """
    n = len(ys)
    if n != len(xs):
        raise ValueError(f"ols_residuals: {n} ys vs {len(xs)} xs")
    if n == 0:
        return []
    mx, my = st.fmean(xs), st.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return [y - my for y in ys]
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    return [y - (intercept + slope * x) for y, x in zip(ys, xs)]


def df_matched_pairs(a_vals: Sequence[float], a_dfs: Sequence[float],
                     b_vals: Sequence[float], b_dfs: Sequence[float], *,
                     tol: float) -> Tuple[List[float], List[float]]:
    """The paired subsample whose two `add` words have (log) document frequency within `tol`.

    Frequency control, method 1: instead of modelling the confound away, keep only the pairs
    where it is not there. Pairing is preserved -- both halves keep the same scene indices --
    so the resulting contrast is the same paired test on a cleaner subsample.

    **THIS PRIMITIVE IS NULL-ENRICHING AND MUST NOT BE READ ALONE.** When the model emits the
    SAME `add` word under both settings the two document frequencies are identical, so
    ``|dlog df| = 0`` and the pair is ALWAYS kept -- and its distance difference is exactly
    0.0, because it is the same word scored against the same context. Every scene where the
    dial did nothing therefore survives the filter, while a scene where the dial changed the
    word survives only if the two words happen to be about equally common. Measured on this
    run, 78-92% of the retained pairs were identical-word pairs contributing a structural
    zero. Use `df_matched_contrast`, which splits those out.
    """
    keep = [i for i in range(len(a_vals)) if abs(a_dfs[i] - b_dfs[i]) <= tol]
    return [a_vals[i] for i in keep], [b_vals[i] for i in keep]


def df_matched_contrast(a_vals: Sequence[float], a_dfs: Sequence[float],
                        a_words: Sequence[Optional[str]], b_vals: Sequence[float],
                        b_dfs: Sequence[float], b_words: Sequence[Optional[str]], *,
                        tol: float, label: str) -> Dict[str, Any]:
    """The df-matched contrast WITH its null-enrichment made visible, and corrected for.

    THE BUG THIS EXISTS TO FIX. An earlier version of this eval published the raw matched
    contrast and concluded that the frequency-matched effect was 3-5x smaller than the
    residualised one, i.e. that "a large share of the raw movement IS frequency". That reading
    was **wrong**, and it was wrong because of the FILTER, not the model: the matched
    subsample is dominated by scenes where the dial emitted the identical word under both
    settings, whose distance difference is 0.0 by construction. Averaging a real effect over a
    pool padded with structural zeros shrinks it by exactly the padding ratio and says nothing
    at all about frequency.

    So the contrast is reported twice: over everything kept (comparable to the old number, and
    uninterpretable on its own), and over the INFORMATIVE pairs -- those where the dial
    actually changed the word, so the difference CAN be non-zero. The informative mean is the
    one that answers the question.

    `n_informative` is also the honest power statement: a step with a few dozen informative
    pairs is UNDERPOWERED, not refuted.
    """
    if not (len(a_vals) == len(b_vals) == len(a_dfs) == len(b_dfs)
            == len(a_words) == len(b_words)):
        raise ValueError("df_matched_contrast: unpaired input")
    keep = [i for i in range(len(a_vals)) if abs(a_dfs[i] - b_dfs[i]) <= tol]
    av = [a_vals[i] for i in keep]
    bv = [b_vals[i] for i in keep]
    same = [i for i in keep if a_words[i] is not None and a_words[i] == b_words[i]]
    info = [i for i in keep if b_vals[i] - a_vals[i] != 0.0]
    out = paired_contrast(av, bv, direction="b_greater", label=label)
    out["kept_pairs_of"] = len(a_vals)
    out["log_df_tolerance"] = tol
    out["n_same_add_word"] = len(same)
    out["n_exact_zero_difference"] = len(keep) - len(info)
    out["n_informative"] = len(info)
    out["structural_zero_share_of_kept"] = (round((len(keep) - len(info)) / len(keep), 4)
                                            if keep else None)
    out["informative_only"] = (
        paired_contrast([a_vals[i] for i in info], [b_vals[i] for i in info],
                        direction="b_greater", label=f"{label} (informative pairs only)")
        if len(info) >= 2 else None)
    out["READ_THE_INFORMATIVE_ROW"] = (
        "`mean_delta` on this row is averaged over a pool in which "
        f"{out['structural_zero_share_of_kept']} of the pairs are the SAME `add` word under "
        "both settings and therefore contribute an exact 0.0 by construction. That is not a "
        "smaller effect; it is the same effect divided by the padding. `informative_only` is "
        "the interpretable number, and `n_informative` is the honest power statement.")
    return out


def frequency_control_agreement(raw: Dict[str, Any], residualised: Dict[str, Any],
                                matched: Dict[str, Any]) -> Dict[str, Any]:
    """Do the two frequency controls agree, and how much of the raw effect IS frequency?

    A named function because an earlier draft answered this in prose, inside the artifact, and
    got it backwards. The comparison must be residualised-vs-**informative**-matched: setting
    the residualised effect against the null-padded matched mean is a category error of
    exactly the kind this project's notes warn about -- two averages that no longer contain
    the same population.

    `near_lt_far` is the reference contrast: largest effect, most informative pairs, so it is
    the step where the two methods can actually be compared.
    """
    raw_d = raw["near_lt_far"]["mean_delta"]
    res_d = residualised["near_lt_far"]["mean_delta"]
    inf = matched["near_lt_far"].get("informative_only")
    inf_d = inf["mean_delta"] if inf else None
    surviving = (res_d / raw_d) if (raw_d not in (None, 0) and res_d is not None) else None
    agree = None
    if inf_d is not None and res_d is not None and raw_d:
        agree = abs(inf_d - res_d) <= 0.25 * abs(raw_d)
    return {
        "reference_contrast": "near_lt_far",
        "raw_mean_delta": raw_d,
        "residualised_mean_delta": res_d,
        "df_matched_informative_mean_delta": inf_d,
        "df_matched_informative_n": (inf["n_pairs"] if inf else None),
        "share_of_raw_surviving_frequency_control": (round(surviving, 4)
                                                     if surviving is not None else None),
        "share_of_raw_attributable_to_log_add_df": (round(1.0 - surviving, 4)
                                                    if surviving is not None else None),
        "the_two_controls_agree": agree,
        "agreement_rule": ("the two informative estimates differ by no more than a quarter of "
                           "the raw effect"),
        "READ_THIS": (
            "About half the raw movement is document frequency and about half survives it -- "
            "and the two frequency controls, once the matched one is read over its INFORMATIVE "
            "pairs rather than over a pool padded with identical-word zeros, land in the same "
            "place. A step that is not significant under matching with only a few dozen "
            "informative pairs is UNDERPOWERED, not refuted."),
    }


def realised_frequency_profile(log_dfs_by_setting: Dict[str, Sequence[float]],
                               order: Sequence[str],
                               spearman_here: Optional[float]) -> Dict[str, Any]:
    """Is the frequency confound monotone across the dial IN THIS RUN? COMPUTED, never quoted.

    THE BUG THIS EXISTS TO FIX. This block used to carry a hard-coded ``not_monotone: True``
    next to the DERIVATION's per-bucket medians (near 16,591 / mid 51,269 / far 39,367), under
    a key named `frequency_confound_HERE`. Both were false of this run: the realised `add_df`
    the model produced is strictly monotone across near/mid/far. A literal that describes a
    different population, published under a key that claims to describe this one, is the exact
    shape of the "trust the subject, verify the instrument" failure -- so the answer is
    computed from the run's own data and the derivation's numbers are labelled as the
    derivation's.

    It matters beyond tidiness: the spec's "mid-vs-far is the cleaner contrast" framing was
    derived from the derivation's non-monotone distribution. If the realised confound is
    monotone, that framing does not transfer, and this function is what says so.
    """
    means = {k: (st.fmean(log_dfs_by_setting[k]) if log_dfs_by_setting[k] else None)
             for k in order}
    vals = [means[k] for k in order]
    monotone = all(v is not None for v in vals) and all(
        vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    return {
        "measured_on": "the add words THIS RUN generated, not the derivation's gold ones",
        "mean_log_add_df_by_setting": {k: (round(v, 4) if v is not None else None)
                                       for k, v in means.items()},
        "monotone_across_the_dial": monotone,
        "not_monotone": not monotone,
        "spearman_log_add_df_vs_distance_in_this_run": spearman_here,
        "consequence_for_the_specs_cleaner_contrast_framing": (
            "The spec named `mid` vs `far` the CLEANER contrast because the DERIVATION's "
            "per-bucket median add_df is non-monotone (near 16,591 / mid 51,269 / far 39,367 "
            "-- `mid` commonest). That framing is a property of the gold label distribution. "
            + ("It does NOT transfer to this run: the realised add_df the model produced is "
               "strictly monotone across the dial, so every step is exposed to the confound "
               "in the same direction and no step is privileged as 'cleaner'. The frequency "
               "control below, not the choice of contrast, is what separates them."
               if monotone else
               "It does transfer: the realised add_df is non-monotone here too.")),
    }


def dial_or_frequency_dial(raw: Dict[str, Any], controlled: Dict[str, Any],
                           matched: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Is this a REACH dial, a FREQUENCY dial, or no dial at all?

    Per the spec amendment: a dial that moves raw distance but NOT frequency-controlled
    distance is a frequency dial -- a real finding about NPMI rather than about the model --
    and must be published as that. This is the function that makes that call, so it is
    fixture-tested rather than decided inside `main`.
    """
    if raw["monotone"] and controlled["monotone"]:
        verdict = "REACH DIAL"
    elif raw["monotone"] and not controlled["monotone"]:
        verdict = "FREQUENCY DIAL"
    elif not raw["monotone"] and controlled["monotone"]:
        verdict = "PARTIAL: frequency-controlled monotone, raw not"
    else:
        verdict = "NO DIAL"
    out = {
        "verdict": verdict,
        "raw_monotone": raw["monotone"],
        "frequency_controlled_monotone": controlled["monotone"],
        "raw_steps_significant": raw["n_significant_steps"],
        "frequency_controlled_steps_significant": controlled["n_significant_steps"],
        "rule": ("raw AND controlled -> REACH DIAL; raw only -> FREQUENCY DIAL (published as "
                 "such); neither -> NO DIAL. The GATE is the residualised control, which is "
                 "what this function was written against before any data existed."),
    }
    if matched is not None:
        # REPORTED, NOT GATING, and said so out loud. The spec's amendment requires all three
        # frequency reports; it does not say which of the two control METHODS gates, and this
        # function's gate was fixed before the data existed. Changing it now, after seeing
        # that the two methods disagree, is precisely the move this project forbids -- so the
        # disagreement is published beside the verdict instead of quietly deciding it.
        steps = ("near_lt_mid", "mid_lt_far", "near_lt_far")
        n_sig_info = sum(1 for k in steps
                         if (matched[k].get("informative_only") or {}).get("significant"))
        out["second_frequency_control_df_matched_subsample"] = {
            "steps_significant_over_informative_pairs": n_sig_info,
            "per_step": {
                k: {"kept": matched[k]["n_pairs"],
                    "n_same_add_word": matched[k]["n_same_add_word"],
                    "n_exact_zero_difference": matched[k]["n_exact_zero_difference"],
                    "structural_zero_share_of_kept": matched[k][
                        "structural_zero_share_of_kept"],
                    "n_informative": matched[k]["n_informative"],
                    "mean_delta_over_kept_UNINTERPRETABLE": matched[k]["mean_delta"],
                    "mean_delta_over_informative": (
                        (matched[k].get("informative_only") or {}).get("mean_delta")),
                    "t_over_informative": (
                        (matched[k].get("informative_only") or {}).get("t")),
                    "significant_over_informative": (
                        (matched[k].get("informative_only") or {}).get("significant")),
                    }
                for k in steps},
            "READ_THIS": (
                "REPORTED, NOT GATING. Read the `_over_informative` columns and ignore "
                "`mean_delta_over_kept_UNINTERPRETABLE`. The matching filter keeps EVERY scene "
                "where the dial emitted the same `add` word under both settings -- identical "
                "word means identical document frequency, so |dlog df| is 0 and the pair "
                "always passes, while its distance difference is 0.0 by construction. Those "
                "structural zeros are the large majority of what is kept, so the kept-pool "
                "mean is the real effect divided by the padding, NOT evidence that the effect "
                "is smaller. Over the informative pairs the matched estimate agrees with the "
                "residualised one (see effects.frequency_control_agreement). A step that is "
                "not significant here with only a few dozen informative pairs is UNDERPOWERED, "
                "not refuted."),
        }
    return out


def eureka_verdict(*, dial_kind: Dict[str, Any], coherence: Dict[str, Any],
                   adherence: Dict[str, Any], nonsense: Dict[str, Any],
                   nodial: Dict[str, Any]) -> Dict[str, Any]:
    """The whole spec's headline, as ONE CALLED FUNCTION.

    EUREKA requires ALL of:
      * a frequency-controlled monotone dial effect (`REACH DIAL`, not `FREQUENCY DIAL`);
      * coherence surviving at `far` within the pre-declared 0.05 margin;
      * the `add` slot-hit rate holding at every setting;
      * the PRIMARY control (nonsense value on the same arm, same schema) NOT reproducing the
        `far` end -- see `nonsense_control_verdict`. If a meaningless value gets you the same
        distance `far` does, the finding is "perturbing the reach line moves generation", not
        "the dial works".

    The `nodial` arm is reported and carried in the reasons but is NOT a gate: it is off-
    schema two ways at once and so cannot separate value-sensitivity from malformed-block
    sensitivity. It is the secondary control on purpose.
    """
    reasons: List[str] = []
    if dial_kind["verdict"] != "REACH DIAL":
        reasons.append(f"dial verdict is {dial_kind['verdict']}, not REACH DIAL")
    if not coherence["passes"]:
        reasons.append(f"coherence guard failed: groundedness fell "
                       f"{coherence['drop_far_below_near']} at far, margin "
                       f"{coherence['declared_margin']}")
    if not adherence["passes"]:
        # NAME THE SETTINGS. "fell 0.0896 between settings" sitting beside a coherence-at-`far`
        # gate and a `far`-end narrative reads as a `far`-side collapse; on this run the worst
        # setting is `near` and `far` is ABOVE it. A reason string that a reader can misread
        # into the opposite finding is a reporting bug, not a wording preference.
        reasons.append(
            f"add slot-hit rate fell {adherence['shortfall']} from the best setting "
            f"{adherence['best_setting']} ({adherence['rates'][adherence['best_setting']]}) "
            f"to the WORST setting {adherence['worst_setting']} "
            f"({adherence['rates'][adherence['worst_setting']]}), margin "
            f"{adherence['declared_margin']}. NOTE THE DIRECTION: "
            + ("the worst setting is NOT `far`, so this is not a far-end collapse -- see "
               "adherence_guard.READ_THIS."
               if not adherence["worst_setting"].endswith("far")
               else "the worst setting IS `far`, the failure the guard was written for."))
    if nonsense.get("reproduces_the_dial_pattern"):
        reasons.append("PRIMARY CONTROL FAILED: an off-vocabulary dial value reproduces the "
                       "`far` end, so the effect is 'any token in the reach line' rather "
                       "than the dial's VALUE")
    met = not reasons
    return {
        "eureka_criterion_met": met,
        "dial_kind_verdict": dial_kind["verdict"],
        "QUOTING_EITHER_ALONE_IS_MISLEADING": (
            f"`eureka_criterion_met` is {met} AND `dial_kind.verdict` is "
            f"{dial_kind['verdict']}. These are different questions and on this run they have "
            f"different answers: the DIAL EFFECT is established (monotone raw and after "
            f"frequency control), while the pre-declared EUREKA criterion additionally "
            f"requires coherence, adherence and the primary control, and it is the adherence "
            f"guard that decides it. Quoting `dial_kind.verdict` alone overstates; quoting "
            f"`eureka_criterion_met` alone understates. Report both."),
        "reasons_against": reasons,
        "gates": {
            "frequency_controlled_monotone_dial": dial_kind["verdict"] == "REACH DIAL",
            "coherence_survives_at_far": coherence["passes"],
            "add_slot_hit_rate_holds": adherence["passes"],
            "primary_control_null": not nonsense.get("reproduces_the_dial_pattern"),
        },
        "secondary_control_nodial_monotone": nodial.get("monotone"),
        "secondary_control_note": ("reported, NOT a gate. `nodial` never saw a `reach` slot, "
                                   "so a forced one is off-schema in two ways at once and "
                                   "movement there cannot be attributed to the dial's VALUE."),
    }


def check_step_matches_manifests(manifests: Dict[str, Dict[str, Any]], step: int) -> None:
    """RAISE unless every arm's manifest describes the checkpoint being evaluated.

    A named function rather than three lines in `main()`, because it decides whether the
    artifact's entire provenance block is true. Point `--step 9000` at the 3000-step
    directory and the numbers are the 3000-step model's while every `lr`, `stop_reason`,
    `val_loss_last` and convergence paragraph beside them describes a different run -- and
    nothing else in this file would notice.

    (Found by a mutation that deleted the check and survived: the guard was unreachable from
    any test because it lived in the driver. Same class as the composer mutants.)
    """
    for arm, m in manifests.items():
        got = m.get("steps")
        if got is None or int(got) != int(step):
            raise RuntimeError(
                f"{arm} manifest says steps={got} but the checkpoint being evaluated is "
                f"step_{step}: the checkpoint and the manifest describe different runs, and "
                f"every provenance field in the artifact would be wrong")


def prior_step_of(prior: Dict[str, Any]) -> Optional[int]:
    """Which checkpoint step a previously published artifact describes.

    `design.checkpoint.step` is where it lives now. The FIRST published artifact predates that
    field, and re-running it just to add one would rewrite a reviewed measurement -- so this
    also reads the step out of the limitations block, which that artifact does carry. Returns
    None when neither is present, and the caller names the block `versus_prior_steps` rather
    than guessing a number.
    """
    step = prior.get("design", {}).get("checkpoint", {}).get("step")
    if step is not None:
        return int(step)
    for lim in prior.get("limitations", []):
        if isinstance(lim, dict) and lim.get("steps") is not None:
            return int(lim["steps"])
    return None


def versus_previous(this: Dict[str, Any], prior: Dict[str, Any]) -> Dict[str, Any]:
    """Diff this run against a previously published one, with the deltas COMPUTED.

    The two questions a longer-trained rerun exists to answer, answered rather than left for a
    reader to subtract:

      1. **Does the effect grow with training?** The shorter run's effects were published as a
         FLOOR. This puts the residualised `near<far` side by side with its delta.
      2. **Does the failing adherence gate resolve?** If it passes at the longer budget, the
         failure was an undertraining artefact and this says so plainly. If it still fails,
         undertraining is REFUTED as the explanation and the gate is telling us something about
         the model -- which is the more interesting outcome and the one that must not be
         softened.

    Comparability is CHECKED, not assumed: same scenes, same conditions, same alpha. Two runs
    over different held-out sets are not two readings of one experiment, and this project's
    notes carry a case where exactly that comparison cost a day.
    """
    def scenes(a):
        return sorted(r["story_id"] for r in a.get("stored", {}).get("rows", []))

    same_scenes = scenes(this) == scenes(prior)
    same_alpha = (this["thresholds"]["critical_t"] == prior["thresholds"]["critical_t"]
                  and this["thresholds"]["n_tests"] == prior["thresholds"]["n_tests"])
    same_conditions = (this["design"]["conditions"] == prior["design"]["conditions"])

    def eff(a, block, step):
        return a["effects"][block][step]

    rows = {}
    for block in ("raw_distance", "frequency_residualised_distance"):
        for stp in ("near_lt_mid", "mid_lt_far", "near_lt_far"):
            t_, p_ = eff(this, block, stp), eff(prior, block, stp)
            rows[f"{block}.{stp}"] = {
                "prior_mean_delta": p_["mean_delta"], "this_mean_delta": t_["mean_delta"],
                "change": round(t_["mean_delta"] - p_["mean_delta"], 6),
                "prior_t": p_["t"], "this_t": t_["t"],
                "prior_significant": p_["significant"],
                "this_significant": t_["significant"],
            }

    ta, pa = this["adherence_guard"], prior["adherence_guard"]
    adherence = {
        "prior_rates": pa["rates"], "this_rates": ta["rates"],
        "prior_shortfall": pa["shortfall"], "this_shortfall": ta["shortfall"],
        "shortfall_change": round(ta["shortfall"] - pa["shortfall"], 6),
        "prior_worst_setting": pa["worst_setting"], "this_worst_setting": ta["worst_setting"],
        "prior_passes": pa["passes"], "this_passes": ta["passes"],
        "resolved_with_more_training": bool(ta["passes"] and not pa["passes"]),
        "verdict": (
            "RESOLVED: the guard failed at the shorter budget and passes here, so that failure "
            "WAS an undertraining artefact."
            if (ta["passes"] and not pa["passes"]) else
            "STILL FAILS at 3x the training budget. Undertraining is REFUTED as the "
            "explanation: the guard is telling us something real about the model, not about "
            "the step count. Note the direction before reading it as a far-end collapse."
            if (not ta["passes"] and not pa["passes"]) else
            "the guard passed at the shorter budget too" if ta["passes"] else
            "REGRESSED: it passed at the shorter budget and fails here."),
    }

    # DID THE MODEL ACTUALLY CHANGE? When two runs report effects that agree to four decimal
    # places, the first thing to rule out is that the same checkpoint was evaluated twice --
    # a wrong `--arm-root`, a stale HF work-dir, a copied store. This measures it directly
    # instead of trusting the provenance strings: the fraction of the 7,000 stored
    # continuations that differ. A near-zero churn means the instrument, not the model, is
    # producing the agreement, and the comparison must be thrown away.
    ga = this.get("stored", {}).get("generations", {})
    gb = prior.get("stored", {}).get("generations", {})
    churn = None
    if ga and gb and set(ga) == set(gb):
        tot = sum(len(ga[k]) for k in ga)
        dif = sum(1 for k in ga for x, y in zip(ga[k], gb[k]) if x != y)
        churn = {"generations_compared": tot, "generations_that_differ": dif,
                 "fraction_changed": round(dif / tot, 4) if tot else None,
                 "the_model_really_did_change": bool(tot and dif / tot > 0.05),
                 "why": ("two runs whose effects agree to four decimal places are first "
                         "suspected of being the SAME checkpoint evaluated twice. This rules "
                         "that out from the stored text rather than from a provenance string. "
                         "A high churn beside unchanged aggregates is the interesting "
                         "outcome: the model's individual choices moved a lot and the dial's "
                         "aggregate effect did not."),
                 }

    tc, pc = this["coherence_guard"], prior["coherence_guard"]
    return {
        "generation_churn": churn,
        "compared_against_step": prior_step_of(prior),
        "comparable": {
            "same_held_out_scenes": same_scenes,
            "same_conditions": same_conditions,
            "same_alpha_and_critical_t": same_alpha,
            "all_three": bool(same_scenes and same_conditions and same_alpha),
            "why": ("a longer-trained rerun is only a rerun if the scenes, the conditions and "
                    "the thresholds are identical. Checked here rather than asserted."),
        },
        "question_1_does_the_effect_grow_with_training": {
            "headline": rows["frequency_residualised_distance.near_lt_far"],
            "all_steps": rows,
            "answer": (
                "the frequency-controlled near->far effect moved from "
                f"{rows['frequency_residualised_distance.near_lt_far']['prior_mean_delta']} to "
                f"{rows['frequency_residualised_distance.near_lt_far']['this_mean_delta']} "
                f"(change "
                f"{rows['frequency_residualised_distance.near_lt_far']['change']:+.6f}). The "
                "shorter run's effects were published as a floor; this says by how much."),
        },
        "question_2_does_the_adherence_gate_resolve": adherence,
        "dial_kind": {"prior": prior["dial_kind"]["verdict"],
                      "this": this["dial_kind"]["verdict"]},
        "eureka": {"prior": prior["headline"]["eureka_criterion_met"],
                   "this": this["headline"]["eureka_criterion_met"]},
        "coherence": {"prior_drop": pc["drop_far_below_near"],
                      "this_drop": tc["drop_far_below_near"],
                      "prior_passes": pc["passes"], "this_passes": tc["passes"]},
        "nonsense_control": {
            "prior_reproduces_the_dial_pattern":
                prior["controls"]["nonsense_value_PRIMARY"]["reproduces_the_dial_pattern"],
            "this_reproduces_the_dial_pattern":
                this["controls"]["nonsense_value_PRIMARY"]["reproduces_the_dial_pattern"]},
        "nodial_control": {
            "prior_monotone": prior["controls"]["nodial_arm_secondary"]["monotone"],
            "this_monotone": this["controls"]["nodial_arm_secondary"]["monotone"]},
    }


# ---------------------------------------------------------------------------------------
# Honest disclosures. Facts that weaken the story and must appear anyway.
# ---------------------------------------------------------------------------------------
def convergence_framing(manifests: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """What "not converged" actually means for THIS recipe -- which is not "run it longer".

    THE FRAMING THAT CHANGED. The 3000-step artifact said "neither arm is converged, so a null
    could be undertraining", which is correct as far as it goes and quietly implies the fix is
    more steps. It is not, because **the learning rate is CONSTANT for every step of this
    recipe -- no decay, no warmup, no schedule of any kind.** A constant-LR run does not
    converge to anything; it asymptotes, and it keeps making small monotone improvements
    indefinitely because SGD noise at a fixed step size never anneals. "Val loss was still
    falling" is therefore a property of the RECIPE, not evidence that the budget was the
    binding constraint.

    That changes what a null MEANS at the longer budget. At 3000 steps "undertrained" was a
    live explanation for a failing guard. Once 3x the budget buys only a small further
    improvement on a curve that cannot flatten by construction, "undertrained" stops being a
    live explanation and "this recipe has plateaued in practice" takes its place.

    Read off the manifests rather than typed, so an arm trained with an actual schedule stops
    getting this paragraph attached to it.
    """
    dial = manifests["dial"]
    lrs = {a: m.get("lr") for a, m in manifests.items()}
    constant_lr = len(set(lrs.values())) == 1 and all(v is not None for v in lrs.values())
    first, last = dial.get("val_loss_first"), dial.get("val_loss_last")
    improvement = None
    if isinstance(first, list) and isinstance(last, list) and first[1]:
        improvement = round((first[1] - last[1]) / first[1], 4)
    return {
        "limitation": ("the arms ASYMPTOTE rather than converge, and that is a property of the "
                       "RECIPE, not a shortage of steps"),
        "lr": lrs["dial"],
        "lr_is_constant_for_every_step": constant_lr,
        "lr_schedule": "NONE -- no decay, no warmup",
        "steps": dial.get("steps"),
        "stop_reason": dial.get("stop_reason"),
        "early_stopping_triggered": dial.get("early_stopping", {}).get("triggered"),
        "val_loss_first": first,
        "val_loss_last": last,
        "relative_val_improvement": improvement,
        "why_it_matters": (
            "The learning rate is CONSTANT at "
            f"{lrs['dial']} for all {dial.get('steps')} steps -- there is no decay and no "
            "schedule. A constant-LR run cannot converge; it asymptotes, and it goes on "
            "producing small monotone validation improvements indefinitely because the step "
            "size never anneals. So 'val loss was still falling' and 'early stopping never "
            "fired' are NOT evidence that more steps would help; they are what this recipe "
            "does. State it that way rather than repeating 'not converged' as though the fix "
            "were a bigger budget. The consequence for reading this artifact: a null or a "
            "failing guard here is no longer plausibly 'undertrained' -- it is 'this recipe "
            "has plateaued in practice'."),
        "what_would_actually_test_convergence": (
            "an LR schedule (cosine or linear decay to ~0) so the run has a defined endpoint, "
            "or a budget sweep with the schedule held fixed. Neither is in scope here."),
    }


def build_limitations(derive: Dict[str, Any], manifests: Dict[str, Dict[str, Any]],
                      *, particles: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every fact a reader needs in order to disbelieve us, assembled as ONE function.

    Stage 2's artifact omitted the derivation drop rate until a final review caught it, and
    the spec now makes it mandatory above 50%. Anything here that reads as an argument
    against the headline is here BECAUSE it is.
    """
    dial, nodial = manifests["dial"], manifests["nodial"]
    rc = dial.get("ruling_c_reapplied", {}).get("measured_before_dropping", {})
    out = [
        {"limitation": "the FILTER chose this population, not the model",
         "drop_rate": derive.get("drop_rate"),
         "stories_scanned": derive.get("stories"),
         "kept": derive.get("kept"),
         "drops_by_rule": derive.get("drops_by_rule"),
         "gate_order": derive.get("gate_order"),
         "gate_order_note": derive.get("gate_order_note"),
         "why_it_matters": ("98.06% of stories were dropped. Every rate in this artifact is "
                            "conditional on a 1.94% slice selected for having five quoted "
                            "utterances, an acceptable offer at every model turn and NPMI "
                            "evidence for every `add` word. Mandatory disclosure above 50%.")},
        convergence_framing(manifests),
        {"limitation": "the arms' training losses are NOT comparable",
         "warning": dial.get("loss_comparability_WARNING"),
         "why_it_matters": ("the dial arm supervises an extra slot line, so the two arms "
                            "supervise different token sets. Do not subtract them. This is "
                            "also why the measurement in this file is WITHIN-model and "
                            "scene-paired rather than between-arm.")},
        {"limitation": "`add` on dialogue turns is mostly PARTICLES",
         **particles,
         "why_it_matters": ("if the dial works, it works on this vocabulary. A reader must be "
                            "able to see that the thing being 'reached for' is very often a "
                            "discourse particle rather than a bold new idea.")},
        {"limitation": "the same-speaker filter cannot catch the case that motivated it",
         "residual_same_voice_rate_after_the_filter": derive.get(
             "same_speaker_filter", {}).get("subject_reading", {}).get(
             "risky_pair_fraction"),
         "filter": derive.get("same_speaker_filter", {}).get("subject_reading", {}).get("gate"),
         "what_it_cannot_catch": derive.get("same_speaker_filter", {}).get(
             "what_this_filter_CANNOT_catch"),
         "why_it_matters": ("`handback` is the slot that depends on 'the partner turn is a "
                            "different voice', and that premise holds only probabilistically. "
                            "This eval does not publish a handback effect, which is why the "
                            "two handback tests in the pre-declared family of 11 are not run "
                            "-- keeping them counted only makes alpha stricter.")},
        {"limitation": "the `nodial` control's prompts are off-schema in THREE ways",
         "why_it_matters": ("prompts are byte-identical across arms, so the `nodial` arm also "
                            "receives a `reach` line in the earlier teacher-forced block, on "
                            "top of the unknown slot name and the extra line in the measured "
                            "block. Within-arm pairing survives (all three settings share "
                            "it), but this arm is far out of distribution and its null is "
                            "weak evidence. It is the SECONDARY control and not a gate; the "
                            "PRIMARY control keeps the schema perfectly well-formed and "
                            "varies only the value.")},
        {"limitation": "`offer` and `accept` are TEACHER-FORCED, and so are the earlier turns",
         "why_it_matters": ("the model is handed the row's own `offer` and `accept` and then "
                            "the forced dial, so it is not generating a plan from scratch. "
                            "That is deliberate -- it is what makes the three settings differ "
                            "in the dial value and NOTHING else -- but it means these numbers "
                            "describe a dial applied to a well-specified plan, not a dial "
                            "applied to free generation.")},
    ]
    return out


def particle_profile(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How concentrated the derived `add` vocabulary is, measured on the real artifact.

    A property of SCALE, so it is computed over every row rather than asserted on a fixture.
    """
    from collections import Counter
    c: Counter = Counter()
    for r in rows:
        for b in r["blocks"]:
            w = add_word_of_value(b["add"])
            if w:
                c[w] += 1
    total = sum(c.values())
    top = c.most_common(15)
    return {
        "observations": total,
        "distinct_add_words": len(c),
        "top_15": [{"word": w, "n": n, "share": round(n / total, 5)} for w, n in top],
        "top_15_share": round(sum(n for _, n in top) / total, 5) if total else None,
    }


# ---------------------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------------------
def generate_all(rows: Sequence[Dict[str, Any]], *, work_dir: Path, max_new_tokens: int,
                 batch_size: int, arm_dirs: Optional[Dict[str, Path]] = None,
                 step: int = DEFAULT_STEP
                 ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Every condition's raw continuation for every scene. CPU only -- no ttml, no device.

    `arm_dirs`/`step` select the checkpoint. Everything else about the procedure is fixed, so
    a longer-trained rerun differs from the published one in the checkpoint and in nothing
    else -- which is the only way the two are comparable.
    """
    arm_dirs = arm_dirs or ARM_DIRS
    from scripts.eval_improv import (generate_batched_from_ids, load_hf,
                                     sft_checkpoint_to_hf)

    out: Dict[str, List[str]] = {}
    meta: Dict[str, Any] = {}
    for arm in ("dial", "nodial"):
        hf_dir = work_dir / arm
        cfg = sft_checkpoint_to_hf(arm_dirs[arm] / f"step_{step}.pkl",
                                   warm_start_ckpt=WARM_START_CKPT,
                                   tokenizer_dir=TOKENIZER_DIR, out_dir=hf_dir)
        tok, model = load_hf(hf_dir)
        if arm == "dial":
            meta["checkpoints"] = {
                a: _repo_relative(arm_dirs[a] / f"step_{step}.pkl")
                for a in ("dial", "nodial")}
            meta["step"] = step
            meta["prompt_parity"] = check_prompt_parity(tok, rows)
            meta["model_config"] = {k: cfg[k] for k in
                                    ("hidden_size", "num_hidden_layers",
                                     "num_attention_heads", "vocab_size")}
        for a, value in CONDITIONS:
            if a != arm:
                continue
            key = condition_key(arm, value)
            prompts = [encode_segments(tok, prompt_segments(r, value)) for r in rows]
            print(f"  [{key}] generating {len(prompts)} continuations "
                  f"(max_new_tokens={max_new_tokens}) ...", flush=True)
            out[key] = generate_batched_from_ids(tok, model, prompts,
                                                 max_new_tokens=max_new_tokens,
                                                 do_sample=False, batch_size=batch_size)
        del model, tok
    return out, meta


# ---------------------------------------------------------------------------------------
# Scoring / assembly. Pure: same stored generations -> same numbers (that is --rescore-from).
# ---------------------------------------------------------------------------------------
def build_observations(rows: Sequence[Dict[str, Any]],
                       generations: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """One row per scene, carrying the parse of every condition. No corpus, no model."""
    obs: List[Dict[str, Any]] = []
    for j, r in enumerate(rows):
        ctx = list(content_words(r["prefix"]))
        for k in range(MEASURED_TURN):
            ctx.extend(content_words(r["turns"][k]))
        per: Dict[str, Any] = {}
        add_words: Set[str] = set()
        for arm, value in CONDITIONS:
            key = condition_key(arm, value)
            g = parse_forced_generation(generations[key][j])
            w = add_word_of_value(g.add_value)
            if w:
                add_words.add(w)
            per[key] = {"raw": generations[key][j], "closed_block": g.closed_block,
                        "add_value": g.add_value, "add_word": w, "stakes": g.stakes,
                        "handback": g.handback, "turn": g.turn,
                        "add_hit": bool(w and _mentions(g.turn, w))}
        nxt = (r["turns"][MEASURED_TURN + 1]
               if MEASURED_TURN + 1 < len(r["turns"]) else None)
        for k, c in per.items():
            # `handback` scored against the REAL following partner turn, the same way
            # scripts/score_skits.py:score_block scores it. Cheap, and capability 2 of 2 in
            # the spec; leaving it unmeasured while publishing the dial would have shipped
            # half the design silently.
            c["handback_hit"] = bool(nxt and c["handback"]
                                     and _mentions(nxt, c["handback"]))
            c["handback_scorable"] = nxt is not None
        gold_w = add_word_of_value(r["blocks"][MEASURED_BLOCK]["add"])
        if gold_w:
            add_words.add(gold_w)
        obs.append({
            "index": j, "story_id": r["story_id"],
            "context_words": ctx, "add_words": sorted(add_words),
            "gold_add_word": gold_w,
            "gold_reach": r["blocks"][MEASURED_BLOCK]["reach"],
            "gold_distance": r["reach_distances"][MEASURED_BLOCK],
            "gold_add_df": r["add_df"][MEASURED_BLOCK],
            "prev_turn": r["turns"][MEASURED_TURN - 1],
            "conditions": per,
        })
    return obs


def gold_reproduction(obs: Sequence[Dict[str, Any]], assoc: Association,
                      own: Dict[int, FrozenSet[str]]) -> Dict[str, Any]:
    """Re-derive every gold row's STORED `reach_distances` from the targeted table.

    THE INSTRUMENT CHECK. It proves three things at once, at real corpus scale, on real data:
    the targeted counting reproduces the full table's counts; the leave-one-out is applied the
    way derivation applied it; and `reach_distance` here is the same function that produced
    the labels. An exact match on thousands of observations is not something a wrong table can
    fake. Also checks the stored `add_df` against `uni`, which is the unigram half of the same
    claim.
    """
    d_err: List[float] = []
    df_err = 0
    n_none = 0
    worst = 0.0
    for o in obs:
        if not o["gold_add_word"]:
            continue
        got = reach_distance_loo(o["gold_add_word"], o["context_words"], assoc,
                                 own.get(o["story_id"]))
        if got is None:
            n_none += 1
            continue
        e = abs(got - o["gold_distance"])
        d_err.append(e)
        worst = max(worst, e)
        if assoc.uni.get(o["gold_add_word"], 0) != o["gold_add_df"]:
            df_err += 1
    return {
        "observations_rederived": len(d_err),
        "max_abs_distance_error": worst,
        "mean_abs_distance_error": (st.fmean(d_err) if d_err else None),
        "add_df_mismatches": df_err,
        "gold_words_with_no_evidence_in_the_targeted_table": n_none,
        "matches": bool(d_err and worst < 1e-12 and df_err == 0 and n_none == 0),
        "why": ("the full 31.9M-pair table is not persisted and costs ~34 minutes to "
                "rebuild. This proves the targeted pass reproduces it exactly on every "
                "value the dial was derived from, rather than asserting it in a comment."),
    }


def per_setting_table(obs: Sequence[Dict[str, Any]], complete: Sequence[int],
                      dist: Dict[str, Dict[int, float]], dfs: Dict[str, Dict[int, float]],
                      grounded: Dict[str, Dict[int, float]],
                      resid: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    """Raw distance, frequency-matched distance, realised add_df, groundedness, add-hit rate.

    Every row carries `add_df`, per the spec amendment: an eval can control for a number, it
    cannot control for a warning.
    """
    table: Dict[str, Any] = {
        "_disclosures": {
            "add_word_already_in_the_context_rate": (
                "DEGENERATE WAY TO BE NEAR, disclosed inline. The cheapest way to make the "
                "`add` word close to the scene is to COPY a word already in it, and a copied "
                "word is not a reach at all. A `near` setting whose rate here is much higher "
                "than `far`'s is partly measuring copying rather than proximity. Read the "
                "near end of the dial against this column."),
            "content_word_repeat_rate_mean": (
                "the generation's own degeneration floor, per setting. Greedy decoding on an "
                "unconverged 123M model loops ('Snowmen are snowmen and snowmen'), and a "
                "setting that loops more has a different effective sample of language than "
                "one that loops less. A 4-gram version of this was tried first and scored "
                "exactly 0.0000 everywhere -- a skit turn is one sentence and rarely has four "
                "content words -- so it is type/token over content words instead."),
            "realised_add_df_median": (
                "THE FREQUENCY CONFOUND, beside the effect rather than in a warning. NPMI "
                "caps a common word's score low, so a farther-scoring word is partly just a "
                "commoner word."),
            "add_slot_hit_rate": (
                "does the generated turn USE the word the block named? A dial that moves the "
                "declaration without moving the behaviour is decorative."),
            "handback_hit_rate": (
                "does the REAL following partner turn contain the word the block handed back? "
                "Scored exactly as scripts/score_skits.py:score_block scores it. Reported "
                "because `handback` is capability 2 of 2 in the spec and the stated payoff of "
                "the whole dialogue rebuild -- but NO handback effect is claimed and none of "
                "the pre-declared family's two handback tests is run against it. It rests on "
                "the premise that the following turn is a DIFFERENT VOICE, which Ruling A's "
                "filter holds only probabilistically (residual same-voice rate 0.0097, see "
                "limitations), and it is a rate against no floor. Read it as a descriptive "
                "number, not as a result."),
        }
    }
    for arm, value in CONDITIONS:
        key = condition_key(arm, value)
        d = [dist[key][i] for i in complete]
        f = [dfs[key][i] for i in complete]
        g = [grounded[key][i] for i in complete]
        rr = [resid[key][i] for i in complete]
        hits = [obs[i]["conditions"][key]["add_hit"] for i in complete]
        ctx_sets = [set(obs[i]["context_words"]) for i in complete]
        copied = [obs[i]["conditions"][key]["add_word"] in ctx_sets[n]
                  for n, i in enumerate(complete)]
        repeats = [content_word_repeat_rate(obs[i]["conditions"][key]["turn"])
                   for i in complete]
        closed = [obs[i]["conditions"][key]["closed_block"] for i in complete]
        named = [obs[i]["conditions"][key]["add_word"] is not None for i in complete]
        words = [obs[i]["conditions"][key]["add_word"] for i in complete]
        table[key] = {
            "arm": arm, "forced_reach": value, "n": len(complete),
            "raw_distance_mean": round(st.fmean(d), 6),
            "raw_distance_sd": round(st.stdev(d), 6) if len(d) > 1 else None,
            "frequency_residualised_distance_mean": round(st.fmean(rr), 6),
            "realised_add_df_median": st.median(f),
            "realised_add_df_mean": round(st.fmean(f), 1),
            "realised_log_add_df_mean": round(st.fmean([math.log(max(x, 1)) for x in f]), 4),
            "groundedness_mean": round(st.fmean(g), 6),
            "add_slot_hit_rate": round(sum(1 for h in hits if h) / len(hits), 6),
            "handback_hit_rate": round(
                sum(1 for i in complete
                    if obs[i]["conditions"][key]["handback_hit"]) / len(complete), 6),
            "block_closed_rate": round(sum(1 for c in closed if c) / len(closed), 6),
            "named_an_add_rate": round(sum(1 for c in named if c) / len(named), 6),
            "distinct_add_words": len(set(words)),
            "add_word_already_in_the_context_rate": round(
                sum(1 for c in copied if c) / len(copied), 6),
            "content_word_repeat_rate_mean": round(st.fmean(repeats), 6),
            "top_add_words": [{"word": w, "n": n} for w, n in
                              _top(words, 8)],
            "realised_bucket_mix": _bucket_mix(d),
        }
    return table


def _top(words: Sequence[Optional[str]], k: int) -> List[Tuple[str, int]]:
    from collections import Counter
    c: Counter = Counter(w for w in words if w)
    return c.most_common(k)


def _bucket_mix(distances: Sequence[float], *, lo: Optional[float] = None,
                hi: Optional[float] = None) -> Dict[str, Any]:
    """Which dial bucket the REALISED distances land in, using the TRAIN-FITTED cut points.

    The cut points are read from `derive_manifest.json`, never re-fitted on eval: a dial whose
    buckets move between train and eval measures nothing.
    """
    if lo is None or hi is None:
        lo, hi = _CUTS["lo"], _CUTS["hi"]
    counts = {v: 0 for v in REACH_VALUES}
    for d in distances:
        counts[reach_bucket(d, lo, hi)] += 1
    n = len(distances)
    return {"counts": counts,
            "fractions": {k: round(v / n, 4) for k, v in counts.items()} if n else None}


#: Filled from derive_manifest.json in `main`/`analyse` before any bucketing. Module state so
#: `_bucket_mix` cannot silently fall back to a hard-coded pair.
_CUTS: Dict[str, float] = {}


def analyse(rows: Sequence[Dict[str, Any]], generations: Dict[str, List[str]], *,
            corpus: Path, corpus_limit: Optional[int], assoc_skits: int,
            all_rows: Sequence[Dict[str, Any]], derive: Dict[str, Any],
            manifests: Dict[str, Dict[str, Any]], gen_meta: Dict[str, Any],
            df_match_tol: float, progress: int = 200000,
            generate_cmd: str = "python scripts/eval_reach.py",
            step: int = DEFAULT_STEP,
            compare_to: Optional[Path] = None) -> Dict[str, Any]:
    """Every scored number and every verdict, as ONE CALLED FUNCTION.

    Pure apart from reading the corpus: given the same stored generations it reproduces the
    same numbers, which is what makes `--rescore-from` a re-analysis rather than a rerun.
    Nothing that decides a published claim lives in `main`.
    """
    global _CUTS
    _CUTS = {"lo": derive["reach"]["cut_points"]["lo"],
             "hi": derive["reach"]["cut_points"]["hi"]}

    # The pairing proof, CHECKED. Both arms consumed identical skits in identical order or
    # the two arms' numbers are not two readings of one experiment.
    fp_dial = manifests["dial"].get("batch_order_fingerprint", {}).get("sha256")
    fp_nodial = manifests["nodial"].get("batch_order_fingerprint", {}).get("sha256")
    if not fp_dial or fp_dial != fp_nodial:
        raise RuntimeError(f"the arms' batch-order fingerprints differ ({fp_dial} vs "
                           f"{fp_nodial}); they did not consume the same skits in the same "
                           f"order and are not paired")
    if manifests["dial"]["n_examples"] != manifests["nodial"]["n_examples"]:
        raise RuntimeError("the arms trained on different-sized training sets; not paired")

    obs = build_observations(rows, generations)
    needed_uni, by_add = needed_lookups(obs)
    story_ids = {o["story_id"] for o in obs}
    print(f"[assoc] one corpus pass for {len(needed_uni):,} unigrams and "
          f"{sum(len(v) for v in by_add.values()):,} pairs over {len(by_add):,} add words ...",
          flush=True)
    assoc, own = collect_targeted_association(corpus, needed_uni=needed_uni, by_add=by_add,
                                              story_ids=story_ids, limit=corpus_limit,
                                              progress=progress)
    print(f"  documents {assoc.n_docs:,}  unigrams held {len(assoc.uni):,}  "
          f"pairs held {len(assoc.co):,}", flush=True)

    repro = gold_reproduction(obs, assoc, own)
    print(f"[assoc] gold reproduction: {repro['observations_rederived']} rows, max abs error "
          f"{repro['max_abs_distance_error']:.3e}, add_df mismatches "
          f"{repro['add_df_mismatches']} -> matches={repro['matches']}", flush=True)
    if corpus_limit is None and not repro["matches"]:
        # A REFUSAL, not a warning. If the table this run counted does not reproduce the
        # distances the dial's own labels were derived from, every distance below is measured
        # on a different instrument than the one that defined `near`/`mid`/`far`.
        raise RuntimeError(
            f"the targeted association table does not reproduce the derived distances "
            f"(max abs error {repro['max_abs_distance_error']:.3e}, "
            f"{repro['add_df_mismatches']} add_df mismatches, "
            f"{repro['gold_words_with_no_evidence_in_the_targeted_table']} gold words with no "
            f"evidence). Refusing to publish numbers measured on a different instrument.")

    # ---- distances, dfs, groundedness -------------------------------------------------
    harm, closure = load_harm_lexicon(), load_closure_lexicon()
    g_pairs: List[Tuple[str, str]] = []
    for r in all_rows[:assoc_skits]:
        if r.get("split") != "train":
            continue
        for t_idx in MODEL_TURNS:
            prev = r["turns"][t_idx - 1] if t_idx > 0 else r["prefix"]
            g_pairs.append((prev, r["turns"][t_idx]))
    g_assoc = build_groundedness_assoc(g_pairs)

    keys = [condition_key(a, v) for a, v in CONDITIONS]
    dist: Dict[str, Dict[int, float]] = {k: {} for k in keys}
    dfs: Dict[str, Dict[int, float]] = {k: {} for k in keys}
    grounded: Dict[str, Dict[int, float]] = {k: {} for k in keys}
    for o in obs:
        i = o["index"]
        for k in keys:
            c = o["conditions"][k]
            w = c["add_word"]
            if not w:
                continue
            d = reach_distance_loo(w, o["context_words"], assoc, own.get(o["story_id"]))
            if d is None:
                continue
            dist[k][i] = d
            dfs[k][i] = float(assoc.uni.get(w, 0))
            grounded[k][i] = score_pair(o["prev_turn"], c["turn"], harm=harm, assoc=g_assoc,
                                        closure=closure).groundedness

    complete = [o["index"] for o in obs if all(o["index"] in dist[k] for k in keys)]
    #: Scenes complete in the DIAL ARM's four conditions only. The `nodial` arm fails to name
    #: a scorable `add` word far more often (it is off-schema three ways), so requiring every
    #: condition makes the dial arm's own contrast conditional on the CONTROL arm's success --
    #: a selection the design never asked for. Both denominators are reported.
    dial_only_keys = [condition_key("dial", v) for v in (*REACH_VALUES, NONSENSE_VALUE)]
    complete_dial = [o["index"] for o in obs
                     if all(o["index"] in dist[k] for k in dial_only_keys)]
    definedness = {k: {"defined": len(dist[k]), "of": len(obs),
                       "rate": round(len(dist[k]) / len(obs), 4) if obs else None}
                   for k in keys}
    if len(complete) < 2:
        raise RuntimeError(f"only {len(complete)} scene(s) have a defined distance in every "
                           f"condition; nothing paired can be measured")

    # ---- frequency control ------------------------------------------------------------
    # Residualise on log add_df over the POOLED dial-arm observations, so one line is removed
    # from all three settings rather than a different line per setting (which would absorb the
    # very between-setting difference under test).
    dial_keys = [condition_key("dial", v) for v in REACH_VALUES]
    pooled_y: List[float] = []
    pooled_x: List[float] = []
    index: List[Tuple[str, int]] = []
    for k in keys:
        for i in complete:
            pooled_y.append(dist[k][i])
            pooled_x.append(math.log(max(dfs[k][i], 1.0)))
            index.append((k, i))
    res_all = ols_residuals(pooled_y, pooled_x)
    resid: Dict[str, Dict[int, float]] = {k: {} for k in keys}
    for (k, i), v in zip(index, res_all):
        resid[k][i] = v
    freq_slope_spearman = spearman(pooled_x, pooled_y)

    # SENSITIVITY: the same control with the nuisance line fitted on the DIAL ARM'S three
    # settings only, so the correction applied to the treatment cannot have been shaped by the
    # control conditions. If the verdict differs between the two fits, the "control" was doing
    # the work and that has to be visible rather than absorbed.
    d_index = [(k, i) for k in dial_keys for i in complete]
    d_y = [dist[k][i] for k, i in d_index]
    d_x = [math.log(max(dfs[k][i], 1.0)) for k, i in d_index]
    d_res_all = ols_residuals(d_y, d_x)
    resid_dial_only: Dict[str, Dict[int, float]] = {k: {} for k in dial_keys}
    for (k, i), v in zip(d_index, d_res_all):
        resid_dial_only[k][i] = v

    table = per_setting_table(obs, complete, dist, dfs, grounded, resid)

    def series(k: str, src: Dict[str, Dict[int, float]]) -> List[float]:
        return [src[k][i] for i in complete]

    def words_of(k: str) -> List[Optional[str]]:
        """The `add` word each complete-case scene produced under condition `k`.

        `df_matched_contrast` needs these to report how much of its retained pool is the SAME
        word under both settings -- the pairs whose distance difference is 0.0 by construction
        and which otherwise silently dominate the matched mean.
        """
        return [obs[i]["conditions"][k]["add_word"] for i in complete]

    raw_dial = monotonicity_verdict(series(dial_keys[0], dist), series(dial_keys[1], dist),
                                    series(dial_keys[2], dist), label="dial raw distance")
    res_dial = monotonicity_verdict(series(dial_keys[0], resid), series(dial_keys[1], resid),
                                    series(dial_keys[2], resid),
                                    label="dial distance residualised on log add_df")
    res_dial["fitted_on"] = ("one line over EVERY condition's pooled observations -- the "
                             "distance~log(add_df) relation is a property of NPMI, not of an "
                             "arm, so one correction is applied everywhere")
    res_dial_only = monotonicity_verdict(series(dial_keys[0], resid_dial_only),
                                        series(dial_keys[1], resid_dial_only),
                                        series(dial_keys[2], resid_dial_only),
                                        label="dial distance residualised, dial-arm slope")
    res_dial_only["fitted_on"] = "the dial arm's three settings only (sensitivity check)"
    res_dial["sensitivity_agrees_with_a_dial_only_slope"] = (
        res_dial["monotone"] == res_dial_only["monotone"])

    # df-matched paired subsamples, per contrast
    logdf = {k: {i: math.log(max(dfs[k][i], 1.0)) for i in complete} for k in keys}
    matched: Dict[str, Any] = {}
    for a_key, b_key, name in ((dial_keys[0], dial_keys[1], "near_lt_mid"),
                               (dial_keys[1], dial_keys[2], "mid_lt_far"),
                               (dial_keys[0], dial_keys[2], "near_lt_far")):
        matched[name] = df_matched_contrast(
            series(a_key, dist), series(a_key, logdf), words_of(a_key),
            series(b_key, dist), series(b_key, logdf), words_of(b_key),
            tol=df_match_tol, label=f"df-matched {name}")
    matched["all_three_significant_over_informative_pairs"] = all(
        (matched[n].get("informative_only") or {}).get("significant")
        for n in ("near_lt_mid", "mid_lt_far", "near_lt_far"))

    # COMPUTED, not quoted: is the realised confound monotone in THIS run? (It was published
    # as a hard-coded `not_monotone: true` once, and that literal was false here.)
    freq_profile = realised_frequency_profile({k: series(k, logdf) for k in dial_keys},
                                              dial_keys, freq_slope_spearman)
    freq_agreement = frequency_control_agreement(raw_dial, res_dial, matched)

    # ---- controls ---------------------------------------------------------------------
    nonsense_key = condition_key("dial", NONSENSE_VALUE)
    nonsense = nonsense_control_verdict(series(dial_keys[0], dist), series(dial_keys[2], dist),
                                        series(nonsense_key, dist))
    nonsense["condition"] = nonsense_key
    nonsense["raw_distance_mean"] = table[nonsense_key]["raw_distance_mean"]
    nonsense["vs_mid"] = paired_contrast(series(dial_keys[1], dist),
                                         series(nonsense_key, dist),
                                         direction="b_greater", label="nonsense vs mid")
    nod_keys = [condition_key("nodial", v) for v in REACH_VALUES]
    nodial = monotonicity_verdict(series(nod_keys[0], dist), series(nod_keys[1], dist),
                                  series(nod_keys[2], dist), label="nodial raw distance")
    nodial["role"] = ("SECONDARY CONTROL, and weaker than it looks -- see "
                      "prompt_parity.nodial_note.")
    nodial["how_off_schema_this_arm_is"] = (
        "THREE ways, not one. The prompts are byte-identical across arms (that is what makes "
        "the two arms' numbers comparable at all), so the `nodial` arm receives (1) an "
        "unknown slot NAME, (2) an extra LINE in the measured block, and (3) a `reach` line "
        "in the earlier TEACHER-FORCED block too. Within the arm all three settings share "
        "every one of those, so the near/mid/far contrast below is still internally paired -- "
        "but the arm is far enough out of distribution that a null here is weak evidence and "
        "a movement here is not attributable to the dial's value. Arm-native five-slot "
        "earlier blocks would be a cleaner secondary control and were not run: it is another "
        "3,000 CPU generations and this arm is not a gate on the headline.")

    coherence = coherence_guard(series(dial_keys[0], grounded), series(dial_keys[2], grounded))
    adherence = adherence_readings({k: table[k]["add_slot_hit_rate"] for k in dial_keys},
                                    dial_keys[0], dial_keys[1], dial_keys[2])
    # SENSITIVITY on the denominator, because the gate's rates are computed on `complete`,
    # which is partly selected by the CONTROL arm's ability to name a scorable `add` word.
    # Over every row where the dial arm named one -- no distance requirement, no `nodial`
    # requirement -- the gate must still fail, or the verdict is an artefact of the selection.
    unsel: Dict[str, float] = {}
    for k in dial_keys:
        named = [o for o in obs if o["conditions"][k]["add_word"]]
        unsel[k] = (sum(1 for o in named if o["conditions"][k]["add_hit"]) / len(named)
                    if named else 0.0)
    unsel_guard = adherence_guard({k: round(v, 6) for k, v in unsel.items()})
    adherence["unselected_denominator_sensitivity"] = {
        "n_rows_considered": len(obs),
        "rates": {k: round(v, 6) for k, v in unsel.items()},
        "shortfall": unsel_guard["shortfall"],
        "worst_setting": unsel_guard["worst_setting"],
        "passes": unsel_guard["passes"],
        "agrees_with_the_gate": unsel_guard["passes"] == adherence["passes"],
        "why": ("the headline rates use the scenes complete in ALL SEVEN conditions, which "
                "the `nodial` arm partly selects. These use every row where the DIAL arm "
                "named an `add` word. If the two disagreed, the failing gate would be a "
                "property of the denominator rather than of the model."),
    }
    dial_kind = dial_or_frequency_dial(raw_dial, res_dial, matched)
    eureka = eureka_verdict(dial_kind=dial_kind, coherence=coherence, adherence=adherence,
                            nonsense=nonsense, nodial=nodial)
    # BOTH DIRECTIONS. `dial_kind.verdict` and `headline.eureka_criterion_met` answer different
    # questions and on this run they have different answers, so either one quoted alone is
    # materially misleading. Each now carries the other.
    dial_kind["eureka_criterion_met"] = eureka["eureka_criterion_met"]
    dial_kind["QUOTING_EITHER_ALONE_IS_MISLEADING"] = eureka[
        "QUOTING_EITHER_ALONE_IS_MISLEADING"]

    # SENSITIVITY on the denominator: the same raw contrast over the dial arm's own complete
    # cases, so a reader can see whether conditioning on the control arm's success moved
    # anything. If these two disagree the headline denominator is doing work it should not.
    def dseries(k: str) -> List[float]:
        return [dist[k][i] for i in complete_dial]
    raw_dial_own = monotonicity_verdict(dseries(dial_keys[0]), dseries(dial_keys[1]),
                                        dseries(dial_keys[2]),
                                        label="dial raw distance, dial-complete cases")
    raw_dial_own["n_scenes"] = len(complete_dial)
    raw_dial_own["vs_headline_denominator"] = (
        f"the headline uses the {len(complete)} scenes complete in ALL SEVEN conditions; this "
        f"uses the {len(complete_dial)} complete in the dial arm's four. The `nodial` arm "
        f"names a scorable `add` word far less often (see definedness_per_condition), so the "
        f"headline set is partly selected by the CONTROL arm's success.")
    raw_dial_own["agrees_with_the_headline"] = (
        raw_dial_own["monotone"] == raw_dial["monotone"])

    result = {
        "measurement": "reach dial -- the EUREKA measurement (task 4)",
        "spec": "docs/superpowers/specs/2026-08-23-reach-dial-design.md",
        "headline": eureka,
        "dial_kind": dial_kind,
        "thresholds": {
            "n_tests": N_TESTS, "bonferroni_alpha": BONFERRONI_ALPHA,
            "critical_t": CRITICAL_T,
            "critical_t_exact_normal_quantile": round(critical_t_for(BONFERRONI_ALPHA), 5),
            "coherence_margin": COHERENCE_MARGIN,
            "adherence_margin": ADHERENCE_MARGIN,
            "defined_locally": ("BONFERRONI_ALPHA and CRITICAL_T are defined in "
                                "scripts/eval_reach.py and NOT imported from eval_improv.py "
                                "(0.01 / 2.576) or eval_skits.py. In stage 2 two effects "
                                "landed between 2.576 and 2.843; an import would have "
                                "manufactured both."),
            "n_tests_note": ("the pre-declared family is 3 dial contrasts + 3 coherence "
                             "guards + 3 adherence checks + 2 handback checks = 11. The two "
                             "handback tests are NOT run here (no handback effect is "
                             "published); keeping them in the family can only make alpha "
                             "stricter, never looser."),
        },
        "design": {
            "within_model_scene_paired": True,
            "measured_block_index": MEASURED_BLOCK,
            "measured_model_turn": MEASURED_TURN,
            "why_this_turn": ("the only model turn with a real partner turn both before and "
                              "after it; context is prefix + turn 0 + turn 1."),
            "teacher_forced": ["prefix", "turn 0 block", "turn 0", "turn 1",
                               "the measured block's offer and accept", "the reach value"],
            "generated": ["add", "stakes", "handback", "</think>", "the model turn"],
            "conditions": [condition_key(a, v) for a, v in CONDITIONS],
            "scenes_requested": len(rows),
            "scenes_complete_in_every_condition": len(complete),
            "scenes_complete_in_the_dial_arms_four_conditions": len(complete_dial),
            "definedness_per_condition": definedness,
            "definedness_note": (
                "a scene is `defined` for a condition when the generation named an `add` word "
                "AND that word has co-occurrence evidence with some context word. The `nodial` "
                "arm's rate is much lower -- it is off-schema three ways -- which is why the "
                "dial-arm-only denominator is reported beside the headline one."),
            "eval_split": "rows whose own `split` field is `eval`",
            "checkpoint": {
                "step": step,
                "paths": gen_meta.get("checkpoints"),
                "lr": manifests["dial"].get("lr"),
                "lr_schedule": ("CONSTANT -- there is no decay and no warmup in this recipe. "
                                "See limitations."),
                "stop_reason": manifests["dial"].get("stop_reason"),
                "note": ("the ONLY variable between this artifact and any sibling "
                         "reach-dial*.json is the checkpoint. Same scenes, same forced-dial "
                         "procedure, same controls, same locally-defined alpha."),
            },
            "cut_points": {**_CUTS,
                           "fitted_on": derive["reach"]["cut_points"]["fitted_on"],
                           "n_fitted_on": derive["reach"]["cut_points"]["n_fitted_on"],
                           "refitted_on_eval": False,
                           "note": derive["reach"]["cut_points"]["eval_must_not_refit"]},
            "n_examples_from_train_manifests": {
                "dial": manifests["dial"]["n_examples"],
                "nodial": manifests["nodial"]["n_examples"],
                "note": ("read from the TRAIN manifests, not the derivation manifest: 36,387 "
                         "trained, 36,913 was the pre-ruling-C count.")},
            "permutation_fingerprint": {
                "dial": fp_dial, "nodial": fp_nodial,
                "identical": fp_dial == fp_nodial and fp_dial is not None,
                "n": manifests["dial"].get("batch_order_fingerprint", {}).get("n"),
                "note": ("identical in both arms IS the pairing proof, and it is CHECKED "
                         "(analyse raises if they differ) rather than quoted.")},
        },
        "per_setting": table,
        "effects": {
            "raw_distance": raw_dial,
            "frequency_residualised_distance": res_dial,
            "frequency_residualised_distance_dial_only_slope": res_dial_only,
            "frequency_matched_subsample": matched,
            "frequency_control_agreement": freq_agreement,
            "raw_distance_dial_complete_cases_sensitivity": raw_dial_own,
            "realised_frequency_profile": freq_profile,
            "frequency_confound_in_the_DERIVATION_not_here": {
                "corpus_scale_spearman_raw_df_vs_distance": derive["reach"][
                    "frequency_confound"]["spearman_df_vs_distance"],
                "corpus_scale_per_bucket_median_add_df": {
                    b: derive["reach"]["frequency_confound"]["per_bucket"][b]["median_add_df"]
                    for b in REACH_VALUES},
                "these_describe_the_GOLD_LABELS": (
                    "the derivation's tercile population, NOT the words this model generated. "
                    "This block used to be keyed `frequency_confound_here` and carried a "
                    "hard-coded `not_monotone: true` beside these numbers; both were false of "
                    "this run. What is true of this run is computed in "
                    "`realised_frequency_profile` above. The corpus-scale figure is also on "
                    "RAW df while this run's is on LOG df, so the two coefficients are not "
                    "directly comparable in magnitude."),
            },
        },
        "controls": {"nonsense_value_PRIMARY": nonsense, "nodial_arm_secondary": nodial},
        "coherence_guard": coherence,
        "adherence_guard": adherence,
        "instrument_checks": {
            "association_table": {
                "documents": assoc.n_docs,
                "expected_documents": derive["reach"]["association_table"]["documents"],
                "documents_match": assoc.n_docs == derive["reach"][
                    "association_table"]["documents"],
                "counted": "target-first (see collect_targeted_association)",
                "unigrams_held": len(assoc.uni), "pairs_held": len(assoc.co),
                "full_table_size": {
                    "vocabulary": derive["reach"]["association_table"]["vocabulary"],
                    "pairs": derive["reach"]["association_table"]["pairs"]},
            },
            "gold_distance_reproduction": repro,
            "prompt_parity": gen_meta.get("prompt_parity"),
            "holdout": ("EXACT leave-one-out: a pair is scored with the story's own document "
                        "subtracted only when the story actually contributes to it (both "
                        "words present). On gold rows that is every pair, so this reduces to "
                        "derivation's holdout=True -- which is what the gold reproduction "
                        "above proves."),
        },
        "add_vocabulary": particle_profile(all_rows),
        "limitations": build_limitations(derive, manifests,
                                         particles=particle_profile(all_rows)),
        "reproduce": {
            "generate": generate_cmd,
            "rescore_from_this_artifact": ("python scripts/eval_reach.py --rescore-from "
                                           "docs/measurements/reach-dial.json"),
            "rescore_from_the_side_file": ("python scripts/eval_reach.py --rescore-from "
                                           "artifacts/reach/eval-generations.json"),
            "why_the_generations_are_EMBEDDED_here": (
                "artifacts/ is git-ignored in this repo (artifacts/.gitignore is `*`), so a "
                "side file cannot make a committed measurement re-derivable. `stored` below "
                "carries every scene AND every raw generation, so every number in this file "
                "can be recomputed from this file -- no model, no tokenizer, no device, and "
                "no regeneration. Stage 2 embedded its `generated_turns` for the same reason "
                "and it proved its worth."),
        },
        # Every scene and every raw continuation. Deliberately last in the file so a reader
        # reaches the verdicts first.
        "stored": {"rows": list(rows), "generations": generations,
                   "gen_meta": gen_meta},
    }
    if compare_to is not None:
        prior = json.loads(Path(compare_to).read_text())
        key = f"versus_{prior_step_of(prior) or 'prior'}_steps"
        result[key] = versus_previous(result, prior)
        result[key]["source"] = _repo_relative(Path(compare_to))
    return result


# ---------------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------------
def load_rows(skits: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in skits.read_text().splitlines() if l.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: derived from --arm-root/--step by `default_paths`, so a "
                         "longer-trained rerun CANNOT overwrite a published measurement")
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT,
                    help="directory holding ckpt-dial/ and ckpt-nodial/")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP,
                    help="checkpoint step to evaluate (step_<N>.pkl)")
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="a previously published artifact to diff against; emits a top-level "
                         "`versus_<step>_steps` block with the deltas COMPUTED")
    ap.add_argument("--skits", type=Path, default=DEFAULT_SKITS)
    ap.add_argument("--n-scenes", type=int, default=800,
                    help="eval-labelled scenes to use; 0 = every one of them")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--corpus-limit", type=int, default=0,
                    help="stories to scan for the association table; 0 = the whole corpus, "
                         "which is the ONLY setting whose counts match the table the dial "
                         "was derived from")
    ap.add_argument("--assoc-skits", type=int, default=6000,
                    help="training skits for the groundedness NPMI table (stage-2 default)")
    ap.add_argument("--df-match-tol", type=float, default=0.25,
                    help="log-add_df tolerance for the frequency-matched paired subsample")
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--rescore-from", type=Path, default=None,
                    help="re-analyse stored generations: no model, no tokenizer, no device")
    args = ap.parse_args(argv)

    # A relative --arm-root is resolved against the REPO ROOT, not the cwd: every other path
    # in this file is repo-relative and a cwd-relative one would silently point elsewhere
    # depending on where the command was typed.
    args.arm_root = (args.arm_root if args.arm_root.is_absolute()
                     else (ROOT / args.arm_root)).resolve()
    arm_dirs = arm_dirs_for(args.arm_root)
    paths = default_paths(args.arm_root, args.step)
    out_path = args.out or paths["out"]
    store_path = args.store or paths["store"]
    work_dir = args.work_dir or paths["work_dir"]

    all_rows = load_rows(args.skits)
    derive = json.loads(DERIVE_MANIFEST.read_text())
    manifests = {arm: json.loads((arm_dirs[arm] / "train_manifest.json").read_text())
                 for arm in arm_dirs}
    check_step_matches_manifests(manifests, args.step)
    corpus_limit = args.corpus_limit or None

    # A REFUSAL, not a warning. Overwriting a published measurement with a different
    # checkpoint's numbers is silent and irreversible, and every test would still pass.
    if out_path.is_file():
        prior = json.loads(out_path.read_text())
        prior_step = (prior.get("design", {}).get("checkpoint", {}).get("step"))
        if prior_step is not None and int(prior_step) != args.step:
            raise RuntimeError(
                f"{out_path} is the step-{prior_step} measurement and --step is {args.step}. "
                f"Refusing to overwrite a published artifact with another checkpoint's "
                f"numbers. Pass an explicit --out if that is really what you want.")

    if args.rescore_from:
        print(f"[rescore] re-analysing {args.rescore_from} -- no generation, no model ...")
        blob = json.loads(args.rescore_from.read_text())
        # Accept either the side file (`{"rows", "generations"}` at the top level) or a
        # published artifact, which carries the same payload under `stored`. One flag, both
        # shapes, so the committed measurement is re-derivable from itself.
        stored = blob.get("stored", blob)
        rows = stored["rows"]
        gens = stored["generations"]
        gen_meta = stored.get("gen_meta", blob.get("gen_meta", {}))
    else:
        eval_rows = [r for r in all_rows if r.get("split") == "eval"]
        print(f"[1/4] {len(eval_rows)} eval-labelled rows of {len(all_rows)}")
        rows = eval_rows[:args.n_scenes] if args.n_scenes else eval_rows
        train_ids = {r["story_id"] for r in all_rows if r.get("split") == "train"}
        overlap = sorted({r["story_id"] for r in rows} & train_ids)
        if overlap:
            raise RuntimeError(f"eval rows overlap training story_ids {overlap[:10]} -- the "
                               f"measurement would be partly memorisation")
        print(f"  using {len(rows)} scenes, story_id overlap with train: 0")
        print("[2/4] converting checkpoints -> HF (CPU only, no ttml, no device) ...")
        gens, gen_meta = generate_all(rows, work_dir=work_dir,
                                      max_new_tokens=args.max_new_tokens,
                                      batch_size=args.batch_size,
                                      arm_dirs=arm_dirs, step=args.step)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        gen_meta["generate_cmd"] = (
            f"python scripts/eval_reach.py --arm-root {_repo_relative(args.arm_root)} "
            f"--step {args.step} --n-scenes {len(rows)} "
            f"--batch-size {args.batch_size} --max-new-tokens {args.max_new_tokens}")
        store_path.write_text(json.dumps({"rows": rows, "generations": gens,
                                          "gen_meta": gen_meta}) + "\n")
        print(f"  stored every generation -> {store_path}")

    print("[3/4] scoring ...")
    result = analyse(rows, gens, corpus=args.corpus, corpus_limit=corpus_limit,
                     assoc_skits=args.assoc_skits, all_rows=all_rows, derive=derive,
                     manifests=manifests, gen_meta=gen_meta,
                     df_match_tol=args.df_match_tol, step=args.step,
                     compare_to=args.compare_to,
                     # The command that ACTUALLY produced these generations, carried through a
                     # rescore. The default `--n-scenes` is 800 and this run used 1,000, so a
                     # printed default is a reproduction instruction that does not reproduce.
                     generate_cmd=gen_meta.get(
                         "generate_cmd",
                         f"python scripts/eval_reach.py --n-scenes {len(rows)}"))
    if corpus_limit:
        result["instrument_checks"]["association_table"]["SMOKE_RUN_WARNING"] = (
            f"--corpus-limit {corpus_limit}: the association table is NOT the one the dial "
            f"was derived from. Not a publishable measurement.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[4/4] wrote {out_path}")

    print("\n--- per setting ---")
    for k, row in result["per_setting"].items():
        if k.startswith("_"):
            continue
        print(f"  {k:>14}  dist {row['raw_distance_mean']:.4f}  resid "
              f"{row['frequency_residualised_distance_mean']:+.4f}  add_df median "
              f"{row['realised_add_df_median']:>9.0f}  grounded {row['groundedness_mean']:.4f}"
              f"  add-hit {row['add_slot_hit_rate']:.3f}")
    print("\n--- monotonicity ---")
    for name, blk in (("raw", result["effects"]["raw_distance"]),
                      ("residualised", result["effects"]["frequency_residualised_distance"]),
                      ("nodial", result["controls"]["nodial_arm_secondary"])):
        for step in ("near_lt_mid", "mid_lt_far", "near_lt_far"):
            s = blk[step]
            print(f"  {name:>13} {step:>12}  d={s['mean_delta']:+.5f}  t={s['t']}  "
                  f"significant={s['significant']}")
    print(f"\nEUREKA: {result['headline']['eureka_criterion_met']}  "
          f"({result['dial_kind']['verdict']})")
    for r in result["headline"]["reasons_against"]:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
