# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The `reach` dial: how far the `add` word is from what is already in play.

WHAT THIS IS FOR
================
Stage 2's one publishable slot was `add` (+0.326 over a context-matched floor), and the
reason is mechanical: `add`'s value is absent from the visible context ~40-44% of the time
where `accept`'s is present 92.8% of the time. `add` is the slot where the model names
something that is not in front of it. This module turns that into a **dial**: the think-block
declares `reach: near|mid|far`, derived from how distant the `add` word is from the scene, and
at inference the declaration becomes an INPUT -- set `reach: far`, get a more distant beat.

THREE INSTRUMENT PROBLEMS: TWO KNOWN BEFORE THIS MODULE EXISTED, ONE FOUND BUILDING IT
======================================================================================

**1. DIRECTION. High NPMI means TIGHTLY ASSOCIATED, which is a NEAR reach.** Verified against
a real association table: ``npmi(dog, bark) = +0.1161``, ``npmi(dog, tail) = +0.1296``,
``npmi(cat, meow) = +0.0978``. A bucketer that sorts association descending and calls the top
tercile `far` is INVERTED, and would produce a perfectly significant, perfectly backwards
result.

The structural defence, not just a comment: **nothing in this module ranks an association.**
`reach_distance` returns ``1 - npmi``, an actual distance, so every comparison downstream
reads in the natural direction -- small distance is `near`, large is `far`, `reach_bucket`
uses plain ``<``, and there is no descending sort anywhere to get backwards.
`test_a_hand_built_near_pair_and_a_hand_built_far_pair_land_in_the_expected_buckets` pins
it, and dies if the comparisons flip.

**2. ZERO-INFLATION. `npmi` returns exactly 0.0 for BOTH "no association" and "never
co-occurred in the table".** Those are opposite claims -- one is a measurement, the other is
our own ignorance -- and binning ignorance as `far` would make the dial partly a dial on
table coverage. Two things are done about it:

  * The table is built over **ALL within-story content-word co-occurrence** (document = one
    whole story, symmetric, every pair), not over the sparse directional
    ``(prefix_word, turn_word)`` pairs that produced the original diagnosis. That alone cut
    the exactly-zero rate on real observations from **6.7% to 0.54%** -- measured over a table
    of 50,000 stories scoring skits drawn from 400,000, so most scored stories were OUTSIDE
    the table and the 0.54% is a genuine coverage figure. (When the scored story IS in the
    table the rate collapses to a structural 0.0000, which is the tautology the holdout below
    exists to undo.) The shipped run re-measures it and writes it into the manifest.
  * What remains is EXCLUDED, never binned. `reach_distance` returns **None** when no context
    word has any co-occurrence evidence with the `add` word, and the caller drops that skit
    under its own named rule. A pair that HAS evidence and still scores 0.0 (negatively
    associated, clamped) is genuinely far and is kept -- `pair_has_evidence` is what tells
    the two apart.
  * Every distance is computed **LEAVE-ONE-OUT** (`holdout`), because the scored story is
    itself a document of the table. Without that the fix above becomes a tautology: the
    no-evidence rate falls to a structural 0.0000 and a rare `add` word paired with a rare
    context word scores a PERFECT association manufactured by the very scene being scored.
    See `pair_counts` for the measured size of that effect.

**3. THE DIAL IS PARTLY A RARITY DIAL, and it runs opposite to the obvious worry.** NPMI is
not frequency-neutral: two RARE words that co-occur at all score high, while a COMMON word's
NPMI with anything is bounded low by its own marginal. So a rare `add` word tends to score
NEAR and a common one tends to score FAR -- and `add` is chosen as the rarest fresh word in
the turn. Measured on the shipped artifact: ``spearman(add_df, distance) = +0.2078``, with
per-bucket median document frequency near 16,591 / mid 51,269 / far 39,367 -- so the effect is
concentrated in the near vs not-near contrast rather than spread monotonically across the dial.

This one is NOT fixed here, because it is a property of the measure rather than a bug in the
plumbing. It is PUBLISHED as a number instead -- `frequency_confound`, per bucket, plus an
`add_df` on every written row -- so an eval can control for it. The consequence is worth
stating plainly: **a monotone, significant near < mid < far on realised distance is not by
itself evidence the model reached further. It is equally consistent with the model reaching
for a commoner word.**

WHY `reach` IS RENDERED BEFORE `add`
-----------------------------------
`REACH_SLOT_NAMES` puts `reach` ahead of `add`, which is load-bearing rather than cosmetic.
The block is generated left to right, so a dial that is declared AFTER the word it is
supposed to govern cannot govern it: forcing `reach: far` at inference would only relabel a
choice already made. `test_reach_is_declared_before_add` pins the order.

NO IMPORTS FROM `scripts/`
--------------------------
Deliberate, and the same shape as `derive_skit_from_turns` taking `intensity` as a callable:
this module owns the metric, the derivation script owns the corpus plumbing. `scripts.
score_improv.npmi` is the same FORMULA over a different table (documents are
(prefix, continuation) pairs there, whole stories here) and a different storage (that one
keys both directions, this one canonicalises ``a < b`` to halve the memory over 2.1M
documents). `test_npmi_agrees_with_score_improv_on_the_same_documents` builds both tables
from one document set and asserts value-for-value agreement, so "one formula, two storages"
is a live check rather than a claim.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict, fields
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from train.improv import STAKES_EPSILON, Slots, content_words
from train.skit import MODEL_TURNS, SKIT_ROLES, Skit

#: The dial's three settings, nearest first. Order is meaningful: `REACH_VALUES.index`
#: is a monotone function of distance, which is what the eval's near < mid < far contrast
#: is stated in.
REACH_VALUES: Tuple[str, ...] = ("near", "mid", "far")

#: The v3 think-block. `reach` sits BEFORE `add` -- see the module docstring; a dial declared
#: after the word it governs is decoration. `stakes` is carried as a continuous signed delta
#: rather than the up/level/down label that stage 2 withdrew as mostly confounded.
REACH_SLOT_NAMES: Tuple[str, ...] = ("offer", "accept", "reach", "add", "stakes", "handback")


@dataclass(frozen=True)
class ReachSlots:
    """The v3 think-block's six slots. Field ORDER is the rendered order.

    `train.improv.render_think` reads the field order off the dataclass, so this ordering --
    not a separate list -- is what reaches the training data.
    `test_reach_slot_names_match_the_dataclass_fields` pins the two together.
    """
    offer: str
    accept: str
    reach: str
    add: str
    stakes: str
    handback: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def parse_reach_think(text: str) -> Optional[ReachSlots]:
    """Parse a v3 SIX-slot think-block out of `text`, or None if it is malformed.

    Deliberately a separate function from `train.improv.parse_think` rather than a change to
    it. That one validates the FIVE-slot schema and requires `stakes` to be one of
    up/level/down, and it sits behind the published stage-1 and stage-2 adherence rates; a v3
    block fails it, which is CORRECT -- they are different schemas and conflating them would
    make two published rates incomparable.

    None rather than a partial object, for the same reason: schema adherence is reported as a
    RATE, and a partial parse would inflate it. A `reach` outside `REACH_VALUES` is malformed,
    and a `stakes` that is not a signed number is malformed -- both are things a GENERATED
    block will do, which is precisely when this is called.
    """
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    if not m:
        return None
    found: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key in REACH_SLOT_NAMES and value:
            found[key] = value
    if set(found) != set(REACH_SLOT_NAMES):
        return None
    if found["reach"] not in REACH_VALUES:
        return None
    if parse_stakes_delta(found["stakes"]) is None:
        return None
    return ReachSlots(**found)


#: The v3 block with the dial REMOVED -- the `nodial` control arm's schema. Derived from
#: `REACH_SLOT_NAMES` by subtraction rather than written out again, so the two schemas
#: cannot drift into differing on a slot other than `reach`, which is the only difference
#: the control is allowed to have.
NODIAL_SLOT_NAMES: Tuple[str, ...] = tuple(n for n in REACH_SLOT_NAMES if n != "reach")


@dataclass(frozen=True)
class NoDialSlots:
    """The `nodial` control arm's five-slot block: v3 minus `reach`, nothing else changed.

    WHY THIS EXISTS, and why it is not `train.improv.Slots`
    ------------------------------------------------------
    `Slots` has these same five field names in this same order, so it would RENDER
    identically -- and that is exactly the trap. `Slots` is the published v2 schema whose
    `stakes` is one of ``up``/``level``/``down`` and whose parser
    (`train.improv.parse_think`) enforces that; the reach derivation carries `stakes` as a
    continuous signed delta (``"+0.4"``). Reusing `Slots` would put a v3 value inside a v2
    dataclass, and the first thing that validated it -- an adherence rate, say -- would
    silently disagree with the training data. A separate name keeps the two schemas
    separable at the type level, at the cost of one dataclass.

    The control's whole claim is "identical to the dial arm except the dial is absent", so
    the field ORDER here is `REACH_SLOT_NAMES` with `reach` deleted in place -- not a
    re-listing. `train.improv.render_think` reads the field order off the object, so this
    ordering is what reaches the training data.
    `test_the_nodial_block_is_the_dial_block_minus_exactly_the_reach_line` pins it against
    the rendered dial block, on a real skit.
    """
    offer: str
    accept: str
    add: str
    stakes: str
    handback: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def drop_reach(slots: ReachSlots) -> NoDialSlots:
    """A `ReachSlots` with the dial removed and every other value carried across verbatim.

    THE CONTROL ARM'S INDEPENDENT VARIABLE, in one named place. `nodial` is a valid negative
    control for "forcing a `reach` token does nothing to a model that never learned one" only
    if it differs from `dial` in the dial and in nothing else, so this copies the other five
    values by NAME rather than by position -- a positional copy would survive a field
    reordering in either dataclass while quietly re-labelling the slots.

    Reads the field names off `NoDialSlots` rather than hard-coding them, so adding a slot to
    the v3 schema is a `TypeError` here (a missing argument) instead of a silently dropped
    value.
    """
    return NoDialSlots(**{f.name: getattr(slots, f.name) for f in fields(NoDialSlots)})


# --------------------------------------------------------------------------------------
# the association table  (document = one whole story)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Association:
    """Symmetric within-document co-occurrence counts.

    `uni[w]`   -- documents containing `w`.
    `co[(a,b)]` -- documents containing both, keyed with ``a < b`` ONLY. Half the memory of
                   the both-directions form, which matters at 2.1M documents; every reader
                   goes through `pair_key` so the asymmetry cannot leak.
    `n_docs`   -- documents.
    """
    uni: Dict[str, int]
    co: Dict[Tuple[str, str], int]
    n_docs: int


def pair_key(a: str, b: str) -> Tuple[str, str]:
    """The canonical, order-free key for a word pair."""
    return (a, b) if a <= b else (b, a)


def build_association(documents: Iterable[str]) -> Association:
    """Count within-document co-occurrence over `documents`. One document = one story.

    Every unordered pair of DISTINCT content words in a document is counted once, however
    often either word repeats -- document frequency, not term frequency, exactly as
    `scripts.score_improv.build_association` does it, so the NPMI values are comparable.

    The word set is `sorted(set(...))`, which is what makes the pair keys canonical without
    a second `pair_key` call per pair, and makes the whole table independent of
    `PYTHONHASHSEED`.
    """
    uni: Dict[str, int] = {}
    co: Dict[Tuple[str, str], int] = {}
    n = 0
    for doc in documents:
        words = sorted(set(content_words(doc)))
        n += 1
        for w in words:
            uni[w] = uni.get(w, 0) + 1
        for i, a in enumerate(words):
            for b in words[i + 1:]:
                k = (a, b)
                co[k] = co.get(k, 0) + 1
    return Association(uni=uni, co=co, n_docs=n)


def pair_counts(a: str, b: str, assoc: Association, *,
                holdout: bool = False) -> Tuple[int, int, int, int]:
    """``(count(a), count(b), count(a & b), n_docs)``, optionally LEAVING ONE DOCUMENT OUT.

    LEAVE-ONE-OUT, AND WHY IT IS NOT OPTIONAL IN PRACTICE. The association table's documents
    are whole stories, and a skit's `add` word and its context words all come from ONE story
    -- which is itself a document of the table. So without a holdout every pair is guaranteed
    at least one co-occurrence: **its own**. Two consequences, both bad, both measured:

      * the "no evidence" rate becomes structurally 0.0000 -- a tautology dressed as a clean
        instrument, not an achievement;
      * a rare `add` word paired with a rare context word gets ``count(a) = count(b) =
        count(a & b) = 1``, which is a PERFECT association (npmi 1.0, distance 0.0) produced
        entirely by the scene being scored. And `add` is chosen as the highest-IDF fresh word,
        i.e. the rarest word in the turn, so this lands exactly where it does most harm.

    Measured on 20,000 stories: 0.45% of observations have their nearest context pair
    supported by a single own-story co-occurrence, and the zero-evidence rate goes from
    0.0000 (without holdout, structurally) to 0.0045 (with it, a real measurement).

    `holdout` subtracts one document from all four counts. The caller must pass True only when
    the scored story IS a document of this table -- `scripts/derive_dialogue_skits.py` knows
    that from the story's index against the table's population.
    """
    ca, cb = assoc.uni.get(a, 0), assoc.uni.get(b, 0)
    cab = assoc.co.get(pair_key(a, b), 0)
    n = assoc.n_docs
    if holdout and cab and ca and cb and n:
        ca, cb, cab, n = ca - 1, cb - 1, cab - 1, n - 1
    return ca, cb, cab, n


def pair_has_evidence(a: str, b: str, assoc: Association, *,
                      holdout: bool = False) -> bool:
    """Did `a` and `b` co-occur in a document of the table (other than the held-out one)?

    THE ZERO-INFLATION GUARD. `npmi` cannot answer this: it returns 0.0 both for a pair that
    never co-occurred (no evidence, our ignorance) and for a pair that co-occurs less often
    than chance (evidence of distance, clamped at the floor). Callers that treat a 0.0 as
    "far" without asking this question are measuring table coverage as if it were semantics.
    """
    ca, cb, cab, n = pair_counts(a, b, assoc, holdout=holdout)
    return bool(ca and cb and cab and n)


def npmi(a: str, b: str, assoc: Association, *, holdout: bool = False) -> float:
    """Normalised PMI in [0, 1]. Same formula as `scripts.score_improv.npmi`.

    0.0 means EITHER no association OR no evidence; `pair_has_evidence` separates them and
    every caller here goes through `reach_distance`, which asks. See `pair_counts` for
    `holdout`.
    """
    ca, cb, cab, n = pair_counts(a, b, assoc, holdout=holdout)
    if not (ca and cb and cab and n):
        return 0.0
    p_ab = cab / n
    denom = -math.log(p_ab)
    if denom <= 0:
        return 0.0
    return max(0.0, math.log(p_ab / ((ca / n) * (cb / n))) / denom)


# --------------------------------------------------------------------------------------
# the distance, and the buckets
# --------------------------------------------------------------------------------------
def reach_distance(add_word: str, context_words: Sequence[str], assoc: Association, *,
                   holdout: bool = False) -> Optional[float]:
    """How far `add_word` is from the nearest thing already in play, or None for no evidence.

    ``1 - max NPMI(add_word, c)`` over the context, computed as ``min(1 - npmi)`` so the code
    never sorts an association descending -- see the module docstring's DIRECTION note. The
    result is in [0, 1]: **0 is on top of the scene, 1 is as far as this table can say.**

    Context words with no co-occurrence evidence are SKIPPED rather than scored 1.0, and when
    that leaves nothing the answer is **None** -- "this table cannot say", which the caller
    must drop rather than bin. Returning 1.0 there would put every word the table has never
    seen into `far` and turn the dial into a dial on our own coverage.

    `add_word` is never itself in the context: `add` is by construction a word that is NOT in
    `established` (see `train.skit._slots_for_turn`), and
    `test_the_add_word_is_never_in_its_own_context_on_real_skits` checks that on the real
    artifact rather than assuming it.
    """
    best: Optional[float] = None
    for c in context_words:
        if not pair_has_evidence(add_word, c, assoc, holdout=holdout):
            continue
        d = 1.0 - npmi(add_word, c, assoc, holdout=holdout)
        if best is None or d < best:
            best = d
    return best


def fit_reach_terciles(distances: Sequence[float]) -> Tuple[float, float]:
    """The two cut points that split `distances` into three, as ``(lo, hi)``.

    Nearest-rank on the sorted values: ``lo = s[n//3]``, ``hi = s[2n//3]``. `reach_bucket`
    then uses strict ``<``, so the element AT a cut falls in the upper bucket and the split
    is exactly reproducible from the fitted values alone -- which is the point, because these
    two numbers go into the manifest and **eval must not re-fit them**. A dial whose buckets
    move between train and eval measures nothing.

    Ties are not smoothed away. If the distribution is lumpy the three buckets come out
    unbalanced, and that is reported (`bucket_balance`) rather than hidden by interpolation:
    the specific failure that killed `stakes` in stage 2 was 85.3% of mass in one class, and
    it was invisible until the balance was printed.

    Raises ValueError below three values -- a fit that cannot separate three buckets must not
    return numbers that look as if it did.
    """
    if len(distances) < 3:
        raise ValueError(f"cannot fit terciles from {len(distances)} value(s); need >= 3")
    s = sorted(distances)
    n = len(s)
    return s[n // 3], s[(2 * n) // 3]


def reach_bucket(distance: Optional[float], lo: float, hi: float) -> str:
    """Which dial setting `distance` belongs to: `near`, `mid` or `far`.

    THE DECISION FUNCTION OF THIS WHOLE SPEC. Small distance is `near`, large is `far`; the
    comparisons are plain ``<`` against the fitted cut points and there is deliberately no
    sorting, no negation and no reversal anywhere in the path from `reach_distance` to here.

    `distance is None` RAISES. None is `reach_distance`'s "no evidence" answer, and the
    caller's job is to drop that skit; silently returning `far` for it is exactly the
    zero-inflation trap this module exists to avoid.
    """
    if distance is None:
        raise ValueError("distance is None (no co-occurrence evidence): the caller must "
                         "DROP this observation, not bucket it -- see reach_distance")
    if not lo <= hi:
        raise ValueError(f"cut points out of order: lo={lo} must not exceed hi={hi}")
    if distance < lo:
        return "near"
    if distance < hi:
        return "mid"
    return "far"


def bucket_balance(buckets: Sequence[str]) -> Dict[str, object]:
    """Counts and fractions per dial setting, plus the largest share.

    A property of SCALE, so it is asserted against the real artifact rather than a fixture
    (`test_bucket_balance_on_the_real_artifact`). `max_fraction` is the number to read: at
    0.853 `stakes` was 85.3% one class with 26 points of headroom above chance, and no amount
    of significance testing rescues that.
    """
    n = len(buckets)
    counts = {v: sum(1 for b in buckets if b == v) for v in REACH_VALUES}
    unknown = n - sum(counts.values())
    return {
        "n": n,
        "counts": counts,
        "fractions": {v: (round(counts[v] / n, 4) if n else None) for v in REACH_VALUES},
        "max_fraction": (round(max(counts.values()) / n, 4) if n else None),
        "unknown_values": unknown,
        "chance_floor_note": "max_fraction IS the majority-class floor any classifier of "
                             "this label must beat. Balanced terciles put it near 0.34.",
    }


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation, TIE-AWARE, or None when either side has no spread.

    Tie handling is not a detail here: the `x` this is used on is the `add` word's document
    frequency, and the same word recurs across hundreds of observations, so a naive
    ordinal-position ranking would invent an ordering inside every tied block and report a
    correlation that is partly an artefact of input order. Ties get their average rank.

    None rather than 0.0 when a side is constant -- "no variance to correlate" and "no
    correlation" are different claims, and a published 0.0 would hide a degenerate input.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None

    def ranks(vs: Sequence[float]) -> List[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return None
    return num / den


def frequency_confound(add_dfs: Sequence[int], distances: Sequence[float],
                       buckets: Sequence[str]) -> Dict[str, object]:
    """How much of the dial is a RARITY dial rather than a semantic-distance dial.

    THE MOST IMPORTANT CAVEAT THIS MODULE PUBLISHES, and it runs opposite to the obvious
    worry. NPMI is not frequency-neutral: two RARE words that co-occur at all score high
    (their expected co-occurrence under independence is tiny), while a COMMON word's NPMI with
    anything is bounded low by its own marginal. So a rare `add` word tends to score NEAR and a
    common one tends to score FAR -- which means `reach: far` partly selects COMMON words.

    Measured on the shipped artifact (2.1M stories, 123,042 observations):
    ``spearman(add_df, distance) = +0.2078``, with per-bucket median document frequency
    **near 16,591 / mid 51,269 / far 39,367**. Read the shape, not just the sign: the confound is
    concentrated in the NEAR vs NOT-NEAR contrast and is NOT monotone across the dial -- `mid` is
    the commonest bucket, not `far`. So a near-vs-far comparison is also a rare-vs-common
    comparison, while mid-vs-far is much less exposed. (At 20,000 stories the correlation was
    +0.306 and looked monotone. It is not, which is its own argument against inheriting a number
    measured on a different population.)

    Reported here, per bucket, so an eval can CONTROL for it rather than merely be warned:
    every written row also carries its own `add_df`, so the covariate needs no re-derivation.
    A positive `spearman_df_vs_distance` means commoner `add` words score as farther.
    """
    per_bucket: Dict[str, object] = {}
    for v in REACH_VALUES:
        dfs = sorted(df for df, b in zip(add_dfs, buckets) if b == v)
        per_bucket[v] = {
            "n": len(dfs),
            "median_add_df": (dfs[len(dfs) // 2] if dfs else None),
            "mean_add_df": (round(sum(dfs) / len(dfs), 1) if dfs else None),
        }
    rho = spearman(add_dfs, distances)
    return {
        "spearman_df_vs_distance": (round(rho, 4) if rho is not None else None),
        "sign_means": "POSITIVE = commoner `add` words score as FARTHER, i.e. `reach: far` "
                      "partly selects common words. This is a property of NPMI (rare words "
                      "that co-occur at all score high; a common word's NPMI is bounded low "
                      "by its own marginal), not of this implementation.",
        "per_bucket": per_bucket,
        "eval_must_control_for_this":
            "A near<mid<far result on realised distance is NOT by itself evidence the model "
            "reached further, because it could equally be evidence the model reached for a "
            "commoner word. Every row carries `add_df`; match or covary on it.",
    }


# --------------------------------------------------------------------------------------
# stakes, as a continuous delta  (ruling D)
# --------------------------------------------------------------------------------------
def stakes_delta(turn: str, prev_turn: str, intensity) -> float:
    """Intensity of `turn` minus intensity of `prev_turn`: harm hits per 100 content words.

    The same quantity `train.skit._slots_for_turn` computes and then throws away by binning
    it into up/level/down. Stage 2 withdrew that label as mostly confounded; ruling D carries
    the delta forward CONTINUOUSLY and tests it on magnitude, so the withdrawal has a
    successor instead of a silent disappearance. It is not a headline slot.

    `intensity` is injected, exactly as `derive_skit_from_turns` injects it -- this module
    does not import `scripts.score_improv`.
    """
    return intensity(turn) - intensity(prev_turn)


def format_stakes_delta(delta: float) -> str:
    """The delta as a signed one-decimal string: ``+0.0``, ``+12.5``, ``-3.4``.

    Always signed, so the slot's direction is legible in the rendered block and a generated
    value is parseable. ``-0.0`` is normalised to ``+0.0``: it is the same number, and two
    spellings of zero in the training data would be two tokens for one fact.
    """
    v = round(float(delta), 1)
    if v == 0:
        v = 0.0
    return f"{v:+.1f}"


def parse_stakes_delta(text: str) -> Optional[float]:
    """Read back what `format_stakes_delta` wrote, or None if it is not a signed number.

    Used by the round-trip test and by any scorer that has to read a GENERATED block, where
    the model may well emit something that is not a number at all -- None is that answer, not
    an exception.
    """
    t = text.strip()
    if not t or t[0] not in "+-":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def stakes_label(delta: float) -> str:
    """The stage-2 up/level/down label for `delta`, kept only so the two can be compared.

    Not rendered anywhere. It exists so a test can show the continuous slot preserves the
    label's information (`test_the_continuous_delta_still_recovers_the_old_label`) and so a
    future comparison against the stage-2 measurement has one implementation of the old rule
    rather than a re-typed threshold.
    """
    return ("up" if delta > STAKES_EPSILON
            else "down" if delta < -STAKES_EPSILON else "level")


# --------------------------------------------------------------------------------------
# putting the slot into a skit
# --------------------------------------------------------------------------------------
def block_context_words(prefix: str, turns: Sequence[str], t_idx: int) -> List[str]:
    """Every content word ALREADY IN PLAY before model turn `t_idx`: prefix + earlier turns.

    A RE-WALK of `train.skit.derive_skit_from_turns`' `established`, for the same reason
    `classify_turn_failure` re-walks its gates: the derivation collapses the value into a
    yes/no and the reach metric needs the words themselves. It is a re-walk and not a second
    definition, and the two are held together by a check on the real artifact rather than by
    this docstring -- `add` is by construction absent from `established`, so
    `test_the_add_word_is_never_in_its_own_context_on_real_skits` fails the moment the two
    drift apart.
    """
    words = list(content_words(prefix))
    for j in range(t_idx):
        words.extend(content_words(turns[j]))
    return words


def add_word_of(slots) -> str:
    """The single word the `add` slot names.

    `_slots_for_turn` renders `add` as ``", ".join(fresh_ranked[:1])`` -- one word today, and
    a comma-joined list if that slice ever widens. Taking the first item keeps this correct
    either way, instead of assuming the string is a bare word.
    """
    return slots.add.split(",")[0].strip().lower()


def skit_reach_distances_per_block(skit: Skit, assoc: Association, *,
                                  holdout: bool = False) -> List[Optional[float]]:
    """One reach distance per model turn, `None` where that block has no evidence.

    The NON-short-circuiting form. It exists so the drop table can say HOW MANY OBSERVATIONS
    lacked evidence rather than only how many skits did -- the zero fraction the pre-flight
    asked to have re-measured is an observation-level rate, and a skit-level count cannot be
    converted into one.
    """
    out: List[Optional[float]] = []
    for i, t_idx in enumerate(MODEL_TURNS):
        ctx = block_context_words(skit.prefix, skit.turns, t_idx)
        out.append(reach_distance(add_word_of(skit.blocks[i]), ctx, assoc,
                                  holdout=holdout))
    return out


def skit_reach_distances(skit: Skit, assoc: Association, *,
                         holdout: bool = False) -> Optional[List[float]]:
    """One reach distance per model turn, or None if ANY block has no evidence.

    Whole-or-nothing, matching `derive_skit_from_turns`' drop rule: a skit whose second block
    carries no dial reading cannot be trained on with a dial in its first, and a think-block
    whose `reach` is a guess teaches the guess.
    """
    per_block = skit_reach_distances_per_block(skit, assoc, holdout=holdout)
    if any(d is None for d in per_block):
        return None
    return [d for d in per_block if d is not None]


def skit_stakes_deltas(skit: Skit, intensity) -> List[float]:
    """One continuous stakes delta per model turn: this turn against the one before it."""
    return [stakes_delta(skit.turns[t_idx],
                         skit.turns[t_idx - 1] if t_idx > 0 else skit.prefix,
                         intensity)
            for t_idx in MODEL_TURNS]


def with_reach(skit: Skit, *, distances: Sequence[float], deltas: Sequence[float],
               lo: float, hi: float) -> Skit:
    """The same skit with v3 six-slot blocks: `reach` bucketed, `stakes` continuous.

    Text is untouched -- prefix, turns and the other four slot values are carried across
    verbatim. Only the block schema changes, so the label rule and the positional
    nine-segment supervision mask in `scripts.derive_skits.build_skit_example` keep working
    on the result unchanged (they read `skit_segments`, which reads the block through
    `render_think`, which reads the dataclass's own field order).
    """
    if not (len(distances) == len(deltas) == len(skit.blocks)):
        raise ValueError(f"need one distance and one delta per block: "
                         f"{len(distances)} / {len(deltas)} / {len(skit.blocks)}")
    blocks = tuple(
        ReachSlots(offer=b.offer, accept=b.accept,
                   reach=reach_bucket(distances[i], lo, hi), add=b.add,
                   stakes=format_stakes_delta(deltas[i]), handback=b.handback)
        for i, b in enumerate(skit.blocks)
    )
    return Skit(story_id=skit.story_id, prefix=skit.prefix, turns=skit.turns, blocks=blocks)


def reach_slot_names_of(slots) -> Tuple[str, ...]:
    """The rendered slot order of `slots`, read off the dataclass.

    One helper so the tests, the manifest and `render_think` all read the order from the same
    place -- a hard-coded second list is how a reordering would ship silently.
    """
    return tuple(f.name for f in fields(slots))


__all__ = ["REACH_VALUES", "REACH_SLOT_NAMES", "ReachSlots", "parse_reach_think",
           "NODIAL_SLOT_NAMES", "NoDialSlots", "drop_reach", "Association", "pair_key", "pair_counts",
           "build_association", "pair_has_evidence", "npmi", "reach_distance",
           "fit_reach_terciles", "reach_bucket", "bucket_balance", "spearman",
           "frequency_confound", "stakes_delta",
           "format_stakes_delta", "parse_stakes_delta", "stakes_label",
           "block_context_words", "add_word_of", "skit_reach_distances",
           "skit_reach_distances_per_block",
           "skit_stakes_deltas", "with_reach", "reach_slot_names_of",
           "Slots", "SKIT_ROLES"]
