"""The improv think-block: schema, extraction, rendering, parsing.

Slots hold SPANS LIFTED FROM THE TEXT, never paraphrases. There is no validated
generator here to paraphrase with, and putting an unvalidated model inside the data
pipeline would make every downstream number unattributable.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional

from train.dialogue import split_sentences_dialogue

SLOT_NAMES = ("offer", "accept", "add", "stakes", "handback")
STAKES_VALUES = ("up", "level", "down")

#: Small closed-class list. Deliberately not a package dependency — the corpus is simple
#: prose and a 40-word list is auditable where an opaque stopword set is not.
STOPWORDS = frozenset("""
a an the and or but if then than so as of to in on at by for with from into onto over
is was were be been being am are it its it's this that these those there here he she
they them his her their him us we you your i me my not no nor do did does done have
has had will would can could should may might must very just
""".split())

_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class Slots:
    offer: str
    accept: str
    add: str
    stakes: str
    handback: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


def split_sentences(text: str) -> List[str]:
    r"""Sentence split. Delegates to `train.dialogue.split_sentences_dialogue`.

    CLEAN CUTOVER, 2026-08-23. This used to be ``re.split(r'(?<=[.!?"])\s+')``, which broke
    ``'"It catches the light!" said her friend.'`` into two units -- the quote and its own
    attribution tag -- and cost the stage-2 eval population 43% of the corpus's dialogue
    (54.6% -> 31.0% of units). Approved in
    ``docs/superpowers/specs/2026-08-23-reach-dial-design.md`` ("The splitter, fixed
    properly"): ONE splitter in the tree, and stage 1 re-derived and REPUBLISHED with the
    new numbers beside the old rather than silently overwritten -- see
    ``docs/measurements/improv-stage1.json``'s ``superseded_by`` and
    ``derivation_republished_2026_08_23``.

    A thin delegation, not a copy: a second splitter in the tree is exactly what the spec
    forbids. Everything reached through this name (``train.skit``, ``scripts.score_improv``,
    ``scripts.derive_traces``, ``scripts.eval_skits``) picks up the new behaviour from here.
    """
    return split_sentences_dialogue(text)


def content_words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in STOPWORDS]


def render_think(slots) -> str:
    """Render a think-block from any slots dataclass, in its OWN field order.

    GENERALISED 2026-08-23 (reach dial, task 2). This used to iterate the module-level
    `SLOT_NAMES`, which is exactly the five fields of `Slots`; it now reads the field order
    off the object, so `train.reach.ReachSlots` (six slots, `reach` ahead of `add`) renders
    through the same function. Behaviour for `Slots` is unchanged --
    `test_render_think_reads_the_dataclass_field_order` pins
    ``fields(Slots) == SLOT_NAMES``, so the generalisation cannot drift into a different
    rendering of the published schema.

    Why generalise rather than add a second renderer: `train.skit.skit_segments` and
    `scripts.derive_skits.build_skit_example` reach the block through this one name, and the
    pre-shifted label rule and the positional nine-segment supervision mask must stay
    byte-identical between the two schemas. One renderer is how that is guaranteed instead of
    asserted.
    """
    body = "\n".join(f"{f.name}: {getattr(slots, f.name)}" for f in fields(slots))
    return f"<think>\n{body}\n</think>\n"


def parse_think(text: str) -> Optional[Slots]:
    """Parse a think-block, or None if malformed.

    Returns None rather than a partial object on purpose: schema adherence is reported as
    a rate, and a partial parse would inflate it.
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
        if key in SLOT_NAMES and value:
            found[key] = value
    if set(found) != set(SLOT_NAMES):
        return None
    if found["stakes"] not in STAKES_VALUES:
        return None
    return Slots(**found)


#: Below this the delta is noise rather than escalation. FIX 5(c) (task-6-report.md):
#: this was previously commented as "Calibrated in Task 3 against the corpus and recorded
#: there" -- that calibration never happened (no such analysis appears in
#: task-3-report.md or anywhere else in this plan's history). This value is an
#: unvalidated default, not a measured one. Treat any result sensitive to its exact
#: value with correspondingly less confidence until it is actually calibrated.
STAKES_EPSILON = 0.5


def extract_slots(prefix: str, continuation: str, *, idf: Dict[str, float],
                  intensity: Callable[[str], float]) -> Optional[Slots]:
    """Derive a think-block from a real continuation, or None to DROP the example."""
    p_sents = split_sentences(prefix)
    if not p_sents or not continuation.strip():
        return None

    last = p_sents[-1]
    p_words = content_words(prefix)
    c_words = content_words(continuation)
    if not c_words:
        return None

    # accept: the longest run of shared content words between the final prefix sentence
    # and the continuation. Falls back to the commonest prefix word that reappears.
    last_words = content_words(last)
    carried = [w for w in last_words if w in set(c_words)]
    if not carried:
        # Tie-break alphabetically: set() iteration order depends on Python's
        # per-process string hash randomization (PYTHONHASHSEED), so a bare count key
        # lets ties resolve differently across runs on identical input. The corpus
        # this produces must be reproducible, so the key must be total (unique) —
        # negate the count to keep most-frequent-first under the plain ascending sort.
        carried = [w for w in sorted(set(p_words), key=lambda w: (-p_words.count(w), w))
                   if w in set(c_words)][:1]
    if not carried:
        return None                      # nothing carried forward -> a block -> drop

    fresh = [w for w in c_words if w not in set(p_words)]
    if not fresh:
        return None                      # nothing added -> also a block -> drop
    # Same reproducibility concern as the `carried` sort above: append the word itself
    # to the key so IDF ties resolve alphabetically instead of by hash-randomized
    # set() iteration order, which would otherwise vary between processes.
    fresh_ranked = sorted(set(fresh), key=lambda w: (-idf.get(w, 0.0), w))
    add = ", ".join(fresh_ranked[:1])

    delta = intensity(continuation) - intensity(prefix)
    stakes = "up" if delta > STAKES_EPSILON else "down" if delta < -STAKES_EPSILON else "level"

    c_sents = split_sentences(continuation)
    tail = content_words(c_sents[-1]) if c_sents else []
    introduced = [w for w in tail if w in set(fresh)]
    handback = introduced[-1] if introduced else "open"

    return Slots(
        offer=" ".join(last_words[:12]) or last[:60],
        accept=" ".join(carried[:6]),
        add=add,
        stakes=stakes,
        handback=handback,
    )
