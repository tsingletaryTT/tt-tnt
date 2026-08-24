# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Does the `add` slot name a THING or an ACTION? -- the content-word gate.

WHY THIS MODULE EXISTS
======================
The reach dial worked statistically (frequency-controlled monotone, t=9.0) and read as
nothing at all, because `train.skit._slots_for_turn` picks `add` as the highest-IDF word the
turn introduces, and a TinyStories dialogue turn is often three words long. Measured on
`artifacts/reach-skits/` (123,042 observations, 6,442 distinct words), the 25 commonest `add`
values were 18.5% of the whole slot and looked like this::

    look please hi hello love what wow why come yes thank ok okay want mine
    let's doing where our mean can't what's give course hey

That is a discourse-particle vocabulary, so `reach: far` reached into a different set of
PARTICLES and the CLI printed ``add=what's`` where a reader expected ``add=comet``. This
module is the gate that makes `add` name something.

THREE INSTRUMENT PROBLEMS, ALL THREE MEASURED BEFORE ANY FILTER WAS WRITTEN
==========================================================================

**1. Tagging a bare token decides nothing.** ``nltk.pos_tag(["please"])``, ``["wow"]``,
``["comet"]`` and ``["hello"]`` all return ``NN`` on this machine's tagger
(averaged_perceptron_tagger_eng). An isolated tag carries no information here at all, so
every tag this module reads is taken from the word INSIDE ITS TURN.

**2. In-context tagging separates them FOR THE WRONG REASON.** Tagged as written, the
particles pile onto ``NNP`` -- ``hi`` 120/120, ``hello`` 118/120, ``wow`` 115/120,
``hey`` 118/120 -- purely because they open a quoted utterance and are therefore capitalised.
An NN/NNS-excluding-NNP filter would have scored well on this corpus and collapsed the first
time a particle appeared mid-utterance ("I said hello"). Worse, the same artifact tags
``look`` NNP 93/120 and ``come`` NNP 84/120 -- two words that ARE genuine verbs and must be
kept -- so the "right" answer and the wrong one came from the same accident.

The defence is structural, not a comment: `pos_tags_in_turn` LOWERCASES EVERY TOKEN before
tagging, so capitalisation cannot be a cue anywhere in the utterance, for any word, in either
direction. Re-measured that way the particles move to ``NN`` and become indistinguishable
from ``comet`` -- which is the honest finding: **part-of-speech cannot separate a greeting
from a noun, and this module does not ask it to.** What POS is good for here is rejecting
adjectives, adverbs, numerals and prepositions ("you are very *kind*", "so *pretty*", "*five*
more minutes"), and doing it PER INSTANCE. Over the 646 real turns where ``mean`` is the
`add` word the tagger calls it a verb in 304 and an adjective in 311, and the gate follows:
kept in "We did not mean to be bad." (VB), rejected in "You are mean. Give it back to me."
(JJ). It is not perfect at it -- "Why are you so mean to me?" comes back VB and is wrongly
kept -- which is why this is the LAST gate and not the one holding the particles out.

**3. What actually separates a particle from a thing is WHERE IT LIVES.** A referential word
is used by the NARRATOR too; a phatic one is almost never used outside quotation marks.
Measured over the whole 2,119,489-story corpus, as the fraction of a word's occurrences that
fall outside quotes (`narration_rate`)::

    KEEP  look .503  come .509  mean .534  give .584  want .477  doing .403  love .293
    DROP  okay .237  thank .182  ok .139  course .118  our .114  can't .104  mine .097
          hi .070  please .066  hey .041  what's .035  wow .033  let's .014

`NARRATION_FLOOR` is 0.20: below `love` (0.293), the lowest-rate word the task's own ground
truth says must be KEPT, and above `thank` (0.182), the highest-rate one it says must go. It
is not tuned on the hand-labelled sample -- see `scripts/validate_content_add.py`, where
moving it over 0.10..0.30 moves the derivation's yield by 0.4 percentage points.

THE GATES, AND WHAT EACH ONE IS ACTUALLY FOR
============================================
Five, in this order (`content_add_reasons` returns ALL that fire, not the first, because the
manifest reports which gate is SOLELY responsible for each rejection):

  ``clitic``            -- ``can't``, ``what's``, ``let's``, ``you're``, ``timmy's``. A rule,
                           not a list, so it generalises to possessives the corpus has not
                           shown us yet.
  ``function_word``     -- wh-words, reflexives, possessive pronouns, numerals, deictic time
                           adverbs. `train.improv.STOPWORDS` is a deliberately small 40-word
                           list and lets ``what``/``where``/``why``/``our``/``mine`` through;
                           this is that list's closed-class continuation, NOT a stoplist of
                           frequent words. Nothing in it is a noun or a verb.
  ``interjection``      -- greetings, backchannels and politeness formulas. Also a closed
                           class. It contains ``hello`` and ``okay``; it deliberately does
                           NOT contain ``look``, ``love``, ``come``, ``want``, ``give``,
                           ``doing`` or ``mean``, which are the hard cases and are verbs.
  ``low_narration_rate``-- the generalising gate. Over the full artifact it is SOLELY
                           responsible for rejecting ``ow``, ``mmm``, ``yuck``, ``yum``,
                           ``woof``, ``whee``, ``ew``, ``amen``, ``sir``, ``dear``,
                           ``sweetie`` -- the long tail of sound effects and vocatives that
                           no list of ours would have contained.
  ``not_noun_or_verb``  -- the per-instance POS check. Solely responsible for ``pretty``,
                           ``cool``, ``nice``, ``silly``, ``smart``, ``wrong``, ``ready``:
                           adjectives, which name a property and not a thing or an action.

VALIDATED, NOT ASSERTED
=======================
250 real `add` observations were hand-labelled in their own turns BEFORE this classifier was
run on them (`docs/measurements/reach-content-add-labels.jsonl`), and
`scripts/validate_content_add.py` reports precision, recall and the full confusion against
them, plus the same numbers against the 25-word ground truth quoted above. Both land in the
derivation manifest. The known residual errors are named there rather than smoothed away.
"""
from __future__ import annotations

import importlib.util
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from train.improv import content_words

#: A word must be used outside quotation marks at least this often to count as naming
#: something. See the module docstring for how the number was chosen and what it costs.
NARRATION_FLOOR = 0.20

#: The tags that mean "a thing" and "an action". Everything else -- JJ, JJR, RB, CD, IN, UH,
#: PRP, DT, MD, EX -- is rejected: a property, a manner, a number or a function word is not
#: what this slot is for.
NOUN_TAGS: Tuple[str, ...] = ("NN", "NNS", "NNP", "NNPS")
VERB_TAGS: Tuple[str, ...] = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")

#: Clitic endings, checked as a RULE rather than enumerated as forms. `train.improv._WORD`
#: keeps the apostrophe, so contractions and possessives arrive here intact.
CLITIC_SUFFIXES: Tuple[str, ...] = ("'s", "'t", "'re", "'ll", "'ve", "'d", "'m")

#: The closed-class continuation of `train.improv.STOPWORDS`. Every entry is a wh-word, a
#: pronoun/reflexive/possessive, a quantifier or degree word, a numeral, or a deictic
#: time/place adverb. NOTHING here is a noun or a verb, which is the property that makes it a
#: closed-class list rather than a frequency stoplist.
FUNCTION_WORDS = frozenset("""
what where when why how which who whom whose
mine yours ours theirs hers myself yourself yourselves himself herself itself
ourselves themselves
our
all any some every each both few many much more most less least enough only also too
else such same other another own here there now then again ever never always sometimes
maybe perhaps almost already anymore still yet even quite really rather
one two three four five six seven eight nine ten eleven twelve hundred thousand million
today tonight tomorrow yesterday
about after before because down up out off under while until since though although
""".split())

#: Greetings, backchannels, politeness formulas and sound effects: a closed class, and the
#: one list in this module that a reader should check for smuggling. It does NOT contain
#: `look`, `love`, `come`, `want`, `give`, `doing` or `mean` -- the seven words in the
#: measured top 25 that are genuine verbs -- and `tests/test_content_add.py` asserts that,
#: because a filter that hit its numbers by quietly stoplisting those would be worthless.
INTERJECTIONS = frozenset("""
hi hello hey heya bye goodbye goodnight goodmorning
yes yeah yep yup nope nah ok okay okey alright
oh ah aha ooh oops ouch wow whoa yay hooray hurray hurrah
ha haha hehe hmm hm huh shh sh psst ugh eek
please thanks thank welcome sorry sure fine course
""".split())

#: The rejection reasons, in the order `content_add_reasons` returns them. Named so the
#: manifest's per-gate attribution and the tests read from one list.
GATE_NAMES: Tuple[str, ...] = ("clitic", "function_word", "interjection",
                               "low_narration_rate", "not_noun_or_verb")

_QUOTE = re.compile('"')


# --------------------------------------------------------------------------------------
# the corpus evidence: where does this word live?
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SpeechProfile:
    """Content-word token counts split by NARRATION vs QUOTED SPEECH, over a corpus.

    `narration[w]` -- times `w` occurs outside quotation marks.
    `speech[w]`    -- times `w` occurs inside them.
    `stories`      -- documents scanned, carried so a manifest names a population.

    TOKEN counts, not document frequency, and that is deliberate. The task's design
    constraint forbids filtering on document frequency, because NPMI is already
    frequency-biased (`spearman(add_df, distance) = +0.2078` on the old artifact) and a DF cut
    would be partly the same axis as the thing the dial measures. What this profile carries is
    a RATIO of one word's own counts against itself; `scripts/validate_content_add.py`
    measures its rank correlation with `add_df` and the derivation manifest publishes it.
    """
    narration: Dict[str, int]
    speech: Dict[str, int]
    stories: int


def split_narration_and_speech(story: str) -> Tuple[str, str]:
    """`(text outside quotation marks, text inside them)` for one story.

    Quotes are paired as a TOGGLE -- 1st with 2nd, 3rd with 4th -- which is exactly what
    `train.dialogue.quoted_utterances` does, so the two agree about where an utterance starts
    and stops. An odd final quote opens a span that never closes; its text is left in
    NARRATION rather than guessed at, matching `quoted_utterances`, which discards it.

    The attribution tag between two utterances (``," she said, "``) is outside both quote
    pairs and therefore lands in narration, which is correct: the narrator wrote it.
    """
    positions = [m.start() for m in _QUOTE.finditer(story)]
    narration: List[str] = []
    speech: List[str] = []
    prev = 0
    for k in range(0, len(positions) - 1, 2):
        s, e = positions[k], positions[k + 1]
        narration.append(story[prev:s])
        speech.append(story[s + 1:e])
        prev = e + 1
    narration.append(story[prev:])
    return " ".join(narration), " ".join(speech)


def build_speech_profile(stories: Iterable[str]) -> SpeechProfile:
    """Count every content word on both sides of the quotation marks.

    `content_words` is the same tokenizer/stoplist the rest of the pipeline uses, so a word
    that can never be an `add` value never appears in the profile either.

    Independent of `PYTHONHASHSEED`: `Counter.update` accumulates counts, and nothing here
    depends on iteration order.
    """
    narration: Counter = Counter()
    speech: Counter = Counter()
    n = 0
    for story in stories:
        n_txt, s_txt = split_narration_and_speech(story)
        narration.update(content_words(n_txt))
        speech.update(content_words(s_txt))
        n += 1
    return SpeechProfile(narration=dict(narration), speech=dict(speech), stories=n)


def narration_rate(word: str, profile: SpeechProfile) -> float:
    """Fraction of `word`'s occurrences that the NARRATOR wrote, or 0.0 if never seen.

    0.0 for an unseen word is a REJECTION by `content_add_reasons`, and that is the intended
    reading: a word this corpus has never used outside a quotation mark has shown no evidence
    of naming anything. It is not the same claim as "far", which is why the reach metric has
    its own separate no-evidence guard (`train.reach.pair_has_evidence`).
    """
    n = profile.narration.get(word, 0)
    s = profile.speech.get(word, 0)
    total = n + s
    return (n / total) if total else 0.0


def profile_meta(profile: SpeechProfile) -> dict:
    """How the profile was built, plus a fingerprint, so a reader can rebuild and compare.

    Same reasoning as `scripts.derive_dialogue_skits.association_meta`: the table itself is
    not written to disk, so the manifest carries what it takes to reproduce it.
    """
    import hashlib
    top = sorted(profile.narration.items(), key=lambda kv: (-kv[1], kv[0]))[:200]
    h = hashlib.md5()
    h.update(f"{profile.stories}|{len(profile.narration)}|{len(profile.speech)}|".encode())
    h.update("|".join(f"{w}:{c}" for w, c in top).encode())
    return {
        "builder": "train.content_add.build_speech_profile",
        "unit": "TOKEN counts of train.improv.content_words, split by the quote toggle",
        "stories": profile.stories,
        "narration_vocabulary": len(profile.narration),
        "speech_vocabulary": len(profile.speech),
        "narration_tokens": sum(profile.narration.values()),
        "speech_tokens": sum(profile.speech.values()),
        "fingerprint_md5": h.hexdigest(),
        "fingerprint_inputs": "stories|narration vocab|speech vocab|top-200 narration words",
        "not_document_frequency":
            "Token counts, and only ever read as the RATIO narration/(narration+speech) for "
            "one word against itself. A document-frequency cut is forbidden here because "
            "NPMI is already frequency-biased and a DF filter would be partly the same axis "
            "as the dial. The manifest publishes this ratio's rank correlation with add_df.",
    }


# --------------------------------------------------------------------------------------
# the five gates -- each named, each fixture-tested
# --------------------------------------------------------------------------------------
def is_clitic_form(word: str) -> bool:
    """``can't``, ``what's``, ``let's``, ``you're``, ``timmy's`` -- a contraction or possessive.

    A rule and not a list, so it also catches the possessives no list of ours would contain.
    The cost is real and accepted: ``dog's`` is rejected even though it names a dog, because
    the bare ``dog`` is the form this slot should be naming and is ranked separately.
    """
    return "'" in word and any(word.endswith(c) for c in CLITIC_SUFFIXES)


def is_function_word(word: str) -> bool:
    """A closed-class word `train.improv.STOPWORDS` does not happen to list. See FUNCTION_WORDS."""
    return word in FUNCTION_WORDS


def is_interjection(word: str) -> bool:
    """A greeting, backchannel, politeness formula or sound effect. See INTERJECTIONS."""
    return word in INTERJECTIONS


def _require_nltk():
    """Import nltk, or say plainly what is missing and how to get it.

    `importlib.util.find_spec` rather than a bare try/import at module scope: this module is
    imported by `train.skit`'s callers on a machine where the content gate may not be in use,
    and an optional dependency must not turn into an ImportError at import time.
    """
    if importlib.util.find_spec("nltk") is None:
        raise RuntimeError(
            "the content-word gate needs nltk (pip install 'tt-tnt[content]') and its "
            "averaged_perceptron_tagger_eng + punkt_tab data "
            "(python -m nltk.downloader averaged_perceptron_tagger_eng punkt_tab).")
    import nltk
    return nltk


def pos_tags_in_turn(turn: str) -> Dict[str, Tuple[str, ...]]:
    """`{lowercased token: every tag it got}` for one turn, tagged CASE-BLIND.

    EVERY TOKEN IS LOWERCASED BEFORE TAGGING. That is the whole point of this function and
    the reason it exists as a named thing rather than as two lines inside the classifier:
    tagged as written, a quoted utterance's first word is capitalised, the tagger reads
    capitalisation as a proper-noun cue, and `hi`/`hello`/`wow`/`hey` come back NNP at
    115-120 out of 120 -- along with `look` at 93 and `come` at 84. A filter reading that
    signal would separate particles from things by an accident of typography and would invert
    the moment a particle appeared mid-utterance.

    Lowercasing removes the cue for every word in the utterance, in both directions, so the
    tag this module reads is the same one it would read for the same word anywhere else.
    `test_the_verdict_does_not_move_when_the_capitalisation_does` pins it.

    Cost of the choice, stated: a genuine proper name in `add` position ("Faye") loses its
    NNP and is usually rejected. Accepted -- a character name is a poor `add` value anyway,
    and the guarantee is worth more than the handful of names.

    A word can appear more than once in a turn with different tags; all of them are kept and
    `has_content_pos` accepts if ANY is a noun or a verb.
    """
    nltk = _require_nltk()
    tokens = [t.lower() for t in nltk.word_tokenize(turn)]
    out: Dict[str, List[str]] = {}
    for token, tag in nltk.pos_tag(tokens):
        out.setdefault(token, []).append(tag)
    return {k: tuple(v) for k, v in out.items()}


def has_content_pos(word: str, tags: Dict[str, Tuple[str, ...]]) -> bool:
    """Was `word` used as a noun or a verb in this turn?

    False when the word is absent from `tags` at all, which happens when nltk's tokenizer
    splits it differently from `train.improv._WORD` (``don't`` -> ``do`` + ``n't``). Absent
    means unverified, and unverified is rejected: this gate is the one holding adjectives
    out, and a word it cannot see is a word it cannot vouch for.
    """
    got = tags.get(word, ())
    return any(t in NOUN_TAGS or t in VERB_TAGS for t in got)


# --------------------------------------------------------------------------------------
# the classifier
# --------------------------------------------------------------------------------------
def content_add_reasons(word: str, turn: str, profile: SpeechProfile, *,
                        floor: float = NARRATION_FLOOR,
                        tags: Optional[Dict[str, Tuple[str, ...]]] = None
                        ) -> Tuple[str, ...]:
    """EVERY gate `word` fails in `turn`, in `GATE_NAMES` order. Empty means content.

    All of them and not the first, deliberately, and the opposite of
    `scripts.derive_dialogue_skits.screen_candidate`, which reports the first gate because its
    gates are ordered by cost and each count is conditional on the one before. Here the
    manifest needs to say which gate is SOLELY responsible for a rejection -- that is how
    `low_narration_rate` was shown to be earning its place (it alone rejects ``ow``, ``mmm``,
    ``yuck``, ``sir``, ``dear``) rather than duplicating the two lists.

    `tags` is an escape hatch for the caller that has already tagged this turn:
    `content_add_filter` tags each turn once and reuses it for every candidate word, which is
    the difference between one tagging per turn and one per fresh word.
    """
    out: List[str] = []
    if is_clitic_form(word):
        out.append("clitic")
    if is_function_word(word):
        out.append("function_word")
    if is_interjection(word):
        out.append("interjection")
    if narration_rate(word, profile) < floor:
        out.append("low_narration_rate")
    if tags is None:
        tags = pos_tags_in_turn(turn)
    if not has_content_pos(word, tags):
        out.append("not_noun_or_verb")
    return tuple(out)


def is_content_add(word: str, turn: str, profile: SpeechProfile, *,
                   floor: float = NARRATION_FLOOR,
                   tags: Optional[Dict[str, Tuple[str, ...]]] = None) -> bool:
    """Does `word`, as used in `turn`, name a thing or an action?"""
    return not content_add_reasons(word, turn, profile, floor=floor, tags=tags)


def content_add_filter(profile: SpeechProfile, *,
                       floor: float = NARRATION_FLOOR) -> Callable[[str, str], bool]:
    """An `add_filter` for `train.skit.derive_skit_from_turns`: `(word, turn) -> bool`.

    Carries a ONE-ENTRY tag cache. `train.skit.choose_add_word` walks a turn's fresh words in
    rank order calling this for each, so without the cache a turn with nine fresh words is
    tagged nine times; with it, once. One entry and not an LRU because the access pattern is
    strictly one turn at a time -- a bigger cache would hold the whole corpus's turns for no
    gain.

    The cache cannot change a verdict: it is keyed on the exact turn string and
    `pos_tags_in_turn` is a pure function of it.
    """
    cache: Dict[str, Dict[str, Tuple[str, ...]]] = {}

    def accept(word: str, turn: str) -> bool:
        if turn not in cache:
            cache.clear()
            cache[turn] = pos_tags_in_turn(turn)
        return is_content_add(word, turn, profile, floor=floor, tags=cache[turn])

    return accept


def gate_attribution(observations: Sequence[Tuple[str, str]], profile: SpeechProfile, *,
                     floor: float = NARRATION_FLOOR) -> dict:
    """Per-gate counts over `(word, turn)` observations: fired-at-all, and SOLELY responsible.

    The `sole` column is the one that says whether a gate is pulling its weight. Three of
    these gates overlap heavily -- `let's` is a clitic AND below the narration floor -- so a
    gate can look busy in `any` while removing nothing that the others would not have removed.
    """
    any_: Counter = Counter()
    sole: Dict[str, Counter] = {g: Counter() for g in GATE_NAMES}
    kept = 0
    for word, turn in observations:
        why = content_add_reasons(word, turn, profile, floor=floor)
        if not why:
            kept += 1
            continue
        for g in why:
            any_[g] += 1
        if len(why) == 1:
            sole[why[0]][word] += 1
    return {
        "observations": len(observations),
        "kept": kept,
        "kept_fraction": round(kept / len(observations), 4) if observations else None,
        "per_gate": {g: {"fired": any_.get(g, 0),
                         "solely_responsible": sum(sole[g].values()),
                         "top_words_it_alone_rejected":
                             [w for w, _ in sole[g].most_common(12)]}
                     for g in GATE_NAMES},
        "reading": "`fired` counts every rejection a gate took part in; "
                   "`solely_responsible` counts the ones no other gate would have caught. A "
                   "gate with a large `fired` and a near-zero `solely_responsible` is "
                   "redundant, not effective.",
    }


__all__ = ["NARRATION_FLOOR", "NOUN_TAGS", "VERB_TAGS", "CLITIC_SUFFIXES",
           "FUNCTION_WORDS", "INTERJECTIONS", "GATE_NAMES", "SpeechProfile",
           "split_narration_and_speech", "build_speech_profile", "narration_rate",
           "profile_meta", "is_clitic_form", "is_function_word", "is_interjection",
           "pos_tags_in_turn", "has_content_pos", "content_add_reasons", "is_content_add",
           "content_add_filter", "gate_attribution"]
