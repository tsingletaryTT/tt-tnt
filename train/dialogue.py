# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Dialogue-aware sentence splitting, and turn extraction by alternation.

TWO PROBLEMS, ONE MODULE
========================

**1. The splitter.** Both variants this project has tried are wrong, in opposite
directions, and each is right where the other is wrong:

    pattern                 '"It catches the light!" said her friend.'   'He said "no." She left.'
    (?<=[.!?"])\\s+  (old)   2  -- WRONG (splits the tag off the quote)   2  -- right
    (?<=[.!?])\\s+   (tried) 1  -- right                                  1  -- WRONG (runs on)

Neither can be fixed by tuning the character class, because the two strings differ only in
what FOLLOWS the closing quote: an attribution clause continues the sentence, a new
capitalised clause starts a new one. A lookbehind cannot see that. `split_sentences_dialogue`
below scans instead, tracking quote state, and asks a named question -- `continues_as_attribution`
-- at exactly the one ambiguous position.

Why it matters beyond tidiness: the old splitter fragmented dialogue-with-attribution, those
fragments failed the skit accept/add gates, and the stage-2 eval population ended up with
**43% less dialogue than the corpus it was drawn from** (54.6% -> 31.0% of units). We trained
on the monologic residue. See `docs/superpowers/specs/2026-08-23-reach-dial-design.md`,
"The splitter, fixed properly".

**2. Turn extraction.** Stage 2 built "partner turns" by slicing consecutive sentences, so the
"partner" was the same narrator continuing ("*The cherry tree was envious of the big trees.*").
`handback_anticipation` topped out at 0.119 on GROUND TRUTH because there was no second voice
to anticipate. `extract_dialogue_turns` takes real quoted utterances and alternates them by
POSITION.

NO SPEAKER ATTRIBUTION, DELIBERATELY
------------------------------------
A probe that attributed speakers by name got them BACKWARDS: vocatives inside the quote fool
any first-capitalised-name heuristic -- *"Fluffy, the word is 'repeat'"* is Timmy addressing
Fluffy, not Fluffy speaking. A skit needs the VOICE TO CHANGE, not to know whose it is, so
attribution is unnecessary here and dropping it drops that bug with it. Nothing in this module
reads a name, and `test_extraction_does_not_attribute_by_name` pins that with a fixture built
so a name heuristic would invert the roles.

KNOWN, ACCEPTED IMPRECISIONS (measured, not guessed -- see the task report)
--------------------------------------------------------------------------
* `continues_as_attribution` looks at most `_LOOKAHEAD_WORDS` words past the closing quote, so
  `'"Stop!" Amy was told to go.'` merges (``told`` is in `SPEECH_VERBS`). Rare, and it errs
  toward keeping a quote with its tail rather than fragmenting it -- the direction that costs
  us nothing.
* Double quotes only, and they are treated as a TOGGLE. Nested doubles are not modelled; a
  single unbalanced `"` would otherwise swallow the rest of the story, so the scan gives up on
  quote state once no closing quote remains (`_last_quote`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

#: Sentence-final punctuation. A run of these (``!?``, ``...``) counts as one terminator.
_TERMINALS = frozenset(".!?")

#: Verbs that mark a clause as a dialogue attribution rather than a new sentence. Includes a
#: few that are not literally verbs of speech (``smiled``, ``nodded``) because the corpus uses
#: them as tags, and includes both tenses because TinyStories mixes them. Kept short and
#: auditable on purpose -- same reasoning as train.improv.STOPWORDS.
SPEECH_VERBS = frozenset("""
said says say asked asks ask replied replies reply answered answers answer told tells tell
shouted shouts shout cried cries cry whispered whispers whisper yelled yells yell
called calls call added adds exclaimed exclaims murmured mumbled muttered
laughed giggled smiled sighed nodded agreed agrees thought thinks
continued repeated screamed groaned wondered wonders explained explains begged
warned warns warn
""".split())

#: How many words past a closing quote `continues_as_attribution` may inspect. Three covers
#: `said her friend`, `Amy said` and `the little girl said`; more starts reading into the next
#: sentence and merging things that are not attributions.
_LOOKAHEAD_WORDS = 3

#: Max words in an interposed clause for `is_interposed_attribution` to accept it. Six covers
#: `she said,` / `the little girl whispered softly,` and excludes a whole narrative sentence.
_MAX_INTERPOSED_WORDS = 6

#: A skit is five turns; a story needs at least this many utterances to supply them.
MIN_UTTERANCES = 5

#: Roles are assigned by index PARITY -- even index is the model, odd is the partner. This is
#: the same shape as train.skit.SKIT_ROLES and is asserted equal to it in the tests, but it is
#: spelled out here so this module does not import from skit (skit imports the splitter).
DIALOGUE_TURNS = 5

_WORD_RE = re.compile(r"[A-Za-z']+")


# --------------------------------------------------------------------------------------
# The splitter
# --------------------------------------------------------------------------------------
def continues_as_attribution(rest: str) -> bool:
    """Does `rest` continue the sentence a closing quote just ended, as an attribution?

    `rest` is everything after a closing `"` that was itself preceded by sentence-final
    punctuation -- i.e. the ONE position where the two historical splitters disagree.

    True  ->  ``said her friend.``   ``Amy said.``   ``the little girl whispered.``
    False ->  ``She left.``          ``"Goodbye!"``  ``The dog ran off.``

    Decision rules, in order:
      1. Nothing left, or the next character is not a letter (another quote, a dash, a
         digit) -> a new unit. An attribution clause always starts with a word.
      2. Next character is LOWER CASE -> attribution. Nothing else starts a clause in
         lower case in this corpus.
      3. Otherwise (capitalised) it is an attribution only if a speech verb appears within
         the next `_LOOKAHEAD_WORDS` words of THIS clause -- ``Amy said`` yes,
         ``The dog ran`` no. The clause is clipped at the next terminator first so the
         lookahead cannot borrow a verb from the sentence after next.
    """
    s = rest.lstrip()
    if not s or not s[0].isalpha():
        return False
    if s[0].islower():
        return True
    clause = _clip_to_clause(s)
    return any(w.lower() in SPEECH_VERBS for w in _WORD_RE.findall(clause)[:_LOOKAHEAD_WORDS])


def _clip_to_clause(s: str) -> str:
    """`s` up to (and including) its first sentence terminator, so a lookahead stays local."""
    for i, ch in enumerate(s):
        if ch in _TERMINALS:
            return s[: i + 1]
    return s


def split_sentences_dialogue(text: str) -> List[str]:
    """Split `text` into sentences, keeping every quoted span with its attribution clause.

    Drop-in replacement for the regex `split_sentences` used to be: same signature, same
    stripping, same "no empty units" guarantee.

    The scan carries one bit of state -- whether we are inside a double-quoted span -- and
    recognises exactly two kinds of boundary:

      * a terminator run at quote depth 0, followed by whitespace. (``He said "no." She
        left.`` breaks here after the closing quote, not at the ``.`` inside it.)
      * a closing quote whose preceding character is a terminator, followed by whitespace --
        but ONLY if `continues_as_attribution` says no attribution follows.

    A terminator inside an open quote is never a boundary, which is what keeps a
    multi-sentence utterance (``"Hi, I am Fin. Do you want to play?" asked the fish.``)
    whole, and what makes the interrupted form (``"Don't cry," she said, "I can help."``)
    one sentence: neither `,"` nor ` she said, ` presents a boundary at all.
    """
    text = text.strip()
    if not text:
        return []

    # Once no closing quote remains, an open quote can only be an unbalanced one; from that
    # point on we stop honouring quote state so a stray `"` cannot swallow the rest.
    last_quote = text.rfind('"')

    out: List[str] = []
    start = 0
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if ch == '"':
            if depth and i > 0 and text[i - 1] in _TERMINALS:
                # A closing quote that ended a sentence: the ambiguous position.
                j = i + 1
                if j >= n:
                    depth = 0
                    i = j
                    continue
                if text[j].isspace() and not continues_as_attribution(text[j:]):
                    out.append(text[start:j].strip())
                    start = j
                    depth = 0
                    i = j
                    continue
            depth = 1 - depth
            i += 1
            continue

        if ch in _TERMINALS and (depth == 0 or i > last_quote):
            j = i + 1
            while j < n and text[j] in _TERMINALS:
                j += 1
            if j < n and text[j].isspace():
                out.append(text[start:j].strip())
                start = j
                depth = 0
            i = j
            continue

        i += 1

    if text[start:].strip():
        out.append(text[start:].strip())
    return [s for s in out if s]


# --------------------------------------------------------------------------------------
# Turn extraction
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Utterance:
    """One spoken turn: the words inside the quotes, and where they sat in the story.

    `start`/`end` are character offsets into the story the utterance came from -- `start` is
    the opening quote, `end` is one past the final closing quote. They exist so a caller can
    take the story text BEFORE the first utterance as a skit prefix without re-scanning.
    """
    text: str
    start: int
    end: int


def is_interposed_attribution(left_content: str, gap: str) -> bool:
    """Is `gap` an attribution that INTERRUPTS one utterance, rather than separating two?

    `left_content` is what was inside the quotes on the left; `gap` is the unquoted text
    between that closing quote and the next opening quote.

    True  ->  ``"Don't cry,"`` + ``  she said,  `` + ``"I can help."``     ONE utterance
    False ->  ``"Hello."``     + ``  said Amy.  `` + ``"Goodbye."``        TWO utterances

    Both commas are load-bearing and are what make this tight. A left side ending in `.`/`!`/
    `?` is a finished utterance, and a gap ending in `.` is a finished narrative clause;
    either one alone rules out interruption. The word cap keeps a whole narrative sentence
    that happens to end in a comma from qualifying.

    Getting this wrong is the rough edge the probe hit: it truncated interrupted quotes into
    two fragments, so a five-utterance story became a ten-fragment one and the roles shifted
    under the parity assignment.
    """
    left = left_content.rstrip()
    g = gap.strip()
    if not left or not g:
        return False
    if left[-1] in _TERMINALS:
        return False
    if not g.endswith(","):
        return False
    return len(_WORD_RE.findall(g)) <= _MAX_INTERPOSED_WORDS


def is_tag_only_gap(gap: str) -> bool:
    """Is the text between two ADJACENT utterances nothing but an attribution tag?

    A DIAGNOSTIC, not a gate. Nothing in this module or in
    `scripts/derive_dialogue_skits.py` drops or merges anything because of it.

    It exists to measure the one real cost of refusing to attribute speakers. The corpus
    writes a single speaker's two-sentence turn as two quoted spans::

        "Don't cry, Binky," he said. "I can help you get your tail out."

    which is one voice, and it writes a genuine exchange the same way::

        "Hello," said Amy. "Goodbye," said Ben.

    which is two. The difference is entirely in the tag's SUBJECT, so telling them apart
    needs exactly the name attribution that got the probe's speakers backwards. We do not
    do it, so the alternation assumption is violated by some unknown fraction of pairs, and
    the honest response is to publish an UPPER BOUND on that fraction rather than a silent
    assumption. `is_tag_only_gap` is that upper bound's indicator: True for both examples
    above, so counting it over-counts the failure and cannot flatter the result.

    (`is_interposed_attribution` is the different, unambiguous case: there the LEFT span is
    unfinished, which no separate second utterance ever is.)
    """
    words = _WORD_RE.findall(gap)
    if not words or len(words) > _MAX_INTERPOSED_WORDS:
        return False
    return any(w.lower() in SPEECH_VERBS for w in words)


#: Words that are capitalised only because they open a clause, never because they name
#: anyone. Every entry is also in `train.improv.STOPWORDS` -- and
#: `test_non_name_capitals_is_a_stopword_subset` pins that -- but it is spelled out here
#: rather than imported because `train.improv` imports THIS module, so importing back
#: would be a cycle. Kept short and auditable, same reasoning as SPEECH_VERBS.
_NON_NAME_CAPITALS = frozenset("""
the a an he she they it we you i and but then so there this that his her their my your
""".split())


def tag_identifies_a_subject(gap: str) -> bool:
    """Does `gap` contain a word that could be the tag's SUBJECT -- anybody at all?

    BLIND TO WHO, DELIBERATELY. It answers "is there a subject here", never "whose voice is
    this". Swap every name in the gap for a different name and the answer is unchanged
    (`test_subject_detection_is_blind_to_which_name`), which is the property that keeps this
    out of the speaker attribution that got an earlier probe's speakers BACKWARDS.

    True  ->  ``said Amy.``  ``Amy said.``  ``The fox replied,``  ``Her mom said,``
    False ->  ``he said.``   ``she asked.``  ``they said,``

    A "subject" is any word that is neither a stopword-shaped function word nor a verb of
    speech. That admits some non-subjects (``they asked together.`` qualifies on
    ``together``), which errs toward KEEPING a pair -- the direction that costs data rather
    than corrupting it.
    """
    for w in _WORD_RE.findall(gap):
        lw = w.lower()
        if lw in _NON_NAME_CAPITALS or lw in SPEECH_VERBS:
            continue
        return True
    return False


def tag_names_a_proper_noun(gap: str) -> bool:
    """Stricter variant: does `gap` carry a CAPITALISED token that is not a clause-opener?

    NOT the filter this module gates on -- `same_voice_risk` uses
    `tag_identifies_a_subject`. This exists because the literal wording of the filter
    proposed in task 1 was "no NEW CAPITALISED token", and the two readings cost very
    different amounts of data, so both are measured and the choice is recorded in the
    manifest rather than asserted.

    Measured on 200,000 stories: gating on this one drops 28.9% of skits against 3.6% for
    `tag_identifies_a_subject` -- and its drops include scenes that alternate perfectly, e.g.
    the fox/rabbit exchange whose four gaps are ``The rabbit looked up and said,`` /
    ``The fox replied,`` / ``The rabbit said,`` / ``The fox smiled and said,``. Those name a
    speaker with a common noun, so this variant is closer to a GENRE filter (it discards
    animal dialogue) than to a voice-safety one. Kept callable, and reported, not used.
    """
    for w in _WORD_RE.findall(gap):
        lw = w.lower()
        if w[0].isupper() and lw not in _NON_NAME_CAPITALS and lw not in SPEECH_VERBS:
            return True
    return False


def same_voice_risk(gap: str, *, strict_names: bool = False) -> bool:
    """Does the text between two ADJACENT utterances fail to evidence a change of voice?

    THE CONSERVATIVE FILTER. `is_tag_only_gap` is the diagnostic that measures the risk;
    this is the gate that acts on it, and `scripts/derive_dialogue_skits.py` drops the whole
    skit when any of its four adjacent gaps trips it. The premise of the dialogue path is
    that the partner turn is a DIFFERENT VOICE, so a pair with no evidence of a change is
    dropped rather than trained on.

    True (no evidence of a change) when the gap is nothing but an attribution tag
    (`is_tag_only_gap`) whose subject cannot distinguish anybody: ``he said.``,
    ``she asked.``, ``they said,``.

    False (keep) when the gap carries narrative, or names a subject.

    AN EMPTY GAP IS NOT A RISK, and that is a measured decision rather than an oversight.
    Two quotes flush against each other look like one speaker continuing, but in this corpus
    the attribution frequently sits AFTER the second utterance instead of between the two:
    ``"Can we make a boat with it?" "Sure, Ben, that's a great log ..." dad said, smiling.``
    is a genuine change of voice with nothing at all in the gap (story 760 of the kept
    population). `is_tag_only_gap` already returns False for a wordless gap, so this function
    inherits the right answer; an earlier draft special-cased it as risky and dropped
    `STORY_B` -- the interrupted-quote fixture -- for it.

    `strict_names=True` swaps the subject test for `tag_names_a_proper_noun`; see there for
    why that is measured but not the default.

    WHAT THIS CANNOT DO, and it is the honest limit of an attribution-free filter: a gap that
    carries narrative and then RE-INTRODUCES THE SAME SPEAKER is invisible to it. Real
    example from the kept population (story 685): ``"I already did, Mommy! Can we go now?"``
    then ``They went to the airport ... Lily got bored and said,`` then ``"Mommy, can we bake
    cookies while we wait?"`` -- both Lily, so parity puts `model` on the daughter at turn 2
    having put it on the mother at turn 0. Separating that from a genuine change needs the
    tag's subject compared against the PREVIOUS tag's subject, which is speaker attribution.
    Published as a limitation; see the manifest's `same_speaker_filter` block.
    """
    if not is_tag_only_gap(gap):
        return False
    names = tag_names_a_proper_noun if strict_names else tag_identifies_a_subject
    return not names(gap)


def adjacent_gaps(story: str, n_turns: int = MIN_UTTERANCES) -> List[str]:
    """The unquoted text between each adjacent pair of the first `n_turns` utterances.

    `n_turns - 1` strings, in order, or fewer if the story has fewer utterances. Shared by
    the diagnostic (`tag_only_gap_count`) and the gate (`voice_changes_throughout`) so the
    two cannot disagree about which spans they are looking at.
    """
    utts = quoted_utterances(story)[:n_turns]
    return [story[utts[k].end:utts[k + 1].start] for k in range(len(utts) - 1)]


def voice_changes_throughout(story: str, *, n_turns: int = MIN_UTTERANCES,
                             strict_names: bool = False) -> bool:
    """Does EVERY adjacent pair among the first `n_turns` utterances evidence a new voice?

    Whole-or-nothing, matching `derive_skit_from_turns`' drop rule: all four pairs of a
    five-turn skit carry supervision on one side or the other, so one bad pair is one bad
    skit. Returns True for a story with fewer than two utterances (no pair can fail); the
    caller has already gated on `MIN_UTTERANCES` long before this.
    """
    return not any(same_voice_risk(g, strict_names=strict_names)
                   for g in adjacent_gaps(story, n_turns))


def quoted_utterances(text: str) -> List[Utterance]:
    """Every spoken turn in `text`, in order, quotes stripped, interruptions rejoined.

    Quotes are paired as a TOGGLE (1st with 2nd, 3rd with 4th, ...); an odd final quote has
    no closer and is discarded rather than guessed at. Adjacent pairs are then merged while
    `is_interposed_attribution` holds, so `"Don't cry," she said, "I can help."` yields one
    utterance -- `Don't cry, I can help.` -- and not two fragments.

    Empty or whitespace-only utterances are dropped: `""` is not a turn.
    """
    positions = [m.start() for m in re.finditer('"', text)]
    raw: List[Tuple[int, int]] = [(positions[k], positions[k + 1] + 1)
                                  for k in range(0, len(positions) - 1, 2)]

    out: List[Utterance] = []
    k = 0
    while k < len(raw):
        s, e = raw[k]
        content = text[s + 1:e - 1]
        # Absorb every following pair that this one is interrupted by.
        while k + 1 < len(raw):
            ns, ne = raw[k + 1]
            gap = text[e:ns]
            if not is_interposed_attribution(content, gap):
                break
            content = f"{content.rstrip()} {text[ns + 1:ne - 1].lstrip()}"
            e = ne
            k += 1
        if content.strip():
            out.append(Utterance(text=content.strip(), start=s, end=e))
        k += 1
    return out


def extract_dialogue_turns(story: str) -> Optional[List[str]]:
    """The five alternating turns of a skit, or None if `story` cannot supply them.

    Roles come from POSITION and nothing else: index 0/2/4 are the model's turns, 1/3 are the
    partner's (`train.skit.SKIT_ROLES`, `MODEL_TURNS`, `PARTNER_TURNS`). No name is read; see
    this module's docstring for why that is a feature.

    Returns exactly `DIALOGUE_TURNS` strings -- the FIRST five utterances, so the pairing with
    the roles tuple is positional and the prefix (`dialogue_prefix`) is the text before the
    scene starts. Returns None when the story has fewer than `MIN_UTTERANCES`, which on
    TinyStories is most of them; the caller must report that drop rate.
    """
    utts = quoted_utterances(story)
    if len(utts) < MIN_UTTERANCES:
        return None
    return [u.text for u in utts[:DIALOGUE_TURNS]]


def dialogue_prefix(story: str) -> str:
    """The story text before its first spoken turn -- a skit's context-only prefix.

    Empty when the story opens on dialogue (there is no context yet) or has no dialogue at
    all. An empty prefix cannot carry an `offer` for block 0, so the caller drops it; that
    drop is reported under its own rule rather than folded into the derivation failures.
    """
    utts = quoted_utterances(story)
    if not utts:
        return ""
    return story[:utts[0].start].strip()
