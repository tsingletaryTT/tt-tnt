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
