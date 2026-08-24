# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for train/dialogue.py: the splitter, and turn extraction by alternation.

FIXTURE DISCIPLINE. Two fixtures in this project were vacuous because one word repeated
through every sentence, and a third could not tell two candidate spans apart because both
shared a word. Every turn in `STORY` below therefore carries its own distinctive noun
(`repeat`, `puzzle`, `fountain`, `turnip`, `treehouse`), so a mis-assignment shows up as a
different STRING and not merely as a different index into look-alike text.

The fixture is also built so the assertions can fail in both directions:

  * utterance 0 (`"Fluffy, the word is 'repeat'!"`) is spoken BY Timmy and names Fluffy, so
    a first-capitalised-name heuristic assigns the roles backwards -- which is exactly the
    bug the probe hit and the reason this module attributes nothing.
  * utterances 1-4 end in commas with attribution tails that end in FULL STOPS, so an
    over-eager interposed-attribution rule (one that forgets to require the gap's trailing
    comma) collapses four turns into one and the story stops qualifying at all.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.dialogue import (MIN_UTTERANCES, _NON_NAME_CAPITALS,  # noqa: E402
                            adjacent_gaps, continues_as_attribution,
                            dialogue_prefix, extract_dialogue_turns,
                            is_interposed_attribution, quoted_utterances,
                            same_voice_risk, split_sentences_dialogue,
                            tag_identifies_a_subject, tag_names_a_proper_noun,
                            voice_changes_throughout)
from train.improv import STOPWORDS, split_sentences  # noqa: E402
from train.skit import MODEL_TURNS, PARTNER_TURNS, SKIT_ROLES  # noqa: E402

# --------------------------------------------------------------------------------------
# A. the splitter
# --------------------------------------------------------------------------------------

#: `(text, expected units)`. The first two rows are the two cases in the spec's table that
#: no single lookbehind pattern can satisfy at once
#: (docs/superpowers/specs/2026-08-23-reach-dial-design.md, "The splitter, fixed properly").
SPLIT_TABLE = [
    # id                          text                                            expected
    ("quote_then_attribution",
     '"It catches the light!" said her friend.',
     ['"It catches the light!" said her friend.']),
    ("quote_inside_then_new_sentence",
     'He said "no." She left.',
     ['He said "no."', 'She left.']),
    ("nested_single_quotes",
     '"Fluffy, the word is \'repeat\'!" said Timmy.',
     ['"Fluffy, the word is \'repeat\'!" said Timmy.']),
    ("interrupted_quote",
     '"Don\'t cry," she said, "I can help."',
     ['"Don\'t cry," she said, "I can help."']),
    ("comma_before_attribution",
     '"I am here to help," said the ghost.',
     ['"I am here to help," said the ghost.']),
    ("multi_sentence_quote_stays_whole",
     '"Hi, I am Fin. Do you want to play?" asked the little fish.',
     ['"Hi, I am Fin. Do you want to play?" asked the little fish.']),
    ("attribution_first_then_new_sentence",
     'Amy said, "Hello." Then she left.',
     ['Amy said, "Hello."', 'Then she left.']),
    ("two_bare_quotes_are_two_units",
     '"Hello!" "Goodbye!"',
     ['"Hello!"', '"Goodbye!"']),
    ("plain_prose_three_terminators",
     'One day. Two days! Three?',
     ['One day.', 'Two days!', 'Three?']),
    ("capitalised_name_attribution",
     '"Stop!" Amy said. The dog ran away.',
     ['"Stop!" Amy said.', 'The dog ran away.']),
    ("capitalised_non_attribution_is_a_new_sentence",
     '"Stop!" The dog ran away.',
     ['"Stop!"', 'The dog ran away.']),
    ("unbalanced_quote_does_not_swallow_the_rest",
     'He said "no. She left.',
     ['He said "no.', 'She left.']),
    ("multi_terminator_run",
     '"Wait...!" cried Ben. He ran.',
     ['"Wait...!" cried Ben.', 'He ran.']),
]


@pytest.mark.parametrize("text,expected",
                         [(t, e) for _, t, e in SPLIT_TABLE],
                         ids=[i for i, _, _ in SPLIT_TABLE])
def test_split_sentences_dialogue_table(text, expected):
    """One splitter, both rows of the spec's table, and eleven more cases around them."""
    assert split_sentences_dialogue(text) == expected


def test_no_lookbehind_pattern_can_satisfy_both_rows_but_this_splitter_does():
    """The table in the spec, executed rather than quoted.

    This is the test that says WHY the module exists. It runs both historical patterns on
    both strings, pins each one's wrong answer, and then pins ours as right on both. Revert
    `split_sentences_dialogue` to either regex and this fails; keep both regexes honest and
    it stays a live claim rather than a comment.
    """
    old = re.compile(r'(?<=[.!?"])\s+')          # shipped through stage 1 and stage 2
    attempted = re.compile(r'(?<=[.!?])\s+')     # tried, reverted

    def n(pat, t):
        return len([s for s in pat.split(t.strip()) if s.strip()])

    a = '"It catches the light!" said her friend.'      # must be ONE sentence
    b = 'He said "no." She left.'                       # must be TWO

    assert (n(old, a), n(old, b)) == (2, 2), "the old pattern's known behaviour changed"
    assert (n(attempted, a), n(attempted, b)) == (1, 1), "the attempted pattern changed"
    assert (len(split_sentences_dialogue(a)), len(split_sentences_dialogue(b))) == (1, 2)


def test_improv_split_sentences_is_the_same_splitter_not_a_second_one():
    """The spec's "no second splitter in the tree". `train.improv.split_sentences` must BE
    the dialogue splitter, so every downstream consumer (skit, score_improv, derive_traces,
    eval_skits) moved with it and none of them kept the old behaviour."""
    for _, text, expected in SPLIT_TABLE:
        assert split_sentences(text) == expected


@pytest.mark.parametrize("rest,expected", [
    ("said her friend.", True),          # lowercase verb of speech
    ("she whispered to the cat.", True),  # lowercase non-verb still continues the clause
    ("Amy said.", True),                 # capitalised name + verb within 3 words
    ("the little girl said.", True),
    ("She left.", False),                # capitalised, no speech verb
    ("The dog ran away.", False),
    ('"Goodbye!"', False),               # a new quoted turn, not an attribution
    ("", False),
    ("The dog ran. Amy said hi.", False),  # the verb belongs to the NEXT sentence
])
def test_continues_as_attribution(rest, expected):
    """The one decision the two historical patterns could not make, tested on its own.

    The last row is the reason `_clip_to_clause` exists: without it the lookahead reaches
    past the sentence boundary, finds `said`, and merges two unrelated sentences.
    """
    assert continues_as_attribution(rest) is expected


def test_splitter_is_total_on_empty_and_whitespace():
    assert split_sentences_dialogue("") == []
    assert split_sentences_dialogue("   \n ") == []
    assert split_sentences_dialogue("no terminator here") == ["no terminator here"]


# --------------------------------------------------------------------------------------
# B. turn extraction
# --------------------------------------------------------------------------------------

#: Six utterances. Turn 0 is Timmy speaking a line that NAMES Fluffy first, so any
#: first-capitalised-name heuristic reads the scene backwards.
STORY = (
    'Timmy carried a lantern through the orchard. Fluffy hopped behind him. '
    '"Fluffy, the word is \'repeat\'!" said Timmy. '
    '"Repeat is a marvellous puzzle," answered the rabbit. '
    '"Then say it beside the fountain," said Timmy. '
    '"I would rather nibble the turnip," grumbled the rabbit. '
    '"Bring the turnip to the treehouse," said Timmy. '
    '"The treehouse is too tall," sighed the rabbit.'
)

#: Identical quoted content; every NAME outside the quotes swapped. `extract_dialogue_turns`
#: must not be able to tell these two stories apart.
STORY_NAMES_SWAPPED = (
    'Fluffy carried a lantern through the orchard. Timmy hopped behind him. '
    '"Fluffy, the word is \'repeat\'!" said the rabbit. '
    '"Repeat is a marvellous puzzle," answered Timmy. '
    '"Then say it beside the fountain," said the rabbit. '
    '"I would rather nibble the turnip," grumbled Timmy. '
    '"Bring the turnip to the treehouse," said the rabbit. '
    '"The treehouse is too tall," sighed Timmy.'
)

EXPECTED_TURNS = [
    "Fluffy, the word is 'repeat'!",
    "Repeat is a marvellous puzzle,",
    "Then say it beside the fountain,",
    "I would rather nibble the turnip,",
    "Bring the turnip to the treehouse,",
]


def test_extract_dialogue_turns_alternates_by_position():
    """Five turns, in order of appearance, quotes stripped.

    Pinned as STRINGS, not as a count: every turn carries its own noun, so a shifted or
    reversed assignment changes the values and cannot pass by accident.
    """
    assert extract_dialogue_turns(STORY) == EXPECTED_TURNS


def test_roles_come_from_index_parity_and_match_the_skit_contract():
    """`model, partner, model, partner, model` -- the same tuple train.skit uses, so the
    dialogue path and the sentence-slicing path supervise the same positions."""
    turns = extract_dialogue_turns(STORY)
    assert SKIT_ROLES == ("model", "partner", "model", "partner", "model")
    roles = tuple("model" if i % 2 == 0 else "partner" for i in range(len(turns)))
    assert roles == SKIT_ROLES
    assert tuple(i for i, r in enumerate(SKIT_ROLES) if r == "model") == MODEL_TURNS
    assert tuple(i for i, r in enumerate(SKIT_ROLES) if r == "partner") == PARTNER_TURNS


def test_extraction_does_not_attribute_by_name():
    """Swap every speaker name OUTSIDE the quotes: the turns must be byte-identical.

    This is the probe's bug written as a test. The probe attributed speakers by the first
    capitalised name it found and got them backwards, because turn 0 here is Timmy
    ADDRESSING Fluffy -- the vocative sits inside the quote. A skit needs the voice to
    change, not to know whose it is, so nothing in this module may read a name.
    """
    assert extract_dialogue_turns(STORY) == extract_dialogue_turns(STORY_NAMES_SWAPPED)
    # And the vocative really is misleading: turn 0 names Fluffy but is not Fluffy's turn.
    assert extract_dialogue_turns(STORY)[0].startswith("Fluffy,")
    # ...and the two stories really do disagree about who said it: turn 0's attribution tag
    # is "said Timmy" in one and "said the rabbit" in the other. If that ever stopped being
    # true the equality above would be trivially satisfied.
    assert STORY.split('"')[2] != STORY_NAMES_SWAPPED.split('"')[2]
    assert "Timmy" in STORY.split('"')[2]
    assert "Timmy" not in STORY_NAMES_SWAPPED.split('"')[2]


def test_extra_utterances_are_truncated_from_the_front_in_order():
    """Six utterances in the fixture, five turns in a skit: the FIRST five, in order.

    Taking the last five (or reversing) still returns five plausible strings, which is why
    this asserts values: `utts[-5:]` would start the skit at "Repeat is a marvellous
    puzzle," and flip every role.
    """
    utts = quoted_utterances(STORY)
    assert len(utts) == 6
    assert extract_dialogue_turns(STORY) == [u.text for u in utts[:5]]
    assert extract_dialogue_turns(STORY)[0] == utts[0].text


def test_fewer_than_five_utterances_is_none():
    four = ('Ben opened the crate. '
            '"Where is the compass?" asked Ben. '
            '"Under the anchor," said Nan. '
            '"The anchor is heavy," said Ben. '
            '"Then use the lever," said Nan.')
    assert len(quoted_utterances(four)) == 4
    assert extract_dialogue_turns(four) is None
    assert extract_dialogue_turns("A story with no dialogue at all. None here.") is None


def test_exactly_five_utterances_is_kept():
    """The gate is `>= MIN_UTTERANCES`. `>` drops every five-utterance story, which is the
    modal usable story in the corpus."""
    five = ('Ben opened the crate. '
            '"Where is the compass?" asked Ben. '
            '"Under the anchor," said Nan. '
            '"The anchor is heavy," said Ben. '
            '"Then use the lever," said Nan. '
            '"The lever snapped," said Ben.')
    assert MIN_UTTERANCES == 5
    assert len(quoted_utterances(five)) == 5
    turns = extract_dialogue_turns(five)
    assert turns is not None and len(turns) == 5
    assert turns[0] == "Where is the compass?"
    assert turns[4] == "The lever snapped,"


# --------------------------------------------------------------------------------------
# C. the interrupted quote -- the probe's known rough edge
# --------------------------------------------------------------------------------------
def test_interrupted_quote_is_one_utterance_not_two_fragments():
    """`"Don't cry, Binky," she said, "I can help."` is ONE turn.

    The probe truncated these, so a five-utterance story presented as ten fragments and
    every role shifted under parity assignment. The joined text keeps both halves: a rule
    that kept only the first fragment would lose "I can help" and this asserts on it.
    """
    story = ('Binky wept beside the pond. '
             '"Don\'t cry, Binky," she said, "I can help."')
    utts = quoted_utterances(story)
    assert len(utts) == 1
    assert utts[0].text == "Don't cry, Binky, I can help."


def test_interrupted_quotes_do_not_change_the_turn_count():
    """The same five turns, with turns 0 and 3 interrupted by an attribution.

    Under the probe's behaviour this story has SEVEN fragments, so the parity assignment
    lands `model` on what is really the partner's line. Asserting the strings catches that;
    asserting `len(...) == 5` alone would not, because seven fragments truncated to five is
    still five.
    """
    story = ('Nell stood by the kiln. Pip watched the smoke. '
             '"Fetch the bellows," said Nell, "before the ember dies." '
             '"The bellows are cracked," said Pip. '
             '"Then bring the ladle," said Nell. '
             '"The ladle is hot," warned Pip, "so wrap it in the apron." '
             '"Wrap it and hurry," said Nell.')
    assert extract_dialogue_turns(story) == [
        "Fetch the bellows, before the ember dies.",
        "The bellows are cracked,",
        "Then bring the ladle,",
        "The ladle is hot, so wrap it in the apron.",
        "Wrap it and hurry,",
    ]


def test_separate_utterances_are_not_merged():
    """The other direction, and the reason both commas are load-bearing.

    `"Yes, Amy," said Ben loudly. "No," said Amy.` is TWO turns: the left content ends in a
    comma, so only the GAP's trailing full stop distinguishes it from an interruption. Drop
    that guard and this fixture's five turns collapse into one.
    """
    story = 'Ben nodded. "Yes, Amy," said Ben loudly. "No," said Amy.'
    assert [u.text for u in quoted_utterances(story)] == ["Yes, Amy,", "No,"]
    # STORY's turns 1-4 all have this shape, so the same mistake would break it too.
    assert len(quoted_utterances(STORY)) == 6


@pytest.mark.parametrize("left,gap,expected", [
    ("Don't cry,", " she said, ", True),
    ("Don't cry,", " the little rabbit whispered, ", True),
    ("Hello.", " she said, ", False),          # left side is a finished utterance
    ("Hello!", " she said, ", False),
    ("Hello?", " she said, ", False),
    ("Yes, Amy,", " said Ben loudly. ", False),   # gap is a finished narrative clause
    ("Yes, Amy,", " Ben walked to the gate and looked at the sky and sighed, ", False),
    ("Yes, Amy,", "", False),
    ("", " she said, ", False),
])
def test_is_interposed_attribution(left, gap, expected):
    assert is_interposed_attribution(left, gap) is expected


def test_unpaired_trailing_quote_is_discarded_not_guessed():
    """An odd `"` has no closer; treat it as absent rather than running to end of story."""
    story = 'Ann spoke. "First," said Ann. "Second," said Bo. "Third'
    assert [u.text for u in quoted_utterances(story)] == ["First,", "Second,"]


# --------------------------------------------------------------------------------------
# D. the prefix
# --------------------------------------------------------------------------------------
def test_dialogue_prefix_is_the_text_before_the_first_utterance():
    assert dialogue_prefix(STORY) == ("Timmy carried a lantern through the orchard. "
                                      "Fluffy hopped behind him.")


def test_dialogue_prefix_is_empty_when_the_story_opens_on_dialogue():
    assert dialogue_prefix('"Hello!" said Amy. "Hi," said Bo.') == ""
    assert dialogue_prefix("No dialogue here at all.") == ""


# --------------------------------------------------------------------------------------
# D. RULING A -- the conservative same-speaker filter
# --------------------------------------------------------------------------------------
# Same fixture, ONE gap changed: turn 1's tag becomes a bare pronoun, so there is no
# subject in it that could distinguish anybody. Everything else -- every turn string, every
# other tag -- is identical, which is what makes the two stories' different verdicts
# attributable to the gap and not to the scene.
STORY_PRONOUN_TAG = STORY.replace('" answered the rabbit. "', '" he answered. "')

#: `(gap, identifies_a_subject, names_a_proper_noun, same_voice_risk)`.
#: The rows are chosen so the two readings DISAGREE on three of them -- if they agreed
#: everywhere the manifest's "what the alternative would have cost" would be meaningless.
GAP_TABLE = [
    (" said Amy. ",                          True,  True,  False),
    (" Amy said. ",                          True,  True,  False),
    (" he said. ",                           False, False, True),
    (" she asked. ",                         False, False, True),
    (" they said, ",                         False, False, True),
    (" The fox replied, ",                   True,  False, False),
    (" The rabbit looked up and said, ",     True,  False, False),
    (" Her mom said, ",                      True,  False, False),
    (" they asked together. ",               True,  False, False),
    ("",                                     False, False, False),
    ("   ",                                  False, False, False),
    (" She went to Max's room and opened the door. She said, ",
                                             True,  True,  False),
]


@pytest.mark.parametrize("gap,subject,proper,risk",
                         GAP_TABLE, ids=[repr(r[0]) for r in GAP_TABLE])
def test_gap_indicator_table(gap, subject, proper, risk):
    assert tag_identifies_a_subject(gap) is subject
    assert tag_names_a_proper_noun(gap) is proper
    assert same_voice_risk(gap) is risk


def test_the_two_readings_really_do_disagree():
    """Guards GAP_TABLE from being vacuous. If `tag_identifies_a_subject` and
    `tag_names_a_proper_noun` agreed on every row, the strict variant would be untested and
    the manifest's cost comparison would be comparing a function with itself."""
    disagree = [g for g, s, p, _ in GAP_TABLE if s != p]
    assert len(disagree) >= 3, disagree
    strict_only = [g for g, _, _, r in GAP_TABLE
                   if same_voice_risk(g, strict_names=True) and not r]
    assert " The fox replied, " in strict_only


def test_subject_detection_is_blind_to_which_name():
    """The property that keeps ruling A's filter out of speaker attribution.

    The probe that attributed by name got the speakers BACKWARDS. This filter is allowed to
    notice that SOMEBODY is named; it must never behave differently depending on WHO, and it
    must not care which side of the tag the name sits on.
    """
    variants = [" said Amy. ", " said Ben. ", " said Zog. ", " Amy said. ", " Zog said. "]
    assert len(set(variants)) == len(variants), "the fixture must really vary the name"
    assert {tag_identifies_a_subject(g) for g in variants} == {True}
    assert {same_voice_risk(g) for g in variants} == {False}


def test_non_name_capitals_is_a_stopword_subset():
    """The local capitalised-opener list is not inventing exclusions of its own.

    It cannot import `train.improv.STOPWORDS` (improv imports dialogue, so that would be a
    cycle), so this test is what holds the hand-copied list to its source.
    """
    assert _NON_NAME_CAPITALS <= STOPWORDS, _NON_NAME_CAPITALS - STOPWORDS


def test_an_empty_gap_is_not_a_same_voice_risk():
    """MEASURED DECISION, not an oversight -- see `same_voice_risk`'s docstring.

    Two quotes flush together look like one speaker continuing, but in this corpus the
    attribution often sits AFTER the second utterance instead of between the two, and story
    760 of the kept population is a genuine change of voice with an empty gap. An earlier
    draft treated it as risky and dropped the interrupted-quote fixture for it.
    """
    story760 = ('"Look, dad, I found a big log!" Ben shouted, dragging a long piece of wood '
                'to the shore. "Can we make a boat with it?" "Sure, Ben, that\'s a great '
                'log for a boat," dad said, smiling.')
    gaps = adjacent_gaps(story760, 3)
    assert gaps[1].strip() == "", gaps
    assert same_voice_risk(gaps[1]) is False
    assert voice_changes_throughout(story760, n_turns=3) is True


def test_adjacent_gaps_are_the_spans_between_the_first_five_utterances():
    gaps = adjacent_gaps(STORY)
    assert len(gaps) == MIN_UTTERANCES - 1
    assert [g.strip() for g in gaps] == ["said Timmy.", "answered the rabbit.",
                                         "said Timmy.", "grumbled the rabbit."]


def test_voice_changes_throughout_is_whole_or_nothing():
    """One bad pair is one bad skit: all four pairs of a five-turn scene carry supervision
    on one side or the other, so the gate matches `derive_skit_from_turns`' drop rule."""
    assert voice_changes_throughout(STORY) is True
    assert voice_changes_throughout(STORY_PRONOUN_TAG) is False
    # exactly ONE gap differs, and it is the one that trips
    risky = [i for i, g in enumerate(adjacent_gaps(STORY_PRONOUN_TAG))
             if same_voice_risk(g)]
    assert risky == [1], [g for g in adjacent_gaps(STORY_PRONOUN_TAG)]
    assert extract_dialogue_turns(STORY) == extract_dialogue_turns(STORY_PRONOUN_TAG), \
        "the two fixtures must differ ONLY in the tag, not in the turns"


def test_the_filter_reads_no_name_at_the_story_level_either():
    """Swap every name OUTSIDE the quotes and the verdict must not move.

    `STORY_NAMES_SWAPPED` is the same fixture task 1 used to pin that extraction attributes
    nothing; reusing it here extends that guarantee to the filter, which is the one place a
    name-based heuristic would be most tempting.
    """
    assert STORY != STORY_NAMES_SWAPPED
    assert voice_changes_throughout(STORY) == voice_changes_throughout(STORY_NAMES_SWAPPED)
    assert [same_voice_risk(g) for g in adjacent_gaps(STORY)] == \
           [same_voice_risk(g) for g in adjacent_gaps(STORY_NAMES_SWAPPED)]


def test_strict_names_is_the_stricter_gate_on_the_same_fixture():
    """The switch really is a switch: the fox/rabbit shape survives one reading and not the
    other, which is the whole reason both costs are recorded in the manifest."""
    fox = ('The forest was quiet. A fox met a rabbit by the stream. '
           '"Do you know what I study every day?" The rabbit looked up and said, '
           '"No, I do not know, what do you study?" The fox replied, '
           '"I like to study the animals here." The rabbit said, '
           '"That sounds interesting! I wish I could study too." The fox smiled and said, '
           '"You can. Come with me tomorrow."')
    assert len(quoted_utterances(fox)) >= MIN_UTTERANCES
    assert voice_changes_throughout(fox) is True
    assert voice_changes_throughout(fox, strict_names=True) is False
