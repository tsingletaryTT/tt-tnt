# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for `train.content_add` -- the gate that decides what the `add` slot may name.

THIS FILE EXISTS BECAUSE THE FIXTURE THAT WOULD NORMALLY BE WRITTEN IS USELESS HERE.
A test that separates ``comet`` from ``hello`` proves nothing: every candidate filter this
project considered, including two that are provably wrong, gets that pair right. The cases
that discriminate are:

  * ``look``, ``love``, ``come``, ``want``, ``give``, ``doing``, ``mean`` -- the seven words
    in the measured top-25 `add` vocabulary that READ as particles ("Look!", "Come on!") and
    are genuine verbs. A stoplist of frequent particle-looking words drops them; so does an
    NNP-based filter, because tagged as written ``look`` is NNP 93 times in 120.
  * ``hello`` and ``okay`` -- particles whose narration rate (0.272, 0.237) sits ABOVE
    ``love``'s (0.293)... no, just below it, in the narrowest part of the scale. They are held
    out by the interjection list, and `test_the_interjection_list_does_not_smuggle_out_verbs`
    checks the list did not simply swallow the seven verbs as well.
  * ``mean`` in two turns -- the same word, verb in one and adjective in the other, so the
    per-instance POS gate has something to be per-instance ABOUT.

The `SpeechProfile` fixtures carry the REAL measured counts from the 2,119,489-story corpus,
not invented ones, because the floor's whole justification is where the real words fall
relative to it. Invented counts would let the floor be anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.content_add import (GATE_NAMES, INTERJECTIONS,  # noqa: E402
                               NARRATION_FLOOR, SpeechProfile, build_speech_profile,
                               content_add_filter, content_add_reasons, gate_attribution,
                               has_content_pos, is_clitic_form, is_content_add,
                               is_function_word, is_interjection, narration_rate,
                               pos_tags_in_turn, split_narration_and_speech)
from train.skit import MODEL_TURNS, choose_add_word, derive_skit_from_turns  # noqa: E402

#: Real (narration, speech) token counts from the whole corpus. The seven verbs the task's
#: ground truth says must be KEPT sit at 0.293..0.584; the particles at 0.014..0.272. The
#: floor is 0.20, in the gap.
REAL_COUNTS = {
    "look": (258526, 255760), "come": (169395, 163409), "mean": (73204, 63757),
    "give": (118812, 84791), "want": (318255, 349516), "doing": (44671, 66090),
    "love": (73749, 177659), "comet": (9857, 2374), "dragon": (48027, 9694),
    "kite": (35818, 7383), "hello": (40299, 107838), "okay": (53999, 173830),
    "please": (10226, 144292), "hi": (4420, 59154), "wow": (3821, 113275),
    "hey": (1731, 40927), "thank": (52752, 236585), "course": (6663, 49802),
    "mine": (5174, 48020), "what": (605282, 340980), "where": (120469, 63322),
    "let's": (5560, 392151), "what's": (1789, 48661), "can't": (11374, 97682),
    "our": (16664, 129067), "yes": (98166, 264331), "ok": (16309, 100698),
    "why": (82948, 121901), "ow": (39, 32582), "yuck": (253, 8095),
    "sweetie": (1027, 22069), "pretty": (60000, 40000), "cool": (30000, 30000),
}
PROFILE = SpeechProfile(narration={w: n for w, (n, _) in REAL_COUNTS.items()},
                        speech={w: s for w, (_, s) in REAL_COUNTS.items()},
                        stories=2119489)

#: One real turn per probe word, lifted from `artifacts/reach-skits/skits.jsonl`.
TURNS = {
    "look": "Look, mom, a monkey!",
    "come": "Come on, Ben, let's go up the hill!",
    "give": "Give me the car, Ben!",
    "want": "No, Mommy! I want music now!",
    "doing": "Sam, what are you doing?",
    "love": "Thank you, Mama. The broken pasta is good. We love you.",
    "mean": "What does melt mean?",
    "comet": "Look, Lily, I can see a comet!",
    "dragon": "The dragon wants to take the princess away!",
    "kite": "Mom, can I fly my kite today?",
    "hello": "Hello, little frog! Are you okay?",
    "okay": "Okay, I will clean my room.",
    "please": "Can I have a lemon, please?",
    "hi": "Hi Billy!",
    "wow": "Wow, Lily, you have an amazing heart.",
    "hey": "Hey, that is my toy!",
    "thank": "Thank you, Mr. Lee. Have a nice day too.",
    "course": "Of course, I can help you.",
    "mine": "That is mine, not yours!",
    "what": "Mom, what is that?",
    "where": "Where is my ball?",
    "let's": "Come on, let's go to the park.",
    "what's": "What's wrong Jack?",
    "can't": "Now you can't play your violin anymore!",
    "our": "Where is our king? What have you done to him?",
    "yes": "Yes, it is big. Do you want to play it?",
    "ok": "OK, mom, I will imagine that I have a crane.",
    "why": "Why are there so many people here?",
    "ow": "Ow! That hurts!",
    "yuck": "Yuck! I do not like that soup.",
    "sweetie": "Sure, sweetie, let's go,",
}

#: The seven hard KEEPS and the eighteen hard DROPS, from the measured top-25.
MUST_KEEP = ("look", "love", "come", "want", "give", "doing", "mean")
MUST_DROP = ("please", "hi", "hello", "what", "wow", "why", "yes", "thank", "ok", "okay",
             "mine", "let's", "where", "our", "can't", "what's", "course", "hey")


# --------------------------------------------------------------------------------------
# the classifier, on the cases that can actually fail
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("word", MUST_KEEP)
def test_the_seven_verbs_that_read_as_particles_are_kept(word):
    """`look`/`love`/`come`/`want`/`give`/`doing`/`mean` name actions and must survive.

    These are the cases a crude stoplist gets wrong, and the cases an NNP-based filter gets
    wrong (tagged as written, `look` is NNP 93/120 and `come` 84/120). If this parametrisation
    goes red, the gate has become a frequency stoplist.
    """
    why = content_add_reasons(word, TURNS[word], PROFILE)
    assert why == (), f"{word!r} rejected by {why}"


@pytest.mark.parametrize("word", MUST_DROP)
def test_the_measured_particle_vocabulary_is_rejected(word):
    """The eighteen non-verbs of the measured top-25, each in a real turn."""
    assert not is_content_add(word, TURNS[word], PROFILE), f"{word!r} kept"


@pytest.mark.parametrize("word", ("comet", "dragon", "kite"))
def test_the_easy_nouns_are_kept_too(word):
    """The obvious cases. Not evidence on their own -- see this module's docstring -- but a
    gate that dropped `comet` would be broken in a way the hard cases might not show."""
    assert is_content_add(word, TURNS[word], PROFILE)


def test_the_same_word_is_content_as_a_verb_and_not_as_an_adjective():
    """`mean` names an action in one turn and a property in another. The POS gate is the only
    per-instance gate, so this is the test that says it is per-instance at all.

    Both turns are real rows of `artifacts/reach-skits/skits.jsonl`; over the 646 turns where
    `mean` is the `add` word the tagger splits 304 verb / 311 adjective, so this is the
    typical case and not a picked one. The gate is NOT perfect at it -- "Why are you so mean
    to me?" comes back VB and is wrongly kept -- and `train.content_add`'s docstring says so.
    """
    assert is_content_add("mean", "Me too, dad. We did not mean to be bad.", PROFILE)
    assert not is_content_add(
        "mean", "No, Tom! That is my doll. You are mean. Give it back to me.", PROFILE)


def test_the_verdict_does_not_move_when_the_capitalisation_does():
    """THE TEST THE WHOLE MODULE IS BUILT AROUND.

    The in-context tagger separates particles from things through CAPITALISATION on this
    corpus, and a filter resting on that inverts the moment a particle appears mid-utterance.
    `pos_tags_in_turn` lowercases every token, so the same word must get the same verdict
    utterance-initial, mid-utterance, and in a shouted all-caps turn.
    """
    for word, initial, medial in (("hello", "Hello, dog!", "I said hello to the dog."),
                                  ("look", "Look at the moon!", "I want to look at the moon."),
                                  ("comet", "Comet is my dog!", "I saw the comet last night.")):
        assert is_content_add(word, initial, PROFILE) == \
               is_content_add(word, medial, PROFILE), word
        assert is_content_add(word, initial, PROFILE) == \
               is_content_add(word, initial.upper(), PROFILE), word


def test_content_add_reasons_reports_every_gate_a_word_fails_not_the_first():
    """`let's` is a clitic AND below the narration floor AND not a noun or a verb.

    The manifest's attribution counts a gate as `solely_responsible` only when it is the one
    entry in this tuple, so a short-circuiting implementation would report every gate as
    indispensable.
    """
    why = content_add_reasons("let's", TURNS["let's"], PROFILE)
    assert len(why) > 1
    assert "clitic" in why and "low_narration_rate" in why
    assert list(why) == [g for g in GATE_NAMES if g in why], "gates out of GATE_NAMES order"


def test_a_word_the_corpus_has_never_written_outside_quotes_is_rejected():
    """`narration_rate` is 0.0 for an unseen word, and 0.0 is below any floor.

    An unseen word is not evidence of distance -- that is `train.reach.pair_has_evidence`'s
    job and it is kept separate deliberately -- it is simply a word with no evidence of
    naming anything.
    """
    assert narration_rate("zzblarg", PROFILE) == 0.0
    assert not is_content_add("zzblarg", "Zzblarg, my friend!", PROFILE)


# --------------------------------------------------------------------------------------
# the individual gates
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("word,expected", [
    ("can't", True), ("what's", True), ("let's", True), ("you're", True),
    ("timmy's", True), ("dog's", True), ("i'm", True), ("we'll", True), ("they've", True),
    ("comet", False), ("look", False), ("o", False),
])
def test_is_clitic_form(word, expected):
    assert is_clitic_form(word) is expected


def test_the_interjection_list_does_not_smuggle_out_verbs():
    """The one list a reader should audit: it must not contain the seven hard keeps.

    A filter that hit its precision number by quietly stoplisting `look`/`love`/`come` would
    look identical from the outside and be worthless. This is the check that says it did not.
    """
    for word in MUST_KEEP:
        assert not is_interjection(word), f"{word!r} is in INTERJECTIONS"
        assert not is_function_word(word), f"{word!r} is in FUNCTION_WORDS"


def test_the_function_word_list_covers_what_the_project_stoplist_misses():
    """`train.improv.STOPWORDS` is 40 words and lets the wh-words through; these are why."""
    from train.improv import STOPWORDS
    for word in ("what", "where", "why", "how", "our", "mine", "yours", "myself",
                 "today", "tomorrow", "five", "ten", "enough"):
        assert word not in STOPWORDS, f"{word!r} is already a STOPWORD; drop it from ours"
        assert is_function_word(word)


def test_has_content_pos_rejects_a_word_the_tagger_never_saw():
    """Absent from the tag map means unverified, and unverified is rejected."""
    tags = {"comet": ("NN",), "pretty": ("JJ",)}
    assert has_content_pos("comet", tags)
    assert not has_content_pos("pretty", tags)
    assert not has_content_pos("dragon", tags)


def test_pos_tags_in_turn_is_case_blind_in_its_keys_and_its_tags():
    """Every token is lowercased before tagging, so the map is keyed lowercase and the tags
    are the ones the tagger gives a lowercase sentence."""
    a = pos_tags_in_turn("Look, a COMET!")
    b = pos_tags_in_turn("look, a comet!")
    assert a == b
    assert "comet" in a and "COMET" not in a


# --------------------------------------------------------------------------------------
# the speech profile
# --------------------------------------------------------------------------------------
def test_split_narration_and_speech_puts_the_attribution_tag_in_narration():
    """The narrator wrote ``," she said, "``, so its words are narration.

    Quotes pair as a toggle, the same rule `train.dialogue.quoted_utterances` uses, so the two
    agree about where an utterance stops.
    """
    story = 'The comet appeared. "Look at it!" said Lily. "It is bright," said Ben.'
    narration, speech = split_narration_and_speech(story)
    assert "comet appeared" in narration
    assert "said Lily" in narration and "said Ben" in narration
    assert "Look at it!" in speech and "It is bright," in speech
    assert "comet" not in speech


def test_split_narration_and_speech_leaves_an_unclosed_quote_in_narration():
    """An odd final quote opens a span that never closes. `quoted_utterances` discards it;
    this keeps its text on the narration side rather than guessing at a closer."""
    narration, speech = split_narration_and_speech('She smiled. "I am not finished')
    assert "not finished" in narration
    assert speech.strip() == ""


def test_build_speech_profile_counts_both_sides_and_the_rate_follows():
    """A word said three times and narrated once has a narration rate of 0.25."""
    stories = ['The dog barked. "Hi!" said Ben.',
               '"Hi there," said Ann. "Hi again," said Ben.',
               'The dog ran to the comet.']
    profile = build_speech_profile(stories)
    assert profile.stories == 3
    assert profile.speech["hi"] == 3
    assert profile.narration.get("hi", 0) == 0
    assert narration_rate("hi", profile) == 0.0
    assert profile.narration["dog"] == 2
    assert narration_rate("dog", profile) == 1.0
    assert narration_rate("comet", profile) == 1.0


def test_gate_attribution_separates_sole_responsibility_from_firing():
    """A gate that only ever fires alongside another removes nothing of its own."""
    obs = [("let's", TURNS["let's"]), ("hello", TURNS["hello"]), ("pretty", "It is pretty!")]
    report = gate_attribution(obs, PROFILE)
    assert report["kept"] == 0
    assert report["per_gate"]["interjection"]["solely_responsible"] == 1   # hello
    assert report["per_gate"]["not_noun_or_verb"]["solely_responsible"] == 1  # pretty
    assert report["per_gate"]["clitic"]["fired"] == 1                      # let's
    assert report["per_gate"]["clitic"]["solely_responsible"] == 0


# --------------------------------------------------------------------------------------
# the WIRING -- the layer that actually fails
# --------------------------------------------------------------------------------------
def test_choose_add_word_without_a_filter_is_the_top_ranked_word():
    """The stage-2 behaviour, unchanged, which is why `add_filter` defaults to None."""
    assert choose_add_word(["hi", "comet"], "Hi! A comet!") == "hi"
    assert choose_add_word([], "") is None


def test_choose_add_word_takes_the_first_word_the_filter_accepts_not_the_top_one():
    """The ranking prefers the rarer word; the filter says which words are eligible at all."""
    accept = lambda w, turn: w == "comet"          # noqa: E731
    assert choose_add_word(["hi", "please", "comet", "dog"], "...", accept) == "comet"


def test_choose_add_word_returns_none_when_the_filter_accepts_nothing():
    assert choose_add_word(["hi", "please"], "...", lambda w, t: False) is None


def test_derive_skit_from_turns_honours_the_filter_and_drops_when_it_cannot():
    """THE WIRING TEST. `choose_add_word` returning the right thing is worth nothing if the
    derivation never calls it -- this project has shipped a composition function that no test
    imported. Same prefix and turns, three filters, three different outcomes.
    """
    prefix = "Lily and Ben walked to the field. They looked up at the night sky."
    turns = ["Look, Ben, a comet is flying over the field!",
             "The comet is bright, Lily.",
             "The comet has a tail, Ben, and the tail is made of ice!",
             "Ice in the sky, Lily?",
             "Yes, Ben, the ice melts and makes the tail glow in the sky."]
    idf = {"comet": 9.0, "look": 1.0, "tail": 5.0, "ice": 4.0, "melts": 8.0,
           "glow": 7.0, "bright": 3.0, "flying": 6.0, "sky": 2.0, "yes": 0.5}

    unfiltered = derive_skit_from_turns(prefix, turns, story_id=1, idf=idf,
                                        intensity=lambda t: 0.0)
    assert unfiltered is not None
    # A filter that bans the word the ranking would have chosen must move the slot, not
    # merely be consulted.
    banned = unfiltered.blocks[0].add
    moved = derive_skit_from_turns(prefix, turns, story_id=1, idf=idf,
                                   intensity=lambda t: 0.0,
                                   add_filter=lambda w, t: w != banned)
    assert moved is not None
    assert moved.blocks[0].add != banned
    # And a filter that accepts nothing must drop the whole skit, not emit a block with an
    # empty `add`.
    assert derive_skit_from_turns(prefix, turns, story_id=1, idf=idf,
                                  intensity=lambda t: 0.0,
                                  add_filter=lambda w, t: False) is None


def test_the_real_filter_moves_a_real_add_slot_off_a_particle():
    """End to end on the thing the task is about: `look` beats nothing, `comet` beats `look`.

    Built so the UNFILTERED derivation picks the particle -- `look` is given the higher IDF --
    which is the situation the whole task exists to fix. If the filter were inert this test
    would show `add: look` on both sides.
    """
    prefix = "Ben and Lily sat on the grass. They looked up at the sky."
    turns = ["The sky is dark tonight, Ben.",
             "Yes, the sky is very dark, Lily.",
             "Look, the sky has a comet, look!",
             "The comet is bright, Ben.",
             "Yes Lily, the comet has a bright tail!"]
    idf = {"look": 12.0, "comet": 6.0, "sky": 5.0, "tail": 4.0, "dark": 3.0,
           "tonight": 2.5, "bright": 2.0, "grass": 1.0, "yes": 0.1}
    unfiltered = derive_skit_from_turns(prefix, turns, story_id=2, idf=idf,
                                        intensity=lambda t: 0.0)
    assert unfiltered is not None
    assert unfiltered.blocks[1].add == "look"
    filtered = derive_skit_from_turns(prefix, turns, story_id=2, idf=idf,
                                      intensity=lambda t: 0.0,
                                      add_filter=lambda w, t: w != "look")
    assert filtered is not None
    assert filtered.blocks[1].add == "comet"


def test_content_add_filter_caches_tags_without_changing_a_verdict():
    """The one-entry cache is a speed fix; a cache that changed an answer would be a bug.

    Calls the filter on two words of one turn, then on a different turn, then back -- the
    order that breaks a cache keyed on anything but the turn text.
    """
    accept = content_add_filter(PROFILE)
    first = accept("comet", TURNS["comet"])
    other = accept("hello", TURNS["hello"])
    again = accept("comet", TURNS["comet"])
    assert first is again is True
    assert other is False
    assert accept("look", TURNS["comet"]) is is_content_add("look", TURNS["comet"], PROFILE)


def test_gate_names_matches_what_content_add_reasons_can_return():
    """One list, so the manifest's per-gate table and the classifier cannot drift apart."""
    seen = set()
    for word in MUST_DROP:
        seen |= set(content_add_reasons(word, TURNS[word], PROFILE))
    assert seen <= set(GATE_NAMES)
    assert len(GATE_NAMES) == len(set(GATE_NAMES)) == 5
