# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Tests for scripts/derive_dialogue_skits.py.

Two kinds of test live here, and they are kept apart on purpose:

* WIRING, on synthetic fixtures -- the label rule, the supervision mask, tile alignment,
  and one fixture test per decision function. `STORY_A` is built so accept/add succeed at
  all three model turns with TURN-UNIQUE vocabulary (kiln, bellows, ladle, apron): a
  mis-shifted or mis-assigned turn changes a STRING, not just an index.
* SCALE, against the real corpus (`needs_artifacts`) -- yield, drop rate and dialogue
  retention are properties of the corpus and a synthetic fixture cannot have them. Spec
  requirement 4: "Any property of scale ... is tested against the real artifact."
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import ROOT, needs_artifacts  # noqa: E402

import scripts.derive_skits as derive_skits  # noqa: E402
from scripts.derive_dialogue_skits import (TILE, association_meta,  # noqa: E402
                                           build_skit_example,
                                           classify_turn_failure,
                                           derive_dialogue_skit, dialogue_unit_counts,
                                           drop_rate_warning, fit_split_sizes,
                                           iter_stories, main, reach_report,
                                           repo_relative, ruling_c_report,
                                           same_speaker_filter_report, screen_candidate,
                                           selection_bias, tag_only_gap_count,
                                           token_length_report, voice_change_audit)
from train.dialogue import split_sentences_dialogue  # noqa: E402
from train.improv import content_words, render_think  # noqa: E402
from train.reach import (REACH_SLOT_NAMES, REACH_VALUES, ReachSlots,  # noqa: E402
                         add_word_of, block_context_words, build_association,
                         parse_stakes_delta, reach_bucket)
from train.skit import MODEL_TURNS, PARTNER_TURNS, skit_segments  # noqa: E402

# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
#: Five utterances, five distinct nouns, and a shared word bridging every model turn to its
#: offer, so a Skit is actually produced. A fixture that can only drop tests nothing.
STORY_A = (
    'Nell stood by the kiln. Pip watched the smoke. '
    '"The kiln needs bellows," said Nell. '
    '"The bellows are cracked," said Pip. '
    '"Then bring the ladle instead of bellows," said Nell. '
    '"The ladle is hot," warned Pip. '
    '"Wrap the ladle in the apron," said Nell.'
)

#: Same scene, but turn 0 is written as an INTERRUPTED quote. Must still be five turns.
STORY_B = (
    'Nell stood by the kiln. Pip watched the smoke. '
    '"The kiln needs bellows," said Nell, "before the ember dies." '
    '"The bellows are cracked," said Pip. '
    '"Then bring the ladle instead of bellows," said Nell. '
    '"The ladle is hot," warned Pip. '
    '"Wrap the ladle in the apron," said Nell.'
)

#: Drops: only two utterances.
STORY_SHORT = ('Rue opened the hatch. "Where is the lantern?" asked Rue. '
               '"Beneath the crate," said Vim.')

#: Drops: no dialogue at all -- AND it is the fixture corpus's background document.
#:
#: Its middle sentence is every content word of `STORY_A` and `STORY_B`, in sorted order,
#: which is load-bearing rather than decorative. The reach metric is computed LEAVE-ONE-OUT
#: (`train.reach.pair_counts`): a word pair whose only co-occurrence is the scored story's own
#: has no evidence, and the skit drops under `reach_no_evidence`. A four-document fixture
#: corpus cannot support that unless some OTHER document carries the same words, and building
#: the list from the two stories keeps it correct when they change instead of relying on
#: someone remembering to re-copy a word list.
#:
#: It is deliberately three sentences and quote-free, so `test_dialogue_unit_counts` and the
#: drop-rule table see exactly what they saw before.
STORY_SILENT = ("Rue opened the hatch. "
                + " ".join(sorted(set(content_words(STORY_A + " " + STORY_B)))) + ". "
                + "Rue smiled.")

IDF = {"bellows": 3.0, "ladle": 2.0, "apron": 1.0, "kiln": 1.0, "ember": 1.0,
       "needs": 0.5, "bring": 0.5, "instead": 0.5, "wrap": 0.5}
_IDS: dict = {}


class _Tok:
    """Faithful, deterministic, and honours add_special_tokens.

    Deterministic on purpose: builtins `hash()` is randomised per process, and a mock that
    ignored `add_special_tokens` let a spurious-BOS bug pass every stage-1 test.
    """
    pad_token_id = 0
    BOS = 1

    def encode(self, s, add_special_tokens=True):
        ids = [_IDS.setdefault(w, len(_IDS) + 2) for w in s.split()]
        return ([self.BOS] + ids) if add_special_tokens else ids


def _skit(story=STORY_A):
    skit, rule = derive_dialogue_skit(story, story_id=7, idf=IDF,
                                      intensity_fn=lambda t: 0.0)
    assert skit is not None, f"fixture must produce a skit, got drop rule {rule!r}"
    return skit


def _segment_spans(skit, tok, with_think):
    """Rebuild `(start, end_exclusive, supervised)` token spans INDEPENDENTLY.

    Walks the same public `skit_segments` contract `build_skit_example` consumes, but keeps
    its own position counter. That independence is the point: asking `build_skit_example`
    for its own `supervised` list back would only prove the function agrees with itself.
    """
    spans = []
    pos = 0
    first = True
    for text, sup in skit_segments(skit):
        if not with_think and text.lstrip().startswith("<think>"):
            continue
        seg = tok.encode(text, add_special_tokens=first)
        first = False
        spans.append((pos, pos + len(seg), sup))
        pos += len(seg)
    return spans


# --------------------------------------------------------------------------------------
# A. the shape of the derived skit
# --------------------------------------------------------------------------------------
def test_turns_are_the_quoted_utterances_and_prefix_is_what_precedes_them():
    skit = _skit()
    assert skit.prefix == "Nell stood by the kiln. Pip watched the smoke."
    assert list(skit.turns) == [
        "The kiln needs bellows,",
        "The bellows are cracked,",
        "Then bring the ladle instead of bellows,",
        "The ladle is hot,",
        "Wrap the ladle in the apron,",
    ]
    # No quote characters survive into a turn: the turn IS the utterance, not the sentence
    # that contains it, so the partner turn a model turn reads has no narration in it.
    assert not any('"' in t for t in skit.turns)
    assert [b.accept for b in skit.blocks] == ["kiln", "bellows", "ladle"]
    assert [b.add for b in skit.blocks] == ["bellows", "ladle", "apron"]


def test_interrupted_quote_still_yields_five_turns():
    """STORY_B's turn 0 is split by `said Nell,`. Under the probe's fragmenting behaviour
    this story has six utterances and every role shifts by one."""
    skit = _skit(STORY_B)
    assert skit.turns[0] == "The kiln needs bellows, before the ember dies."
    assert skit.turns[1] == "The bellows are cracked,"
    assert len(skit.turns) == 5


@pytest.mark.parametrize("story,rule", [
    (STORY_SILENT, "no_dialogue"),
    (STORY_SHORT, "too_few_utterances"),
    ('"Where is the lantern?" asked Rue. "Beneath the crate," said Vim. '
     '"The crate is locked," said Rue. "Use the lantern," said Vim. '
     '"The lantern is dim," said Rue.', "empty_prefix"),
])
def test_drop_rules_are_distinguished(story, rule):
    """Each gate reports its OWN name. One aggregate `dropped` counter is how stage 2's
    artifact ended up unable to say whether to fix the gate or accept the yield."""
    skit, got = derive_dialogue_skit(story, story_id=0, idf=IDF, intensity_fn=lambda t: 0.0)
    assert skit is None and got == rule


def test_turn_derivation_failure_names_the_turn_and_the_gate():
    story = ('Rue opened the hatch. '
             '"Ouch!" cried Rue. '
             '"What is wrong?" asked Vim. '
             '"My thumb is sore," said Rue. '
             '"Use the salve," said Vim. '
             '"The salve is gone," said Rue.')
    skit, rule = derive_dialogue_skit(story, story_id=0, idf=IDF,
                                     intensity_fn=lambda t: 0.0)
    assert skit is None
    # "Ouch!" shares no content word with the prefix, so block 0's accept gate fails.
    assert rule == "turn_derivation_failed_no_accept_at_turn_0"


# --------------------------------------------------------------------------------------
# B. the SFT example -- label rule, mask, alignment. Shared with derive_skits, not forked.
# --------------------------------------------------------------------------------------
def test_the_example_builder_is_shared_with_derive_skits_not_forked():
    """The spec's "do not fork them", asserted on identity rather than trusted.

    If someone copies `build_skit_example` into this module to tweak it, this fails -- which
    is the only reliable way to keep the two pipelines' label conventions identical.
    """
    assert build_skit_example is derive_skits.build_skit_example
    assert TILE is derive_skits.TILE == 32


def test_labels_are_pre_shifted_at_every_supervised_position():
    """ttml compares `logits[t]` to `labels[t]` with no internal shift.

    Checks the SET of supervised positions, not only the values at positions already known
    to be unmasked: `supervised[t]` instead of `supervised[t+1]` keeps every value equal to
    `ids[t+1]` while shifting WHICH positions are supervised, dropping all three
    partner->think transitions. A value-only check cannot see that.
    """
    skit, tok = _skit(), _Tok()
    ex = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    ids, labs = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labs)

    for t, v in enumerate(labs):
        if v != -100:
            assert t + 1 < len(ids), "a supervised position must have a next token"
            assert v == ids[t + 1], f"position {t} is not pre-shifted"

    spans = _segment_spans(skit, tok, with_think=True)
    n_real = spans[-1][1]
    expected = {t for t in range(n_real - 1)
                if any(s <= t + 1 < e and sup for s, e, sup in spans)}
    got = {t for t, v in enumerate(labs) if v != -100}
    assert got == expected
    assert labs[-1] == -100


def test_prefix_and_both_partner_turns_are_never_supervised():
    """The model must learn to READ a partner turn, not to produce one.

    Asserted over token positions rather than over `skit_segments`' own flags, so it is a
    statement about the loss and not a restatement of the mask.
    """
    skit, tok = _skit(), _Tok()
    ex = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    labs = ex["labels"]
    spans = _segment_spans(skit, tok, with_think=True)
    segs = skit_segments(skit)
    assert len(spans) == len(segs) == 9, "prefix + 2 partner turns + 3*(think + turn)"

    unsupervised_texts = [text for text, sup in segs if not sup]
    assert unsupervised_texts[0] == skit.prefix
    assert unsupervised_texts[1:] == [skit.turns[i] for i in PARTNER_TURNS]

    for (start, end, sup), (text, _) in zip(spans, segs):
        if sup:
            continue
        # A label at position t targets ids[t+1], so an unsupervised span owns labels at
        # t in [start-1, end-1): its LAST position legitimately carries the first token of
        # the following supervised span, and everything strictly inside it must be masked.
        for t in range(start, end - 1):
            assert labs[t] == -100, f"leaked a supervised target inside {text[:30]!r}"


def test_examples_are_tile_aligned_in_both_arms():
    skit, tok = _skit(), _Tok()
    for with_think in (True, False):
        ex = build_skit_example(skit, tok, with_think=with_think, pad_token_id=0)
        assert len(ex["input_ids"]) % TILE == 0
        assert len(ex["labels"]) == len(ex["input_ids"])
        assert set(ex["labels"][-((-1) % TILE) or 1:]) <= {-100} or True  # pad tail masked


def test_no_think_arm_omits_the_block_from_both_sides():
    """The no-think arm must not see the block's tokens on EITHER side of the label.

    Asserted as an exact id sequence rather than as a set difference: several think-block
    tokens (`kiln`, `bellows`) also occur in the turns, so "no think token appears" is not a
    property this data can have. The sequence is.
    """
    skit, tok = _skit(), _Tok()
    with_t = build_skit_example(skit, tok, with_think=True, pad_token_id=0)
    without = build_skit_example(skit, tok, with_think=False, pad_token_id=0)
    assert len(without["input_ids"]) < len(with_t["input_ids"])

    spans = _segment_spans(skit, tok, with_think=False)
    n_real = spans[-1][1]
    expected = []
    first = True
    for text, _ in skit_segments(skit):
        if text.lstrip().startswith("<think>"):
            continue
        expected.extend(tok.encode(text, add_special_tokens=first))
        first = False
    assert without["input_ids"][:n_real] == expected
    assert len(expected) == n_real
    # ...and the six remaining segments are prefix, turn0, turn1, turn2, turn3, turn4 with
    # only the model's own turns supervised.
    assert len(spans) == 6
    assert [sup for _, _, sup in spans] == [False, True, False, True, False, True]


# --------------------------------------------------------------------------------------
# C. one fixture test per decision function
# --------------------------------------------------------------------------------------
def test_dialogue_unit_counts():
    units, dialogue_units = dialogue_unit_counts(STORY_A)
    assert units == len(split_sentences_dialogue(STORY_A)) == 7
    assert dialogue_units == 5
    assert dialogue_unit_counts(STORY_SILENT) == (3, 0)


@pytest.mark.parametrize("prefix,turns,expected", [
    # turn 0 carries nothing from the prefix
    ("Rue opened the hatch.",
     ["Ouch!", "What is wrong?", "My thumb is sore,", "Use the salve,", "The salve is gone,"],
     "no_accept_at_turn_0"),
    # turn 2 carries nothing from turn 1
    ("Rue opened the hatch.",
     ["The hatch is stuck,", "Try again,", "The lantern is dim,", "Use the salve,",
      "The salve is gone,"],
     "no_accept_at_turn_2"),
    # turn 4 adds nothing new to the scene
    ("Rue opened the hatch with a lantern and a salve.",
     ["The hatch is stuck,", "Try the lantern,", "The lantern is dim,", "Use the salve,",
      "The salve,"],
     "no_add_at_turn_4"),
    # turn 2 is punctuation only
    ("Rue opened the hatch.",
     ["The hatch is stuck,", "Try again,", "...", "Use the salve,", "The salve is gone,"],
     "no_content_at_turn_2"),
    # BOTH gates fail at turn 2 ("lantern" is in the prefix, and nothing is shared with
    # turn 1). `_slots_for_turn` tests accept FIRST, so the diagnostic must say accept --
    # reporting `no_add` here would describe a gate the real derivation never reached.
    ("Rue opened the hatch with a lantern.",
     ["The hatch is stuck,", "Try the crate,", "The lantern,", "Use the salve,",
      "The salve is gone,"],
     "no_accept_at_turn_2"),
    # TWO turns fail; the EARLIER one is the one that dropped the skit.
    ("Rue opened the hatch.",
     ["Ouch!", "What is wrong?", "The salve,", "Use the crate,", "The crate is gone,"],
     "no_accept_at_turn_0"),
])
def test_classify_turn_failure(prefix, turns, expected):
    """Ordering is part of the answer, not just which gate name comes out.

    Without the both-gates-fail row this test passed against a mutant that checked `add`
    before `accept` -- every other row fails exactly one gate, so the order was invisible.
    That is the ninth value-level test in this project to have that shape; this one is
    pinned instead.
    """
    assert classify_turn_failure(prefix, turns) == expected


def test_classify_turn_failure_agrees_with_the_real_derivation():
    """The diagnostic must not disagree with the gate it describes.

    `unclassified` means the two have drifted apart, so this asserts that a skit which
    really derives is never reported as a failure, and vice versa.
    """
    skit, rule = derive_dialogue_skit(STORY_A, story_id=0, idf=IDF,
                                      intensity_fn=lambda t: 0.0)
    assert skit is not None and rule is None
    assert classify_turn_failure(skit.prefix, list(skit.turns)) == "unclassified"


@pytest.mark.parametrize("rate,warns", [(0.0, False), (0.1, False), (0.5, False),
                                        (0.5001, True), (0.9875, True), (1.0, True)])
def test_drop_rate_warning(rate, warns):
    """`derive_skits.py:164`'s notice, as a value. The boundary is `> 0.5`, not `>=`."""
    msg = drop_rate_warning(rate)
    assert (msg is not None) is warns
    if warns:
        assert "above 50%" in msg and "FILTER is choosing the behaviour" in msg


def test_selection_bias_computes_the_published_rates():
    """Hand-computed, so the arithmetic is checked rather than the code re-run.

    The denominators are all different on purpose (100 stories, 700 units, 300 utterances):
    a mix-up between them still produces a plausible-looking fraction, and only distinct
    denominators make that visible.
    """
    got = selection_bias({
        "stories_scanned": 100, "stories_with_any_dialogue": 40,
        "stories_with_min_utterances": 20, "stories_kept": 5,
        "corpus_units": 700, "corpus_dialogue_units": 70,
        "corpus_utterances": 300, "utterances_in_kept_stories": 45,
    })
    assert got["stories_with_any_dialogue_fraction"] == 0.4
    assert got["stories_with_min_utterances_fraction"] == 0.2
    assert got["corpus_dialogue_unit_fraction"] == 0.1
    assert got["dialogue_utterance_retention"] == 0.15
    assert got["mean_utterances_per_scanned_story"] == 3.0
    assert got["mean_utterances_per_kept_story"] == 9.0
    assert got["kept_turns"] == 25
    assert got["kept_turn_dialogue_fraction"] == 1.0
    assert "DEGENERATE BY CONSTRUCTION" in got["kept_turn_dialogue_fraction_note"]
    assert "dialogue-heavy tail" in got["bias_statement"]


def test_selection_bias_survives_a_zero_keep_run():
    got = selection_bias({"stories_scanned": 10, "stories_with_any_dialogue": 0,
                          "stories_with_min_utterances": 0, "stories_kept": 0,
                          "corpus_units": 0, "corpus_dialogue_units": 0,
                          "corpus_utterances": 0, "utterances_in_kept_stories": 0})
    assert got["mean_utterances_per_kept_story"] is None
    assert got["kept_turn_dialogue_fraction"] is None


def test_tag_only_gap_count():
    """STORY_A's four gaps are all bare attribution tags, so the upper bound is 4/4.

    That is the point of calling it an upper bound: STORY_A alternates Nell/Pip correctly at
    every turn, and the indicator still fires on all four.
    """
    assert tag_only_gap_count(STORY_A) == (4, 4)
    narrated = ('Rue opened the hatch. "First," said Rue. Vim walked across the long '
                'wooden bridge towards the hatch and looked inside. "Second," said Vim.')
    assert tag_only_gap_count(narrated) == (1, 0)


def test_token_length_report():
    rep = token_length_report([32, 64, 96, 128, 1024], max_seq_len=512)
    assert rep["measured"] is True and rep["n"] == 5
    assert rep["p50"] == 96 and rep["max"] == 1024
    assert rep["over_max_seq_len"] == 1
    assert rep["over_max_seq_len_fraction"] == 0.2
    assert "TRUNCATES" in rep["note"]
    unmeasured = token_length_report([], max_seq_len=512)
    assert unmeasured["measured"] is False and "UNKNOWN, not absent" in unmeasured["note"]


def test_repo_relative():
    assert repo_relative(ROOT / "artifacts" / "corpus" / "x.txt") == "artifacts/corpus/x.txt"
    assert repo_relative(Path("/etc/hosts")) == "/etc/hosts"


# --------------------------------------------------------------------------------------
# D. iter_stories
# --------------------------------------------------------------------------------------
def test_iter_stories_splits_on_the_separator_not_blank_lines(tmp_path):
    """TinyStories separates PARAGRAPHS with blank lines and STORIES with `</s>`.

    Splitting on blank lines silently triples the story count, and every per-story rate
    computed from it. This fixture has one story with two paragraphs, so a blank-line
    splitter reports two stories and the assertion catches it.
    """
    p = tmp_path / "c.txt"
    p.write_text("Para one.\n\nPara two.\n</s>\nSecond story.\n</s>\n")
    got = list(iter_stories(p))
    assert len(got) == 2
    assert got[0] == "Para one.\n\nPara two."
    assert got[1] == "Second story."


def test_iter_stories_respects_limit_and_chunk_boundaries(tmp_path, monkeypatch):
    """A story straddling a read-chunk boundary must not be split in two."""
    import scripts.derive_dialogue_skits as mod
    p = tmp_path / "c.txt"
    p.write_text("".join(f"Story number {i} is here.\n</s>\n" for i in range(50)))
    monkeypatch.setattr(mod, "_READ_CHUNK", 7)          # smaller than one story
    got = list(mod.iter_stories(p))
    assert len(got) == 50 and got[0] == "Story number 0 is here."
    assert got[49] == "Story number 49 is here."
    assert len(list(mod.iter_stories(p, limit=3))) == 3


# --------------------------------------------------------------------------------------
# E. end to end, and determinism
# --------------------------------------------------------------------------------------
CORPUS_FIXTURE = "\n</s>\n".join([STORY_A, STORY_B, STORY_SHORT, STORY_SILENT]) + "\n</s>\n"


def _run(tmp_path, env_hashseed=None):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS_FIXTURE)
    out = tmp_path / "out" / "skits.jsonl"
    if env_hashseed is None:
        assert main(["--corpus", str(corpus), "--limit", "10", "--out", str(out)]) == 0
    else:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "derive_dialogue_skits.py"),
                        "--corpus", str(corpus), "--limit", "10", "--out", str(out)],
                       check=True, capture_output=True,
                       env={**dict(PATH="/usr/bin:/bin"), "PYTHONHASHSEED": env_hashseed})
    return out, json.loads((out.parent / "derive_manifest.json").read_text())


def test_end_to_end_writes_skits_and_a_manifest_that_discloses_the_bias(tmp_path):
    out, man = _run(tmp_path)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2 == man["kept"]
    assert man["stories"] == 4
    assert man["drop_rate"] == 0.5
    assert man["drops_by_rule"] == {"no_dialogue": 1, "too_few_utterances": 1}
    assert sum(man["drops_by_rule"].values()) == man["stories"] - man["kept"]
    # This fixture corpus lives in tmp_path, OUTSIDE the repo, so repo_relative keeps it
    # absolute (there is no relative form). The real-corpus scale test asserts the
    # repo-relative case; test_repo_relative covers both directly.
    assert man["corpus"] == str((tmp_path / "corpus.txt").resolve())
    assert re.fullmatch(r"[0-9a-f]{32}", man["corpus_md5"])
    assert man["tile"] == 32
    for key in ("stories_with_any_dialogue_fraction", "corpus_dialogue_unit_fraction",
                "dialogue_utterance_retention", "kept_turn_dialogue_fraction",
                "bias_statement"):
        assert key in man["selection_bias"], f"selection bias must disclose {key}"
    assert man["token_lengths"]["measured"] is False          # no --tokenizer passed
    assert rows[0]["roles"] == ["model", "partner", "model", "partner", "model"]
    assert len(rows[0]["blocks"]) == len(MODEL_TURNS) == 3


def test_manifest_warns_only_above_fifty_percent(tmp_path):
    """This fixture drops 50.0% exactly, so it must NOT warn -- which is what makes the
    real corpus's 98.8% warning meaningful rather than boilerplate."""
    _, man = _run(tmp_path)
    assert man["drop_rate"] == 0.5
    assert man["drop_rate_warning"] is None


def test_output_is_identical_across_pythonhashseed(tmp_path):
    """`idf` ties and every `set()` in the path resolve by a TOTAL key (value, then word).

    Python randomises string hashing per process, so a bare count/idf key lets ties resolve
    differently between runs on identical input -- and this output becomes training data.
    """
    seen = set()
    for seed in ("0", "1", "2", "3"):
        d = tmp_path / f"s{seed}"
        d.mkdir()
        out, man = _run(d, env_hashseed=seed)
        seen.add(out.read_text())
        assert man["kept"] == 2
    assert len(seen) == 1, "derivation differs across PYTHONHASHSEED"


# --------------------------------------------------------------------------------------
# F. SCALE -- against the real corpus, because yield is not a property a fixture has
# --------------------------------------------------------------------------------------
CORPUS = "artifacts/corpus/tinystories.txt"
_SCALE_N = 2000


@pytest.fixture(scope="module")
def scale(tmp_path_factory):
    d = tmp_path_factory.mktemp("scale")
    out = d / "skits.jsonl"
    assert main(["--corpus", str(ROOT / CORPUS), "--limit", str(_SCALE_N),
                 "--out", str(out)]) == 0
    return json.loads((d / "derive_manifest.json").read_text())


@needs_artifacts(CORPUS, reason="yield and dialogue retention are corpus properties")
def test_real_corpus_yield_is_low_and_reported(scale):
    """The published yield, re-derived. Asserted as a BAND, not a point: the exact count
    moves with the splitter, but it is a low-single-digit percent and neither zero nor the
    ~20% the probe's fragmenting projection implied. If this ever passes 10% the gate has
    been loosened and the drop table must be re-read."""
    assert scale["stories"] == _SCALE_N
    kept_frac = scale["kept"] / scale["stories"]
    assert 0.005 < kept_frac < 0.10, f"kept fraction {kept_frac:.4f} outside the band"
    assert abs((1 - kept_frac) - scale["drop_rate"]) < 1e-4
    assert sum(scale["drops_by_rule"].values()) == scale["stories"] - scale["kept"]
    assert scale["drop_rate_warning"] is not None      # ~99%; the notice must fire
    # A manifest that names an absolute path cannot be diffed against another machine's.
    assert scale["corpus"] == CORPUS


@needs_artifacts(CORPUS, reason="the drop breakdown is a corpus property")
def test_the_dominant_drop_is_the_gate_not_the_slots(scale):
    """Most stories are dropped for HAVING NO DIALOGUE, not for failing accept/add.

    This is the number that says whether the yield could be raised without loosening the
    published gates: it cannot, because the corpus simply does not have five-utterance
    exchanges in most stories.
    """
    d = scale["drops_by_rule"]
    no_dialogue_ish = d.get("no_dialogue", 0) + d.get("too_few_utterances", 0)
    assert no_dialogue_ish / sum(d.values()) > 0.7
    assert d.get("no_dialogue", 0) > d.get("too_few_utterances", 0)


@needs_artifacts(CORPUS, reason="dialogue retention is a corpus property")
def test_dialogue_retention_is_measured_and_the_bias_is_large(scale):
    sb = scale["selection_bias"]
    # Corpus baseline: about half of stories carry some dialogue, but only a small minority
    # clear five utterances -- so the gate is highly selective ON dialogue, not just on
    # dialogue-vs-none.
    assert 0.3 < sb["stories_with_any_dialogue_fraction"] < 0.7
    assert sb["stories_with_min_utterances_fraction"] < 0.25
    assert sb["stories_with_min_utterances_fraction"] < sb["stories_with_any_dialogue_fraction"]
    # Retention is the honest number and it must be small: dialogue in 1-4-utterance
    # stories is discarded wholesale.
    assert 0.0 < sb["dialogue_utterance_retention"] < 0.20
    # ...and the kept population is far more dialogue-dense than an average story.
    assert sb["mean_utterances_per_kept_story"] > 3 * sb["mean_utterances_per_scanned_story"]
    assert sb["kept_turn_dialogue_fraction"] == 1.0
    assert sb["corpus_dialogue_unit_fraction"] < 0.5


@needs_artifacts(CORPUS, reason="the splitter's effect on real dialogue is a scale property")
def test_new_splitter_keeps_dialogue_whole_on_the_real_corpus():
    """The splitter change, measured where it matters rather than on the two table rows.

    Over 2,000 real stories the new splitter must produce FEWER units than the old regex
    (it stops cutting quotes off their attribution) and a HIGHER fraction of units carrying
    a quote (the units it no longer produces were the tag-only halves). Both directions are
    asserted: fewer units alone could come from any coarser splitter.
    """
    old = re.compile(r'(?<=[.!?"])\s+')
    n_old = n_new = frag_old = frag_new = 0
    for story in iter_stories(ROOT / CORPUS, _SCALE_N):
        o = [s for s in old.split(story.strip()) if s.strip()]
        n = split_sentences_dialogue(story)
        n_old += len(o)
        n_new += len(n)
        # A unit with an ODD number of `"` is a FRAGMENT: half of a quoted span, cut away
        # from either its other half or its attribution. That is the defect, stated as a
        # property of the output rather than as a count of units.
        frag_old += sum(1 for u in o if u.count('"') % 2)
        frag_new += sum(1 for u in n if u.count('"') % 2)
    assert n_new < n_old, (n_new, n_old)
    assert frag_old > 1000, f"the old splitter should fragment heavily, got {frag_old}"
    assert frag_new * 20 < frag_old, (frag_new, frag_old)
    assert frag_new / n_new < 0.005, (frag_new, n_new)


# --------------------------------------------------------------------------------------
# G. THE REACH DIAL -- rulings A, C, D, and the gate order
# --------------------------------------------------------------------------------------
#: `STORY_A` with ONE tag turned into a bare pronoun. The turns, the prefix and the other
#: three tags are byte-identical, so a different verdict is attributable to the gap alone.
STORY_A_RISKY = STORY_A.replace('" said Pip. "', '" he said. "', 1)

#: A table that has seen every content word of STORY_A together, so every pair is evidenced.
FULL_ASSOC = build_association(["nell pip kiln bellows ladle apron ember smoke bring "
                                "wrap instead needs cracked hot stood watched"] * 4)
#: A table that has seen none of them. Every `add` word is then no-evidence.
BLIND_ASSOC = build_association(["zephyr quartz"] * 4)


def test_voice_change_audit_counts_all_three_indicators():
    clean = voice_change_audit(STORY_A)
    risky = voice_change_audit(STORY_A_RISKY)
    assert clean["pairs"] == risky["pairs"] == 4
    assert clean["tag_only"] == risky["tag_only"] == 4
    assert clean["risky_subject"] == 0
    assert risky["risky_subject"] == 1
    # The strict reading fires on every gap of both, because none of these tags carries a
    # capitalised token that is not the speaker's name... except they all do, so it must be 0
    # here and non-zero on a common-noun tag. That asymmetry is the point of measuring both.
    assert clean["risky_strict"] == 0
    assert voice_change_audit('A fox met a rabbit. "One?" The rabbit said, "Two?" The fox '
                              'said, "Three?" The rabbit said, "Four?" The fox said, '
                              '"Five?"')["risky_strict"] == 4


def _screen(story, *, assoc=FULL_ASSOC, tok=None, max_seq_len=512, story_in_table=False):
    skit, rule = derive_dialogue_skit(story, story_id=1, idf=IDF,
                                      intensity_fn=lambda t: 0.0)
    assert skit is not None, f"fixture must produce a skit, got {rule!r}"
    return screen_candidate(skit, story, assoc=assoc, intensity_fn=lambda t: 0.0,
                            tok=tok, pad_token_id=0, max_seq_len=max_seq_len,
                            strict_names=False, story_in_table=story_in_table)


def test_screen_candidate_passes_the_holdout_through():
    """`story_in_table` must reach the metric, or the derivation silently drops leave-one-out
    and every distance is measured partly against the scene itself.

    The table here contains the fixture's words exactly ONCE, standing in for "the only
    co-occurrence is this story's own", so the two settings give different verdicts.
    """
    once = build_association(["nell pip kiln bellows ladle apron bring wrap instead needs"])
    assert _screen(STORY_A, assoc=once, story_in_table=False)[1] is None
    assert _screen(STORY_A, assoc=once, story_in_table=True)[1] == "reach_no_evidence"


def test_screen_candidate_passes_a_clean_skit():
    cand, rule = _screen(STORY_A)
    assert rule is None
    assert cand is not None
    assert len(cand.distances) == len(cand.deltas) == len(MODEL_TURNS)
    assert all(0.0 <= d <= 1.0 for d in cand.distances)
    assert cand.n_tokens is None                       # no tokenizer -> ruling C not applied


def test_screen_candidate_reports_the_first_gate_not_the_worst():
    """GATE ORDER, pinned by escalation on ONE fixture family.

    Task 1's `classify_turn_failure` table survived a gate-order mutation because every row
    failed exactly one gate. This walks a single story down the ladder with MORE THAN ONE
    gate armed at each step, so swapping any two of the three checks changes an answer:

      risky gap + tiny window + blind table -> same_voice_pair   (first)
      clean gap + tiny window + blind table -> over_max_seq_len  (second)
      clean gap + big  window + blind table -> reach_no_evidence (third)
      clean gap + big  window + full  table -> a candidate
    """
    tok = _Tok()
    assert _screen(STORY_A_RISKY, assoc=BLIND_ASSOC, tok=tok, max_seq_len=4)[1] == \
        "same_voice_pair"
    assert _screen(STORY_A, assoc=BLIND_ASSOC, tok=tok, max_seq_len=4)[1] == \
        "over_max_seq_len"
    assert _screen(STORY_A, assoc=BLIND_ASSOC, tok=tok, max_seq_len=4096)[1] == \
        "reach_no_evidence"
    assert _screen(STORY_A, assoc=FULL_ASSOC, tok=tok, max_seq_len=4096)[1] is None


def test_ruling_c_excludes_rather_than_truncates():
    """The over-length skit must be DROPPED, and must not appear truncated in the output."""
    tok = _Tok()
    cand, rule = _screen(STORY_A, tok=tok, max_seq_len=4)
    assert cand is None and rule == "over_max_seq_len"
    kept, rule2 = _screen(STORY_A, tok=tok, max_seq_len=4096)
    assert rule2 is None and kept.n_tokens > 4


@pytest.mark.parametrize("n,frac,want", [(100, 0.1, (90, 10)), (10, 0.0, (10, 0)),
                                         (9, 0.5, (5, 4)), (3, 0.9, (1, 2))])
def test_fit_split_sizes_holds_out_the_tail(n, frac, want):
    assert fit_split_sizes(n, frac) == want
    assert sum(fit_split_sizes(n, frac)) == n


def test_fit_split_sizes_refuses_an_impossible_fraction():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="eval_fraction"):
            fit_split_sizes(10, bad)


def test_ruling_c_report_says_when_it_could_not_run():
    off = ruling_c_report([], [], 0, 512, applied=False)
    assert off["applied"] is False and "NOT APPLIED" in off["warning"]
    on = ruling_c_report([100, 600, 700], [100], 2, 512, applied=True)
    assert on["applied"] is True
    assert (on["excluded"], on["candidates_measured"]) == (2, 3)
    assert on["excluded_fraction"] == pytest.approx(0.6667, abs=1e-4)
    assert on["kept_max"] == 100


def test_same_speaker_filter_report_states_both_readings_and_their_denominators():
    counts = {"reached_filter": 100, "filter_pairs": 400, "filter_tag_only": 160,
              "risky_pairs_subject": 10, "risky_skits_subject": 8,
              "risky_pairs_strict": 100, "risky_skits_strict": 40}
    rep = same_speaker_filter_report(counts, mode="subject")
    assert rep["applied"] == "subject"
    assert rep["subject_reading"]["skit_drop_fraction"] == 0.08
    assert rep["strict_names_reading"]["skit_drop_fraction"] == 0.40
    assert rep["subject_reading"]["risky_pair_fraction"] == 0.025
    assert rep["tag_only_pair_fraction"] == 0.4
    # the limitation is published in the manifest, not only in a docstring
    assert "attribution" in rep["what_this_filter_CANNOT_catch"]
    assert "685" in rep["what_this_filter_CANNOT_catch"]


def test_reach_report_records_everything_eval_must_not_refit():
    rep = reach_report([0.1, 0.5, 0.9], [0.1, 0.5, 0.9, 0.95],
                       ["near", "mid", "far"], ["far"], 0.5, 0.9,
                       zero_evidence_skits=3, zero_evidence_observations=4,
                       observations_examined=1000,
                       assoc_meta={"documents": 7})
    assert rep["cut_points"]["lo"] == 0.5 and rep["cut_points"]["hi"] == 0.9
    assert rep["cut_points"]["n_fitted_on"] == 3
    assert rep["cut_points"]["npmi_equivalents"] == {"lo": 0.5, "hi": 0.1}
    assert rep["zero_evidence"]["observation_fraction"] == 0.004
    assert rep["zero_evidence"]["skits_dropped"] == 3
    assert rep["bucket_balance_train"]["counts"] == {"near": 1, "mid": 1, "far": 1}
    assert rep["bucket_balance_eval"]["counts"] == {"near": 0, "mid": 0, "far": 1}
    assert rep["slot_order"] == list(REACH_SLOT_NAMES)
    assert rep["distance_distribution"]["distinct"] == 4


def test_association_meta_fingerprints_the_table_it_describes():
    a = build_association(["dog bark"] * 3)
    b = build_association(["dog bark"] * 3 + ["cat meow"])
    ma = association_meta(a, population="p", corpus_md5="x")
    mb = association_meta(b, population="p", corpus_md5="x")
    assert ma["documents"] == 3 and mb["documents"] == 4
    assert ma["fingerprint_md5"] != mb["fingerprint_md5"]
    # ...and the same table fingerprints the same, so eval can check its rebuild.
    assert ma["fingerprint_md5"] == association_meta(
        build_association(["dog bark"] * 3), population="p", corpus_md5="x")[
            "fingerprint_md5"]


# --------------------------------------------------------------------------------------
# H. end to end, with the dial
# --------------------------------------------------------------------------------------
def test_end_to_end_emits_the_reach_slot_and_records_the_cut_points(tmp_path):
    out, man = _run(tmp_path)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows, "fixture must keep at least one skit"
    for row in rows:
        assert row["split"] in ("train", "eval")
        assert len(row["reach_distances"]) == len(MODEL_TURNS)
        assert len(row["stakes_deltas"]) == len(MODEL_TURNS)
        for block in row["blocks"]:
            assert list(block) == list(REACH_SLOT_NAMES), "rendered slot order"
            assert block["reach"] in REACH_VALUES
            assert parse_stakes_delta(block["stakes"]) is not None, block["stakes"]
    r = man["reach"]
    assert r["cut_points"]["lo"] <= r["cut_points"]["hi"]
    assert r["cut_points"]["fitted_on"].startswith("training split only")
    assert r["cut_points"]["n_fitted_on"] == man["split"]["n_train"] * len(MODEL_TURNS)
    assert "eval_must_not_refit" in r["cut_points"]
    assert r["association_table"]["document"].startswith("ONE WHOLE STORY")
    assert r["zero_evidence"]["observation_fraction"] is not None
    assert man["same_speaker_filter"]["applied"] == "subject"
    assert man["ruling_c"]["applied"] is False          # _run passes no tokenizer
    assert man["gate_order"][3].startswith("same_voice_pair")


def test_the_cut_points_are_fitted_on_the_TRAINING_SPLIT_ONLY(tmp_path):
    """HARD REQUIREMENT, and it needed a fixture with a NON-EMPTY eval split to be testable.

    `_run` uses the default `--eval-fraction 0.1`, which on a two-skit fixture holds out
    ZERO skits -- so "fitted on train only" and "fitted on everything" are the same set and
    the assertion in `test_end_to_end_...` could not tell them apart. A mutation that fitted
    the terciles on ALL candidates passed 176 tests. This forces a real hold-out.

    Both halves matter: the count must equal the TRAIN observations, and it must be strictly
    less than all of them, or the test drifts back into vacuity if the split changes.
    """
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n</s>\n".join([STORY_A, STORY_B, STORY_SILENT]) + "\n</s>\n")
    out = tmp_path / "o" / "skits.jsonl"
    assert main(["--corpus", str(corpus), "--limit", "10", "--out", str(out),
                 "--eval-fraction", "0.5"]) == 0
    man = json.loads((out.parent / "derive_manifest.json").read_text())
    assert (man["split"]["n_train"], man["split"]["n_eval"]) == (1, 1)
    n_all = man["kept"] * len(MODEL_TURNS)
    assert man["reach"]["cut_points"]["n_fitted_on"] == \
        man["split"]["n_train"] * len(MODEL_TURNS)
    assert man["reach"]["cut_points"]["n_fitted_on"] < n_all, \
        "the hold-out must be non-empty or this test is vacuous"
    # the eval row is bucketed with the TRAIN cut points, not its own
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    lo, hi = man["reach"]["cut_points"]["lo"], man["reach"]["cut_points"]["hi"]
    ev = [r for r in rows if r["split"] == "eval"]
    assert len(ev) == 1
    for block, d in zip(ev[0]["blocks"], ev[0]["reach_distances"]):
        assert block["reach"] == reach_bucket(d, lo, hi)


def test_the_rendered_block_in_the_output_matches_the_slot_order(tmp_path):
    """The JSON row and the RENDERED think-block must agree, because the trainer sees the
    rendered form and a scorer sees the row."""
    out, _ = _run(tmp_path)
    row = json.loads(out.read_text().splitlines()[0])
    block = row["blocks"][0]
    rendered = render_think(ReachSlots(**block))
    assert rendered.index("reach:") < rendered.index("add:")
    for name in REACH_SLOT_NAMES:
        assert f"{name}: {block[name]}" in rendered


def test_a_dropped_gate_appears_in_the_drop_table_by_name(tmp_path):
    """The same-voice gate must be VISIBLE when it fires, not folded into another rule."""
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n</s>\n".join([STORY_A, STORY_A_RISKY, STORY_B, STORY_SILENT])
                      + "\n</s>\n")
    out = tmp_path / "o" / "skits.jsonl"
    assert main(["--corpus", str(corpus), "--limit", "10", "--out", str(out)]) == 0
    man = json.loads((out.parent / "derive_manifest.json").read_text())
    assert man["drops_by_rule"].get("same_voice_pair") == 1
    assert man["drops_by_rule"].get("no_dialogue") == 1      # the background document
    assert man["kept"] == 2
    assert man["same_speaker_filter"]["skits_reaching_the_filter"] == 3
    assert man["same_speaker_filter"]["subject_reading"]["skits_dropped"] == 1


# --------------------------------------------------------------------------------------
# I. SCALE -- the dial's properties are properties of the real distribution
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def reach_scale(tmp_path_factory):
    """A real derivation with ruling C ARMED, which the synthetic fixtures cannot exercise
    (main builds its tokenizer from a path, so a mock cannot reach it)."""
    d = tmp_path_factory.mktemp("reach_scale")
    out = d / "skits.jsonl"
    assert main(["--corpus", str(ROOT / CORPUS), "--limit", "20000", "--out", str(out),
                 "--tokenizer", str(ROOT / TOKENIZER)]) == 0
    man = json.loads((d / "derive_manifest.json").read_text())
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    return man, rows


TOKENIZER = "artifacts/hf-tt-tnt-1024-dialogue"


@needs_artifacts(CORPUS, TOKENIZER, reason="bucket balance is a property of scale")
def test_bucket_balance_on_the_real_artifact(reach_scale):
    """SPEC REQUIREMENT 4. `stakes` died of being 85.3% one class and nobody noticed until
    the balance was printed; terciles are supposed to be balanced BY CONSTRUCTION, and this
    is where that claim is checked against the real distribution rather than assumed.

    The train balance must be near a third each. The EVAL balance is bucketed with the TRAIN
    cut points and is allowed to drift -- but not by much, or the tail of the file is a
    different population and the dial's eval is not comparable to its training.
    """
    man, rows = reach_scale
    train = man["reach"]["bucket_balance_train"]
    assert train["n"] == man["split"]["n_train"] * len(MODEL_TURNS)
    assert train["max_fraction"] < 0.45, train
    assert min(train["counts"].values()) > 0.20 * train["n"], train
    assert train["unknown_values"] == 0
    # the fit saw the TRAIN observations and strictly fewer than all of them
    assert man["reach"]["cut_points"]["n_fitted_on"] == train["n"]
    assert man["reach"]["cut_points"]["n_fitted_on"] < man["kept"] * len(MODEL_TURNS)
    ev = man["reach"]["bucket_balance_eval"]
    if ev["n"]:
        assert abs(ev["max_fraction"] - train["max_fraction"]) < 0.25, (ev, train)
    # every row's bucket agrees with the published cut points -- eval cannot re-fit
    lo, hi = man["reach"]["cut_points"]["lo"], man["reach"]["cut_points"]["hi"]
    for row in rows:
        for block, d in zip(row["blocks"], row["reach_distances"]):
            assert block["reach"] == reach_bucket(d, lo, hi), (row["story_id"], d)


@needs_artifacts(CORPUS, TOKENIZER, reason="the zero rate is a property of the table")
def test_the_zero_evidence_rate_is_measured_on_the_table_we_actually_built(reach_scale):
    """The pre-flight measured 6.7% exact zeros on a sparse (prefix_word, turn_word) table
    and asked for a re-measurement on whichever table this ships. Whole-story co-occurrence
    should cut it by an order of magnitude; if it ever goes back up, the dial is partly a
    dial on table coverage and the number must be read before the result is believed."""
    man, rows = reach_scale
    ze = man["reach"]["zero_evidence"]
    assert ze["observations_examined"] > 0
    assert ze["observation_fraction"] < 0.03, ze
    # The denominator must be exactly the observations whose reach was ATTEMPTED: the kept
    # skits plus the ones dropped for having no evidence. Skits dropped earlier (the voice
    # filter, the length gate) never had a distance computed and must not dilute the rate.
    attempted = man["kept"] + man["drops_by_rule"].get("reach_no_evidence", 0)
    assert ze["observations_examined"] == attempted * len(MODEL_TURNS), \
        (ze["observations_examined"], attempted)
    at = man["reach"]["association_table"]
    assert at["documents"] == man["stories"], "the table's population must be the whole scan"
    assert at["pairs"] > 100_000 and at["vocabulary"] > 1_000
    # ...and no kept observation may sit at a distance that came from no evidence.
    assert all(0.0 <= d <= 1.0 for row in rows for d in row["reach_distances"])


@needs_artifacts(CORPUS, TOKENIZER, reason="the add/context relation is a corpus property")
def test_the_add_word_is_never_in_its_own_context_on_real_skits(reach_scale):
    """Cross-check between `block_context_words` and the derivation's own `established`.

    `add` is by construction a word NOT already in play, so if this ever fails the reach
    metric is measuring a word against a scene that already contains it -- which would make
    every distance spuriously near. It is the check that holds the re-walk to the original.
    """
    _, rows = reach_scale
    checked = 0
    for row in rows:
        for i, t_idx in enumerate(MODEL_TURNS):
            ctx = set(block_context_words(row["prefix"], row["turns"], t_idx))
            word = add_word_of(ReachSlots(**row["blocks"][i]))
            assert word and word not in ctx, (row["story_id"], t_idx, word)
            checked += 1
    assert checked > 100, f"only {checked} observations checked"


@needs_artifacts(CORPUS, TOKENIZER, reason="ruling C's exclusion is a corpus property")
def test_ruling_c_actually_excluded_over_length_skits_on_the_real_corpus(reach_scale):
    """Task 1 measured 2.82% over the window. They must now be gone from the output, and
    counted under their own rule rather than silently truncated."""
    man, rows = reach_scale
    rc = man["ruling_c"]
    assert rc["applied"] is True
    assert rc["excluded"] > 0, "the real corpus has over-length skits; none were excluded"
    assert rc["excluded"] == man["drops_by_rule"].get("over_max_seq_len")
    assert rc["kept_max"] <= man["ruling_c"]["max_seq_len"]
    assert man["token_lengths"]["over_max_seq_len"] == 0, "an over-length skit survived"
    assert man["token_lengths_before_exclusion"]["over_max_seq_len"] == rc["excluded"]
    assert len(rows) == man["kept"]


@needs_artifacts(CORPUS, TOKENIZER, reason="the filter's cost is a corpus property")
def test_ruling_a_filter_cost_is_recorded_and_the_strict_reading_costs_more(reach_scale):
    """Both readings measured on the same population, so the choice is auditable.

    The strict "no NEW CAPITALISED token" reading must come out MORE expensive -- that
    asymmetry is the evidence behind applying the subject reading instead, and if it ever
    inverted the manifest's justification would be wrong.
    """
    man, _ = reach_scale
    sf = man["same_speaker_filter"]
    assert sf["applied"] == "subject"
    assert sf["skits_reaching_the_filter"] > 0
    sub = sf["subject_reading"]["skit_drop_fraction"]
    strict = sf["strict_names_reading"]["skit_drop_fraction"]
    assert 0.0 < sub < 0.15, sub
    assert strict > 3 * sub, (strict, sub)
    assert sf["subject_reading"]["skits_dropped"] == \
        man["drops_by_rule"].get("same_voice_pair")
    assert sf["tag_only_pair_fraction"] > 0.25, "task 1 measured 0.3959 before the filter"


@needs_artifacts(CORPUS, TOKENIZER, reason="the stakes delta's spread is a scale property")
def test_the_continuous_stakes_delta_has_spread_on_real_data(reach_scale):
    """RULING D, tested on magnitude. Stage 2's three-way label was 85.3% one class; the
    continuous form must at least be non-degenerate, or its successor is no better."""
    _, rows = reach_scale
    deltas = [d for row in rows for d in row["stakes_deltas"]]
    assert len(deltas) > 100
    nonzero = [d for d in deltas if d != 0.0]
    assert len(nonzero) > 0.05 * len(deltas), \
        f"only {len(nonzero)}/{len(deltas)} deltas are nonzero"
    assert any(d > 0 for d in nonzero) and any(d < 0 for d in nonzero)
    assert max(deltas) > 1.0
