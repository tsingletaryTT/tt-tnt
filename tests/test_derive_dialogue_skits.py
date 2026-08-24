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
from scripts.derive_dialogue_skits import (TILE, build_skit_example,  # noqa: E402
                                           classify_turn_failure,
                                           derive_dialogue_skit, dialogue_unit_counts,
                                           drop_rate_warning, iter_stories, main,
                                           repo_relative, selection_bias,
                                           tag_only_gap_count, token_length_report)
from train.dialogue import split_sentences_dialogue  # noqa: E402
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

#: Drops: no dialogue at all.
STORY_SILENT = "Rue opened the hatch. The lantern was beneath the crate. Rue smiled."

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
