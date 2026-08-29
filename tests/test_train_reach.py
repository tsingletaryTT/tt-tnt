# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Wiring tests for scripts/train_reach.py -- the reach dial's two paired arms.

WHAT THESE TEST, AND WHAT THEY DELIBERATELY DO NOT.
Nothing here runs a training step; that needs a device and a lease. What can fail silently
on a device is the WIRING around the step, and this task has three joins where a silent
failure would produce two healthy-looking runs that mean nothing:

  1. THE SPLIT. `artifacts/reach-skits/skits.jsonl` carries its own per-row `split` label,
     and the tercile cut points that DEFINE `near`/`mid`/`far` were fitted on the training
     split only. A trainer that held out its own 256-row tail instead would train on ~3,845
     eval-labelled rows, and task 4's dial would then measure bucket boundaries it helped
     choose. Tested against the REAL artifact, not a fixture, because the property is a
     property of that file.
  2. THE BLOCK DIFFERENCE. The arms must differ in the `reach` line and in NOTHING else --
     including not differing in whether a think-block is present at all, which is stage 2's
     experiment, not this one.
  3. THE STOCHASTIC-ROUNDING READ PATH. The guard must read the CONSTRUCTED optimizer's
     state, not the dict handed to the trainer. A trainer whose input dict says True while
     its optimizer says False is the exact shape of the bug that froze 17 gammas for 3000
     steps, and it must raise.

FIXTURES ARE BUILT SO THE ASSERTIONS CAN FAIL. Two fixtures in this project were vacuous
because one word repeated through every sentence. Here every slot of the fixture block
carries a DIFFERENT value, so a `drop_reach` that copied the five surviving values by
position instead of by name is a visible, failing difference rather than an invisible
identity. Every substantive test below was run against a plausible wrong implementation and
seen to fail before being kept; the outputs are in
`.superpowers/sdd/2026-08-23-reach-dial/task-3-report.md`.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import needs_artifacts  # noqa: E402
from scripts.derive_skits import build_skit_example  # noqa: E402
from scripts.train_reach import (ARMS, DEFAULT_DERIVE_MANIFEST, DEFAULT_SKITS,  # noqa: E402
                                 SPLIT_LABELS, TOKENIZATION_NOTE,
                                 assert_split_respected, block_for_arm,
                                 block_schema_report, build_arm_examples,
                                 drop_positions, load_reach_skits,
                                 over_length_indices, partition_by_split,
                                 read_manifest_split, skit_for_arm, split_layout,
                                 with_dial_for_arm)
from scripts.train_skits import assert_stochastic_rounding  # noqa: E402
from train.improv import render_think  # noqa: E402
from train.reach import (NODIAL_SLOT_NAMES, REACH_SLOT_NAMES, REACH_VALUES,  # noqa: E402
                         NoDialSlots, ReachSlots, drop_reach, reach_slot_names_of)
from train.skit import SKIT_ROLES, Skit  # noqa: E402

# Every slot value is DISTINCT and none is a substring of another, so a positional copy, a
# swapped pair or a dropped value all show up as a concrete difference. A fixture whose
# `accept` and `handback` were both "study" could not distinguish those cases at all.
BLOCK = ReachSlots(offer="alpha bravo", accept="charlie", reach="far", add="delta",
                   stakes="+1.5", handback="echo")

TURNS = ("turn zero words", "turn one words", "turn two words", "turn three words",
         "turn four words")


def _skit(story_id=0, *, reach="far"):
    """A three-block skit with v3 six-slot blocks. Blocks differ so an index bug shows."""
    blocks = tuple(
        ReachSlots(offer=f"offer{i}", accept=f"accept{i}", reach=reach,
                   add=f"add{i}", stakes=f"+{i}.0", handback=f"handback{i}")
        for i in range(3)
    )
    return Skit(story_id=story_id, prefix="prefix words here", turns=TURNS, blocks=blocks)


def _straddling_skit(story_id=0):
    """A skit whose two arms land on DIFFERENT padded lengths.

    The 17-word prefix is tuned, and the tuning is the point: `build_skit_example` pads to a
    32-token tile, so for most sizes the three `reach: ...` lines vanish inside the padding
    and both arms come out identical in length. At 17 prefix words the dial arm is 96 tokens
    and the control 64, which is the only shape in which a per-arm-vs-union distinction can
    be observed at all.
    """
    blocks = tuple(
        ReachSlots(offer=f"o{i}", accept=f"a{i}", reach="far", add=f"ad{i}",
                   stakes=f"+{i}.0", handback=f"h{i}")
        for i in range(3)
    )
    return Skit(story_id=story_id, prefix=" ".join(f"p{j}" for j in range(17)),
                turns=tuple(f"t{k}" for k in range(5)), blocks=blocks)


def _row_dict(story_id, split, *, reach="far"):
    d = _skit(story_id, reach=reach).as_dict()
    d["split"] = split
    return d


def _write_rows(tmp_path, spec, name="skits.jsonl"):
    """`spec` is a list of (story_id, split). Written in the order given -- order matters."""
    path = tmp_path / name
    with path.open("w") as fh:
        for sid, split in spec:
            fh.write(json.dumps(_row_dict(sid, split)) + "\n")
    return path


class _Tok:
    """Deterministic, and it honours add_special_tokens.

    Deterministic on purpose: builtins `hash()` is randomised per process, and a mock that
    ignored `add_special_tokens` let a spurious-BOS bug through every stage-1 test.
    """
    pad_token_id = 0
    BOS = 1

    def __init__(self):
        self._ids = {}

    def encode(self, s, add_special_tokens=True):
        ids = [self._ids.setdefault(w, len(self._ids) + 2) for w in s.split()]
        return ([self.BOS] + ids) if add_special_tokens else ids


# --------------------------------------------------------------------------------------
# 2. THE BLOCK DIFFERENCE: the arms differ in the `reach` line and in nothing else
# --------------------------------------------------------------------------------------
def test_arm_maps_to_with_dial_and_rejects_anything_else():
    """This boolean IS the independent variable; inverted or constant, both arms look fine."""
    assert with_dial_for_arm("dial") is True
    assert with_dial_for_arm("nodial") is False
    for bad in ("think", "nothink", "DIAL", "", None):
        with pytest.raises(ValueError):
            with_dial_for_arm(bad)
    assert ARMS == ("dial", "nodial")


def test_the_nodial_block_is_the_dial_block_minus_exactly_the_reach_line():
    """The control's entire claim, asserted on the RENDERED TEXT the model actually sees.

    Compared line by line rather than by dataclass field, because the rendered string is
    what reaches the tokenizer: a difference in whitespace, ordering or a re-labelled slot
    would all be invisible to a field-set comparison and all change the training data.
    """
    dial_lines = render_think(BLOCK).splitlines()
    nodial_lines = render_think(drop_reach(BLOCK)).splitlines()

    removed = [ln for ln in dial_lines if ln not in nodial_lines]
    assert removed == ["reach: far"], f"removed {removed}, expected only the reach line"
    # And nothing else moved: deleting that one line from the dial block must yield the
    # nodial block EXACTLY, in order.
    assert [ln for ln in dial_lines if ln != "reach: far"] == nodial_lines


def test_drop_reach_carries_the_other_five_values_verbatim_and_by_name():
    """A positional copy would survive a field reordering while re-labelling every slot."""
    got = drop_reach(BLOCK)
    assert isinstance(got, NoDialSlots)
    for name in NODIAL_SLOT_NAMES:
        assert getattr(got, name) == getattr(BLOCK, name), f"slot {name} changed"
    assert not hasattr(got, "reach")


def test_the_nodial_schema_is_the_reach_schema_minus_reach_in_the_same_order():
    """Derived by subtraction, so the two cannot drift into differing on another slot."""
    assert NODIAL_SLOT_NAMES == tuple(n for n in REACH_SLOT_NAMES if n != "reach")
    assert reach_slot_names_of(BLOCK) == REACH_SLOT_NAMES
    assert reach_slot_names_of(drop_reach(BLOCK)) == NODIAL_SLOT_NAMES


def test_reach_is_declared_before_add_in_both_the_schema_and_the_rendered_block():
    """Load-bearing: a dial declared after the word it governs can only relabel it."""
    names = list(REACH_SLOT_NAMES)
    assert names.index("reach") < names.index("add")
    body = render_think(BLOCK)
    assert body.index("reach: ") < body.index("add: ")


def test_block_for_arm_and_skit_for_arm_agree_with_with_dial_for_arm():
    assert block_for_arm(BLOCK, with_dial=True) is BLOCK
    assert block_for_arm(BLOCK, with_dial=False) == drop_reach(BLOCK)
    s = _skit(7)
    dial = skit_for_arm(s, with_dial=True)
    nodial = skit_for_arm(s, with_dial=False)
    # Text is untouched -- only the block schema changes.
    assert (dial.prefix, dial.turns, dial.story_id) == (s.prefix, s.turns, s.story_id)
    assert (nodial.prefix, nodial.turns, nodial.story_id) == (s.prefix, s.turns, s.story_id)
    assert all(isinstance(b, ReachSlots) for b in dial.blocks)
    assert all(isinstance(b, NoDialSlots) for b in nodial.blocks)


def test_block_schema_report_records_the_difference_as_exactly_reach():
    rep = block_schema_report(_skit(0))
    assert rep["dial_slots"] == list(REACH_SLOT_NAMES)
    assert rep["nodial_slots"] == list(NODIAL_SLOT_NAMES)
    assert rep["difference"] == ["reach"]
    assert rep["reach_before_add"] is True


def test_BOTH_arms_get_a_think_block_this_is_not_stage_twos_think_vs_nothink():
    """The guard against silently rebuilding stage 2's experiment under this task's name.

    `nodial` is NOT `nothink`. If `build_arm_examples` ever passed `with_think=False` for
    the control, that arm would have no think-block segments at all -- a different
    experiment, and one whose control answers a different question. Asserted by building
    the no-think example explicitly and requiring the nodial arm to differ from it.
    """
    tok, s = _Tok(), _skit(0)
    nodial = build_arm_examples([s], tok, arm="nodial", pad_token_id=0)[0]
    nothink = build_skit_example(skit_for_arm(s, with_dial=False), tok,
                                with_think=False, pad_token_id=0)
    assert nodial["input_ids"] != nothink["input_ids"], (
        "the nodial arm built the same ids as a NO-THINK-BLOCK arm; the control lost its "
        "think-block and this is stage 2's experiment, not task 3's")
    assert len(nodial["input_ids"]) > len(nothink["input_ids"])


def test_the_reach_line_really_reaches_the_dial_arms_token_stream():
    """If the extra slot line did not reach the token stream the dial cannot be learned.

    Asserted on the IDS, not on their length. The first version of this test compared
    ``len(input_ids)`` and was GREEN-BUT-VACUOUS on the small fixture: `build_skit_example`
    pads to a 32-token tile, and the three `reach: far` lines fit inside the padding, so
    both arms came out at exactly 64 tokens. A length comparison here is a test of the
    fixture's size, not of the arms. `_long_skit` below restores the length property with a
    fixture big enough to cross a tile boundary; this one asserts the property that does not
    depend on fixture size at all -- the token streams DIFFER.
    """
    tok, s = _Tok(), _skit(0)
    dial = build_arm_examples([s], tok, arm="dial", pad_token_id=0)[0]
    nodial = build_arm_examples([s], tok, arm="nodial", pad_token_id=0)[0]
    assert dial["input_ids"] != nodial["input_ids"], (
        "both arms tokenized identically -- the reach line never reached the token stream")
    assert dial["labels"] != nodial["labels"], (
        "the reach line is present but unsupervised; the dial could not be learned")
    assert len(dial["input_ids"]) == len(dial["labels"])
    assert len(nodial["input_ids"]) == len(nodial["labels"])


@needs_artifacts("artifacts/reach-skits/skits.jsonl",
                 "artifacts/hf-tt-tnt-1024",
                 reason="the length relation is a property of the real tokenizer and the "
                        "real slot values; a fixture cannot establish it")
def test_the_dial_arm_is_never_shorter_and_is_longer_in_aggregate():
    """The length relation, stated as it is actually true -- and it is NOT per-example.

    MEASURED, and the measurement corrected the claim. `build_skit_example` pads to a
    32-token tile, and the three `reach: ...` lines are only ~12 tokens, so for 227 of the
    first 400 real skits the two arms land on the SAME padded length. "The dial arm's
    examples are longer" is therefore false per-example and true in aggregate.

    That distinction is not pedantry -- it is the difference between a test that passes and
    a test that is vacuous. The first two versions of this assertion compared one fixture's
    padded lengths, and both were green-but-meaningless: the fixture's extra line fit inside
    its padding, so `assert dial > nodial` was comparing 64 to 64 and then 192 to 192.

    So: never SHORTER (a smaller dial example would mean the block lost content), and
    strictly longer summed over the sample (the tokens do reach the stream). The aggregate
    is what the manifest's truncation note rests on, and it is what makes the two arms'
    losses non-comparable.
    """
    from transformers import AutoTokenizer

    from scripts.train_reach import TOKENIZER_DIR

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    train, _ = partition_by_split(load_reach_skits(DEFAULT_SKITS))
    sample = train[:400]
    dial = build_arm_examples(sample, tok, arm="dial", pad_token_id=tok.pad_token_id or 0)
    nodial = build_arm_examples(sample, tok, arm="nodial",
                               pad_token_id=tok.pad_token_id or 0)

    dl = [len(e["input_ids"]) for e in dial]
    nl = [len(e["input_ids"]) for e in nodial]
    assert all(a >= b for a, b in zip(dl, nl)), "a dial example came out SHORTER"
    assert sum(dl) > sum(nl), (
        f"dial total {sum(dl)} !> nodial total {sum(nl)}; the reach lines never reached "
        f"the token stream")
    # And the fixture-independent property, on every real example.
    assert all(d["input_ids"] != n["input_ids"] for d, n in zip(dial, nodial))


def test_both_arms_build_the_same_examples_in_the_same_order():
    """Pairing: same count, same story_ids, same positions. Order is never sorted."""
    tok = _Tok()
    skits = [_skit(sid) for sid in (5, 3, 9, 1)]
    dial = build_arm_examples(skits, tok, arm="dial", pad_token_id=0)
    nodial = build_arm_examples(skits, tok, arm="nodial", pad_token_id=0)
    assert [e["story_id"] for e in dial] == [5, 3, 9, 1]
    assert [e["story_id"] for e in nodial] == [5, 3, 9, 1]


def test_every_example_is_tile_aligned_in_both_arms():
    """Tile-align to 32 or SDPA backward raises TT_FATAL ... u_scaler shape mismatch."""
    tok = _Tok()
    skits = [_skit(sid) for sid in range(6)]
    for arm in ARMS:
        for ex in build_arm_examples(skits, tok, arm=arm, pad_token_id=0):
            assert len(ex["input_ids"]) % 32 == 0, f"{arm}: not tile-aligned"
            assert len(ex["labels"]) == len(ex["input_ids"])


# --------------------------------------------------------------------------------------
# 1. THE SPLIT: the file's own label, never a size
# --------------------------------------------------------------------------------------
def test_load_reach_skits_preserves_file_order_and_labels(tmp_path):
    path = _write_rows(tmp_path, [(4, "train"), (1, "eval"), (7, "train")])
    rows = load_reach_skits(path)
    assert [(s.story_id, lab) for s, lab in rows] == [(4, "train"), (1, "eval"),
                                                      (7, "train")]


def test_load_reach_skits_rejects_a_row_with_no_split_label(tmp_path):
    """A missing label is the exact failure the split guard exists for; catch it on the row."""
    d = _row_dict(0, "train")
    del d["split"]
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="split"):
        load_reach_skits(path)


def test_load_reach_skits_rejects_an_unknown_split_label(tmp_path):
    path = _write_rows(tmp_path, [(0, "validation")])
    with pytest.raises(ValueError, match="split"):
        load_reach_skits(path)


def test_load_reach_skits_rejects_a_five_slot_block(tmp_path):
    """A task-1 file trained here would produce a nodial-shaped arm labelled `dial`."""
    d = _row_dict(0, "train")
    for b in d["blocks"]:
        b.pop("reach")
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="block keys"):
        load_reach_skits(path)


def test_load_reach_skits_rejects_a_reach_value_outside_the_dials_three_settings(tmp_path):
    """A fourth value is a token task 4 could never force."""
    d = _row_dict(0, "train")
    d["blocks"][1]["reach"] = "very_far"
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="reach"):
        load_reach_skits(path)


def test_load_reach_skits_rejects_roles_that_are_not_the_skit_roles(tmp_path):
    d = _row_dict(0, "train")
    d["roles"] = ["model"] * 5
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="roles"):
        load_reach_skits(path)


def test_partition_by_split_reads_the_label_and_not_a_tail(tmp_path):
    """The decisive difference from train_skits.py, on a file where the two DISAGREE.

    The eval-labelled rows here are INTERLEAVED, not a tail. A tail-slicing partition of
    any size gets this wrong, and a filter on the label gets it right -- which is why the
    fixture is deliberately not in the real artifact's layout.
    """
    path = _write_rows(tmp_path, [(0, "train"), (1, "eval"), (2, "train"), (3, "eval"),
                                  (4, "train")])
    rows = load_reach_skits(path)
    train, ev = partition_by_split(rows)
    assert [s.story_id for s in train] == [0, 2, 4]
    assert [s.story_id for s in ev] == [1, 3]


def test_split_layout_reports_a_tail_as_a_tail_and_an_interleaving_as_not(tmp_path):
    tail = load_reach_skits(_write_rows(tmp_path, [(0, "train"), (1, "train"),
                                                  (2, "eval"), (3, "eval")], "a.jsonl"))
    mixed = load_reach_skits(_write_rows(tmp_path, [(0, "train"), (1, "eval"),
                                                    (2, "train")], "b.jsonl"))
    assert split_layout(tail)["eval_is_contiguous_tail"] is True
    assert split_layout(tail)["first_eval_index"] == 2
    assert split_layout(mixed)["eval_is_contiguous_tail"] is False


def test_the_split_guard_raises_when_an_eval_labelled_row_reaches_training(tmp_path):
    """THE assertion. Simulated by handing the guard a train list that includes an eval row.

    This is what a `--val-size` tail hold-out would produce on this artifact: rows the
    derivation marked `eval` sitting in the training list, with every count still looking
    plausible.
    """
    path = _write_rows(tmp_path, [(0, "train"), (1, "eval"), (2, "eval")])
    rows = load_reach_skits(path)
    train, ev = partition_by_split(rows)
    leaky_train = train + [ev[0]]                    # exactly the leak, one row
    with pytest.raises(RuntimeError, match="NOT labelled `train`"):
        assert_split_respected(rows, leaky_train, ev[1:], manifest_split={}, arm="dial")


def test_the_split_guard_raises_when_the_counts_disagree_with_the_derive_manifest(tmp_path):
    """Catches reading the WRONG FIELD -- the failure no per-row check can see.

    A `split` read that found nothing and defaulted every row to `train` passes the
    identity check perfectly: every row really does carry the label it was given. Only the
    cross-check against the derivation's own n_train/n_eval catches it.
    """
    path = _write_rows(tmp_path, [(0, "train"), (1, "train"), (2, "eval")])
    rows = load_reach_skits(path)
    train, ev = partition_by_split(rows)
    with pytest.raises(RuntimeError, match="disagree with the derivation"):
        assert_split_respected(rows, train, ev,
                              manifest_split={"n_train": 2, "n_eval": 2}, arm="dial")


def test_the_split_guard_raises_when_the_partition_does_not_cover_the_file(tmp_path):
    path = _write_rows(tmp_path, [(0, "train"), (1, "eval"), (2, "eval")])
    rows = load_reach_skits(path)
    train, ev = partition_by_split(rows)
    with pytest.raises(RuntimeError, match="does not cover the file"):
        assert_split_respected(rows, train, ev[:1], manifest_split={}, arm="dial")


def test_the_split_guard_passes_and_reports_when_it_agrees_with_the_manifest(tmp_path):
    path = _write_rows(tmp_path, [(0, "train"), (1, "train"), (2, "eval")])
    rows = load_reach_skits(path)
    train, ev = partition_by_split(rows)
    rep = assert_split_respected(rows, train, ev,
                                 manifest_split={"n_train": 2, "n_eval": 1}, arm="dial")
    assert rep["n_train"] == 2 and rep["n_eval"] == 1
    assert rep["eval_labelled_rows_in_training"] == 0
    assert rep["checked_against_derive_manifest"] is True


def test_the_split_guard_says_so_when_no_manifest_was_available(tmp_path):
    """A skipped cross-check must be visible, not silently absent from the evidence."""
    path = _write_rows(tmp_path, [(0, "train"), (1, "eval")])
    rows = load_reach_skits(path)
    train, ev = partition_by_split(rows)
    rep = assert_split_respected(rows, train, ev, manifest_split={}, arm="dial")
    assert rep["checked_against_derive_manifest"] is False


def test_read_manifest_split_returns_empty_for_a_missing_manifest(tmp_path):
    assert read_manifest_split(tmp_path / "nope.json") == {}
    (tmp_path / "m.json").write_text(json.dumps({"split": {"n_train": 9, "n_eval": 1}}))
    assert read_manifest_split(tmp_path / "m.json") == {"n_train": 9, "n_eval": 1}


# --------------------------------------------------------------------------------------
# 3. THE STOCHASTIC-ROUNDING READ PATH
# --------------------------------------------------------------------------------------
class _FakeOptimizer:
    def __init__(self, state):
        self._state = state

    def get_state_dict(self):
        return dict(self._state)


class _FakeTrainer:
    """A trainer whose INPUT dict and whose OPTIMIZER can disagree. That is the whole point."""

    def __init__(self, *, optimizer_state, requested):
        self.optimizer = _FakeOptimizer(optimizer_state)
        self.optimizer_config = dict(requested)


def test_the_guard_raises_when_the_optimizer_ignored_a_flag_that_was_asked_for():
    """The exact bug: the request says True, the built optimizer says False.

    Checking the dict we passed in would pass this case happily, and 3000 steps of frozen
    gammas is what that costs. The guard must read the optimizer.
    """
    t = _FakeTrainer(optimizer_state={"lr": 1e-5, "stochastic_rounding": False},
                     requested={"stochastic_rounding": True})
    with pytest.raises(RuntimeError, match="stochastic_rounding is NOT enabled"):
        assert_stochastic_rounding(t, "dial")


def test_the_guard_raises_when_the_optimizer_state_omits_the_key_entirely():
    """A renamed or unsupported key leaves the request sitting there looking correct."""
    t = _FakeTrainer(optimizer_state={"lr": 1e-5},
                     requested={"stochastic_rounding": True})
    with pytest.raises(RuntimeError, match="stochastic_rounding is NOT enabled"):
        assert_stochastic_rounding(t, "nodial")


def test_the_guard_passes_only_when_the_optimizer_itself_reports_it_on():
    t = _FakeTrainer(optimizer_state={"lr": 1e-5, "stochastic_rounding": True},
                     requested={})
    assert_stochastic_rounding(t, "dial") is None


# --------------------------------------------------------------------------------------
# PROPERTIES OF SCALE -- tested against the REAL artifact, per the spec's rule 4
# --------------------------------------------------------------------------------------
@needs_artifacts("artifacts/reach-skits/skits.jsonl",
                 "artifacts/reach-skits/derive_manifest.json",
                 reason="the split IS a property of this file; a fixture cannot stand in")
def test_the_real_artifact_splits_exactly_as_its_derive_manifest_says():
    rows = load_reach_skits(DEFAULT_SKITS)
    train, ev = partition_by_split(rows)
    want = read_manifest_split(DEFAULT_DERIVE_MANIFEST)
    assert len(train) == want["n_train"] == 36913
    assert len(ev) == want["n_eval"] == 4101
    assert len(rows) == 41014


@needs_artifacts("artifacts/reach-skits/skits.jsonl",
                 "artifacts/reach-skits/derive_manifest.json")
def test_no_eval_labelled_row_enters_training_on_the_real_artifact():
    """The guard, run on the file the arms actually train on."""
    rows = load_reach_skits(DEFAULT_SKITS)
    train, ev = partition_by_split(rows)
    rep = assert_split_respected(rows, train, ev,
                                 manifest_split=read_manifest_split(
                                     DEFAULT_DERIVE_MANIFEST),
                                 arm="dial")
    assert rep["eval_labelled_rows_in_training"] == 0
    assert rep["checked_against_derive_manifest"] is True
    labels = {id(s): lab for s, lab in rows}
    assert all(labels[id(s)] == "train" for s in train)


@needs_artifacts("artifacts/reach-skits/skits.jsonl")
def test_the_real_artifact_uses_all_three_dial_settings_in_both_splits():
    """A dial with a dead setting is not a dial. Bucket balance is a property of scale."""
    rows = load_reach_skits(DEFAULT_SKITS)
    train, ev = partition_by_split(rows)
    for name, group in (("train", train), ("eval", ev)):
        seen = {b.reach for s in group for b in s.blocks}
        assert seen == set(REACH_VALUES), f"{name} split uses only {seen}"


@needs_artifacts("artifacts/reach-skits/skits.jsonl")
def test_on_the_real_artifact_the_two_arms_differ_only_by_the_reach_line():
    """Rendered-text diff over real blocks, not a fixture: 300 real skits, every block."""
    rows = load_reach_skits(DEFAULT_SKITS)
    for skit, _ in rows[:300]:
        for b in skit.blocks:
            dial = render_think(b).splitlines()
            nodial = render_think(drop_reach(b)).splitlines()
            assert [ln for ln in dial if not ln.startswith("reach: ")] == nodial
            assert len(dial) - len(nodial) == 1


@needs_artifacts("artifacts/reach-skits/skits.jsonl")
def test_the_default_skits_file_is_the_full_corpus_reach_derivation():
    """Pin the default. Pointing this at task 1's file trains a dial with no dial in it."""
    assert DEFAULT_SKITS.name == "skits.jsonl"
    assert DEFAULT_SKITS.parent.name == "reach-skits"
    first = json.loads(DEFAULT_SKITS.open().readline())
    assert tuple(first["blocks"][0]) == REACH_SLOT_NAMES
    assert first["split"] in SPLIT_LABELS


def test_the_tokenization_note_records_the_space_prefixed_think_boundary():
    """Task 4 builds prompts from this note; a note that lost the space is a broken prompt."""
    assert "SEGMENT-WISE" in TOKENIZATION_NOTE
    assert "Ġ<" in TOKENIZATION_NOTE          # the space-prefixed '<', measured
    assert "think" in TOKENIZATION_NOTE


# --------------------------------------------------------------------------------------
# RULING C, re-applied: the derivation's length gate measured the pre-with_reach block
# --------------------------------------------------------------------------------------
def test_over_length_indices_takes_the_UNION_across_the_two_arms():
    """The union is what keeps the arms paired; a per-arm drop unpairs them silently.

    Built so the two arms DISAGREE: the fixture's example is over the cap in the dial arm
    and under it in the nodial arm, which is the real artifact's situation (526 vs 180). A
    per-arm rule keeps this row in the nodial arm and drops it from the dial arm, leaving
    the two arms with different lengths -- hence different dataloader permutations, hence no
    pairing. The union must drop it from both.
    """
    tok, s = _Tok(), _straddling_skit(0)
    dial_len = len(build_arm_examples([s], tok, arm="dial", pad_token_id=0)[0]["input_ids"])
    nodial_len = len(build_arm_examples([s], tok, arm="nodial",
                                       pad_token_id=0)[0]["input_ids"])
    # The gap is the whole point of the fixture, so it is ASSERTED, not skipped past. A tie
    # here would make the cap below meaningless and the test vacuous -- which is exactly
    # what happened with the first fixture, whose 17-word prefix was tuned to produce this
    # gap precisely because the untuned one tied at 64 == 64.
    assert dial_len > nodial_len, (
        f"fixture must straddle a tile boundary: dial {dial_len}, nodial {nodial_len}")
    cap = nodial_len          # over in the dial arm, exactly at the cap in the control
    assert over_length_indices([s], tok, pad_token_id=0, max_seq_len=cap) == [0]


def test_over_length_indices_is_empty_when_everything_fits():
    tok = _Tok()
    skits = [_skit(i) for i in range(4)]
    assert over_length_indices(skits, tok, pad_token_id=0, max_seq_len=100_000) == []


def test_over_length_indices_flags_every_row_when_nothing_fits():
    tok = _Tok()
    skits = [_skit(i) for i in range(4)]
    assert over_length_indices(skits, tok, pad_token_id=0, max_seq_len=1) == [0, 1, 2, 3]


def test_drop_positions_removes_exactly_those_positions_and_keeps_order():
    skits = [_skit(i) for i in range(5)]
    assert [s.story_id for s in drop_positions(skits, [1, 3])] == [0, 2, 4]
    assert [s.story_id for s in drop_positions(skits, [])] == [0, 1, 2, 3, 4]
    # Out-of-range positions are inert rather than an error -- the caller's positions come
    # from over_length_indices over the same list, so a stray index means a bug elsewhere
    # and must not silently shorten the data.
    assert len(drop_positions(skits, [99])) == 5


@needs_artifacts("artifacts/reach-skits/skits.jsonl", "artifacts/hf-tt-tnt-1024")
def test_the_derivations_length_gate_really_did_miss_rows_on_the_real_artifact():
    """The defect this re-application exists for, asserted on the real file.

    The derive manifest claims `over_max_seq_len: 0`. If that were true of what trains, this
    test would find no over-length rows and the drop would be dead code. It finds them,
    and it finds MORE in the dial arm than the control -- which is the part that makes
    truncation a confound rather than only a loss.
    """
    from transformers import AutoTokenizer

    from scripts.train_reach import MAX_SEQ_LEN, TOKENIZER_DIR

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad = tok.pad_token_id or 0
    train, _ = partition_by_split(load_reach_skits(DEFAULT_SKITS))
    sample = train[:4000]

    def n_over(arm):
        return sum(1 for e in build_arm_examples(sample, tok, arm=arm, pad_token_id=pad)
                   if len(e["input_ids"]) > MAX_SEQ_LEN)

    dial_over, nodial_over = n_over("dial"), n_over("nodial")
    assert dial_over > 0, ("no over-length rows found; if the derivation's gate were "
                           "correct the ruling-C re-application would be dead code")
    assert dial_over > nodial_over, (
        f"dial {dial_over} !> nodial {nodial_over}; the asymmetry that makes silent "
        f"truncation a confound between the arms is absent")
    # And the union drops at least as many as the worse arm, from both arms alike.
    union = over_length_indices(sample, tok, pad_token_id=pad, max_seq_len=MAX_SEQ_LEN)
    assert len(union) >= dial_over
