# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Wiring tests for scripts/train_skits.py.

WHAT THESE TEST, AND WHAT THEY DELIBERATELY DO NOT.
Nothing here runs a training step -- that needs a device and a lease. What can fail
silently on a device, and did in stage 1, is the WIRING around the step: an optimizer that
ignored `stochastic_rounding`, an eval that was never enabled, a curve that recorded two
endpoints, an arm flag that was inverted, a skits file that got reordered so the two arms
were no longer paired. Every test below asserts on one of those joins, not on the
arithmetic underneath it, and every one of them was run against a plausible wrong
implementation and seen to fail before being kept (see task-4-report.md for the outputs).

The fixture story is NOT the dialogue-with-attribution shape ('"..." said X.'): that is
train/improv.py's documented over-split (train/skit.py:31-38) and `derive_skit` genuinely
returns None on it, so a fixture in that shape would produce no Skit and test nothing.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import needs_artifacts  # noqa: E402
from scripts.train_skits import (DEFAULT_SKITS, EXPECTED_GAMMAS, LossRecorder,  # noqa: E402
                                 assert_eval_wired, assert_gammas_moved,
                                 assert_stochastic_rounding, build_arm_examples,
                                 compare_gammas, length_stats, load_skits,
                                 read_arm_gammas, with_think_for_arm)
from train.improv import render_think  # noqa: E402
from train.skit import MODEL_TURNS, derive_skit  # noqa: E402

STORY = ("Lily found a shiny rock. She showed it to her friend. "
         "The rock sparkled on the windowsill. "
         "Her friend loved the windowsill shine. "
         "The shine made a rainbow appear. "
         "They admired the rainbow glow. "
         "The glow lasted through the evening.")

_IDS = {}


class _Tok:
    """Faithful, deterministic, and honours add_special_tokens.

    Deterministic on purpose: builtins hash() is randomised per process, and a mock that
    ignored add_special_tokens let a spurious-BOS bug through every stage-1 test.
    """
    pad_token_id = 0
    BOS = 1

    def encode(self, s, add_special_tokens=True):
        ids = [_IDS.setdefault(w, len(_IDS) + 2) for w in s.split()]
        return ([self.BOS] + ids) if add_special_tokens else ids


def _skit(story_id=0):
    s = derive_skit(STORY, story_id=story_id, idf={"windowsill": 1.0, "rainbow": 1.0},
                    intensity=lambda t: 0.0)
    assert s is not None, "fixture story must derive, or these tests test nothing"
    return s


def _write_skits(tmp_path, story_ids):
    path = tmp_path / "skits.jsonl"
    with path.open("w") as fh:
        for sid in story_ids:
            fh.write(json.dumps(_skit(sid).as_dict()) + "\n")
    return path


# --------------------------------------------------------------------------------------
# Loading: fidelity and, above all, ORDER
# --------------------------------------------------------------------------------------
def test_load_skits_round_trips_every_field_including_the_blocks():
    """A Skit that loses its blocks still trains -- on an arm with no think-blocks at all.

    That is the failure this guards: `blocks` is the only field distinguishing the two
    arms, and a loader that dropped or emptied it would turn the think arm into a second
    no-think arm while every count, length and loss stayed plausible.
    """
    original = _skit(42)
    tmp = Path(__file__).parent / "_tmp_round_trip.jsonl"
    try:
        tmp.write_text(json.dumps(original.as_dict()) + "\n")
        (loaded,) = load_skits(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    assert loaded == original, "round trip must reproduce the Skit exactly"
    assert len(loaded.blocks) == len(MODEL_TURNS)
    assert loaded.blocks[0].add == original.blocks[0].add


def test_load_skits_preserves_file_order(tmp_path):
    """PAIRING DEPENDS ON THIS. The dataloader permutes POSITIONS, not story ids, so two
    arms that loaded the same file in different orders would train on different examples
    at every step while reporting identical counts -- an unpaired comparison that looks
    exactly like a paired one."""
    path = _write_skits(tmp_path, [7, 3, 9, 1])
    assert [s.story_id for s in load_skits(path)] == [7, 3, 9, 1]


def test_load_skits_rejects_a_file_whose_roles_are_not_the_skit_roles(tmp_path):
    """Roles decide which turns are supervised. A file that disagrees is not a skits file,
    and training on it would supervise partner turns with no other symptom."""
    rec = _skit(0).as_dict()
    rec["roles"] = ["model", "model", "model", "partner", "model"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="roles"):
        load_skits(path)


# --------------------------------------------------------------------------------------
# The independent variable
# --------------------------------------------------------------------------------------
def test_arm_maps_to_with_think_and_rejects_anything_else():
    assert with_think_for_arm("think") is True
    assert with_think_for_arm("nothink") is False
    with pytest.raises(ValueError):
        with_think_for_arm("Think")


def test_think_blocks_reach_only_the_think_arm(tmp_path):
    """The whole experiment is this one difference. An inverted or constant `with_think`
    produces two runs that train cleanly, converge, and mean nothing."""
    skits = load_skits(_write_skits(tmp_path, [0]))
    tok = _Tok()
    with_t = build_arm_examples(skits, tok, arm="think", pad_token_id=0)[0]
    without = build_arm_examples(skits, tok, arm="nothink", pad_token_id=0)[0]
    block_ids = tok.encode(render_think(skits[0].blocks[0]), add_special_tokens=False)

    def contains(hay, needle):
        return any(list(hay[i:i + len(needle)]) == list(needle)
                   for i in range(len(hay) - len(needle) + 1))

    assert contains(with_t["input_ids"], block_ids), "think arm must carry the block"
    assert not contains(without["input_ids"], block_ids), "no-think arm must not"


def test_both_arms_build_the_same_examples_in_the_same_order(tmp_path):
    """The paired-comparison invariant, asserted on the wiring rather than assumed from
    the fact that one function built both lists."""
    skits = load_skits(_write_skits(tmp_path, [5, 2, 8]))
    tok = _Tok()
    think = build_arm_examples(skits, tok, arm="think", pad_token_id=0)
    nothink = build_arm_examples(skits, tok, arm="nothink", pad_token_id=0)
    assert [e["story_id"] for e in think] == [5, 2, 8]
    assert [e["story_id"] for e in nothink] == [5, 2, 8]
    assert len(think) == len(nothink) == len(skits), (
        "an arm that dropped examples would give the two arms different dataset "
        "lengths, and InMemoryDataloader's shuffle is a function of that length -- the "
        "arms would silently visit different examples at every step")


# --------------------------------------------------------------------------------------
# Truncation accounting
# --------------------------------------------------------------------------------------
def test_length_stats_counts_examples_the_collate_function_would_truncate():
    """`sft_collate_fn` slices to max_seq_len silently. The think arm's examples are the
    longer of the two by construction, so truncation is a systematic between-arm
    difference that is not the think-block -- it has to be counted, not assumed zero."""
    examples = [{"input_ids": [0] * 32, "labels": [-100] * 32},
                {"input_ids": [0] * 544, "labels": [-100] * 544}]
    stats = length_stats(examples, max_seq_len=512)
    assert stats["n"] == 2
    assert stats["max"] == 544
    assert stats["over_max_seq_len"] == 1


# --------------------------------------------------------------------------------------
# Guard 1: stochastic rounding, read back off the optimizer
# --------------------------------------------------------------------------------------
class _FakeOptimizer:
    def __init__(self, state):
        self._state = state

    def get_state_dict(self):
        return dict(self._state)


class _FakeTrainerConfig:
    def __init__(self, eval_interval=0):
        self.eval_interval = eval_interval
        # The DECOY. This is what a guard that checked "the config we passed in" would
        # read, and it says exactly what we asked for -- which is why reading it proves
        # nothing about what the optimizer factory did with the request.
        self.optimizer = {"type": "AdamW", "stochastic_rounding": True}


class _FakeTrainer:
    def __init__(self, *, optimizer_state, eval_interval=0, eval_dataloader=None):
        self.optimizer = _FakeOptimizer(optimizer_state)
        self.config = _FakeTrainerConfig(eval_interval)
        self.eval_dataloader = eval_dataloader


def test_stochastic_rounding_guard_raises_when_the_optimizer_ignored_the_flag():
    """THE stage-1 bug. The optimizer was asked for stochastic rounding and silently did
    not have it; 17 gammas then sat bit-identical for 3000 steps in both arms.

    The fake reports `stochastic_rounding` absent from the optimizer's own state while its
    config still carries the request, so a guard that inspected the passed-in dict would
    pass here. It must RAISE, not warn: stage 1 had a warning for this exact condition and
    two full arms went past it.
    """
    trainer = _FakeTrainer(optimizer_state={"type": "AdamW", "lr": 1e-5})
    assert trainer.config.optimizer["stochastic_rounding"] is True, (
        "the decoy must claim the flag is on, or this test cannot discriminate")
    with pytest.raises(RuntimeError, match="stochastic_rounding is NOT enabled"):
        assert_stochastic_rounding(trainer, "think")


def test_stochastic_rounding_guard_passes_when_the_optimizer_really_has_it():
    trainer = _FakeTrainer(optimizer_state={"stochastic_rounding": True})
    assert_stochastic_rounding(trainer, "think")   # must not raise


def test_stochastic_rounding_guard_raises_on_an_explicit_false():
    trainer = _FakeTrainer(optimizer_state={"stochastic_rounding": False})
    with pytest.raises(RuntimeError):
        assert_stochastic_rounding(trainer, "nothink")


# --------------------------------------------------------------------------------------
# Guard: the eval is actually wired
# --------------------------------------------------------------------------------------
def test_eval_guard_raises_when_a_val_split_exists_but_eval_is_disabled():
    """Two independent ways to lose the whole validation curve silently: no eval
    dataloader, or eval_interval=0. Both are checked off the constructed trainer."""
    with pytest.raises(RuntimeError, match="never evaluate"):
        assert_eval_wired(_FakeTrainer(optimizer_state={}, eval_interval=0,
                                       eval_dataloader=object()),
                          val_size=256, arm="think")
    with pytest.raises(RuntimeError, match="never evaluate"):
        assert_eval_wired(_FakeTrainer(optimizer_state={}, eval_interval=250,
                                       eval_dataloader=None),
                          val_size=256, arm="think")


def test_eval_guard_passes_when_wired_and_is_silent_when_no_split_was_asked_for():
    assert_eval_wired(_FakeTrainer(optimizer_state={}, eval_interval=250,
                                   eval_dataloader=object()), val_size=256, arm="think")
    # val_size == 0 is a deliberate choice, not a misconfiguration.
    assert_eval_wired(_FakeTrainer(optimizer_state={}, eval_interval=0,
                                   eval_dataloader=None), val_size=0, arm="think")


# --------------------------------------------------------------------------------------
# Guard 2: the gammas moved
# --------------------------------------------------------------------------------------
def _gammas(values, n=EXPECTED_GAMMAS):
    return {f"llama/llama_block_{i}/attention_norm/gamma":
            np.array(values, dtype=np.float32) for i in range(n)}


def test_compare_gammas_reports_a_bit_identical_set_as_frozen():
    base = _gammas([1.0, 1.0, 1.0])
    report = compare_gammas(base, {k: v.copy() for k, v in base.items()})
    assert report["all_moved"] is False
    assert len(report["frozen"]) == EXPECTED_GAMMAS
    assert report["total_changed"] == 0
    with pytest.raises(RuntimeError, match="BIT-IDENTICAL"):
        assert_gammas_moved(report, arm="think", base_path=Path("base"),
                            arm_path=Path("arm"))


def test_compare_gammas_detects_a_single_ulp_move():
    """EXACT comparison, not a tolerance. A single bfloat16 ulp is the SMALLEST real
    movement this parameter can make, and it is exactly what stochastic rounding exists to
    let accumulate -- a tolerant comparison would score it as no movement and so would
    stamp a working run void, or (worse, in the other direction) let a frozen run pass on
    float noise."""
    base = _gammas([1.0, 1.0, 1.0])
    arm = {k: v.copy() for k, v in base.items()}
    # The next representable float32 above 1.0 -- one ulp, nothing smaller exists.
    one_ulp_up = np.nextafter(np.float32(1.0), np.float32(2.0))
    assert one_ulp_up != np.float32(1.0)
    for k in arm:
        arm[k][0] = one_ulp_up
    report = compare_gammas(base, arm)
    assert report["all_moved"] is True, "a one-ulp change must count as movement"
    assert report["total_changed"] == EXPECTED_GAMMAS
    assert 0 < report["per_tensor"][sorted(arm)[0]]["max_abs_delta"] < 1e-6
    assert_gammas_moved(report, arm="think", base_path=Path("b"), arm_path=Path("a"))


def test_compare_gammas_flags_one_frozen_tensor_among_many_that_moved():
    """Partial freezing is the shape that a summary statistic hides: 16 of 17 tensors
    moving gives a healthy-looking aggregate delta while one norm is dead."""
    base = _gammas([1.0, 1.0, 1.0])
    arm = {k: v.copy() for k, v in base.items()}
    for k in sorted(arm)[1:]:
        arm[k][0] += 0.01
    report = compare_gammas(base, arm)
    assert report["all_moved"] is False
    assert report["frozen"] == [sorted(arm)[0]]
    with pytest.raises(RuntimeError, match="BIT-IDENTICAL"):
        assert_gammas_moved(report, arm="nothink", base_path=Path("b"),
                            arm_path=Path("a"))


def test_compare_gammas_refuses_to_compare_an_intersection():
    """An arm checkpoint missing a gamma must fail loudly. Comparing only shared names
    would report '0 frozen of 0' for a checkpoint with no gammas at all."""
    base = _gammas([1.0])
    arm = {k: v.copy() for k, v in base.items()}
    arm.pop(sorted(arm)[0])
    with pytest.raises(ValueError, match="name sets differ"):
        compare_gammas(base, arm)


def test_compare_gammas_requires_all_seventeen():
    base = _gammas([1.0], n=EXPECTED_GAMMAS - 1)
    with pytest.raises(ValueError, match=f"expected {EXPECTED_GAMMAS} gamma"):
        compare_gammas(base, {k: v.copy() for k, v in base.items()})


def test_read_arm_gammas_rejects_a_file_that_is_not_an_sft_checkpoint(tmp_path):
    """The warm-start base and an SFTTrainer checkpoint are DIFFERENT formats (a ttml
    record stream vs a plain `{step, model_state}` pickle). Handing one reader the other
    file must fail rather than return an empty dict that compares as 'no gammas'."""
    path = tmp_path / "not_a_ckpt.pkl"
    path.write_bytes(pickle.dumps({"header": {}, "manifest": {}}))
    with pytest.raises(ValueError, match="model_state"):
        read_arm_gammas(path)


# --------------------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------------------
def test_loss_recorder_writes_every_step_not_a_sample(tmp_path):
    """Two endpoints cannot distinguish a fast early collapse (template memorisation) from
    a smooth decline (learning), and neither can a curve sampled every 100 steps -- the
    distinction is sharpest in exactly the early region a sparse sample skips."""
    curve = tmp_path / "loss_curve.jsonl"
    rec = LossRecorder(curve)
    for step in range(1, 8):
        rec.on_step_end(None, step, 5.0 - 0.1 * step, 1e-5)
    rec.close()
    rows = [json.loads(l) for l in curve.read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 8)), (
        "every step must appear; a sampled curve is not a trajectory")
    assert all(r["split"] == "train" for r in rows)
    assert all(r["lr"] == 1e-5 for r in rows)
    assert rows[0]["loss"] == pytest.approx(4.9)


def test_loss_recorder_flushes_so_a_killed_run_still_has_its_curve(tmp_path):
    curve = tmp_path / "loss_curve.jsonl"
    rec = LossRecorder(curve)
    rec.on_step_end(None, 1, 3.0, 1e-5)
    # deliberately NOT closed -- this is the mid-run-kill case
    assert curve.read_text().strip(), "the curve must be on disk before close()"


def test_loss_recorder_keeps_train_and_val_points_distinguishable(tmp_path):
    curve = tmp_path / "loss_curve.jsonl"
    rec = LossRecorder(curve)
    rec.on_step_end(None, 1, 3.0, 1e-5)
    rec.on_eval_end(None, 1, 2.5)
    rec.close()
    rows = [json.loads(l) for l in curve.read_text().splitlines()]
    assert [r["split"] for r in rows] == ["train", "val"]
    assert rec.eval_history == [(1, 2.5)]


# --------------------------------------------------------------------------------------
# The dataset choice (plan Ruling 10)
# --------------------------------------------------------------------------------------
def test_the_default_skits_file_is_the_200k_derivation():
    """`artifacts/skits/skits.jsonl` holds 1,921 skits -- a tenth of stage 1's 18,791
    training examples, from a 20,000-story derivation with a 90.4% drop rate. Training an
    arm on it would make the two stages incomparable on data volume. The default must be
    the 200,000-story derivation."""
    assert DEFAULT_SKITS.parent.name == "skits-200k", (
        f"default skits file is {DEFAULT_SKITS}, not the 200k derivation")


@needs_artifacts("artifacts/skits-200k/skits.jsonl",
                 reason="the real derivation is the subject: its SIZE is what is asserted")
def test_the_default_skits_file_really_holds_the_larger_derivation():
    with DEFAULT_SKITS.open() as fh:
        n = sum(1 for line in fh if line.strip())
    assert n > 15_000, (
        f"{DEFAULT_SKITS} holds {n} skits; stage 1 trained on 18,791 examples and an arm "
        f"trained on an order of magnitude less is not comparable to it")
