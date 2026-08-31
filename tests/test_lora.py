# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""LoRA merge and freeze checks. Pure numpy — no ttml, no ttnn, no device."""
from __future__ import annotations

import numpy as np
import pytest

from train.lora import (
    assert_adapter_moved,
    assert_base_frozen,
    base_parameter_snapshot,
    lora_scaling,
    merge_lora_state,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _state(*, rank=4, in_f=8, out_f=6, seed=0, zero_b=False, leading=True):
    """A one-module checkpoint shaped the way SFTTrainer writes them."""
    rng = np.random.default_rng(seed)
    stem = "llama/llama_block_0/attention/q_linear"
    w = rng.normal(size=(out_f, in_f)).astype(np.float32)
    a = rng.normal(size=(rank, in_f)).astype(np.float32)
    b = (np.zeros((out_f, rank)) if zero_b
         else rng.normal(size=(out_f, rank))).astype(np.float32)
    shape = (lambda x: x[None, None]) if leading else (lambda x: x)
    return stem, {
        f"{stem}/weight": shape(w),
        f"{stem}/lora_A": shape(a),
        f"{stem}/lora_B": shape(b),
        "llama/fc/weight": shape(rng.normal(size=(3, in_f)).astype(np.float32)),
    }


def test_scaling_is_alpha_over_rank_and_is_not_hardcoded():
    assert lora_scaling(rank=8, alpha=16.0) == 2.0
    assert lora_scaling(rank=16, alpha=16.0) == 1.0          # changing rank changes it
    assert lora_scaling(rank=4, alpha=16.0, use_rslora=True) == 8.0
    with pytest.raises(ValueError):
        lora_scaling(rank=0, alpha=16.0)


def test_a_zero_adapter_merges_to_a_bit_identical_model():
    """lora_B initialises to zeros, so an untrained adapter must be a no-op.

    This is the correctness end of the merge: if merging a zero adapter changed the weights,
    the merge would be corrupting every model it touched.
    """
    stem, st = _state(zero_b=True)
    merged, report = merge_lora_state(st, rank=4, alpha=16.0)
    np.testing.assert_array_equal(merged[f"{stem}/weight"], st[f"{stem}/weight"])
    assert report["merged_pairs"] == 1


def test_a_trained_adapter_actually_changes_the_weights():
    """The non-vacuity end: without this, the test above passes against a merge that does
    nothing at all."""
    stem, st = _state(zero_b=False)
    merged, _ = merge_lora_state(st, rank=4, alpha=16.0)
    assert not np.array_equal(merged[f"{stem}/weight"], st[f"{stem}/weight"])


def test_merged_weight_equals_the_lora_forward_pass():
    """W + scaling*(B@A) must reproduce what LoraLinear.forward computes, on real inputs.

    linear(x, M) = x @ M.T, so the LoRA branch is (x @ A.T) @ B.T * scaling. Asserting on the
    OUTPUT rather than on the weight arithmetic is what makes this a test of the claim rather
    than a restatement of the implementation.
    """
    stem, st = _state()
    rank, alpha = 4, 16.0
    w = st[f"{stem}/weight"][0, 0]
    a = st[f"{stem}/lora_A"][0, 0]
    b = st[f"{stem}/lora_B"][0, 0]
    x = np.random.default_rng(7).normal(size=(5, w.shape[1])).astype(np.float32)

    lora_out = x @ w.T + ((x @ a.T) @ b.T) * (alpha / rank)
    merged, _ = merge_lora_state(st, rank=rank, alpha=alpha)
    merged_out = x @ merged[f"{stem}/weight"][0, 0].T

    np.testing.assert_allclose(merged_out, lora_out, rtol=1e-5, atol=1e-5)


def test_merge_drops_the_adapter_tensors_and_keeps_everything_else():
    stem, st = _state()
    merged, _ = merge_lora_state(st, rank=4, alpha=16.0)
    assert f"{stem}/lora_A" not in merged and f"{stem}/lora_B" not in merged
    assert set(merged) == {f"{stem}/weight", "llama/fc/weight"}
    np.testing.assert_array_equal(merged["llama/fc/weight"], st["llama/fc/weight"])


def test_merge_preserves_the_base_weight_shape_including_leading_axes():
    """convert.hf_mapping squeezes leading axes itself; changing them here would be a second,
    invisible edit to every tensor."""
    for leading in (True, False):
        stem, st = _state(leading=leading)
        merged, _ = merge_lora_state(st, rank=4, alpha=16.0)
        assert merged[f"{stem}/weight"].shape == st[f"{stem}/weight"].shape


def test_merging_a_checkpoint_with_no_adapter_raises():
    """A full-parameter checkpoint passed to the merge is a wrong-file mistake, not a no-op."""
    _, st = _state()
    plain = {k: v for k, v in st.items() if "lora_" not in k}
    with pytest.raises(ValueError, match="no lora_A/lora_B pairs"):
        merge_lora_state(plain, rank=4, alpha=16.0)


def test_an_adapter_with_no_base_weight_raises():
    stem, st = _state()
    del st[f"{stem}/weight"]
    with pytest.raises(ValueError, match="no base weight"):
        merge_lora_state(st, rank=4, alpha=16.0)


def test_a_mismatched_rank_between_A_and_B_raises():
    stem, st = _state()
    st[f"{stem}/lora_B"] = st[f"{stem}/lora_B"][:, :, :, :2]
    with pytest.raises(ValueError, match="do not compose"):
        merge_lora_state(st, rank=4, alpha=16.0)


# --- the freeze check -------------------------------------------------------
#
# These fakes do two jobs. They keep the tests off the hardware -- `read_native` does
# `import ttml`, and importing ttml OPENS THE DEVICE, so a test that let it through would
# touch /dev/tenstorrent with no gozer lease. And they reproduce AutocastTensor's lazy
# FULL-precision cache, so a `read_native` that forgot `precision=NATIVE` is caught here
# rather than in a training run. A fake that ignored `precision` would pass either way,
# which is the hollow-test shape this project keeps rediscovering.


@pytest.fixture
def fake_ttml(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("ttml")
    mod.autograd = types.SimpleNamespace(
        PreferredPrecision=types.SimpleNamespace(NATIVE="NATIVE", FULL="FULL")
    )
    monkeypatch.setitem(sys.modules, "ttml", mod)
    return mod


class _CachingTensor:
    """A bf16 parameter with AutocastTensor's cache: NATIVE sees live, FULL sees the first read."""

    def __init__(self, values):
        self._live = np.asarray(values, dtype=np.float32)
        self._cache = None

    def mutate(self, values):
        self._live = np.asarray(values, dtype=np.float32)

    def to_numpy(self, precision=None):
        if precision == "NATIVE":
            return self._live.copy()
        if self._cache is None:
            self._cache = self._live.copy()
        return self._cache.copy()


class _P:
    def __init__(self, values):
        self.tensor = _CachingTensor(values)


class _Model:
    def __init__(self, params):
        self._p = params

    def parameters(self):
        return self._p


def _model(base, lora_b):
    return _Model({
        "llama/fc/weight": _P(base),
        "llama/llama_block_0/attention/q_linear/lora_B": _P(lora_b),
    })


def test_freeze_check_passes_when_base_parameters_are_untouched(fake_ttml):
    m = _model([1.0, 2.0], [0.0])
    before = base_parameter_snapshot(m)
    assert assert_base_frozen(before, m)["moved"] == 0


def test_freeze_check_fails_on_the_smallest_representable_movement(fake_ttml):
    """Exact equality, deliberately: a base parameter that moved at all did not stay frozen,
    and LoRA's anti-forgetting claim rests entirely on this check."""
    m = _model([1.0, 2.0], [0.0])
    before = base_parameter_snapshot(m)
    one_ulp = np.nextafter(np.float32(2.0), np.float32(3.0))
    assert one_ulp != np.float32(2.0)          # the premise, asserted before it is relied on
    m.parameters()["llama/fc/weight"].tensor.mutate([1.0, one_ulp])
    with pytest.raises(RuntimeError, match="freeze did not hold"):
        assert_base_frozen(before, m)


def test_freeze_check_reads_native_so_a_cached_view_cannot_hide_a_thawed_weight(fake_ttml):
    """The failure this guards is not a moving weight -- it is a moving weight that READS as
    frozen. Poison the FULL cache first, exactly as a careless baseline read would, then move
    the parameter. An implementation reading FULL is handed the pre-move copy and reports a
    perfect freeze; reading NATIVE sees the move. This is the bug that made this project
    wrongly conclude LoRA was blocked upstream."""
    m = _model([1.0, 2.0], [0.0])
    t = m.parameters()["llama/fc/weight"].tensor
    t.to_numpy()                                # populate the FULL cache with the old value
    before = base_parameter_snapshot(m)
    t.mutate([1.0, 99.0])
    assert t.to_numpy()[1] == 2.0               # a FULL read still says 2.0 -- the trap
    with pytest.raises(RuntimeError, match="freeze did not hold"):
        assert_base_frozen(before, m)


def test_freeze_check_ignores_adapter_tensors(fake_ttml):
    """lora_B is expected to move; counting it would make the check fail on every good run."""
    m = _model([1.0, 2.0], [0.5])
    before = base_parameter_snapshot(m)
    assert before.keys() == {"llama/fc/weight"}
    assert assert_base_frozen(before, m)["frozen_checked"] == 1


def test_adapter_movement_check_fails_on_an_untrained_adapter(fake_ttml):
    m = _model([1.0, 2.0], [0.0, 0.0])
    with pytest.raises(RuntimeError, match="still exactly zero"):
        assert_adapter_moved(m)


def test_adapter_movement_check_passes_once_lora_b_moves(fake_ttml):
    m = _model([1.0, 2.0], [0.0, 0.25])
    report = assert_adapter_moved(m)
    assert report["lora_B_moved"] == 1 and report["max_abs"] == 0.25


def test_adapter_movement_check_raises_when_injection_did_not_happen(fake_ttml):
    with pytest.raises(RuntimeError, match="injection did not happen"):
        assert_adapter_moved(_Model({"llama/fc/weight": _P([1.0])}))


def test_importing_train_lora_does_not_import_ttml_or_ttnn():
    """`import ttml` opens the device. This module is imported by CPU-only tooling (the merge
    runs on the host), so its ttml imports must stay inside the functions that need one."""
    import subprocess
    import sys

    code = ("import sys; import train.lora; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('ttml','ttnn')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"train.lora pulled in {out.stdout.strip()}"
