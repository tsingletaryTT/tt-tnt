# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The paired context-use comparison.

These tests are built around the two errors this project has actually made with paired
designs: pooling observations that should stay paired, and comparing a paired effect against
an unpaired spread. Every assertion below is checked to fail against the wrong implementation
as well as pass against the right one -- a test that has never been seen to fail is a claim,
not a check.
"""
import math

import numpy as np
import pytest

from scripts.compare_context_use import (
    EARLY_BAND,
    LATE_BAND,
    delta_late,
    paired_stats,
    sign_test_p,
)


def _windows(early: float, late: float, n=8, seq_len=512, jitter=0.0, seed=0):
    """A (n, seq_len) loss array with a known early band and late band value."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, seq_len))
    a[:, EARLY_BAND[0]:EARLY_BAND[1]] = early + jitter * rng.standard_normal((n, 1))
    a[:, LATE_BAND[0]:LATE_BAND[1]] = late + jitter * rng.standard_normal((n, 1))
    return a


def test_delta_late_is_the_early_minus_late_drop():
    d = delta_late(_windows(early=3.0, late=2.5))
    assert np.allclose(d, 0.5), "delta_late must be CE(early) - CE(late)"


def test_delta_late_is_negative_when_loss_rises_with_position():
    """A model that gets WORSE deeper into the window must report a negative delta, not an
    absolute value -- the sign is the entire finding."""
    assert np.all(delta_late(_windows(early=2.5, late=3.0)) < 0)


def test_delta_late_returns_one_number_per_window_not_a_pooled_scalar():
    """THE MUTATION: collapsing to a grand mean destroys the pairing the comparison rests on.
    Pooling has silently rewritten a published claim in this project before."""
    d = delta_late(_windows(early=3.0, late=2.5, n=11))
    assert d.shape == (11,), f"expected one delta per window, got shape {d.shape}"


def test_delta_late_uses_only_its_declared_bands():
    """Positions outside [64,128) and [448,512) must not influence the statistic: a change
    confined to the untouched middle has to leave it exactly alone."""
    a = _windows(early=3.0, late=2.5)
    b = a.copy()
    b[:, 200:300] = 99.0
    assert np.allclose(delta_late(a), delta_late(b))


def test_delta_late_rejects_a_band_that_does_not_fit():
    with pytest.raises(ValueError, match="does not fit"):
        delta_late(np.zeros((4, 128)))


def test_delta_late_rejects_a_non_window_array():
    with pytest.raises(ValueError, match="expected"):
        delta_late(np.zeros(512))


def test_paired_stats_pairs_elementwise_rather_than_comparing_distributions():
    """THE MUTATION: comparing two sets of numbers by their means alone. Here a and b hold the
    SAME values in a different order, so every unpaired summary is identical while the true
    paired difference is not zero for any window."""
    a = [1.0, 2.0, 3.0, 4.0]
    b = [4.0, 3.0, 2.0, 1.0]
    st = paired_stats(a, b)
    assert st["mean_a"] == st["mean_b"]           # unpaired view: no difference at all
    assert st["paired_sd"] > 0                    # paired view: real per-window disagreement
    assert st["n_positive"] == 2 and st["n_negative"] == 2


def test_paired_stats_cancels_shared_per_window_noise():
    """The reason to pair at all: a large common offset per window must not inflate the
    paired sd. If it does, the code is differencing the wrong axis."""
    rng = np.random.default_rng(0)
    common = rng.normal(0, 10.0, size=256)        # huge between-window variance
    a, b = common, common + 0.5                   # constant, tiny true effect
    st = paired_stats(a, b)
    assert st["paired_sd"] < 1e-9, "shared per-window noise survived the pairing"
    assert math.isclose(st["paired_mean"], 0.5, abs_tol=1e-9)


def test_paired_stats_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="unpaired"):
        paired_stats([1.0, 2.0], [1.0, 2.0, 3.0])


def test_minimum_detectable_is_reported_so_a_null_is_not_read_as_zero():
    """A null has to be reportable as 'smaller than this design can see'."""
    st = paired_stats([0.0] * 64, [0.001] * 64 + [])
    assert "minimum_detectable" in st
    st2 = paired_stats(list(np.random.default_rng(1).normal(0, 1, 64)),
                       list(np.random.default_rng(2).normal(0, 1, 64)))
    assert st2["minimum_detectable"] > 0


@pytest.mark.parametrize("n_pos,n,expected", [
    (5, 5, 2 * (1 / 32)),      # every window one way
    (0, 5, 2 * (1 / 32)),      # every window the other way, same p
    (4, 8, 1.0),               # a dead heat cannot be significant
])
def test_sign_test_matches_the_exact_binomial(n_pos, n, expected):
    assert math.isclose(sign_test_p(n_pos, n), min(1.0, expected), rel_tol=1e-9)


def test_sign_test_is_symmetric_under_flipping_the_direction():
    for n_pos in range(0, 13):
        assert math.isclose(sign_test_p(n_pos, 12), sign_test_p(12 - n_pos, 12))


def test_the_bands_are_the_ones_the_spec_declared():
    """The spec fixed these before any run existed. If they move, the published statistic is
    not the declared one, and this test is the tripwire that says so."""
    assert EARLY_BAND == (64, 128)
    assert LATE_BAND == (448, 512)


def test_both_models_would_see_identical_windows():
    """The confound this design exists to remove: each arm's own val split is the tail of a
    different blend. One seeded draw, shared by both models, is what makes it paired."""
    from scripts.probe_context_use import sample_windows
    ids = np.random.default_rng(0).integers(0, 32000, size=200_000)
    xa, ya = sample_windows(ids, 512, 16, np.random.default_rng(7))
    xb, yb = sample_windows(ids, 512, 16, np.random.default_rng(7))
    assert np.array_equal(xa, xb) and np.array_equal(ya, yb)
