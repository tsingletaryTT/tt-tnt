# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The run header's wording is a contract, not prose.

These assertions look pedantic — they pin colons and their absence. They are pinned because
the consumer matches by prefix on the trimmed line, so `Max steps 3000` parses and
`Max steps: 3000` does not, and nothing else in this repo would notice the difference. The
run would train fine and the log would silently stop answering the question.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from train.runlog import count_parameters, format_eval_line, print_run_header


def _header(**kw):
    out = []
    base = dict(size_name="1024", model_config=Path("train/configs/model/tt-tnt-1024.yaml"),
                steps=3000, batch_size=8, seq_len=512, param_count=122962944,
                print_fn=out.append)
    base.update(kw)
    print_run_header(**base)
    return out


def test_the_summary_line_carries_steps_batch_and_seq_len_together():
    """The only place a ttml-driven run states its step budget: ttml's trainer prints
    nothing per step, so without this there is no progress fraction and no ETA."""
    line = next(l for l in _header() if l.startswith("tt-tnt training"))
    assert "steps=3000" in line and "batch=8" in line and "seq_len=512" in line


def test_the_model_config_is_named_with_its_basename_in_parentheses():
    """The consumer resolves the YAML beneath the trainer's cwd, so the NAME is what
    matters; a bare path with no parentheses is not matched."""
    line = next(l for l in _header() if "model size:" in l)
    assert line.strip() == "model size: 1024 (tt-tnt-1024.yaml)"


def test_max_steps_and_batch_size_have_no_colon_but_parameters_does():
    """tt-train's own C++ spellings, which the parser matches by prefix. A colon added to
    `Max steps` or dropped from `Number of parameters:` silently stops the line parsing."""
    h = _header()
    assert "Max steps 3000" in h
    assert "Batch size 8" in h
    assert "Number of parameters: 122962944" in h
    assert not any(l.startswith("Max steps:") for l in h)
    assert not any(l.startswith("Batch size:") for l in h)


def test_parameter_count_is_omitted_rather_than_guessed_when_unknown():
    h = _header(param_count=None)
    assert not any("Number of parameters" in l for l in h)
    # everything else still lands
    assert any("Max steps" in l for l in h)


def test_scheduler_is_only_stated_when_there_is_one():
    assert not any("Scheduler type" in l for l in _header())
    assert "Scheduler type constant" in _header(scheduler="constant")


def test_eval_line_matches_the_pretrain_paths_shape():
    """Both halves of the project must spell validation the same way, or a reader has to
    learn two formats and a parser has to match two."""
    line = format_eval_line(1250, 1.7319, 1.7419, lr=2e-4)
    assert "step=   1250" in line
    assert "train_loss=1.7319" in line and "val_loss=1.7419" in line
    assert "lr=2.000e-04" in line


def test_eval_line_omits_lr_when_absent():
    assert "lr=" not in format_eval_line(10, 1.0, 2.0)


# --- parameter counting -----------------------------------------------------


class _T:
    def __init__(self, dims):
        self._dims = dims

    def shape(self):
        return list(self._dims)

    def to_numpy(self, *a, **k):                      # pragma: no cover
        raise AssertionError("counting parameters must not copy tensors off the device")


class _P:
    def __init__(self, dims):
        self.tensor = _T(dims)


class _M:
    def __init__(self, params):
        self._p = params

    def parameters(self):
        return self._p


def test_parameters_are_counted_from_shapes_without_touching_the_data():
    """`to_numpy()` raises in the fake. Under DDP that call moves a distributed tensor to
    the host purely to ask how big it is — the mistake train/enthusiasts.py records."""
    m = _M({"a": _P((1, 1, 4, 8)), "b": _P((1, 1, 3, 3))})
    assert count_parameters(m) == 32 + 9


def test_an_unusual_model_yields_none_rather_than_failing_the_run():
    class _Bad:
        def parameters(self):
            raise RuntimeError("no shapes here")

    assert count_parameters(_Bad()) is None
