# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""scripts/prepare_demo_checkpoints.py: exercises the pure-logic helpers with tmp_path
fixtures -- no real checkpoint or model needed. The real conversion run (calling
scripts.eval_improv.sft_checkpoint_to_hf against this machine's actual checkpoints) is
verified separately by running the script for real, not by a unit test.
"""
from scripts.prepare_demo_checkpoints import already_converted, latest_sft_checkpoint


def test_latest_sft_checkpoint_picks_highest_step(tmp_path):
    (tmp_path / "step_250.pkl").touch()
    (tmp_path / "step_3000.pkl").touch()
    (tmp_path / "step_1000.pkl").touch()
    assert latest_sft_checkpoint(tmp_path) == tmp_path / "step_3000.pkl"


def test_latest_sft_checkpoint_returns_none_when_empty(tmp_path):
    assert latest_sft_checkpoint(tmp_path) is None


def test_latest_sft_checkpoint_ignores_ttml_naming(tmp_path):
    # ttml's own pretrain naming (tt_tnt_step<N>.pkl) is a DIFFERENT format this
    # function must not mistake for an SFT checkpoint.
    (tmp_path / "tt_tnt_step00000500.pkl").touch()
    assert latest_sft_checkpoint(tmp_path) is None


def test_already_converted_true_when_config_json_present(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert already_converted(tmp_path) is True


def test_already_converted_false_when_directory_missing_config(tmp_path):
    assert already_converted(tmp_path) is False
