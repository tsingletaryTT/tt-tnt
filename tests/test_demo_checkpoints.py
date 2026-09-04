# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
from pathlib import Path

from demo_checkpoints import CheckpointEntry, list_available, resolve


def _entry(label, tmp_path, name, config_present, hub_repo_id=None):
    d = tmp_path / name
    d.mkdir()
    if config_present:
        (d / "config.json").write_text("{}")
    return CheckpointEntry(label, d, hub_repo_id)


def test_resolve_returns_local_dir_when_config_json_present(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=True)]
    assert resolve("a", entries) == str(tmp_path / "a")


def test_resolve_falls_back_to_hub_repo_when_local_dir_missing_config(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=False, hub_repo_id="org/model")]
    assert resolve("a", entries) == "org/model"


def test_resolve_returns_none_when_neither_local_nor_hub_available(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=False)]
    assert resolve("a", entries) is None


def test_resolve_returns_none_for_unknown_label(tmp_path):
    entries = [_entry("a", tmp_path, "a", config_present=True)]
    assert resolve("does-not-exist", entries) is None


def test_list_available_includes_only_resolvable_labels(tmp_path):
    entries = [
        _entry("has-local", tmp_path, "x", config_present=True),
        _entry("has-hub-fallback", tmp_path, "y", config_present=False, hub_repo_id="org/m"),
        _entry("has-neither", tmp_path, "z", config_present=False),
    ]
    assert list_available(entries) == ["has-local", "has-hub-fallback"]


def test_default_checkpoints_list_is_a_nonempty_list_of_entries():
    from demo_checkpoints import CHECKPOINTS
    assert len(CHECKPOINTS) > 0
    assert all(isinstance(e, CheckpointEntry) for e in CHECKPOINTS)
