# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Checkpoint discovery/registry for the Gradio demo (app.py).

Resolves a friendly label to either a local `artifacts/hf-*` directory or, if that
directory has no `config.json` (missing entirely, or present but not yet converted), a
published Hub repo id. This is what lets the SAME registry work unmodified on: a bare
local clone (fewer converted checkpoints -> fewer dropdown entries), a machine with
every experimental checkpoint converted, and a HuggingFace Space (only the published
label resolves at all, via its Hub repo id -- see this project's
docs/superpowers/specs/2026-09-04-gradio-demo-design.md, "Portability: HuggingFace
Spaces").
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


@dataclass(frozen=True)
class CheckpointEntry:
    label: str
    local_dir: Path
    hub_repo_id: Optional[str]  # None if this checkpoint was never published


#: Only checkpoints this demo actually offers. Adding a new tab's checkpoint means
#: adding an entry here, not scattering a hardcoded path through app.py.
CHECKPOINTS: List[CheckpointEntry] = [
    CheckpointEntry(
        "tt-tnt-1024 (production)", ARTIFACTS / "hf-tt-tnt-1024", "episod/tt-tnt-1024",
    ),
    CheckpointEntry(
        "tool-calling-s2", ARTIFACTS / "hf-tt-tnt-1024-tool-calling-s2", None,
    ),
    CheckpointEntry(
        "tool-calling-s3", ARTIFACTS / "hf-tt-tnt-1024-tool-calling-s3", None,
    ),
    CheckpointEntry(
        "editor-blend (broken run)", ARTIFACTS / "hf-tt-tnt-1024-editor-blend", None,
    ),
]


def _is_valid_local_dir(path: Path) -> bool:
    return (path / "config.json").is_file()


def resolve(label: str, checkpoints: Optional[List[CheckpointEntry]] = None) -> Optional[str]:
    """What to pass to `from_pretrained` for `label`: the local directory (as a str) if
    it has a real `config.json`, else the label's Hub repo id if it has one, else None
    if this label can't be satisfied at all right now."""
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    for entry in checkpoints:
        if entry.label != label:
            continue
        if _is_valid_local_dir(entry.local_dir):
            return str(entry.local_dir)
        return entry.hub_repo_id
    return None


def list_available(checkpoints: Optional[List[CheckpointEntry]] = None) -> List[str]:
    """Labels `resolve()` can actually satisfy right now, in registry order. Never
    includes a label with neither a valid local directory nor a Hub repo id."""
    checkpoints = CHECKPOINTS if checkpoints is None else checkpoints
    return [e.label for e in checkpoints if resolve(e.label, checkpoints) is not None]
