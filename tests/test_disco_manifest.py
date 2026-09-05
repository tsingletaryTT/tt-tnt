# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Anti-drift test for `.disco/app.yaml`, following this project's own convention
(see train/sizes.py vs its YAML, CLAUDE.md) of pinning coupled constants with a test
so they can't silently drift apart.

pyyaml is not a declared dependency of this project (see pyproject.toml), so this
parses the manifest by hand rather than pulling in a real YAML parser for a file this
simple -- 4 flat `key: value` lines, no nesting, no lists.
"""
from pathlib import Path

import app

MANIFEST_PATH = Path(__file__).resolve().parent.parent / ".disco" / "app.yaml"


def _load_manifest() -> dict:
    manifest = {}
    text = MANIFEST_PATH.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        manifest[key.strip()] = value.strip()
    return manifest


def test_disco_manifest_port_matches_app_port():
    manifest = _load_manifest()
    assert int(manifest["port"]) == app.PORT


def test_disco_manifest_declares_no_chips():
    """The demo process must never open a Tenstorrent device -- confirmed by the
    hard, deliberate absence of a `chips` key in the manifest."""
    manifest = _load_manifest()
    assert "chips" not in manifest
