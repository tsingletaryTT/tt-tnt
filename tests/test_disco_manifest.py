# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Anti-drift test for `.disco/app.yaml`, following this project's own convention
(see train/sizes.py vs its YAML, CLAUDE.md) of pinning coupled constants with a test
so they can't silently drift apart.

pyyaml is not a declared dependency of this project (see pyproject.toml), so this
parses the manifest by hand rather than pulling in a real YAML parser for a file this
simple -- 4 flat `key: value` lines, no nesting, no lists.

Same reasoning for `app.PORT`: a plain `import app` drags in `gradio` (the `ui`
extra) at module scope, which CI deliberately never installs (see
.github/workflows/tests.yml -- "a fresh checkout is 1592 passed, 81 skipped, 0
failed" with no `ui` extra in that install list). Reading `PORT = <int>` out of
app.py's own source text keeps this test running the same everywhere the manifest
itself does, rather than silently skipping the one CI environment it matters most in.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / ".disco" / "app.yaml"
APP_PATH = REPO_ROOT / "app.py"


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


def _app_port() -> int:
    match = re.search(r"^PORT\s*=\s*(\d+)", APP_PATH.read_text(), flags=re.MULTILINE)
    assert match, "app.py no longer declares a module-level PORT = <int>"
    return int(match.group(1))


def test_disco_manifest_port_matches_app_port():
    manifest = _load_manifest()
    assert int(manifest["port"]) == _app_port()


def test_disco_manifest_declares_no_chips():
    """The demo process must never open a Tenstorrent device -- confirmed by the
    hard, deliberate absence of a `chips` key in the manifest."""
    manifest = _load_manifest()
    assert "chips" not in manifest
