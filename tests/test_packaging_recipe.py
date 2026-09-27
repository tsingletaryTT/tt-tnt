# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Gates for ``packaging/package-thin.sh``, the one recipe the v6 bundles are staged from.

A 1-chip serve on a p300 board needs two things the adapter cannot provide by itself: a mesh
descriptor it can find beside itself, and TT_VISIBLE_DEVICES narrowed to one chip (a 1-chip
gozer lease on a p300c grants both chips of the board). Both come from the packaging step,
so these tests run the real script, with a stub ``tt-model`` that does what package-thin does
with the runner (copies ``--model-py`` to the bundle root) and records its arguments, then
check the staged layout and the env the bundle will export. No hardware, no network.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "packaging" / "package-thin.sh"
REV = "0" * 40

STUB_TT_MODEL = """#!/usr/bin/env bash
# Stand-in for `tt-model package-thin`: record argv, stage the runner the way it does.
printf '%s\\n' "$@" > "$STUB_ARGS"
out=""; model=""
while [ $# -gt 0 ]; do
  case "$1" in --out) out="$2"; shift ;; --model-py) model="$2"; shift ;; esac
  shift
done
mkdir -p "$out" && cp "$model" "$out/"
echo '{"mesh": {"fabric": null}}' > "$out/tt_kernel_manifest.json"
"""

# The fabric step runs `<dir of tt-model>/python - <out> <fabric>`; record it instead.
STUB_PYTHON = """#!/usr/bin/env bash
cat > /dev/null
printf '%s\\n' "$@" > "$STUB_FABRIC"
"""


def _exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(stat.S_IRWXU)


def _stage(tmp_path: Path, name: str):
    stub = tmp_path / "stub"
    stub.mkdir()
    _exe(stub / "tt-model", STUB_TT_MODEL)
    _exe(stub / "python", STUB_PYTHON)
    out = tmp_path / "out"
    env = {
        **os.environ,
        "TT_MODEL": str(stub / "tt-model"),
        "WHEELS_DIR": str(tmp_path / "wheels"),
        "STUB_ARGS": str(tmp_path / "args"),
        "STUB_FABRIC": str(tmp_path / "fabric"),
    }
    subprocess.run(["bash", str(SCRIPT), name, REV, str(out)], env=env, check=True,
                   capture_output=True, text=True)
    argv = (tmp_path / "args").read_text().splitlines()
    fabric = (tmp_path / "fabric").read_text().splitlines() if (tmp_path / "fabric").exists() else None
    return out, argv, fabric


def _flag_values(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def _env(argv):
    return dict(v.split("=", 1) for v in _flag_values(argv, "--env"))


@pytest.mark.parametrize("name, descriptor", [("tt-tnt", "mesh-1x1.textproto"),
                                              ("tt-tnt-1024", "mesh-1x4-ring.textproto")])
def test_the_descriptor_is_staged_beside_the_adapter(tmp_path, name, descriptor):
    """The adapter's Patch 4 and the manifest env both resolve the descriptor next to the
    adapter at the bundle root; a bundle without it fails at mesh open."""
    out, argv, _ = _stage(tmp_path, name)
    assert (out / "tt_tnt_adapter.py").is_file()
    assert (out / descriptor).read_text() == (REPO_ROOT / "train/configs/mesh" / descriptor).read_text()
    assert _env(argv)["TT_MESH_GRAPH_DESC_PATH"] == f"$HERE/{descriptor}"


def test_the_one_chip_descriptor_matches_the_adapter_fallback_name(tmp_path):
    out, _, _ = _stage(tmp_path, "tt-tnt")
    adapter = (REPO_ROOT / "bundle" / "tt_tnt_adapter.py").read_text()
    assert 'MESH_1X1_NAME = "mesh-1x1.textproto"' in adapter
    assert (out / "mesh-1x1.textproto").is_file()


@pytest.mark.parametrize("grant, expected", [("4,5", "4"), ("2", "2"), ("", "0")])
def test_the_one_chip_bundle_narrows_a_board_lease_to_one_chip(tmp_path, grant, expected):
    """A 1-chip gozer lease on a p300c exports both chips of the board. Evaluate the exact
    value the bundle exports, the way run.sh's `export` does, under that grant."""
    _, argv, _ = _stage(tmp_path, "tt-tnt")
    value = _env(argv)["TT_VISIBLE_DEVICES"]
    r = subprocess.run(["bash", "-c", f'echo "{value}"'], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], **({"TT_VISIBLE_DEVICES": grant} if grant else {})})
    assert r.stdout.strip() == expected


def test_the_one_chip_bundle_declares_one_device(tmp_path):
    _, argv, fabric = _stage(tmp_path, "tt-tnt")
    assert _flag_values(argv, "--device-count") == ["1"]
    assert _flag_values(argv, "--max-model-len") == ["2048"]
    assert fabric is None  # no fabric on one chip


def test_the_four_chip_bundle_gets_its_fabric_and_pins_weights(tmp_path):
    out, argv, fabric = _stage(tmp_path, "tt-tnt-1024")
    assert _flag_values(argv, "--device-count") == ["4"]
    assert _flag_values(argv, "--max-model-len") == ["512"]
    assert fabric == ["-", str(out), "FABRIC_2D_TORUS_XY"]  # python - <out> <fabric>
    assert _flag_values(argv, "--weights-revision") == [REV]


def test_the_script_refuses_an_unknown_bundle(tmp_path):
    r = subprocess.run(["bash", str(SCRIPT), "tt-nope", REV, str(tmp_path / "o")],
                       env={**os.environ, "WHEELS_DIR": "x"}, capture_output=True, text=True)
    assert r.returncode == 2 and "unknown bundle" in r.stderr
