# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Gates for ``packaging/package-thin.sh``, the one recipe the v6 bundles are staged from.

Two groups of checks.

**The mesh wiring.** A 1-chip serve on a p300 board needs two things the adapter cannot
provide by itself: a mesh descriptor it can find beside itself, and TT_VISIBLE_DEVICES
narrowed to one chip (a 1-chip lease on a p300c grants both chips of the board). The 4-chip
bundle needs its fabric in the manifest. All of it comes from the packaging step.

**The canonical v6 shape** (tt-model-manager ``docs/E2E_RECIPE_V6.md``): stock ttnn, the
model's code shipped in ``extra_code/``, fully resolved constraints, no index options in
requirements.txt, install guards, provenance, and a stage that cannot upload.

The tests run the real script with a stub ``tt-model`` that does what package-thin does
(copies ``--model-py``, ``--extra-code``, ``--provenance`` into the output) and records its
arguments, plus small stand-in wheels. The install guards (``--verify`` statements) are
executed for real against fake bundle layouts, and each is shown to reject a layout it
should reject. No hardware, no network, no ttnn import.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGING = REPO_ROOT / "packaging"
SCRIPT = PACKAGING / "package-thin.sh"
REV = "0" * 40
BUNDLES = [("tt-tnt", "mesh-1x1.textproto"), ("tt-tnt-1024", "mesh-1x4-ring.textproto")]

# Stand-in for `tt-model package-thin`: record argv, stage what the real one stages.
STUB_TT_MODEL = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_ARGS"
out=""; model=""; extra=""; prov=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="$2"; shift ;;
    --model-py) model="$2"; shift ;;
    --extra-code) extra="$2"; shift ;;
    --provenance) prov="$2"; shift ;;
  esac
  shift
done
mkdir -p "$out" && cp "$model" "$out/"
[ -n "$extra" ] && cp -r "$extra" "$out/extra_code"
[ -n "$prov" ] && cp "$prov" "$out/PROVENANCE.json"
echo '{}' > "$out/tt_kernel_manifest.json"
"""


def _exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(stat.S_IRWXU)


def _wheel(path: Path, files: dict[str, str]) -> str:
    """Write a zip with ``files`` and return its sha256."""
    with zipfile.ZipFile(path, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


# The closure stand-in: a tiny `models` tree whose imports all resolve.
CLOSURE_FILES = {
    "models/common/__init__.py": "",
    "models/common/utility.py": "X = 1\n",
    "models/tt_transformers/tt/generator_vllm.py": "from models.common.utility import X\nfrom . import helper\n",
    "models/tt_transformers/tt/helper.py": "import torch  # external, never checked\n",
    # The real adapter imports these two, and the recipe checks it against this tree.
    "models/tt_transformers/tt/model_config.py": "",
    "models/tt_transformers/tt/generator.py": "",
    "tt_tnt_models_closure-0.77.0.dist-info/METADATA": "Name: tt-tnt-models-closure\n",
}


def _stage(tmp_path: Path, name: str, closure_files: dict[str, str] | None = None,
           closure_sha_override: str | None = None):
    """Run the real recipe. Returns (completed process, out dir, argv)."""
    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    _exe(stub / "tt-model", STUB_TT_MODEL)
    wheels = tmp_path / "wheels"
    wheels.mkdir(exist_ok=True)
    sha = {
        "PLUGIN": _wheel(wheels / "vllm_tt_plugin-0.1.0-py3-none-any.whl", {"p.py": "1"}),
        "VLLM": _wheel(wheels / "vllm-0.25.1+empty-cp312-cp312-linux_x86_64.whl", {"v.py": "1"}),
        "CLOSURE": _wheel(wheels / "tt_tnt_models_closure-0.77.0-py3-none-any.whl",
                          closure_files if closure_files is not None else CLOSURE_FILES),
    }
    out = tmp_path / "out"
    env = {
        **os.environ,
        "TT_MODEL": str(stub / "tt-model"),
        "WHEELS_DIR": str(wheels),
        "STUB_ARGS": str(tmp_path / "args"),
        "PLUGIN_SHA256": sha["PLUGIN"],
        "VLLM_SHA256": sha["VLLM"],
        "CLOSURE_SHA256": closure_sha_override or sha["CLOSURE"],
    }
    r = subprocess.run(["bash", str(SCRIPT), name, REV, str(out)], env=env,
                       capture_output=True, text=True)
    argv = (tmp_path / "args").read_text().splitlines() if (tmp_path / "args").exists() else []
    return r, out, argv


def _staged_ok(tmp_path: Path, name: str):
    r, out, argv = _stage(tmp_path, name)
    assert r.returncode == 0, r.stderr + r.stdout
    return out, argv


def _flag_values(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def _env(argv):
    return dict(v.split("=", 1) for v in _flag_values(argv, "--env"))


# ---------------------------------------------------------------------------------------------
# Mesh wiring (unchanged behaviour from the first recipe)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, descriptor", BUNDLES)
def test_the_descriptor_is_staged_beside_the_adapter(tmp_path, name, descriptor):
    """The adapter's Patch 4 and the manifest env both resolve the descriptor next to the
    adapter at the bundle root; a bundle without it fails at mesh open."""
    out, argv = _staged_ok(tmp_path, name)
    assert (out / "tt_tnt_adapter.py").is_file()
    assert (out / descriptor).read_text() == (REPO_ROOT / "train/configs/mesh" / descriptor).read_text()
    assert _env(argv)["TT_MESH_GRAPH_DESC_PATH"] == f"$HERE/{descriptor}"


def test_the_one_chip_descriptor_matches_the_adapter_fallback_name(tmp_path):
    out, _ = _staged_ok(tmp_path, "tt-tnt")
    adapter = (REPO_ROOT / "bundle" / "tt_tnt_adapter.py").read_text()
    assert 'MESH_1X1_NAME = "mesh-1x1.textproto"' in adapter
    assert (out / "mesh-1x1.textproto").is_file()


@pytest.mark.parametrize("grant, expected", [("4,5", "4"), ("2", "2"), ("", "0")])
def test_the_one_chip_bundle_narrows_a_board_lease_to_one_chip(tmp_path, grant, expected):
    """A 1-chip lease on a p300c exports both chips of the board. Evaluate the exact value
    the bundle exports, the way run.sh's `export` does, under that grant."""
    _, argv = _staged_ok(tmp_path, "tt-tnt")
    value = _env(argv)["TT_VISIBLE_DEVICES"]
    r = subprocess.run(["bash", "-c", f'echo "{value}"'], capture_output=True, text=True,
                       env={"PATH": os.environ["PATH"], **({"TT_VISIBLE_DEVICES": grant} if grant else {})})
    assert r.stdout.strip() == expected


def test_the_one_chip_bundle_declares_one_device_and_no_fabric(tmp_path):
    _, argv = _staged_ok(tmp_path, "tt-tnt")
    assert _flag_values(argv, "--device-count") == ["1"]
    assert _flag_values(argv, "--max-model-len") == ["2048"]
    assert "--mesh-fabric" not in argv  # no fabric on one chip


def test_the_four_chip_bundle_gets_its_fabric_and_pins_weights(tmp_path):
    """The fabric goes in through package-thin's own flag, so the manifest and run.sh agree."""
    _, argv = _staged_ok(tmp_path, "tt-tnt-1024")
    assert _flag_values(argv, "--device-count") == ["4"]
    assert _flag_values(argv, "--max-model-len") == ["512"]
    assert _flag_values(argv, "--mesh-fabric") == ["FABRIC_2D_TORUS_XY"]
    assert _flag_values(argv, "--weights-revision") == [REV]


def test_the_script_refuses_an_unknown_bundle(tmp_path):
    r = subprocess.run(["bash", str(SCRIPT), "tt-nope", REV, str(tmp_path / "o")],
                       env={**os.environ, "WHEELS_DIR": "x"}, capture_output=True, text=True)
    assert r.returncode == 2 and "unknown bundle" in r.stderr


# ---------------------------------------------------------------------------------------------
# The canonical shape
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, _descriptor", BUNDLES)
def test_the_models_code_ships_in_extra_code_not_as_a_wheel(tmp_path, name, _descriptor):
    """The model's code rides in extra_code/ at its original import path. It is not a
    --models-wheel and not an --ops-wheel, and the metadata of the retired wheel is left out."""
    out, argv = _staged_ok(tmp_path, name)
    assert _flag_values(argv, "--extra-code"), "the code must ship via --extra-code"
    assert "--models-wheel" not in argv
    assert "--ops-wheel" not in argv  # tt-tnt has no custom kernels at serve time
    for rel in CLOSURE_FILES:
        if rel.startswith("models/"):
            assert (out / "extra_code" / rel).is_file(), rel
    assert not list((out / "extra_code").glob("*.dist-info"))


@pytest.mark.parametrize("name, _descriptor", BUNDLES)
def test_the_recipe_never_uploads(tmp_path, name, _descriptor):
    """A positional repo id makes package-thin push. Only --weights may carry an episod/ id."""
    _, argv = _staged_ok(tmp_path, name)
    assert "--out" in argv
    for i, tok in enumerate(argv):
        if tok.startswith("episod/"):
            assert argv[i - 1] == "--weights", f"{tok!r} would be taken as a repo id"


def test_the_stage_passes_constraints_python_provenance_and_pinned_vllm_requirements(tmp_path):
    out, argv = _staged_ok(tmp_path, "tt-tnt")
    assert _flag_values(argv, "--constraints") == [str(PACKAGING / "constraints.txt")]
    assert _flag_values(argv, "--requirements") == [str(PACKAGING / "requirements.txt")]
    assert re.fullmatch(r"\d+\.\d+\.\d+", _flag_values(argv, "--python")[0])  # exact patch pin
    assert _flag_values(argv, "--vllm-version") == ["0.25.1"]
    assert Path(_flag_values(argv, "--vllm-common-requirements")[0]).is_file()
    prov = json.loads((out / "PROVENANCE.json").read_text())
    assert prov["weights"]["revision"] == REV
    assert prov["ttnn"].startswith("0.77.0")
    assert len(prov["extra_code"]["wheel_sha256"]) == 64


def test_no_parsers_are_declared_for_the_base_models(tmp_path):
    """README.md's feature table says the published weights declare neither parser."""
    _, argv = _staged_ok(tmp_path, "tt-tnt")
    assert "--tool-parser" not in argv and "--reasoning-parser" not in argv


def test_a_wheel_that_is_not_the_pinned_one_is_refused(tmp_path):
    r, _, argv = _stage(tmp_path, "tt-tnt", closure_sha_override="f" * 64)
    assert r.returncode == 3 and "sha256 mismatch" in r.stderr
    assert argv == []  # tt-model was never called


def test_a_closure_with_an_unresolved_import_is_refused(tmp_path):
    """The static import check runs before staging. An import of a module that is not in
    extra_code/ would only fail after a multi-minute boot."""
    bad = {**CLOSURE_FILES,
           "models/tt_transformers/tt/generator_vllm.py": "from models.common.missing import X\n"}
    r, _, argv = _stage(tmp_path, "tt-tnt", closure_files=bad)
    assert r.returncode != 0 and "UNRESOLVED" in r.stdout
    assert argv == []


def test_requirements_are_stock_ttnn_with_no_index_options():
    lines = [ln.split("#")[0].strip() for ln in (PACKAGING / "requirements.txt").read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    assert not [ln for ln in lines if ln.startswith("-")], "index/options lines are not allowed"
    ttnn = [ln for ln in lines if ln.startswith("ttnn")]
    assert ttnn == ["ttnn==0.77.0"]  # exact, from PyPI, no +local fork tag
    assert not [ln for ln in lines if "closure" in ln or "tt-metal-models" in ln]


def test_constraints_pin_everything_and_cover_the_requirements():
    pins = {}
    for ln in (PACKAGING / "constraints.txt").read_text().splitlines():
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[^\s=]+", ln), f"not an exact pin: {ln!r}"
        pins[ln.split("==")[0].lower().replace("_", "-")] = ln.split("==")[1]
    for must in ("ttnn", "torchvision", "pytest", "certifi", "urllib3", "requests", "idna",
                 "charset-normalizer", "transformers", "numpy"):
        assert must in pins, f"{must} is not pinned"
    assert "+" not in pins["ttnn"]
    # Path-installed wheels are not index packages; the retired closure wheel is gone.
    assert not {"vllm", "vllm-tt-plugin", "tt-tnt-models-closure"} & pins.keys()


# ---------------------------------------------------------------------------------------------
# The install guards, executed against fake bundle layouts
# ---------------------------------------------------------------------------------------------

def _guards(tmp_path: Path) -> list[str]:
    tmp_path.mkdir(parents=True)
    _, argv = _staged_ok(tmp_path, "tt-tnt")
    guards = _flag_values(argv, "--verify")
    assert len(guards) == 3
    return guards


def _fake_bundle(root: Path, ttnn_version="0.77.0", with_code=True, stray_models=False,
                 closure_installed=False) -> Path:
    """Lay out <root>/{extra_code,tt_tnt_adapter.py,venv} the way install.sh leaves it."""
    root.mkdir(parents=True)
    (root / "tt_tnt_adapter.py").write_text("")
    if with_code:
        gen = root / "extra_code/models/tt_transformers/tt/generator_vllm.py"
        gen.parent.mkdir(parents=True)
        gen.write_text("")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / "venv")], check=True)
    site = next((root / "venv/lib").glob("python*/site-packages"))

    def dist(name, version):
        d = site / f"{name.replace('-', '_')}-{version}.dist-info"
        d.mkdir()
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")

    dist("ttnn", ttnn_version)
    if closure_installed:
        dist("tt-tnt-models-closure", "0.77.0")
    if stray_models:
        (site / "models").mkdir()
        (site / "models/__init__.py").write_text("")
    return root


def _run_guard(bundle: Path, statement: str) -> subprocess.CompletedProcess:
    """Run one guard the way install.sh does: venv python, extra_code on PYTHONPATH."""
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str(bundle / "extra_code")}
    return subprocess.run([str(bundle / "venv/bin/python"), "-c", statement],
                          env=env, capture_output=True, text=True)


def test_every_guard_passes_on_a_correct_bundle(tmp_path):
    guards = _guards(tmp_path / "stage")
    bundle = _fake_bundle(tmp_path / "good")
    for g in guards:
        r = _run_guard(bundle, g)
        assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("index, kwargs, message", [
    (0, {"ttnn_version": "0.78.0"}, "expected stock 0.77.0"),
    (1, {"with_code": False}, "model code missing"),
    (1, {"stray_models": True}, "resolves outside extra_code"),
    (2, {"closure_installed": True}, "would shadow extra_code"),
])
def test_each_guard_rejects_the_layout_it_exists_for(tmp_path, index, kwargs, message):
    guards = _guards(tmp_path / "stage")
    bundle = _fake_bundle(tmp_path / "bad", **kwargs)
    r = _run_guard(bundle, guards[index])
    assert r.returncode != 0 and message in r.stderr, r.stderr


# ---------------------------------------------------------------------------------------------
# The static import checker
# ---------------------------------------------------------------------------------------------

CHECKER = PACKAGING / "check_static_imports.py"


def _check(tmp_path: Path, files: dict[str, str]):
    root = tmp_path / "code"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return subprocess.run([sys.executable, "-I", str(CHECKER), "--root", str(root)],
                          capture_output=True, text=True)


def test_checker_accepts_resolving_absolute_relative_and_lazy_imports(tmp_path):
    r = _check(tmp_path, {
        "models/a/__init__.py": "",
        "models/a/x.py": "import torch\nfrom models.a import y\nfrom .y import Z\n"
                         "def f():\n    from ..b.w import W\n",
        "models/a/y.py": "Z = 1\n",
        "models/b/w.py": "W = 1\n",
    })
    assert r.returncode == 0, r.stdout
    assert "torch" in r.stdout  # listed as external, not an error


@pytest.mark.parametrize("source", [
    "from models.a.gone import Z\n",          # absolute, missing module
    "import models.a.gone\n",                  # plain import, missing module
    "from .gone import Z\n",                   # relative, missing module
    "def f():\n    from models.zzz import Q\n",  # lazy import, missing module
])
def test_checker_rejects_an_import_that_does_not_resolve(tmp_path, source):
    r = _check(tmp_path, {"models/a/__init__.py": "", "models/a/x.py": source})
    assert r.returncode == 1 and "UNRESOLVED" in r.stdout


def test_checker_warns_about_dynamic_imports(tmp_path):
    r = _check(tmp_path, {"models/a.py": "import importlib\nm = importlib.import_module('models.q')\n"})
    assert r.returncode == 0 and "dynamic import" in r.stdout
