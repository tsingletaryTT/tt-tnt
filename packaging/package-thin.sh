#!/usr/bin/env bash
# Stage the episod/tt-tnt or episod/tt-tnt-1024 v6 thin bundle -- the ONE recipe a repackage
# uses. It stages only; publishing is a separate step after hardware verification.
#
#   WHEELS_DIR=/path/with/the/three/wheels \
#     packaging/package-thin.sh <tt-tnt|tt-tnt-1024> <weights-revision> <out-dir>
#
# WHEELS_DIR holds the three wheels this recipe takes by path. They are too large for git.
# Every one is checked against the sha256 pinned below before it is used:
#   vllm_tt_plugin-0.1.0-py3-none-any.whl
#   vllm-0.25.1+empty-cp312-cp312-linux_x86_64.whl   (stock vLLM built for the empty target)
#   tt_tnt_models_closure-0.77.0-py3-none-any.whl    (SOURCE of extra_code/, see below)
# Where to get them: `hf download episod/tt-tnt --include 'wheels/*'` (read-only), or the
# wheels/ folder of an existing pulled bundle (~/.cache/tt-model/models/episod/tt-tnt/wheels).
# The wheel for stock ttnn is NOT needed: ttnn==0.77.0 installs from PyPI.
#
# <weights-revision> is the full HF commit sha of the weights repo (the bundle repo itself)
# to pin. TT_MODEL overrides which tt-model build stages it.
#
# THE SHAPE (tt-model-manager docs/E2E_RECIPE_V6.md, "The canonical shape")
#   * engine: stock ttnn==0.77.0 from PyPI (requirements.txt). No fork, no +local tag.
#   * constraints.txt pins every resolved package, so an index-ranking mistake fails the
#     install. requirements.txt carries no index options.
#   * the model's code ships inside the bundle in extra_code/ (--extra-code). tt-tnt is a
#     stock HF Llama; the code that computes it is tt-metal v0.77.0's models/tt_transformers
#     and models/common. Those files were a vendored wheel (tt-tnt-models-closure) installed
#     into the venv. They now ride in extra_code/ at the SAME import path (`models.*`), byte
#     for byte, so the adapter's imports do not change.
#   * no custom-op wheel and no tooling-only tt-transformers wheel. tt-tnt serves with no
#     custom kernels (its Gumbel-sampling kernels are training-side, in ttml), so there is
#     nothing for --ops-wheel to carry. The tt-transformers code here is tt-metal v0.77.0's
#     models/ tree, which is not the "tooling only" layout the newer tt_transformers wheel
#     has. See CLAUDE.md, 2026-10-16, for why that is left as is.
#   * no --tool-parser / --reasoning-parser. The published weights are base completion
#     models, and README.md's feature table says they declare neither parser. A hermes
#     tool-calling checkpoint exists in the repo (CLAUDE.md, 2026-08-29) but was not
#     published. If one is published, add --tool-parser hermes here for that bundle.
#
# What this adds beyond package-thin's own flags, and why (tests/test_packaging_recipe.py
# checks every item, and stages a bundle to check the resulting layout):
#   * The mesh descriptor is copied to the bundle ROOT, next to the adapter (package-thin
#     ships --model-py at the root, and the adapter resolves the descriptor beside itself).
#     package-thin has no flag for an extra file.
#   * TT_MESH_GRAPH_DESC_PATH points at that descriptor, so a mesh open never depends on the
#     adapter's own fallback (Patch 4, which only covers a one-device process).
#   * tt-tnt narrows TT_VISIBLE_DEVICES to the first leased chip. On a p300 board a 1-chip
#     lease grants both chips of the board; without this the plugin opens a (1,2) mesh.
#   * tt-tnt-1024's fabric (FABRIC_2D_TORUS_XY) is set with --mesh-fabric, so package-thin
#     records it in the manifest and renders it into run.sh itself. An earlier version of
#     this recipe patched the manifest after staging and re-rendered run.sh; the result is
#     the same run.sh (tests/test_packaging_recipe.py compares them).
#   * install guards (--verify): they fail `install.sh`, not the first request.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
NAME="${1:?usage: package-thin.sh <tt-tnt|tt-tnt-1024> <weights-revision> <out-dir>}"
REV="${2:?weights revision (full 40-hex sha) required}"
OUT="${3:?out-dir required}"
: "${WHEELS_DIR:?set WHEELS_DIR to the dir holding the three wheels the bundle ships}"
TT_MODEL="${TT_MODEL:-tt-model}"

PLUGIN_WHL="vllm_tt_plugin-0.1.0-py3-none-any.whl"
VLLM_WHL="vllm-0.25.1+empty-cp312-cp312-linux_x86_64.whl"
CLOSURE_WHL="tt_tnt_models_closure-0.77.0-py3-none-any.whl"
# sha256 of each wheel, as published in episod/tt-tnt and episod/tt-tnt-1024 (the two repos
# carry identical copies). The three *_SHA256 variables can be overridden by
# tests/test_packaging_recipe.py, which builds small stand-in wheels; leave them alone for a
# real staging.
PLUGIN_SHA256="${PLUGIN_SHA256:-3990ec4d8df47f532d6e29f1f2cc13f0193814cbcc726ba98ba83dc56881001d}"
VLLM_SHA256="${VLLM_SHA256:-3ef923531b77f8e662940bf55b0d8a0a8854617c2003f05ad7beafe94ea94097}"
CLOSURE_SHA256="${CLOSURE_SHA256:-cbbee57b3afcba59359dea7043e309837d5a5d5a32afb6214f3ead74ac5feeb8}"
# The interpreter the bundles were hardware-verified under (uv provisions it at install).
PYTHON_PIN="3.12.12"

case "$NAME" in
  tt-tnt)
    PROFILE=(--mesh P150 --device-count 1 --max-num-seqs 32 --block-size 64 --max-model-len 2048
             --env 'TT_MESH_GRAPH_DESC_PATH=$HERE/mesh-1x1.textproto'
             --env 'TT_VISIBLE_DEVICES=$(echo ${TT_VISIBLE_DEVICES:-0} | cut -d, -f1)')
    MESHFILE=mesh-1x1.textproto ;;
  tt-tnt-1024)
    PROFILE=(--mesh P300x2 --device-count 4 --max-num-seqs 32 --block-size 64 --max-model-len 512
             --mesh-fabric FABRIC_2D_TORUS_XY
             --env 'TT_MESH_GRAPH_DESC_PATH=$HERE/mesh-1x4-ring.textproto')
    MESHFILE=mesh-1x4-ring.textproto ;;
  *) echo "unknown bundle $NAME (expected tt-tnt or tt-tnt-1024)" >&2; exit 2 ;;
esac

# --- 1. Check the three wheels are the ones this recipe was written against -----------------
# A wheel is code that runs in the serving venv, so an unexpected file is refused here.
check_wheel() {  # <file name> <expected sha256>
  local path="$WHEELS_DIR/$1" got
  [ -f "$path" ] || { echo "missing wheel: $path" >&2; exit 3; }
  got="$(sha256sum "$path" | cut -d' ' -f1)"
  [ "$got" = "$2" ] || { echo "sha256 mismatch for $1: got $got, expected $2" >&2; exit 3; }
}
check_wheel "$PLUGIN_WHL" "$PLUGIN_SHA256"
check_wheel "$VLLM_WHL" "$VLLM_SHA256"
check_wheel "$CLOSURE_WHL" "$CLOSURE_SHA256"

# --- 2. Lay out extra_code/: the model's code, at its original import path ------------------
# A wheel is a zip. Unpack its models/ tree and nothing else (the *.dist-info metadata would
# only describe a package that is no longer installed). The scratch dir is removed on exit.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
EXTRA="$SCRATCH/extra_code"
mkdir -p "$EXTRA"
python3 -I - "$WHEELS_DIR/$CLOSURE_WHL" "$EXTRA" <<'PYEOF'
import sys, zipfile
wheel, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as z:
    members = [n for n in z.namelist() if n.startswith("models/") and not n.endswith("/")]
    if not members:
        sys.exit(f"{wheel}: no models/ tree inside, so there is no code to ship")
    # zipfile.extract strips absolute paths and '..' parts, but refuse them outright.
    bad = [n for n in members if n.startswith("/") or ".." in n.split("/")]
    if bad:
        sys.exit(f"{wheel}: unsafe member names {bad[:3]}")
    for n in members:
        z.extract(n, dest)
print(f"unpacked {len(members)} files from {wheel.rsplit('/', 1)[-1]} into extra_code/")
PYEOF

# --- 3. Prove every in-bundle import resolves, with no import of ttnn or torch --------------
# (docs/E2E_RECIPE_V6.md Step 2c.) Exits non-zero on an unresolved `models.*` import.
python3 -I "$HERE/check_static_imports.py" --root "$EXTRA" --file "$REPO_ROOT/bundle/tt_tnt_adapter.py"

# --- 4. Provenance: what this bundle was built from ------------------------------------------
# package-thin stages the file as PROVENANCE.json. It records inputs, not test results.
PROVENANCE="$SCRATCH/provenance.json"
python3 -I - "$PROVENANCE" "$NAME" "$REV" "$REPO_ROOT" \
  "$PLUGIN_SHA256" "$VLLM_SHA256" "$CLOSURE_SHA256" <<'PYEOF'
import json, subprocess, sys
out, name, rev, repo, plugin, vllm, closure = sys.argv[1:8]
def git(*a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()
doc = {
    "what": f"episod/{name}: tt-tnt served through the Tenstorrent vLLM plugin, v6 thin bundle",
    "recipe": "packaging/package-thin.sh in tsingletaryTT/tt-tnt",
    "recipe_commit": git("rev-parse", "HEAD") or None,
    "recipe_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "ttnn": "0.77.0 from PyPI (stock)",
    "extra_code": {
        "source": "models/ tree of the vendored tt-tnt-models-closure 0.77.0 wheel "
                  "(tt-metal v0.77.0 models/common + models/tt_transformers/tt), unchanged",
        "wheel_sha256": closure,
    },
    "wheels_sha256": {"vllm_tt_plugin": plugin, "vllm_empty_target": vllm},
    "vllm": {"version": "0.25.1", "target_device": "empty"},
    "weights": {"repo": f"episod/{name}", "revision": rev},
}
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
PYEOF

# --- 5. Stage ---------------------------------------------------------------------------------
# NEVER pass a positional repo id here: with one, package-thin uploads. This stages only.
#
# Install guards. install.sh runs each one with the venv's python, with extra_code/ on
# PYTHONPATH, and without $HERE exported. The venv lives at <bundle>/venv, so the bundle
# folder is the parent of sys.prefix. None of them imports ttnn: that needs LD_PRELOAD and a
# device-capable host, and the guards must work on a machine with neither.
#   guard 1: stock ttnn 0.77.0 (read from package metadata, not by importing it)
#   guard 2: the model's code is in extra_code/ and is what `models` resolves to
#   guard 3: the retired tt-tnt-models-closure wheel is not installed next to it
GUARD_TTNN='import importlib.metadata as m; v = m.version("ttnn"); assert v == "0.77.0", f"ttnn {v}, expected stock 0.77.0 from PyPI"'
GUARD_CODE='import sys, pathlib, importlib.util as u; b = pathlib.Path(sys.prefix).parent; x = b / "extra_code"; assert (x / "models/tt_transformers/tt/generator_vllm.py").is_file(), f"model code missing from {x}"; assert (b / "tt_tnt_adapter.py").is_file(), "adapter missing"; s = u.find_spec("models"); assert s is not None, "no models package on the path"; locs = [pathlib.Path(p).resolve() for p in s.submodule_search_locations]; assert locs and all(p == (x / "models").resolve() for p in locs), f"models resolves outside extra_code: {locs}"'
GUARD_NO_CLOSURE='import importlib.metadata as m; d = [x for x in m.distributions() if x.metadata["Name"].lower().replace("_", "-") == "tt-tnt-models-closure"]; assert not d, "tt-tnt-models-closure is installed; its models/ would shadow extra_code/"'

rm -rf "$OUT"
cd "$REPO_ROOT"
"$TT_MODEL" package-thin --out "$OUT" --name "$NAME" \
  --model-py bundle/tt_tnt_adapter.py \
  --extra-code "$EXTRA" \
  --requirements "$HERE/requirements.txt" \
  --constraints "$HERE/constraints.txt" \
  --python "$PYTHON_PIN" \
  --plugin-wheel "$WHEELS_DIR/$PLUGIN_WHL" \
  --vllm-wheel   "$WHEELS_DIR/$VLLM_WHL" \
  --vllm-version 0.25.1 \
  --vllm-common-requirements "$HERE/vllm-common-0.25.1.txt" \
  --arch blackhole --arch-name LlamaForCausalLM --main-class tt_tnt_adapter:LlamaForCausalLM \
  --weights "episod/$NAME" --weights-revision "$REV" --tt-metal-version 0.77.0 \
  --verify "$GUARD_TTNN" --verify "$GUARD_CODE" --verify "$GUARD_NO_CLOSURE" \
  --provenance "$PROVENANCE" \
  "${PROFILE[@]}"

# The adapter looks for its mesh descriptor beside itself, at the bundle root.
cp "$REPO_ROOT/train/configs/mesh/$MESHFILE" "$OUT/"

echo "staged $NAME @ weights revision $REV in $OUT"
