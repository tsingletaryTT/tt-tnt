#!/usr/bin/env bash
# Stage the episod/tt-tnt or episod/tt-tnt-1024 v6 thin bundle -- the ONE recipe a repackage
# uses. It stages only; publishing is a separate step after hardware verification.
#
#   WHEELS_DIR=/path/with/the/three/wheels \
#     packaging/package-thin.sh <tt-tnt|tt-tnt-1024> <weights-revision> <out-dir>
#
# WHEELS_DIR holds the wheels the bundle ships, too large for git: vllm_tt_plugin-0.1.0,
# tt_tnt_models_closure-0.77.0, and vllm-0.25.1+empty (the published bundle's wheels/ dir
# works). <weights-revision> is the full HF commit sha of the weights repo (the bundle repo
# itself) to pin. TT_MODEL overrides which tt-model build stages it.
#
# What this adds beyond package-thin's own flags, and why (tests/test_packaging_recipe.py
# checks every item, and stages a bundle to check the resulting layout):
#   * The mesh descriptor is copied to the bundle ROOT, next to the adapter (package-thin
#     ships --model-py at the root, and the adapter resolves the descriptor beside itself).
#     package-thin has no flag for an extra file.
#   * TT_MESH_GRAPH_DESC_PATH points at that descriptor, so a mesh open never depends on the
#     adapter's own fallback (Patch 4, which only covers a one-device process).
#   * tt-tnt narrows TT_VISIBLE_DEVICES to the first leased chip. On a p300 board a 1-chip
#     gozer lease grants both chips of the board; without this the plugin opens a (1,2)
#     mesh. tt-model-manager's newer run.sh also narrows a grant; this keeps the bundle
#     correct when staged with an older tt-model too.
#   * tt-tnt-1024's mesh.fabric is set after staging (package-thin has no --fabric flag) and
#     run.sh is re-rendered with tt-model-manager's own renderer so the two stay in sync.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
NAME="${1:?usage: package-thin.sh <tt-tnt|tt-tnt-1024> <weights-revision> <out-dir>}"
REV="${2:?weights revision (full 40-hex sha) required}"
OUT="${3:?out-dir required}"
: "${WHEELS_DIR:?set WHEELS_DIR to the dir holding the three wheels the bundle ships}"
TT_MODEL="${TT_MODEL:-tt-model}"

case "$NAME" in
  tt-tnt)
    PROFILE=(--mesh P150 --device-count 1 --max-num-seqs 32 --block-size 64 --max-model-len 2048
             --env 'TT_MESH_GRAPH_DESC_PATH=$HERE/mesh-1x1.textproto'
             --env 'TT_VISIBLE_DEVICES=$(echo ${TT_VISIBLE_DEVICES:-0} | cut -d, -f1)')
    MESHFILE=mesh-1x1.textproto
    FABRIC="" ;;
  tt-tnt-1024)
    PROFILE=(--mesh P300x2 --device-count 4 --max-num-seqs 32 --block-size 64 --max-model-len 512
             --env 'TT_MESH_GRAPH_DESC_PATH=$HERE/mesh-1x4-ring.textproto')
    MESHFILE=mesh-1x4-ring.textproto
    FABRIC=FABRIC_2D_TORUS_XY ;;
  *) echo "unknown bundle $NAME (expected tt-tnt or tt-tnt-1024)" >&2; exit 2 ;;
esac

rm -rf "$OUT"
cd "$REPO_ROOT"
"$TT_MODEL" package-thin --out "$OUT" --name "$NAME" \
  --model-py bundle/tt_tnt_adapter.py \
  --requirements "$HERE/requirements.txt" \
  --plugin-wheel "$WHEELS_DIR/vllm_tt_plugin-0.1.0-py3-none-any.whl" \
  --models-wheel "$WHEELS_DIR/tt_tnt_models_closure-0.77.0-py3-none-any.whl" \
  --vllm-wheel   "$WHEELS_DIR/vllm-0.25.1+empty-cp312-cp312-linux_x86_64.whl" \
  --arch blackhole --arch-name LlamaForCausalLM --main-class tt_tnt_adapter:LlamaForCausalLM \
  --weights "episod/$NAME" --weights-revision "$REV" --tt-metal-version 0.77.0 \
  "${PROFILE[@]}"

cp "$REPO_ROOT/train/configs/mesh/$MESHFILE" "$OUT/"

if [ -n "$FABRIC" ]; then
  PY="$(dirname "$(command -v "$TT_MODEL")")/python"
  "$PY" - "$OUT" "$FABRIC" <<'PYEOF'
import json, sys
from pathlib import Path
from tt_kernel import packaging
from tt_kernel.manifest import Manifest
out, fabric = Path(sys.argv[1]), sys.argv[2]
d = json.loads((out / "tt_kernel_manifest.json").read_text())
d["mesh"]["fabric"] = fabric
(out / "tt_kernel_manifest.json").write_text(json.dumps(d, indent=2))
(out / "run.sh").write_text(packaging.render_run_sh(Manifest.model_validate(d)))
PYEOF
fi
echo "staged $NAME @ weights revision $REV in $OUT"
