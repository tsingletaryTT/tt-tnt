#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Post-training analysis for the window-purity control arm.
# Spec: docs/superpowers/specs/2026-09-01-window-purity-control.md
#
# CPU only -- conversion and probing both run on the host. No device, no lease.
#
# Each seed's two arms are compared on BOTH validation splits, reported separately. A
# conclusion that holds under only one of them is not a conclusion: the two splits are tails
# of different blends, and that asymmetry is the confound this design is built to survive.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p docs/measurements/winpurity

for SEED in 5489 20260815 8191; do
  for ARM in v4 v5; do
    CKPT_DIR="artifacts/checkpoints-winpurity-${ARM}-s${SEED}"
    HF="artifacts/hf-winpurity-${ARM}-s${SEED}"
    [ -d "$CKPT_DIR" ] || { echo "skip ${ARM} s${SEED}: no checkpoints"; continue; }
    if [ ! -f "${HF}/config.json" ]; then
      echo "converting ${ARM} s${SEED} -> ${HF}"
      python scripts/convert_checkpoint.py --size 1024 \
        --checkpoint-dir "$CKPT_DIR" --out-dir "$HF" || echo "!!! convert failed ${ARM} s${SEED}"
    fi
  done

  A="artifacts/hf-winpurity-v4-s${SEED}"; B="artifacts/hf-winpurity-v5-s${SEED}"
  [ -f "${A}/config.json" ] && [ -f "${B}/config.json" ] || { echo "skip compare s${SEED}"; continue; }
  for VAL in v4 v5; do
    OUT="docs/measurements/winpurity/context-use-s${SEED}-on-${VAL}val.json"
    echo "=== seed ${SEED}, probed on tokens-${VAL}/val_ids.npy ==="
    python scripts/compare_context_use.py \
      --model-a "$A" --model-b "$B" \
      --label-a "v4-s${SEED}" --label-b "v5-s${SEED}" \
      --tokens "artifacts/tokens-${VAL}/val_ids.npy" \
      --seq-len 512 --n-windows 512 --seed 0 --out "$OUT"
  done
done
echo "=== analysis complete ==="
