#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# The window-purity control arm.
# Spec: docs/superpowers/specs/2026-09-01-window-purity-control.md
#
# Six runs, sequential, each under its own gozer lease so the board is released between
# them rather than held for four hours. Ordered seed-major (both arms of a seed before the
# next seed) so that stopping early always leaves COMPLETE PAIRS -- a half-finished seed
# would be an arm-vs-arm comparison with a seed confound baked in, which is worse than
# fewer seeds.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=10719          # 1.000 epoch on tokens-v5 (0.996 on v4); see the spec's arithmetic
SAVE_EVERY=5000      # deviation from the spec's 2000, on disk grounds; see the run log
VAL_EVERY=500

for SEED in 5489 20260815 8191; do
  for ARM in v4 v5; do
    DIR="artifacts/checkpoints-winpurity-${ARM}-s${SEED}"
    LOG="logs/winpurity/${ARM}-s${SEED}.log"
    if [ -f "${DIR}/val_losses.jsonl" ] && \
       [ "$(wc -l < "${DIR}/val_losses.jsonl")" -ge 21 ]; then
      echo "SKIP ${ARM} seed ${SEED}: already complete"; continue
    fi
    echo "=== $(date -Is) START ${ARM} seed ${SEED} -> ${DIR} ==="
    gozer run --chips 4 \
      --who "claude:window-purity" \
      --reason "control arm: does window purity move the model? ${ARM} seed ${SEED}" -- \
    python train/run.py \
      --size 1024 --seq-len 512 --batch-size 64 --steps "${STEPS}" \
      --seed "${SEED}" --ddp 4 --model-impl python \
      --config train/configs/nanollama3_bpe_v2.yaml \
      --tokens-dir "artifacts/tokens-${ARM}" \
      --val-every "${VAL_EVERY}" --save-every "${SAVE_EVERY}" \
      --checkpoint-dir "${DIR}" > "${LOG}" 2>&1
    RC=$?
    echo "=== $(date -Is) END ${ARM} seed ${SEED} rc=${RC} ==="
    [ $RC -ne 0 ] && echo "!!! ${ARM} seed ${SEED} FAILED rc=${RC}; see ${LOG}"
    tail -4 "${LOG}" | sed 's/^/    /'
  done
done
echo "=== $(date -Is) ALL RUNS DONE ==="
