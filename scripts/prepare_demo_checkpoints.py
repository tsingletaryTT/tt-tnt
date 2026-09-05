#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""One-time CLI: convert the two SFT `.pkl` checkpoints the Gradio demo (app.py) wants
but that have no HF conversion on disk -- tool-calling-s3 (the corrected, best-selected
tool-calling run) and editor-blend (the ORIGINAL broken run, kept specifically for its
dramatic single-word repeat-loop collapse in the demo's "Known Limitations" tab).
Idempotent: skips a checkpoint whose output directory already has a config.json.

    python scripts/prepare_demo_checkpoints.py

CPU-only -- no ttnn/ttml import, no device, same as every other conversion path in
this project (see scripts/eval_improv.py's sft_checkpoint_to_hf, which does the actual
conversion work this script just calls with the right paths).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: Used ONLY to recover an architecture header (vocab_size, seq_len,
#: transformer_config, ...) for HF conversion -- every SFT run this demo converts is
#: dense and this same 1024 shape, so this checkpoint's header describes them all
#: exactly. Its own weights are never used (see scripts/eval_improv.py's own comment
#: on WARM_START_CKPT for the precedent this follows).
WARM_START_CKPT = ROOT / "artifacts" / "checkpoints-v077-beta2-control" / "tt_tnt_step00010764.pkl"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024"

_STEP_RE = re.compile(r"^step_(\d+)\.pkl$")


class DemoConversion(NamedTuple):
    checkpoint_dir: Path
    out_dir: Path


CONVERSIONS = [
    DemoConversion(
        ROOT / "artifacts" / "checkpoints-1024-tool-calling-s3",
        ROOT / "artifacts" / "hf-tt-tnt-1024-tool-calling-s3",
    ),
    DemoConversion(
        ROOT / "artifacts" / "checkpoints-1024-editor-blend",
        ROOT / "artifacts" / "hf-tt-tnt-1024-editor-blend",
    ),
]


def latest_sft_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Highest-step `step_<N>.pkl` in `checkpoint_dir`, or None if there are none.

    SFTTrainer checkpoints use `step_<N>.pkl`, a different naming scheme from ttml's
    own pretrain checkpoints (`tt_tnt_step<N>.pkl`, handled by
    train.checkpoint.latest_checkpoint) -- that function looks for the wrong prefix
    entirely and would silently find nothing in an SFT checkpoint directory.
    """
    best_step = -1
    best_path: Optional[Path] = None
    if not checkpoint_dir.is_dir():
        return None
    for path in checkpoint_dir.glob("step_*.pkl"):
        m = _STEP_RE.match(path.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def already_converted(out_dir: Path) -> bool:
    return (out_dir / "config.json").is_file()


def main() -> int:
    from scripts.eval_improv import sft_checkpoint_to_hf

    if not WARM_START_CKPT.is_file():
        print(f"ERROR: warm-start header checkpoint missing: {WARM_START_CKPT}", file=sys.stderr)
        return 1

    did_work = False
    for conv in CONVERSIONS:
        if already_converted(conv.out_dir):
            print(f"skip {conv.out_dir} (already converted)")
            continue
        step_pkl = latest_sft_checkpoint(conv.checkpoint_dir)
        if step_pkl is None:
            print(f"skip {conv.checkpoint_dir} (no step_*.pkl found -- not on this machine)")
            continue
        print(f"converting {step_pkl} -> {conv.out_dir} ...")
        sft_checkpoint_to_hf(
            step_pkl, warm_start_ckpt=WARM_START_CKPT, tokenizer_dir=TOKENIZER_DIR,
            out_dir=conv.out_dir,
        )
        did_work = True
        print(f"  done: {conv.out_dir}/config.json")

    if not did_work:
        print("nothing to do -- all demo checkpoints already converted or unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
