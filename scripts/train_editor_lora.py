#!/usr/bin/env python3
# scripts/train_editor_lora.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""BLOCKED-CONCLUSION WITHDRAWN (2026-08-31): the "0/24 moved" diagnostic was confounded.

This file previously declared LoRA blocked by a bug inside `AdamW::step()`. That conclusion
rested on before/after comparisons of `lora_B`'s VALUE, and those comparisons were measuring
the instrument. `_create_lora_B` builds the tensor as `ttnn.DataType.BFLOAT16`
(`ttml/modules/lora.py:69`), so `AutocastTensor` stores it in the half-precision slot and
leaves the full-precision slot empty. `t.to_numpy()` takes the binding's default precision,
`FULL`: the `lora_b_before` read TYPECASTS bf16 -> fp32 and CACHES the all-zeros result, and
because the optimizer mutates the bf16 tensor in place without calling `set_tensor()`, nothing
invalidates that cache -- so `lora_b_after`'s read is handed the same all-zeros array back.
`moved == 0/24` was therefore guaranteed by the baseline read itself, whether or not training
worked. tt-metal documents the hazard in `autograd/autocast_tensor.cpp` ("Lazy precision
caching can leave the FULL/FLOAT32 view stale after in-place updates that mutate only the BF16
tensor (e.g. optimizer step). Tracking: #41657"), and it is the same root cause as the
SFT duplicate-checkpoint bug fixed in `train.checkpoint.save_sft_checkpoint`.

The diagnostics that measured GRADIENTS (1-3) stand: gradients are computed and are nonzero.
Only the value-comparison diagnostics (4, 5) are withdrawn. The reads below now pass
`precision=NATIVE`, which bypasses the cache.

**This does not yet prove LoRA works** -- it removes the evidence that it does not. The
outstanding check is a short hardware run confirming `lora_B` moves off zero under a NATIVE
read. Do not treat LoRA as available until that has been seen. See
`docs/upstream-tt-metal-asks.md` entry 5.

The editor objective, via LoRA instead of full-parameter continued training.

Two full-parameter attempts (`scripts/train_editor.py`) established: (1) with 100% task
data and no anti-forgetting slice, the model regresses on repetition/termination; (2) with
a majority base-blend anti-forgetting slice, that regression is fixed, but the editor
capability itself never measurably improved on either attempt (`self_edit()` still fails on
the same negative-result draft, held-out corruption recovery is flat-to-worse). See CLAUDE.md
for both accounts.

LoRA is a different lever for the SAME anti-forgetting problem, structurally rather than by
counterweighting: it freezes 100% of the base weights and trains only a small low-rank
update on the attention projections, so the base model's general fluency literally cannot
move. No base-blend slice is needed here for that reason -- this script trains on 100%
editor+poetry+skits task data, same as the (regressed) first full-parameter attempt, to
isolate LoRA's effect from base-blend's.

**A real naming trap, found by direct inspection before writing any training-critical
code (never assumed):** `ttml.modules.lora.LoraModel` does not override `.parameters()`,
so calling it on the LoRA WRAPPER directly returns names walked from the wrapper's own root
(observed: `LoraModel/model/blocks/0/attention/q_linear/lora_A`) -- NOT the canonical
`llama/llama_block_0/...` names this project's checkpoint format, HF conversion, and every
other consumer expect. The INNER model's own `.parameters()` (accessed via `lora_model.model`)
already returns the correct canonical names -- confirmed directly, not assumed, by printing
both before committing to this design. `TtTntLoraModel` below is a two-line subclass
delegating `.parameters()` to the inner model, the same pattern `train.model.TtTntLlama`
already uses for its own C++-name translation, for exactly the same reason.

`ttml`'s own reference example (`tt-metal/tt-train/sources/examples/lora_llama/`) targets
`["q_linear", "kv_linear", "out_linear"]` -- the three attention projections -- and this
script does the same; MLP projections are not targeted in this first attempt.

    gozer run --chips 1 --who "claude:editor-training-lora" --reason "editor LoRA" -- \
        python3 scripts/train_editor_lora.py --steps 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_editor import (  # noqa: E402
    MAX_SEQ_LEN,
    _current_dialogue_checkpoint,
    _pad_to_max_seq_len,
    build_category_lists,
    load_editor_pairs,
    load_poetry_pairs,
    load_skits,
    stratified_split,
)
from scripts.train_skits import (  # noqa: E402
    LossRecorder,
    assert_eval_wired,
    assert_stochastic_rounding,
)

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
DEFAULT_PAIRS = ROOT / "artifacts" / "editor-pairs" / "pairs.jsonl"
DEFAULT_POETRY_PAIRS = ROOT / "artifacts" / "poetry-pairs" / "pairs.jsonl"
DEFAULT_SKITS = ROOT / "artifacts" / "skits-200k" / "skits.jsonl"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "checkpoints-1024-editor-lora"

#: The same three attention projections `ttml`'s own lora_llama example targets.
LORA_TARGET_MODULES = ["q_linear", "kv_linear", "out_linear"]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--poetry-pairs", type=Path, default=DEFAULT_POETRY_PAIRS)
    ap.add_argument("--skits", type=Path, default=DEFAULT_SKITS)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA conventionally trains at a higher LR than full-parameter "
                         "fine-tuning (this project's full-FT runs use 1e-5) -- only a "
                         "small, randomly-initialized subspace is being learned")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--warm-start", type=Path, default=None,
                    help="defaults to the currently-designated dialogue checkpoint "
                         "(docs/current_model.json), resolved at call time")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the examples and report the length distribution, then "
                         "stop. Touches no device, so it needs no lease.")
    args = ap.parse_args(argv)

    warm_start_ckpt = args.warm_start or _current_dialogue_checkpoint()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_token_id = tok.pad_token_id or 0

    pairs = load_editor_pairs(args.pairs)
    poetry_pairs = load_poetry_pairs(args.poetry_pairs)
    skits = load_skits(args.skits)
    categories = build_category_lists(pairs, poetry_pairs, skits, tok, pad_token_id=pad_token_id)

    task_data_count = sum(len(v) for v in categories.values())
    raw_count = len(pairs) + len(poetry_pairs) + len(skits)
    dropped = raw_count - task_data_count

    examples = [e for exs in categories.values() for e in exs]
    val_size = max(0, min(args.val_size, len(examples) // 4))
    train_examples, val_examples = stratified_split(categories, val_size)

    lengths = sorted(len(e["input_ids"]) for e in examples)
    print(f"pairs={len(pairs):,}  poetry_pairs={len(poetry_pairs):,}  "
          f"skits={len(skits):,}  total_examples={len(examples):,}  "
          f"train={len(train_examples):,}  val={len(val_examples):,}  "
          f"dropped_zero_supervision={dropped:,}")
    print(f"token lengths: min {lengths[0]}  median {lengths[len(lengths)//2]}  "
          f"max {lengths[-1]}")
    print(f"lora: rank={args.rank} alpha={args.alpha} targets={LORA_TARGET_MODULES} "
          f"dropout={args.lora_dropout}")

    if args.dry_run:
        print("dry run: no device opened, nothing trained.")
        return 0

    import ttml  # noqa: F401 -- opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.modules import LoraConfig

    from train.checkpoint import save_sft_checkpoint
    from train.lora import (assert_adapter_moved, assert_base_frozen,
                            base_parameter_snapshot, make_lora_model)
    from train.runlog import count_parameters, print_run_header
    from ttml.trainers import SFTConfig, SFTTrainer

    # (1, 1) ONLY -- matches every other training script in this project; LoRA changes
    # nothing about mesh requirements.
    ttml.open_device_mesh((1, 1))
    try:
        train_examples_padded = [_pad_to_max_seq_len(e, pad_token_id) for e in train_examples]
        val_examples_padded = [_pad_to_max_seq_len(e, pad_token_id) for e in val_examples]

        collate = partial(sft_collate_fn, max_seq_len=MAX_SEQ_LEN, pad_token_id=pad_token_id)
        loader = InMemoryDataloader(train_examples_padded, batch_size=args.batch_size,
                                    collate_fn=collate, shuffle=True, seed=args.seed)
        val_loader = (InMemoryDataloader(val_examples_padded, batch_size=args.batch_size,
                                         collate_fn=collate, shuffle=False, seed=args.seed)
                      if val_examples_padded else None)

        out = Path(args.out_root).resolve()
        out.mkdir(parents=True, exist_ok=True)

        from train.model import create_model

        model_yaml = yaml.safe_load(MODEL_YAML.read_text())
        transformer_config = model_yaml["transformer_config"]
        model = create_model({}, transformer_config)

        from train.enthusiasts import warm_start

        warm_summary = warm_start(
            model, Path(warm_start_ckpt),
            transformer_config=transformer_config, yaml_config={},
            moe_block_indices=[],
        )
        print(f"warm start: {warm_summary}")

        lora_config = LoraConfig(
            rank=args.rank, alpha=args.alpha, target_modules=LORA_TARGET_MODULES,
            # verbose=False: ttml's verbose injection prints one line per wrapped parameter
            # (48 of them here), which buries this script's own summary line. That summary,
            # the freeze check and the adapter check are the output worth reading.
            lora_dropout=args.lora_dropout, verbose=False,
        )
        lora_model = make_lora_model(model, lora_config)

        all_names = list(lora_model.parameters().keys())
        trainable_names = [n for n, t in lora_model.parameters().items() if t.get_requires_grad()]
        print(f"[lora] {len(all_names)} total parameters, {len(trainable_names)} trainable "
              f"(canonical names: {all_names[0]!r} ... confirms the naming fix is active)")
        if not any(n.endswith("/lora_A") or n.endswith("/lora_B") for n in trainable_names):
            raise RuntimeError(
                "no lora_A/lora_B parameters are trainable -- injection or the naming "
                "delegation is broken; aborting before spending a training run on it"
            )

        curve_path = out / "loss_curve.jsonl"
        recorder = LossRecorder(curve_path)
        trainer = SFTTrainer(
            model=lora_model, train_dataloader=loader, eval_dataloader=val_loader,
            config=SFTConfig(max_steps=args.steps, learning_rate=args.lr, seed=args.seed,
                             max_seq_len=MAX_SEQ_LEN, checkpoint_dir=str(out),
                             save_interval=args.save_every,
                             eval_interval=args.eval_every if val_loader else 0,
                             log_interval=1, max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.0,
                       "stochastic_rounding": True},
            # Without this the default saver reads each bf16 parameter at FULL precision and
            # is handed AutocastTensor's stale fp32 cache, so every checkpoint after the
            # first is a byte-identical duplicate. See train.checkpoint.
            checkpoint_saver=save_sft_checkpoint,
            callbacks=[recorder],
        )
        assert_stochastic_rounding(trainer, "editor-lora")
        assert_eval_wired(trainer, val_size=val_size, arm="editor-lora")

        # State the run's shape before training it. See train/runlog.py: the SFT entry
        # points printed almost none of this, so their logs held three thousand
        # progress-bar frames and no record of WHAT was trained.
        print_run_header(
            size_name="1024", model_config=MODEL_YAML, steps=args.steps,
            batch_size=args.batch_size, seq_len=MAX_SEQ_LEN,
            param_count=count_parameters(lora_model),
            scheduler="constant",
        )

        # The freeze is the whole premise of this arm: with 100% of base weights frozen,
        # general fluency CANNOT regress, which is why this script trains on 100% task data
        # with no base-blend counterweight. Snapshot at NATIVE precision -- a FULL read is
        # served from AutocastTensor's cache and would report a perfect freeze and a
        # motionless adapter whether or not either were true.
        base_before = base_parameter_snapshot(lora_model)

        trainer.train()

        frozen = assert_base_frozen(base_before, lora_model)
        moved = assert_adapter_moved(lora_model)
        print(f"[editor-lora] freeze held: {frozen['frozen_checked']} base parameters "
              f"bit-identical, {frozen['moved']} moved")
        print(f"[editor-lora] adapter trained: {moved['lora_B_moved']}/{moved['lora_B_total']} "
              f"lora_B tensors off zero, max |value| {moved['max_abs']:.6e}")

        print(f"training complete -> {out}")
    finally:
        ttml.close_device_mesh()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
