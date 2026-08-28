#!/usr/bin/env python3
# scripts/train_editor_lora.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""BLOCKED (2026-08-27): do not run `main()` for real training yet.

Five independent, hardware-verified diagnostics (see `docs/upstream-tt-metal-asks.md` entry 5
and CLAUDE.md) isolated a real upstream bug: `ttml`'s `SFTTrainer(..., peft_config=LoraConfig(...))`
path -- the exact, documented, un-customized usage from `tt-train/docs/SFT_TRAINER.md` -- computes
correct LoRA gradients (confirmed nonzero immediately before `optimizer.step()`) but never applies
them (`lora_B`'s value is bit-identical before and after `.step()`). This is not fixable from this
repo -- it is inside `AdamW::step()`'s C++ implementation, which we do not build or patch. A real
3000-step run against this mechanism would have applied zero update the entire time; do not run
one until the upstream fix lands. The code below (data loading, category building, stratified
split, the naming-delegation fix for checkpoint compatibility) is unaffected by the defect and
needs no rework once it is fixed -- only the training loop's actual parameter updates are broken.

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
    from ttml.modules import LoraConfig, LoraModel
    from ttml.trainers import SFTConfig, SFTTrainer

    class TtTntLoraModel(LoraModel):
        """`LoraModel` with `.parameters()` delegated to the wrapped model's own
        canonical naming. `LoraModel` itself does not override `.parameters()`, so
        calling it directly walks the C++-registered tree from the WRAPPER's own root
        (observed empirically: `LoraModel/model/blocks/N/...`) instead of the
        canonical `llama/llama_block_N/...` this project's checkpoint format, optimizer
        construction, and HF conversion all expect -- confirmed by printing both before
        writing this class, not assumed. Same fix shape as `train.model.TtTntLlama`'s
        own override, for the same underlying reason.
        """

        def parameters(self):
            return self.model.parameters()

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
            lora_dropout=args.lora_dropout, verbose=True,
        )
        lora_model = TtTntLoraModel(model, lora_config)

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
            callbacks=[recorder],
        )
        assert_stochastic_rounding(trainer, "editor-lora")
        assert_eval_wired(trainer, val_size=val_size, arm="editor-lora")

        # lora_B initialises to all-zeros (ttml.modules.lora._create_lora_B), so this is a
        # real health check, not a formality: if training silently did nothing, every
        # lora_B tensor would still read back as exactly zero.
        import numpy as np
        lora_b_before = {
            n: t.to_numpy().copy() for n, t in lora_model.parameters().items()
            if n.endswith("/lora_B")
        }
        assert all(np.all(v == 0) for v in lora_b_before.values()), (
            "expected every lora_B to start at zero (ttml's own init convention) -- "
            "if this fails, the 'lora_B moved' check below would be meaningless"
        )

        trainer.train()

        lora_b_after = {n: t.to_numpy() for n, t in lora_model.parameters().items()
                         if n.endswith("/lora_B")}
        moved = sum(1 for n in lora_b_after if not np.all(lora_b_after[n] == 0))
        print(f"[editor-lora] lora_B tensors moved from zero: {moved}/{len(lora_b_after)}")
        if moved == 0:
            raise RuntimeError(
                "training completed but every lora_B tensor is still exactly zero -- "
                "this run learned nothing"
            )

        print(f"training complete -> {out}")
    finally:
        ttml.close_device_mesh()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
