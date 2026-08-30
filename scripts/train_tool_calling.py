#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Continued training: tool-calling response modes (train/tool_calling.py's data layer),
warm-started from the published tt-tnt-1024 checkpoint (512 context, tokens-v4 corpus).

Same anti-forgetting discipline `scripts/train_editor.py` learned the hard way: a majority
base-blend refresh alongside the new task data, not the whole budget spent on task data alone
(that shipped a catastrophically broken checkpoint once already -- CLAUDE.md, 2026-08-27).
`--base-blend-ratio` defaults to 0.6 for the same reason it does there.

TWO CONSTANTS THAT MUST TRACK REALITY, NOT BE COPY-PASTED
----------------------------------------------------------
1. `MAX_SEQ_LEN` must equal the PUBLISHED model's context. It has been wrong in both
   directions already: written as 2048 while the 2048-context line was live, then stale at
   2048 after that line was reverted on 2026-08-29 (CLAUDE.md). Padding every example to a
   cap the model was never trained to is silent -- nothing downstream raises. A test pins it
   against `train.sizes` so the two cannot drift apart again.
2. `DEFAULT_BASE_BLEND_TOKENS` here is `artifacts/tokens-v4/train_ids.npy`, not `tokens-v3`.
   tokens-v3 predates the dialogue slice (the exact mistake this project just made and
   corrected for the base checkpoint itself -- see CLAUDE.md's 2026-08-29 entry, "The
   ctx2048 retrain used the wrong corpus generation"). Anti-forgetting refresh sampled from
   the wrong corpus generation would refresh the model on facts it was never trained on in
   the first place, which is not "anti-forgetting" of anything real.

WARM START
----------
Defaults to `artifacts/checkpoints-1024-dialogue/tt_tnt_step00010764.pkl` -- the weights
`docs/current_model.json` currently designates and `episod/tt-tnt-1024` currently serves.
Kept literal rather than resolved from that file at call time so a future re-designation
cannot silently change what this script warm-starts from without someone editing this line.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_skits import (  # noqa: E402
    LossRecorder,
    assert_eval_wired,
    assert_stochastic_rounding,
)
from train.tool_calling import (  # noqa: E402
    ToolCallExample,
    build_corpus,
    render_tool_call,
)

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
DEFAULT_BASE_BLEND_TOKENS = ROOT / "artifacts" / "tokens-v4" / "train_ids.npy"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "checkpoints-1024-tool-calling"
DEFAULT_WARM_START = (
    ROOT / "artifacts" / "checkpoints-1024-dialogue" / "tt_tnt_step00010764.pkl"
)
#: This checkpoint's real trained context. Was 2048 when written; the 2048 line was
#: reverted on 2026-08-29 (see CLAUDE.md) and the published model is 512 again.
MAX_SEQ_LEN = 512
#: How many of the mined (not hand-authored) Question/Answer pairs to expand into training
#: examples. Passed straight through to train.tool_calling.build_corpus's own cap.
DEFAULT_MAX_MINED_PAIRS = 200


def _truncate_to_max_seq_len(input_ids: list, labels: list) -> Dict[str, list] | None:
    """Same convention as scripts/train_editor.py's function of the same name: cap from the
    END only (the prompt -- the question -- is never truncated), and return None if that
    would leave zero real supervised positions."""
    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def build_tool_calling_example(
    example: ToolCallExample, tok, *, pad_token_id: int
) -> Dict[str, list] | None:
    """`{"input_ids", "labels"}` for one tool-call example, pre-shifted for ttml.

    Same boundary convention as every other SFT example in this project's editor/poetry/skits
    lines (`scripts/train_editor.py::build_editor_example`): the LAST prompt position's label
    is the FIRST completion token (supervised -- that transition IS the trained behaviour),
    the LAST completion position is masked. Prompt is `Q: {question}\\nAnswer:` (matching the
    existing dialogue-slice convention this project already established); completion is the
    real hermes-parseable `<tool_call>...</tool_call>` text (`train.tool_calling.render_tool_call`)
    -- training on this EXACT text is what makes the served model's tool call real.
    """
    prompt = f"Q: {example.question}\nAnswer:"
    completion = render_tool_call(example.tool, example.arguments)
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)
    if not p_ids:
        raise ValueError("prompt tokenized to zero ids; label shifting requires at least "
                         "one prompt token")
    input_ids = p_ids + c_ids
    labels = [-100] * (len(p_ids) - 1) + c_ids + [-100]
    return _truncate_to_max_seq_len(input_ids, labels)


def build_base_blend_example(window) -> Dict[str, list]:
    """Identical convention to scripts/train_editor.py's function of the same name (correctly
    shifted: `labels[t] == input_ids[t+1]`, the bug that once broke a checkpoint outright when
    it shipped unshifted -- see that file's own docstring and CLAUDE.md)."""
    ids = list(window)
    return {"input_ids": ids[:-1], "labels": ids[1:]}


def sample_base_blend_examples(
    token_path: Path, n: int, *, seq_len: int = MAX_SEQ_LEN, seed: int
) -> List[dict]:
    """Same convention as scripts/train_editor.py's function of the same name, over THIS
    checkpoint's real seq_len and the CORRECTED corpus generation (tokens-v4, passed by
    the caller via --base-blend-tokens's default)."""
    import numpy as np
    import random as _random

    arr = np.load(token_path, mmap_mode="r")
    if len(arr) < seq_len + 1:
        raise ValueError(
            f"{token_path} has only {len(arr)} tokens, fewer than seq_len+1={seq_len + 1}"
        )
    rng = _random.Random(seed)
    examples = []
    for _ in range(n):
        start = rng.randrange(0, len(arr) - seq_len)
        window = arr[start : start + seq_len + 1].tolist()
        examples.append(build_base_blend_example(window))
    return examples


def _pad_to_max_seq_len(example: dict, pad_token_id: int) -> dict:
    """Same rationale as scripts/train_editor.py's function of the same name: pad every
    example to a FIXED cap up front so batch shape never varies mid-run, which is what a
    real hardware failure (2026-08-27, ttml's SDPA backward kernel sizing a scratch tensor
    from the first batch) was caused by."""
    input_ids = list(example["input_ids"])
    labels = list(example["labels"])
    n = MAX_SEQ_LEN - len(input_ids)
    if n > 0:
        input_ids = input_ids + [pad_token_id] * n
        labels = labels + [-100] * n
    return {"input_ids": input_ids, "labels": labels}


def build_category_lists(
    corpus: List[ToolCallExample], tok, *, pad_token_id: int
) -> Dict[str, List[dict]]:
    """Tool-calling examples as their own category (base_blend is added by main(), same as
    scripts/train_editor.py's pattern, since its size depends on --base-blend-ratio which
    this function doesn't know about)."""
    tool_calling_examples = [
        e for e in (
            build_tool_calling_example(ex, tok, pad_token_id=pad_token_id) for ex in corpus
        )
        if e is not None
    ]
    return {"tool_calling": tool_calling_examples}


def stratified_split(categories: Dict[str, List[dict]], val_size: int) -> tuple:
    """Identical convention to scripts/train_editor.py's function of the same name: an even
    share of val_size held out from the TAIL of each category, deterministic, no shuffle
    before this split -- so a category never silently ends up 0% or 100% of the held-out set."""
    if val_size <= 0 or not categories:
        train = [e for exs in categories.values() for e in exs]
        return train, []
    per_category = max(1, val_size // len(categories))
    train_examples: List[dict] = []
    val_examples: List[dict] = []
    for exs in categories.values():
        n = min(per_category, len(exs) // 4, len(exs))
        if n > 0:
            train_examples += exs[: len(exs) - n]
            val_examples += exs[len(exs) - n :]
        else:
            train_examples += exs
    return train_examples, val_examples


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogue-corpus", type=Path,
                    default=ROOT / "artifacts" / "corpus" / "dialogue.txt")
    ap.add_argument("--max-mined-pairs", type=int, default=DEFAULT_MAX_MINED_PAIRS)
    ap.add_argument("--base-blend-tokens", type=Path, default=DEFAULT_BASE_BLEND_TOKENS)
    ap.add_argument("--base-blend-ratio", type=float, default=0.6,
                    help="target fraction of the FINAL combined example count that should be "
                         "base-blend anti-forgetting refresh -- see module docstring; "
                         "scripts/train_editor.py's first run shipped at 0.0 and measured a "
                         "real regression this ratio is meant to avoid repeating")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START,
                    help="see module docstring: literal, not auto-resolved from "
                         "docs/current_model.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the examples and report the length distribution, then stop. "
                         "Touches no device, so it needs no lease.")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_token_id = tok.pad_token_id or 0

    corpus = build_corpus(dialogue_path=args.dialogue_corpus, max_mined_pairs=args.max_mined_pairs)
    categories = build_category_lists(corpus, tok, pad_token_id=pad_token_id)

    task_data_count = sum(len(v) for v in categories.values())
    dropped = len(corpus) - task_data_count

    ratio = args.base_blend_ratio
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"--base-blend-ratio must be in [0, 1); got {ratio}")
    base_blend_count = round(ratio / (1 - ratio) * task_data_count) if ratio > 0 else 0
    categories["base_blend"] = (
        sample_base_blend_examples(args.base_blend_tokens, base_blend_count, seed=args.seed)
        if base_blend_count > 0 else []
    )

    examples = [e for exs in categories.values() for e in exs]
    val_size = max(0, min(args.val_size, len(examples) // 4))
    train_examples, val_examples = stratified_split(categories, val_size)

    lengths = sorted(len(e["input_ids"]) for e in examples)
    over_max = sum(1 for l in lengths if l > MAX_SEQ_LEN)
    realized_base_blend_share = len(categories["base_blend"]) / len(examples) if examples else 0.0
    print(f"tool_calling_corpus={len(corpus):,}  tool_calling_examples={task_data_count:,}  "
          f"base_blend={len(categories['base_blend']):,} ({realized_base_blend_share:.1%})  "
          f"total_examples={len(examples):,}  train={len(train_examples):,}  "
          f"val={len(val_examples):,}  dropped_zero_supervision={dropped:,}")
    print(f"token lengths: min {lengths[0]}  median {lengths[len(lengths)//2]}  "
          f"max {lengths[-1]}  >{MAX_SEQ_LEN}: {over_max} ({over_max/len(lengths):.2%})")

    if args.dry_run:
        print("dry run: no device opened, nothing trained.")
        return 0

    if not args.warm_start.is_file():
        raise FileNotFoundError(
            f"--warm-start {args.warm_start} does not exist. This is expected until the "
            f"corrected ctx2048 retrain (artifacts/checkpoints-tt-tnt-1024-ctx2048-v4corpus) "
            f"finishes and is verified -- pass --warm-start explicitly once it has."
        )

    import ttml  # noqa: F401 -- opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

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
            model, Path(args.warm_start),
            transformer_config=transformer_config, yaml_config={},
            moe_block_indices=[],
        )
        print(f"warm start: {warm_summary}")

        curve_path = out / "loss_curve.jsonl"
        recorder = LossRecorder(curve_path)
        trainer = SFTTrainer(
            model=model, train_dataloader=loader, eval_dataloader=val_loader,
            config=SFTConfig(max_steps=args.steps, learning_rate=args.lr, seed=args.seed,
                             max_seq_len=MAX_SEQ_LEN, checkpoint_dir=str(out),
                             save_interval=args.save_every,
                             eval_interval=args.eval_every if val_loader else 0,
                             log_interval=1, max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.01,
                       "stochastic_rounding": True},
            callbacks=[recorder],
        )
        assert_stochastic_rounding(trainer, "tool_calling")
        assert_eval_wired(trainer, val_size=len(val_examples), arm="tool_calling")

        trainer.train()
        print(f"loss curve: {curve_path}")
    finally:
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
