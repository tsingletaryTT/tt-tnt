# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""The run header every training entry point prints at startup.

`train/run.py` has always printed its shape — model size, sequence length, step budget —
because a log that does not say what it trained is not evidence of anything. The SFT entry
points (`train_skits.py`, `train_editor.py`, `train_tool_calling.py`, `train_editor_lora.py`)
grew up separately and printed almost none of it, so their logs recorded three thousand
progress-bar frames and no statement of what was being trained.

That is worth fixing on its own. It also happens to be exactly what `tt-toplike`'s Training
view reads: it parses the run's shape out of the log rather than being told, because a
harness that assembles its config in Python has no config file to point at
(`tt-toplike/src/workload/train/parse.rs`). The formats below are that parser's, and it was
written against **this project's** pretrain harness — its own comments name tt-tnt and
`tt-tnt-384.yaml`. The SFT path simply never held up the other end of a contract that
already existed.

Printing these lines does not change training. It changes what the log can be asked
afterwards, which is the same reason `train/run.py` prints them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def print_run_header(
    *,
    size_name: str,
    model_config: Path,
    steps: int,
    batch_size: int,
    seq_len: int,
    param_count: Optional[int] = None,
    arch: str = "blackhole",
    grad_accum: int = 1,
    scheduler: Optional[str] = None,
    print_fn=print,
) -> None:
    """State the run's shape, in the spellings `train/run.py` and the log parser both use.

    Each line is matched by prefix on the *trimmed* line, so leading indentation is free
    but the wording is not — ``Max steps`` has no colon and ``Number of parameters:`` does,
    because those are tt-train's own C++ spellings and the parser accepts both families.
    """
    # One line carrying the step budget, batch and sequence length together. Without a step
    # budget there is no progress fraction and no ETA for a ttml-driven run, because ttml's
    # trainer prints nothing per step.
    print_fn(f"tt-tnt training — steps={steps} batch={batch_size} "
             f"seq_len={seq_len} arch={arch}")
    # The topology YAML, named rather than passed: the consumer resolves it beneath the
    # trainer's own cwd. This is what turns "topology unknown" into a real model card.
    print_fn(f"  model size: {size_name} ({model_config.name})")
    print_fn(f"  seq_len: {seq_len} (SFT harness; examples are padded to this length)")
    if param_count is not None:
        print_fn(f"Number of parameters: {param_count}")
    print_fn(f"Max steps {steps}")
    print_fn(f"Batch size {batch_size}")
    print_fn(f"Gradient accumulation steps {grad_accum}")
    if scheduler:
        print_fn(f"Scheduler type {scheduler}")


def count_parameters(model: Any) -> Optional[int]:
    """Total elements across a model's parameters, from shapes alone.

    Shapes come from each tensor's own accessor, so nothing is copied off the device and no
    precision is coerced — the mistake `train/enthusiasts.py` records is reading a shape by
    way of `to_numpy()`, which under DDP moves a distributed tensor to the host purely to
    ask how big it is. Returns ``None`` rather than raising if a model does not expose
    shapes the way ttml's does; a missing header line is better than a failed run.
    """
    try:
        total = 0
        for _name, p in model.parameters().items():
            tensor = p.tensor if hasattr(p, "tensor") else p
            dims = list(tensor.shape())
            n = 1
            for d in dims:
                n *= int(d)
            total += n
        return total
    except Exception:
        return None


def format_eval_line(step: int, train_loss: float, val_loss: float,
                     lr: Optional[float] = None) -> str:
    """The per-eval line, in `train/run.py`'s exact shape.

    ``SFTTrainer`` reports validation only into a tqdm postfix and a jsonl file, so a run's
    log has no readable record of it at all. This restores one, and it is the same spelling
    the pretrain path already emits, so both halves of the project read alike.
    """
    lr_note = "" if lr is None else f" lr={lr:.3e}"
    return f"  step={step:>7} train_loss={train_loss:.4f} val_loss={val_loss:.4f}{lr_note}"
