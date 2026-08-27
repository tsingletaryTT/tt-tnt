#!/usr/bin/env python3
# scripts/train_editor.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Continued-training stage: editor pairs + poetry-instructions pairs + the existing skits
corpus, warm-started from tt-tnt-1024-dialogue.

Modeled directly on `scripts/train_skits.py`'s structure and quality gates -- this is a new
objective on the SAME dense architecture (no new tokens, no shape change), so every
mechanical piece of that script's device/training path transfers unchanged: `(1, 1)` mesh
only (a four-chip open hard-froze this host once, no reproduction, not retried),
`stochastic_rounding` asserted before and after, gamma-movement verified against the
warm-start base at the end.

Four example TYPES feed the same `InMemoryDataloader`: editor pairs (this file's
`build_editor_example`), poetry-instructions pairs (this file's `build_poetry_example`),
skits (`scripts.derive_skits.build_skit_example`, unchanged), and a base-blend refresh
(this file's `build_base_blend_example`, sampling random windows directly from
`artifacts/tokens-v3/train_ids.npy`). All four return `{"input_ids", "labels"}`, so they mix
freely in one list.

**The base-blend slice was added in a follow-up run (2026-08-27), after the first run
measured a real regression.** Design spec sec.4 mandates a MAJORITY base-blend
anti-forgetting slice; the first run implemented none (0%), training on 100% task data with
no anti-forgetting refresh -- exactly the failure mode spec sec.Risks warned about, and the
leading candidate cause of that run's measured regression (see CLAUDE.md). Unlike the other
three types, base-blend examples are NOT masked SFT pairs -- `labels == input_ids` with no
`-100` anywhere, ordinary next-token language modeling, matching how the base checkpoint
itself was trained. Sampling directly from the pre-tokenized `train_ids.npy` (rather than
re-tokenizing `artifacts/corpus/blend.txt`) is deliberate: that array already reflects the
corpus's settled per-source shares (it is what tt-tnt-1024-dialogue itself trained on), so no
new tokenization or share computation is needed.

    gozer run --chips 1 --who "claude:editor-training" --reason "editor+skits SFT" -- \
        python3 scripts/train_editor.py --steps 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_skits import build_skit_example  # noqa: E402
from scripts.train_skits import (  # noqa: E402
    LossRecorder,
    assert_eval_wired,
    assert_gammas_moved,
    assert_stochastic_rounding,
    compare_gammas,
    read_arm_gammas,
    read_base_gammas,
)
from train.skit import Skit  # noqa: E402

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
DEFAULT_PAIRS = ROOT / "artifacts" / "editor-pairs" / "pairs.jsonl"
DEFAULT_POETRY_PAIRS = ROOT / "artifacts" / "poetry-pairs" / "pairs.jsonl"
DEFAULT_SKITS = ROOT / "artifacts" / "skits-200k" / "skits.jsonl"
DEFAULT_BASE_BLEND_TOKENS = ROOT / "artifacts" / "tokens-v3" / "train_ids.npy"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "checkpoints-1024-editor"
#: The currently-designated dialogue checkpoint -- this run's warm-start base.
#: Read from docs/current_model.json at call time in main(); duplicated here only as the
#: literal default so --help shows something concrete.
DEFAULT_WARM_START = ROOT / "artifacts" / "checkpoints-1024-dialogue" / "tt_tnt_step00010764.pkl"
MAX_SEQ_LEN = 512


def _truncate_to_max_seq_len(input_ids: list, labels: list) -> Dict[str, list] | None:
    """Deterministically caps `input_ids`/`labels` at `MAX_SEQ_LEN`, from the END only --
    the prompt is never truncated, since the tail is where completion tokens (and the final
    masked position) live. `input_ids` and `labels` are the same length by construction, so
    one shared slice is correct.

    This is the same tradeoff `sft_collate_fn` already applies implicitly to skits examples
    that exceed the length limit (`scripts/train_skits.py`'s own docstring: losing the final
    supervised turn is accepted) -- made explicit and deterministic here instead of left to
    collate-time chance, per Task 3's real-data finding that poetry examples' word-count cap
    alone does not reliably bound token count (this corpus's real tokens/word ratio is
    heavy-tailed: median 1.71, p90 1.95, max 3.13).

    Returns `None` if truncation would leave ZERO real supervised positions (every label is
    `-100`) -- i.e. the prompt alone already meets or exceeds `MAX_SEQ_LEN`, so there is no
    completion token left to slice in. Training on such an example would be silent noise: a
    full forward/backward pass with no gradient signal from the loss. Dropping it is a
    genuinely different (better) failure mode than "truncated but some supervision remains".
    """
    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        labels = labels[:MAX_SEQ_LEN]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": input_ids, "labels": labels}


def build_editor_example(pair: Dict[str, str], tok, *, pad_token_id: int) -> Dict[str, list] | None:
    """`{"input_ids", "labels"}` for one (draft, better) pair, pre-shifted for ttml.

    Same boundary convention as every other SFT example in this project
    (`scripts/derive_traces.py::_sft_example_unaligned`): the LAST prompt position's label
    is the FIRST completion token (supervised, not masked -- that transition IS the trained
    behaviour), the LAST completion position is masked (no legitimate next token).

    Truncated to `MAX_SEQ_LEN` from the END ONLY (never the prompt) so every example fits
    the model's training window deterministically, not probabilistically -- see
    `_truncate_to_max_seq_len`. Returns `None` (to be filtered out by
    `build_combined_examples`) if the prompt alone is so long that truncation would leave no
    real supervised label.
    """
    prompt = f"\nDraft: {pair['draft']}\nEdit: "
    completion = pair["better"]
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)
    if not p_ids:
        raise ValueError("prompt tokenized to zero ids; label shifting requires at least "
                         "one prompt token")
    input_ids = p_ids + c_ids
    labels = [-100] * (len(p_ids) - 1) + c_ids + [-100]
    return _truncate_to_max_seq_len(input_ids, labels)


def build_poetry_example(pair: Dict[str, Any], tok, *, pad_token_id: int) -> Dict[str, list] | None:
    """`{"input_ids", "labels"}` for one poetry-instructions pair (Task 3's
    `scripts/build_poetry_pairs.py`). Same masking convention as `build_editor_example`;
    only the prompt template differs by `pair["kind"]`.

    Truncated to `MAX_SEQ_LEN` from the END ONLY (never the prompt), same as
    `build_editor_example` -- see `_truncate_to_max_seq_len`. Necessary in practice: this
    corpus's real tokens/word ratio (median 1.71, p90 1.95, max 3.13 -- archaic/rare
    vocabulary) is heavy-tailed enough that Task 3's word-count cap alone leaves ~55% of
    poetry examples over the model's 512-token window. Returns `None` if the prompt alone
    already consumes the whole window, leaving zero real supervised positions.
    """
    if pair["kind"] == "continuation":
        prompt = f"\nContinue this poem:\n{pair['input']}\nContinuation:\n"
    elif pair["kind"] == "keywords":
        prompt = f"\nWrite a poem about: {', '.join(pair['arg'])}\nPoem:\n"
    else:
        raise ValueError(f"unknown poetry pair kind: {pair['kind']!r}")
    completion = pair["target"]
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion, add_special_tokens=False)
    if not p_ids:
        raise ValueError("prompt tokenized to zero ids; label shifting requires at least "
                         "one prompt token")
    input_ids = p_ids + c_ids
    labels = [-100] * (len(p_ids) - 1) + c_ids + [-100]
    return _truncate_to_max_seq_len(input_ids, labels)


def load_editor_pairs(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_poetry_pairs(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_skits(path: Path) -> List[Skit]:
    """Reverses `Skit.as_dict()` (`train/skit.py`) -- there is no `from_dict` in that
    module, so `blocks` (serialized as plain dicts via each `Slots.as_dict()`) must be
    reconstructed into real `Slots` objects here; `"roles"` is dropped, since it is a
    derived constant (`SKIT_ROLES`) `as_dict()` writes for readability, not a real
    constructor field.
    """
    from train.improv import Slots  # NOT train.skit -- verified against train_skits.py's
                                     # own `from train.improv import Slots` import.

    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [
        Skit(story_id=r["story_id"], prefix=r["prefix"], turns=tuple(r["turns"]),
             blocks=tuple(Slots(**b) for b in r["blocks"]))
        for r in records
    ]


def build_category_lists(
    pairs: List[Dict[str, str]], poetry_pairs: List[Dict[str, Any]], skits: List[Skit],
    tok, *, pad_token_id: int,
) -> Dict[str, List[dict]]:
    """Editor examples, poetry-instructions examples, and skits examples
    (with_think=False -- this stage is about editing/turn-structure/poetry, not the
    separate think-block objective), each returned as its OWN list rather than one
    flattened list -- needed so a caller can stratify a held-out split across types
    (see `_stratified_split`) instead of taking the tail of a type-concatenated list,
    which silently held out 100% skits in this stage's first run.

    `build_editor_example`/`build_poetry_example` may each return `None` when truncation to
    `MAX_SEQ_LEN` would leave zero real supervised positions (prompt alone >= MAX_SEQ_LEN) --
    filtered out here so no zero-signal example reaches the dataloader.
    """
    editor_examples = [
        e for e in (build_editor_example(p, tok, pad_token_id=pad_token_id) for p in pairs)
        if e is not None
    ]
    poetry_examples = [
        e for e in (
            build_poetry_example(p, tok, pad_token_id=pad_token_id) for p in poetry_pairs
        )
        if e is not None
    ]
    skit_examples = [
        e for e in (
            build_skit_example(s, tok, with_think=False, pad_token_id=pad_token_id)
            for s in skits
        )
        if e is not None
    ]
    return {"editor": editor_examples, "poetry": poetry_examples, "skits": skit_examples}


def build_combined_examples(
    pairs: List[Dict[str, str]], poetry_pairs: List[Dict[str, Any]], skits: List[Skit],
    tok, *, pad_token_id: int,
) -> List[dict]:
    """Editor examples, then poetry-instructions examples, then skits examples,
    flattened into one list -- see `build_category_lists` for the per-type breakdown this
    is built from. Kept for callers that don't need per-type access."""
    categories = build_category_lists(pairs, poetry_pairs, skits, tok, pad_token_id=pad_token_id)
    return categories["editor"] + categories["poetry"] + categories["skits"]


def stratified_split(categories: Dict[str, List[dict]], val_size: int) -> tuple:
    """Hold out an even share of `val_size` from the TAIL of each category (deterministic,
    matching `train_skits.py`'s D3 precedent -- no shuffle before this split, so a re-run
    with the same inputs reproduces the same split), rather than the tail of one
    type-concatenated list, which can silently hold out 0% of some types entirely.

    Returns `(train_examples, val_examples)`, both flattened across categories in the
    same dict-iteration order. Each category contributes `val_size // len(categories)` (a
    category smaller than that contributes everything it has, per `min()` below, rather
    than raising or padding).
    """
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


def build_base_blend_example(window) -> Dict[str, list]:
    """Plain next-token language-modeling example from a contiguous corpus window.

    Full supervision (`labels == input_ids`, no `-100` anywhere) -- unlike
    `build_editor_example`/`build_poetry_example`/`build_skit_example`, which mask a
    prompt. That is what makes this an anti-forgetting REFRESH rather than a fourth
    task-specific slice: the loss signal is exactly the one the base checkpoint was
    already trained on, not a new objective.
    """
    ids = list(window)
    return {"input_ids": ids, "labels": list(ids)}


def sample_base_blend_examples(
    token_path: Path, n: int, *, seq_len: int = MAX_SEQ_LEN, seed: int
) -> List[dict]:
    """Sample `n` random `seq_len`-token windows from the pre-tokenized corpus stream at
    `token_path` (independently drawn -- overlap between two windows is possible, not
    systematically avoided). Reads via mmap so the ~1.4GB array is never fully loaded.
    """
    import numpy as np

    arr = np.load(token_path, mmap_mode="r")
    if len(arr) < seq_len:
        raise ValueError(
            f"{token_path} has only {len(arr)} tokens, fewer than seq_len={seq_len}"
        )
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        start = rng.randrange(0, len(arr) - seq_len + 1)
        window = arr[start : start + seq_len].tolist()
        examples.append(build_base_blend_example(window))
    return examples


def _pad_to_max_seq_len(example: dict, pad_token_id: int) -> dict:
    """Right-pads `input_ids`/`labels` to exactly `MAX_SEQ_LEN`.

    Necessary because of a real hardware failure (2026-08-27 training run, step 75):
    `sft_collate_fn` pads each batch only to that batch's OWN longest example (capped
    at `MAX_SEQ_LEN`), so batch shape varies step to step on this dataset's wide length
    distribution (min 12, median 473, max 512). ttml's SDPA backward kernel sizes an
    internal scratch tensor (`u_scaler`) from the FIRST batch it sees and asserts
    (`TT_FATAL: u_scaler shape mismatch`) rather than resizing when a LATER batch is
    larger -- it does not support a growing shape mid-run. Padding every example to the
    cap up front makes every batch exactly `(batch_size, MAX_SEQ_LEN)` from step 1
    onward, so no later batch can ever exceed the first one's shape. `input_ids` is
    padded with `pad_token_id`; `labels` with `-100`, so `sft_collate_fn`'s existing
    loss-masking already ignores every padded position -- same convention it already
    uses for its own per-batch padding, just applied uniformly instead of variably.
    """
    input_ids = list(example["input_ids"])
    labels = list(example["labels"])
    n = MAX_SEQ_LEN - len(input_ids)
    if n > 0:
        input_ids = input_ids + [pad_token_id] * n
        labels = labels + [-100] * n
    return {"input_ids": input_ids, "labels": labels}


def _current_dialogue_checkpoint() -> Path:
    manifest = json.loads((ROOT / "docs" / "current_model.json").read_text())
    checkpoints_dir = ROOT / manifest["current"]["checkpoints"]
    steps = sorted(checkpoints_dir.glob("tt_tnt_step*.pkl"))
    if not steps:
        raise FileNotFoundError(f"no checkpoints found under {checkpoints_dir}")
    return steps[-1]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--poetry-pairs", type=Path, default=DEFAULT_POETRY_PAIRS)
    ap.add_argument("--skits", type=Path, default=DEFAULT_SKITS)
    ap.add_argument("--base-blend-tokens", type=Path, default=DEFAULT_BASE_BLEND_TOKENS,
                    help="pre-tokenized corpus stream to sample the anti-forgetting "
                         "refresh from (design spec sec.4) -- already reflects the "
                         "corpus's settled per-source shares, so no re-tokenization or "
                         "share computation is needed")
    ap.add_argument("--base-blend-ratio", type=float, default=0.6,
                    help="target fraction of the FINAL combined example count that "
                         "should be base-blend refresh -- spec sec.4 calls this a "
                         "MAJORITY share; 0.6 means base-blend outnumbers editor+poetry"
                         "+skits combined by 1.5x. The first run (2026-08-27) shipped "
                         "at 0.0 and measured a real regression this is meant to fix")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
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

    # build_category_lists silently drops zero-supervision examples (prompt alone >=
    # MAX_SEQ_LEN after truncation); recompute the raw count here purely to report how many
    # were dropped, since that's a real measured number the task ruling asked to verify.
    task_data_count = sum(len(v) for v in categories.values())
    raw_count = len(pairs) + len(poetry_pairs) + len(skits)
    dropped = raw_count - task_data_count

    # Base-blend sized so it is `args.base_blend_ratio` of the FINAL combined count:
    # base_blend / (base_blend + task_data) == ratio  =>  base_blend = ratio/(1-ratio) * task_data.
    ratio = args.base_blend_ratio
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"--base-blend-ratio must be in [0, 1); got {ratio}")
    base_blend_count = round(ratio / (1 - ratio) * task_data_count) if ratio > 0 else 0
    categories["base_blend"] = (
        sample_base_blend_examples(args.base_blend_tokens, base_blend_count, seed=args.seed)
        if base_blend_count > 0 else []
    )

    examples = [e for exs in categories.values() for e in exs]

    # Stratified across ALL FOUR types, not a tail-of-one-list split -- the first run's
    # val set was silently 100% skits (see stratified_split's docstring and CLAUDE.md).
    val_size = max(0, min(args.val_size, len(examples) // 4))
    train_examples, val_examples = stratified_split(categories, val_size)

    lengths = sorted(len(e["input_ids"]) for e in examples)
    over_max = sum(1 for l in lengths if l > MAX_SEQ_LEN)
    realized_base_blend_share = len(categories["base_blend"]) / len(examples) if examples else 0.0
    print(f"pairs={len(pairs):,}  poetry_pairs={len(poetry_pairs):,}  "
          f"skits={len(skits):,}  base_blend={len(categories['base_blend']):,} "
          f"({realized_base_blend_share:.1%})  total_examples={len(examples):,}  "
          f"train={len(train_examples):,}  val={len(val_examples):,}  "
          f"dropped_zero_supervision={dropped:,}")
    print(f"token lengths: min {lengths[0]}  median {lengths[len(lengths)//2]}  "
          f"max {lengths[-1]}  >{MAX_SEQ_LEN}: {over_max} ({over_max/len(lengths):.2%})")

    if args.dry_run:
        print("dry run: no device opened, nothing trained.")
        return 0

    import ttml  # noqa: F401 -- opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    # (1, 1) ONLY -- see this file's module docstring and this plan's Global Constraints.
    ttml.open_device_mesh((1, 1))
    try:
        # Pad every example to a FIXED MAX_SEQ_LEN here -- not inside build_category_lists
        # -- so the length report printed above still reflects real, unpadded content.
        # See _pad_to_max_seq_len's docstring for why this is load-bearing, not cosmetic.
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

        # Exact call shape verified against scripts/train_skits.py's own main() -- do not
        # simplify this signature; warm_start requires transformer_config/yaml_config.
        warm_summary = warm_start(
            model, Path(warm_start_ckpt),
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
        # Both guards imported from scripts/train_skits.py -- same exact checks that caught
        # skits stage 1's 0%-adherence bug, reused rather than reimplemented. "editor" is
        # passed as the `arm` label (these functions are generic; the label is just what
        # gets printed).
        assert_stochastic_rounding(trainer, "editor")
        assert_eval_wired(trainer, val_size=len(val_examples), arm="editor")

        trainer.train()
        recorder.close()

        loss_start = recorder.history[0] if recorder.history else (None, None)
        loss_end = recorder.history[-1] if recorder.history else (None, None)
        print(f"loss: step {loss_start[0]}={loss_start[1]}  ->  step {loss_end[0]}={loss_end[1]}")

        # Guard 2 (post-training): did the gammas actually move against the warm-start
        # base? Same read_base_gammas/read_arm_gammas/compare_gammas/assert_gammas_moved
        # pipeline train_skits.py uses -- read_base_gammas expects a ttml-format
        # checkpoint (the dialogue warm-start base), read_arm_gammas expects an
        # SFTTrainer-format `{"step","model_state"}` pickle (this run's own output) -- the
        # two formats are NOT interchangeable, which is exactly why there are two readers.
        final_ckpt = out / f"step_{args.steps}.pkl"
        if final_ckpt.exists():
            report = compare_gammas(read_base_gammas(Path(warm_start_ckpt)),
                                    read_arm_gammas(final_ckpt))
            assert_gammas_moved(report, arm="editor", base_path=Path(warm_start_ckpt),
                                arm_path=final_ckpt)
            print(f"gammas moved: {report['all_moved']}  "
                  f"({report['total_changed']}/{report['total_elements']} elements)")
        else:
            raise FileNotFoundError(f"expected final checkpoint at {final_ckpt}")

        print(f"training complete -> {out}")
        return 0
    finally:
        ttml.close_device_mesh()


if __name__ == "__main__":
    raise SystemExit(main())
