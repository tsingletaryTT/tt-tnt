#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Stage 2: two paired SFT arms over five-turn skits -- `think` vs `nothink`.

PAIRED BY CONSTRUCTION. Both arms read the SAME skits file, build one example per skit in
FILE ORDER, and hand that list to an ``InMemoryDataloader`` with the same ``seed`` and the
same ``batch_size``. That loader's shuffle is a pure function of ``(seed, epoch)`` and the
dataset length (``InMemoryDataloader._epoch_indices``), and both arms have identical
lengths, so both arms visit the same skits in the same order at every step. Per-step noise
therefore cancels in the arm-vs-arm delta, and the ONLY difference between the arms is
whether the think-block segments are present. The manifest records a fingerprint of the
first epoch's index order (read off the constructed loader, not recomputed from a copy of
its formula) so the pairing is checkable after the fact rather than merely asserted here.

    gozer run --chips 1 --who "claude:skits" --reason "skit SFT arm think" -- \
        python3 scripts/train_skits.py --arm think --steps 3000
    gozer run --chips 1 --who "claude:skits" --reason "skit SFT arm nothink" -- \
        python3 scripts/train_skits.py --arm nothink --steps 3000

Run the arms SEQUENTIALLY, on ONE BOARD, with a ``(1, 1)`` mesh. A four-chip mesh open
hard-froze this host once with no OOM, no kernel panic and no pstore record; there is no
reproduction and no fix, so it is not retried.

This is `scripts/train_improv.py` (stage 1, committed and working) with four deltas, each
noted at the code that implements it:

  D1  reads skits from `artifacts/skits-200k/skits.jsonl` and rebuilds `Skit` objects,
      rather than reading flat trace dicts;
  D2  builds examples with `scripts.derive_skits.build_skit_example` (nine alternating
      segments, positional supervision mask) rather than `build_sft_examples`;
  D3  holds out a validation split -- identical in both arms -- and enables
      `SFTConfig.eval_interval`, so the curve can distinguish learning from template
      memorisation. Stage 1 shipped with `eval_interval=0` and no eval dataloader, and
      the whole validation curve was silently absent;
  D4  verifies the 17 RMSNorm gammas actually MOVED against the warm-start base once
      training finishes, and refuses to call the run valid if any of them is
      bit-identical.

WHY `skits-200k` AND NOT `skits`
--------------------------------
`artifacts/skits/skits.jsonl` is a 20,000-story derivation that kept 1,921 skits (90.4%
drop). The three sequential accept/add gates compound multiplicatively (~0.46^3), so the
yield is a property of the gates, not a defect -- and those gates ARE the falsifiable
prediction that distinguishes stage 2 from stage 1, so they must not be relaxed to buy
examples. The fix is more input: `artifacts/skits-200k/skits.jsonl` is the same derivation
over 200,000 stories and keeps 18,610, within 1.0% of stage 1's 18,791 training examples,
which makes the two stages comparable on data volume. Both files are preserved; the one
actually used is recorded in `train_manifest.json`.

THE GUARD THAT MATTERS
----------------------
`stochastic_rounding` defaults **off** in ttml's optimizer registry. In bfloat16 the ulp at
1.0 is 0.0039, an order of magnitude larger than the ~3e-4 Adam updates the 17 RMSNorm
gamma tensors receive, so with deterministic rounding every one of those updates rounds
straight back to 1.0 and is discarded -- silently, for the entire run. Stage 1 shipped both
arms that way and measured 0% schema adherence; the same arms measured 98% once the flag
was on. Two guards here, at the two layers that can fail:

  * before training -- read the flag back out of `trainer.optimizer.get_state_dict()`, the
    optimizer the C++ factory actually built, and RAISE if it is off. Checking the dict we
    passed in would prove intent only, and a warning is what let the bug through the first
    time;
  * after training -- compare the final checkpoint's 17 gammas against the warm-start base
    and RAISE if any is bit-identical. The pre-training guard proves the flag reached the
    optimizer; only this one proves the parameters moved.

Also available without a device, for checking an existing run::

    python3 scripts/train_skits.py --verify-gammas artifacts/skits/ckpt-think/step_3000.pkl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from convert.checkpoint_reader import read_tensors  # noqa: E402
from scripts.derive_skits import build_skit_example  # noqa: E402
from train.improv import Slots  # noqa: E402
from train.skit import SKIT_ROLES, Skit  # noqa: E402

MODEL_YAML = ROOT / "train" / "configs" / "model" / "tt-tnt-1024.yaml"
TOKENIZER_DIR = ROOT / "artifacts" / "hf-tt-tnt-1024"
WARM_START_CKPT = (ROOT / "artifacts" / "checkpoints-v077-beta2-control"
                   / "tt_tnt_step00010764.pkl")

#: D1 -- the 200k derivation (18,610 skits), NOT artifacts/skits/skits.jsonl (1,921).
#: See "WHY skits-200k AND NOT skits" in the module docstring.
DEFAULT_SKITS = ROOT / "artifacts" / "skits-200k" / "skits.jsonl"

#: Checkpoints and curves land here regardless of which skits file was read -- this is the
#: path Task 5's evaluation consumes. The skits file actually used is in the manifest.
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "skits"

#: Every RMSNorm scale in this architecture: 8 blocks x {attention_norm, mlp_norm} plus
#: the final llama/ln_fc = 17. Verified twice, against real files rather than the config:
#: the warm-start checkpoint's manifest holds 66 tensors of which exactly 17 end in
#: "/gamma", and a trained checkpoint enumerates llama_block_{0..7} plus llama/ln_fc.
#: (The plan's ledger says "blocks 0..15", which would be 33 -- it is the count, not the
#: block range, that this constant is pinned to, and the count is what was measured.)
EXPECTED_GAMMAS = 17

MAX_SEQ_LEN = 512


# --------------------------------------------------------------------------------------
# D1: skits.jsonl -> Skit objects
# --------------------------------------------------------------------------------------
def load_skits(path: Path) -> List[Skit]:
    """Rebuild `Skit` objects from a `skits.jsonl` written by `scripts/derive_skits.py`.

    ORDER IS LOAD-BEARING and this function must never sort, filter or deduplicate: the
    arms are paired by consuming the same examples in the same order, and the dataloader's
    shuffle permutes POSITIONS, not story ids. Reordering here would silently unpair the
    two arms while leaving every count identical.

    The `roles` field is validated rather than ignored. It is redundant with `SKIT_ROLES`
    for a well-formed file, which is exactly why an unchecked mismatch would be invisible:
    a file whose roles disagree is not a skits file this trainer understands, and the
    partner/model supervision split downstream would be wrong without any other symptom.
    """
    skits: List[Skit] = []
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            roles = tuple(rec.get("roles", ()))
            if roles != SKIT_ROLES:
                raise ValueError(
                    f"{path}:{lineno}: roles {roles} != {SKIT_ROLES}; this is not a skits "
                    f"file this trainer can supervise correctly")
            turns = tuple(rec["turns"])
            if len(turns) != len(SKIT_ROLES):
                raise ValueError(f"{path}:{lineno}: expected {len(SKIT_ROLES)} turns, "
                                 f"got {len(turns)}")
            skits.append(Skit(
                story_id=rec["story_id"],
                prefix=rec["prefix"],
                turns=turns,
                blocks=tuple(Slots(**b) for b in rec["blocks"]),
            ))
    return skits


def with_think_for_arm(arm: str) -> bool:
    """`--arm` -> the `with_think` flag, in one place so the two cannot drift apart.

    Named and tested separately rather than inlined as `arm == "think"`: this single
    boolean IS the independent variable of the entire experiment, and an inverted or
    constant value produces two runs that look perfectly healthy and mean nothing.
    """
    if arm == "think":
        return True
    if arm == "nothink":
        return False
    raise ValueError(f"unknown arm {arm!r}; expected 'think' or 'nothink'")


def build_arm_examples(skits: List[Skit], tok, *, arm: str, pad_token_id: int) -> List[dict]:
    """D2 -- one `{"input_ids", "labels", "story_id"}` per skit, in the given order.

    `story_id` rides along for provenance; `sft_collate_fn` reads only `input_ids` and
    `labels` and ignores anything else. It is what makes "both arms consumed the same
    skits in the same order" checkable from the two runs' artifacts instead of assumed.
    """
    with_think = with_think_for_arm(arm)
    out: List[dict] = []
    for skit in skits:
        ex = build_skit_example(skit, tok, with_think=with_think,
                                pad_token_id=pad_token_id)
        ex["story_id"] = skit.story_id
        out.append(ex)
    return out


def length_stats(examples: List[dict], max_seq_len: int = MAX_SEQ_LEN) -> Dict[str, Any]:
    """Token-length distribution, and how many examples the collate function will TRUNCATE.

    `sft_collate_fn` caps each batch at `max_seq_len` by slicing, silently. An example
    longer than the cap loses its tail -- which for a skit is its last supervised model
    turn -- and the think arm's examples are the longer of the two by construction, so
    truncation does not fall equally on the arms. That would not be noise; it would be a
    systematic difference between the arms that is not the think-block. Counted and
    recorded, not assumed to be zero.
    """
    lens = np.array([len(e["input_ids"]) for e in examples], dtype=np.int64)
    over = int((lens > max_seq_len).sum())
    return {
        "n": int(lens.size),
        "min": int(lens.min()) if lens.size else 0,
        "median": int(np.median(lens)) if lens.size else 0,
        "p99": int(np.percentile(lens, 99)) if lens.size else 0,
        "max": int(lens.max()) if lens.size else 0,
        "over_max_seq_len": over,
        "over_max_seq_len_frac": round(over / max(lens.size, 1), 6),
        "max_seq_len": max_seq_len,
    }


# --------------------------------------------------------------------------------------
# Loss / eval curve
# --------------------------------------------------------------------------------------
class LossRecorder:
    """Writes EVERY step's `(step, loss, lr)` to `loss_curve.jsonl`, plus each eval point.

    `SFTTrainer.train()` returns nothing and only feeds the loss into a tqdm postfix, so a
    callback on `on_step_end` is the only way to get the numbers out. Duck-typed rather
    than subclassing `TrainerCallback`: the trainer calls exactly the hooks it fires.

    EVERY step, not a sample: the question this instrumentation exists to answer is
    whether the loss falls in a fast early collapse (template memorisation) or declines
    throughout (learning), and a sparse sample misrepresents precisely the early region
    where that distinction is sharpest. Two endpoints cannot show a trajectory at all.
    Flushed after each line, so a mid-run kill still leaves the curve up to that point.

    Train and eval points share one file and are told apart by `"split"`. Eval points are
    only ever written if the trainer actually ran an eval -- `on_eval_end` fires from
    inside the `eval_interval > 0 and eval_dataloader is not None` branch, so an empty
    `val` series in this file is evidence the eval was disabled, not that it was flat.
    """

    def __init__(self, curve_path: Path) -> None:
        self.history: List[Tuple[int, float]] = []
        self.eval_history: List[Tuple[int, float]] = []
        self.curve_path = curve_path
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = curve_path.open("w")

    def _write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def on_train_begin(self, trainer) -> None:
        pass

    def on_after_forward(self, trainer, batch, loss) -> None:
        pass

    def on_after_backward(self, trainer, batch) -> None:
        pass

    def on_step_end(self, trainer, step, *args, **kwargs) -> None:
        # SFTTrainer calls cb.on_step_end(self, self.step, step_loss, lr).
        if not args:
            return
        loss = float(args[0])
        lr = float(args[1]) if len(args) > 1 else None
        self.history.append((step, loss))
        self._write({"step": step, "split": "train", "loss": loss, "lr": lr})

    def on_eval_end(self, trainer, step, eval_loss) -> None:
        loss = float(eval_loss)
        self.eval_history.append((step, loss))
        self._write({"step": step, "split": "val", "loss": loss, "lr": None})

    def on_before_optimizer_step(self, trainer) -> None:
        pass

    def on_save(self, trainer, step, path) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------------------
# Guard 1 (pre-training): the optimizer really has stochastic rounding on
# --------------------------------------------------------------------------------------
def assert_stochastic_rounding(trainer, arm: str) -> None:
    """RAISE unless the CONSTRUCTED optimizer reports stochastic rounding enabled.

    Reads `trainer.optimizer.get_state_dict()` -- the object ttml's C++ optimizer factory
    actually built -- and never the dict passed into `SFTTrainer`. The input dict only
    records what was asked for; an optimizer that ignored the key entirely (a typo, a
    renamed field, a registry that does not accept it on this optimizer type) would still
    leave the request sitting there looking correct. This is the layer that fails, so this
    is where the assertion goes.

    Raises rather than warns. Stage 1 had a warning for this exact condition
    (`train/run.py::_warn_if_stochastic_rounding_disabled`), and 3000 steps of two arms
    went past it unnoticed in a long log.
    """
    state = trainer.optimizer.get_state_dict()
    enabled = bool(state.get("stochastic_rounding", False))
    print(f"[{arm}] optimizer stochastic_rounding (read back from optimizer state): "
          f"{enabled}")
    if not enabled:
        raise RuntimeError(
            f"stochastic_rounding is NOT enabled on the '{arm}' arm's optimizer -- the 17 "
            "RMSNorm gamma parameters cannot move in bfloat16 (the ulp at 1.0 is 0.0039, "
            "an order of magnitude larger than the ~3e-4 updates these gammas receive), "
            "so this run would train with them frozen and would be void. Refusing to "
            "start. Set \"stochastic_rounding\": True in the optimizer dict."
        )


def assert_eval_wired(trainer, *, val_size: int, arm: str) -> None:
    """RAISE if a validation split was asked for but the trainer will never evaluate it.

    D3. `SFTConfig.eval_interval` defaults to 200 but the eval branch ALSO requires
    `eval_dataloader is not None`, and stage 1 passed `eval_dataloader=None` with
    `eval_interval=0` -- two independent ways to get a silently empty validation curve
    that nothing else in the run distinguishes from a curve that was simply flat. The
    guard reads both back off the constructed trainer.
    """
    if val_size <= 0:
        return
    interval = int(getattr(trainer.config, "eval_interval", 0) or 0)
    if trainer.eval_dataloader is None or interval <= 0:
        raise RuntimeError(
            f"[{arm}] a validation split of {val_size} examples was held out, but the "
            f"trainer would never evaluate it (eval_dataloader="
            f"{'None' if trainer.eval_dataloader is None else 'set'}, "
            f"eval_interval={interval}). The validation curve would be silently empty."
        )
    print(f"[{arm}] eval wired: {val_size} held-out examples, every {interval} steps")


# --------------------------------------------------------------------------------------
# Guard 2 (post-training): the gammas actually moved
# --------------------------------------------------------------------------------------
def read_base_gammas(path: Path) -> Dict[str, np.ndarray]:
    """The 17 `/gamma` tensors from a ttml-format checkpoint (the warm-start base).

    Streams record-by-record through `convert.checkpoint_reader`, which is pure stdlib
    pickle plus numpy: no ttml, no ttnn, no device. So this half of the comparison can be
    run on a machine with no hardware, and -- more to the point -- it can be run in the
    same process as a training run without perturbing anything on the device.
    """
    out: Dict[str, np.ndarray] = {}
    for name, arr in read_tensors(Path(path), "model"):
        if name.endswith("/gamma"):
            out[name] = np.asarray(arr)
    return out


def read_arm_gammas(path: Path) -> Dict[str, np.ndarray]:
    """The 17 `/gamma` tensors from an `SFTTrainer` checkpoint.

    NOTE the format difference, which is not cosmetic: `SFTTrainer._save_checkpoint`'s
    default saver writes a plain pickle of `{"step", "model_state"}` with FLOAT32 numpy
    arrays, NOT the ttml `save_checkpoint` record stream that `read_base_gammas` reads.
    Feeding either file to the other reader fails loudly rather than quietly, which is why
    the two readers are separate functions instead of one that sniffs the format.
    """
    with Path(path).open("rb") as fh:
        blob = pickle.load(fh)
    if not isinstance(blob, dict) or "model_state" not in blob:
        raise ValueError(f"{path}: not an SFTTrainer checkpoint "
                         f"(no 'model_state'; keys={sorted(blob) if isinstance(blob, dict) else type(blob)})")
    return {k: np.asarray(v) for k, v in blob["model_state"].items()
            if k.endswith("/gamma")}


def compare_gammas(base: Dict[str, np.ndarray], arm: Dict[str, np.ndarray],
                   *, expected: int = EXPECTED_GAMMAS) -> Dict[str, Any]:
    """Per-tensor EXACT comparison of two gamma sets. A tensor that did not move is fatal.

    EXACT, not `allclose`. The failure being guarded against is not "the gammas moved a
    little less than expected" but "not one bit of one element changed in 3000 steps",
    which is what deterministic bfloat16 rounding produces. A tolerance would classify a
    genuine single-ulp update -- the smallest real movement bfloat16 can represent, and
    the thing stochastic rounding exists to accumulate -- as no movement at all, turning
    the guard into the bug it was written to catch. Casting bfloat16 up to float32 is
    exact, so the two sides are directly comparable.

    The name sets must match EXACTLY. Comparing an intersection would report happily on
    whatever the two files agree about while a renamed or missing tensor went unnoticed --
    including the case where the arm checkpoint has no gammas at all, which would
    otherwise pass as "0 frozen of 0".
    """
    if set(base) != set(arm):
        only_base, only_arm = sorted(set(base) - set(arm)), sorted(set(arm) - set(base))
        raise ValueError(
            f"gamma name sets differ; refusing to compare an intersection. "
            f"only in base: {only_base}; only in arm: {only_arm}")
    if len(base) != expected:
        raise ValueError(f"expected {expected} gamma tensors, found {len(base)}: "
                         f"{sorted(base)}")

    per: Dict[str, Any] = {}
    frozen: List[str] = []
    total_elems = total_changed = 0
    for name in sorted(base):
        b = np.asarray(base[name]).astype(np.float32)
        a = np.asarray(arm[name]).astype(np.float32)
        if b.shape != a.shape:
            raise ValueError(f"{name}: shape {b.shape} (base) != {a.shape} (arm)")
        diff = a - b
        changed = int(np.count_nonzero(diff))
        total_elems += int(b.size)
        total_changed += changed
        per[name] = {
            "elements": int(b.size),
            "changed": changed,
            "changed_frac": round(changed / max(b.size, 1), 6),
            "max_abs_delta": float(np.abs(diff).max()) if b.size else 0.0,
            "mean_abs_delta": float(np.abs(diff).mean()) if b.size else 0.0,
        }
        if changed == 0:
            frozen.append(name)

    return {
        "n_tensors": len(per),
        "frozen": frozen,
        "all_moved": not frozen,
        "total_elements": total_elems,
        "total_changed": total_changed,
        "total_changed_frac": round(total_changed / max(total_elems, 1), 6),
        "per_tensor": per,
    }


def assert_gammas_moved(report: Dict[str, Any], *, arm: str, base_path: Path,
                        arm_path: Path) -> None:
    """RAISE if any gamma is bit-identical to the warm-start base. The run is then void."""
    print(f"[{arm}] gammas vs warm-start base: {report.get('n_tensors', 0)} tensors, "
          f"{report.get('total_changed', 0):,}/{report.get('total_elements', 0):,} "
          f"elements changed ({report.get('total_changed_frac', 0.0):.1%}), "
          f"frozen: {len(report.get('frozen', []))}")
    for name, st in sorted(report.get("per_tensor", {}).items()):
        print(f"    {name:52} {st['changed']:>5}/{st['elements']:<5} changed  "
              f"max|d|={st['max_abs_delta']:.5f}")
    if not report["all_moved"]:
        raise RuntimeError(
            f"[{arm}] {len(report.get('frozen', []))} of {report.get('n_tensors', 0)} "
            f"RMSNorm gamma "
            f"tensors are BIT-IDENTICAL to the warm-start base after training:\n  "
            + "\n  ".join(report.get("frozen", []))
            + f"\n\nbase: {base_path}\narm:  {arm_path}\n"
            "stochastic_rounding did not take effect and this arm is VOID -- do not "
            "evaluate it. Fix the flag and re-run."
        )


def verify_gammas_cli(arm_ckpt: Path, base_ckpt: Path = WARM_START_CKPT) -> int:
    """`--verify-gammas`: run guard 2 alone, on CPU, against an existing checkpoint."""
    report = compare_gammas(read_base_gammas(base_ckpt), read_arm_gammas(arm_ckpt))
    assert_gammas_moved(report, arm=arm_ckpt.parent.name, base_path=base_ckpt,
                        arm_path=arm_ckpt)
    print("OK: every gamma moved.")
    return 0


# --------------------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """Repo-relative path when the path is under the repo, absolute otherwise.

    `--out-root` may legitimately point outside the repo (a scratch dir for a smoke run),
    and `Path.relative_to` RAISES rather than falling back in that case -- which, placed
    inside the manifest-writing block, threw away a completed run's manifest after the
    training itself had succeeded. Observed exactly once, on the 20-step smoke.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _order_fingerprint(loader, epoch: int = 0) -> Dict[str, Any]:
    """A checkable fingerprint of the order this loader will visit examples in.

    Read off the CONSTRUCTED loader (`_epoch_indices`), not recomputed from a copy of its
    permutation formula: a copy would agree with itself forever, including after ttml
    changed how it shuffles. Both arms record this; identical fingerprints across the two
    manifests is the evidence that the comparison is genuinely paired. Private attribute,
    so its absence is reported rather than raised -- a missing fingerprint must not kill a
    3000-step run, it just leaves the pairing unfingerprinted.
    """
    if not hasattr(loader, "_epoch_indices"):
        return {"source": "unavailable (loader has no _epoch_indices)", "sha256": None}
    idx = np.asarray(loader._epoch_indices(epoch))
    return {
        "source": "loader._epoch_indices(0)",
        "n": int(idx.size),
        "first_8": [int(i) for i in idx[:8]],
        "sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["think", "nothink"],
                    help="which paired arm to train (required unless --verify-gammas)")
    ap.add_argument("--skits", type=Path, default=DEFAULT_SKITS,
                    help="skits.jsonl to train on (default: the 200k derivation)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=5489)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--val-size", type=int, default=256,
                    help="skits held out for validation (identical in both arms; 0 = off)")
    ap.add_argument("--eval-every", type=int, default=250,
                    help="SFTConfig.eval_interval -- steps between validation passes. "
                         "0 disables evaluation entirely, which is what silently cost "
                         "stage 1 its whole validation curve.")
    ap.add_argument("--warm-start", type=Path, default=WARM_START_CKPT)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the examples and report the length distribution, then "
                         "stop. Touches no device, so it needs no lease.")
    ap.add_argument("--verify-gammas", type=Path, default=None,
                    help="compare an existing step_*.pkl's gammas against the warm-start "
                         "base and exit. Touches no device.")
    args = ap.parse_args(argv)

    if args.verify_gammas is not None:
        return verify_gammas_cli(args.verify_gammas, args.warm_start)
    if args.arm is None:
        ap.error("--arm is required unless --verify-gammas is given")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_token_id = tok.pad_token_id or 0

    skits = load_skits(args.skits)
    examples = build_arm_examples(skits, tok, arm=args.arm, pad_token_id=pad_token_id)

    # D3 -- the hold-out is the TAIL of the file in file order, so it is a pure function of
    # the skits file and the split size: identical in both arms, and identical on a re-run.
    # No RNG is involved, deliberately -- a seeded shuffle here would be one more thing that
    # has to match across the two arms for the pairing to hold.
    val_size = max(0, min(args.val_size, len(examples) // 4))
    train_examples = examples[:len(examples) - val_size] if val_size else examples
    val_examples = examples[len(examples) - val_size:] if val_size else []

    stats = length_stats(examples)
    print(f"arm={args.arm}  skits={args.skits}  skits_read={len(skits):,}  "
          f"train={len(train_examples):,}  val={len(val_examples):,}  seed={args.seed}")
    print(f"token lengths: min {stats['min']}  median {stats['median']}  "
          f"p99 {stats['p99']}  max {stats['max']}  "
          f">{MAX_SEQ_LEN}: {stats['over_max_seq_len']} "
          f"({stats['over_max_seq_len_frac']:.2%})")
    if stats["over_max_seq_len"]:
        print(f"NOTE: {stats['over_max_seq_len']} example(s) exceed max_seq_len="
              f"{MAX_SEQ_LEN} and WILL be truncated by sft_collate_fn, losing their final "
              f"supervised turn. The think arm's examples are longer by construction, so "
              f"this does not fall equally on the two arms. Recorded in the manifest.")

    if args.dry_run:
        print("dry run: no device opened, nothing trained.")
        return 0

    import ttml  # noqa: F401 -- opens the UMD cluster; MUST run under a gozer lease
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig, SFTTrainer

    # SFTTrainer.__init__ -> _build_loss_fn() calls ttml.mesh(), a module-global that only
    # open_device_mesh populates. This repo's initialize_device()/AutoContext.open_device()
    # opens a device but leaves that global empty ("Device mesh is not initialized").
    # (1, 1) ONLY: one board, one chip's worth of mesh. A 4-chip open hard-froze this host.
    yaml_config: dict = {}
    ttml.open_device_mesh((1, 1))
    try:
        collate = partial(sft_collate_fn, max_seq_len=MAX_SEQ_LEN,
                          pad_token_id=pad_token_id)
        loader = InMemoryDataloader(train_examples, batch_size=args.batch_size,
                                    collate_fn=collate, shuffle=True, seed=args.seed)
        val_loader = (InMemoryDataloader(val_examples, batch_size=args.batch_size,
                                         collate_fn=collate, shuffle=False,
                                         seed=args.seed)
                      if val_examples else None)
        order = _order_fingerprint(loader)
        print(f"batch order fingerprint (must match the other arm): {order}")

        out = Path(args.out_root).resolve() / f"ckpt-{args.arm}"
        out.mkdir(parents=True, exist_ok=True)

        from train.model import create_model

        model_yaml = yaml.safe_load(MODEL_YAML.read_text())
        transformer_config = model_yaml["transformer_config"]
        model = create_model(yaml_config, transformer_config)

        # Dense model, dense checkpoint, identical shape: every parameter name is shared,
        # so moe_block_indices=[] and the whole checkpoint transfers with nothing fresh.
        from train.enthusiasts import warm_start

        warm_summary = warm_start(
            model, Path(args.warm_start),
            transformer_config=transformer_config, yaml_config=yaml_config,
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
                             log_interval=1,  # per-step curve: fires on_step_end every step
                             max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": args.lr, "weight_decay": 0.01,
                       "stochastic_rounding": True},
            callbacks=[recorder],
        )
        assert_stochastic_rounding(trainer, args.arm)
        assert_eval_wired(trainer, val_size=len(val_examples), arm=args.arm)

        trainer.train()
        recorder.close()

        loss_start = recorder.history[0] if recorder.history else (None, None)
        loss_end = recorder.history[-1] if recorder.history else (None, None)
        print(f"loss: step {loss_start[0]}={loss_start[1]}  ->  "
              f"step {loss_end[0]}={loss_end[1]}")
        if recorder.eval_history:
            print(f"val loss: step {recorder.eval_history[0][0]}="
                  f"{recorder.eval_history[0][1]}  ->  step "
                  f"{recorder.eval_history[-1][0]}={recorder.eval_history[-1][1]}")

        # --- guard 2: did the gammas actually move? -------------------------------------
        final_ckpt = out / f"step_{args.steps}.pkl"
        gamma_report: Dict[str, Any]
        if final_ckpt.exists():
            gamma_report = compare_gammas(read_base_gammas(Path(args.warm_start)),
                                          read_arm_gammas(final_ckpt))
        else:
            gamma_report = {"error": f"no final checkpoint at {final_ckpt}",
                            "all_moved": False, "frozen": ["<checkpoint missing>"],
                            "n_tensors": 0}

        manifest = {
            "arm": args.arm,
            "paired_with": "nothink" if args.arm == "think" else "think",
            "seed": args.seed,
            # WHICH skits file this arm trained on. artifacts/skits/skits.jsonl (1,921
            # skits, 20k stories) is too small; the 200k derivation is the default here.
            "skits": str(args.skits),
            "skits_read": len(skits),
            "n_examples": len(train_examples),
            "n_val_examples": len(val_examples),
            "val_split": "tail of the skits file in file order (no RNG; identical in "
                         "both arms)",
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_interval": args.eval_every if val_loader else 0,
            "log_interval": 1,
            # Declared here AND verified before training via
            # trainer.optimizer.get_state_dict(), then verified again after training by
            # comparing the 17 gammas against the warm-start base (gamma_check below).
            "stochastic_rounding": True,
            "max_seq_len": MAX_SEQ_LEN,
            "mesh": "(1,1)",
            "example_token_lengths": stats,
            "batch_order_fingerprint": order,
            "warm_start": _rel(Path(args.warm_start)),
            "warm_start_summary": warm_summary,
            "checkpoint_dir": _rel(out),
            "loss_curve": _rel(curve_path),
            "loss_curve_note": "every step logged (not sampled); split=train|val",
            "loss_step_start": loss_start[0],
            "loss_start": loss_start[1],
            "loss_step_end": loss_end[0],
            "loss_end": loss_end[1],
            "val_loss_first": (recorder.eval_history[0] if recorder.eval_history
                               else None),
            "val_loss_last": (recorder.eval_history[-1] if recorder.eval_history
                              else None),
            "gamma_check": gamma_report,
        }
        (out / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {out / 'train_manifest.json'}")

        # Manifest first, then raise: a void run must still leave its evidence on disk.
        assert_gammas_moved(gamma_report, arm=args.arm, base_path=Path(args.warm_start),
                            arm_path=final_ckpt)
    finally:
        # Mirror of open_device_mesh; also clears ttml's global mesh state. Skipping the
        # teardown aborts in MetalContext::destroy_all_instances.
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
