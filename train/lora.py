# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""LoRA support shared by this project's SFT entry points.

Three things live here, all of which were previously duplicated in (or missing from)
``scripts/train_editor_lora.py``:

1. :class:`TtTntLoraModel` — ``LoraModel`` with canonical parameter naming.
2. :func:`base_parameter_snapshot` / :func:`assert_base_frozen` — the freeze is LoRA's entire
   premise here, so it is verified rather than assumed.
3. :func:`merge_lora_state` — folding the adapter back into the base weights, which is
   mandatory before HF conversion.

**Every read in this module passes ``precision=NATIVE``.** Our parameters are bfloat16, so
``AutocastTensor`` keeps them in the half slot and caches the first ``FULL`` read's
bf16→fp32 typecast forever (tt-metal ``autograd/autocast_tensor.cpp``, #41657). A ``FULL``
read here would report a perfect freeze and a motionless ``lora_B`` whether or not either were
true — which is exactly the artefact that made this project wrongly conclude LoRA was blocked
upstream. See ``docs/upstream-tt-metal-asks.md`` entry 5.

**Why LoRA at this model size at all.** Not efficiency — full fine-tuning this model takes nine
minutes. The reason is the anti-forgetting guarantee: with 100% of base weights frozen, general
fluency cannot regress, so a run may spend 100% of each step on task data. Full-parameter SFT on
100% task data regressed 4-gram repeat to 21.76x the seed floor and longest repeated span to
30.17x; the fix was a 60% base-blend counterweight, which costs 60% of every step.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

#: ttml's own reference example targets the three attention projections
#: (``tt-metal/tt-train/sources/examples/lora_llama/``). MLP projections are deliberately not
#: targeted by default: adding them is the first thing to try if rank capacity binds, and
#: changing two variables at once would make that unreadable.
DEFAULT_TARGET_MODULES = ["q_linear", "kv_linear", "out_linear"]

_LORA_A = "/lora_A"
_LORA_B = "/lora_B"


def read_native(tensor) -> np.ndarray:
    """Host copy of ``tensor`` read at NATIVE precision, as float32.

    NATIVE returns the value as stored, ignoring any lazily-cached other-precision view. This
    is the read ``ttml.checkpointing`` and ``ttml.sharding.Sharding.gather`` use, for the same
    reason: a FULL read of a bf16 parameter is served from a cache that the optimizer's in-place
    update never invalidates.
    """
    import ttml

    return np.asarray(
        tensor.to_numpy(precision=ttml.autograd.PreferredPrecision.NATIVE),
        dtype=np.float32,
    )


def make_lora_model(model, config):
    """``model`` wrapped for LoRA, with this project's canonical parameter names.

    ``ttml.modules.LoraModel`` does not override ``parameters()``, so calling it on the wrapper
    walks the registered tree from the *wrapper's* root (``LoraModel/model/blocks/0/...``)
    rather than the canonical ``llama/llama_block_0/...`` that this repo's checkpoint format, HF
    conversion and optimizer construction all expect. The inner model's own ``parameters()``
    already returns canonical names — including the injected ``lora_A``/``lora_B``, because
    injection replaces modules *inside* it. Same fix shape as ``train.model.TtTntLlama``'s.
    """
    from ttml.modules import LoraModel

    class TtTntLoraModel(LoraModel):
        def parameters(self):
            return self.model.parameters()

    return TtTntLoraModel(model, config)


def lora_scaling(*, rank: int, alpha: float, use_rslora: bool = False) -> float:
    """The factor ``LoraLinear.forward`` multiplies its low-rank branch by.

    Mirrors ``ttml/modules/lora.py:85`` exactly. Never hardcode the value this returns: at the
    default rank 8 / alpha 16 it is 2.0, and a hardcoded 2.0 silently produces a half- or
    double-strength merged model the first time anyone changes rank.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    return alpha / np.sqrt(rank) if use_rslora else alpha / rank


def base_parameter_snapshot(model) -> Dict[str, np.ndarray]:
    """NATIVE host copies of every parameter that LoRA is supposed to freeze.

    Adapter tensors are excluded — they are the ones expected to move.
    """
    return {
        name: read_native(p.tensor if hasattr(p, "tensor") else p)
        for name, p in model.parameters().items()
        if not (name.endswith(_LORA_A) or name.endswith(_LORA_B))
    }


def assert_base_frozen(before: Dict[str, np.ndarray], model) -> Dict[str, Any]:
    """Raise unless every non-adapter parameter is bit-identical to ``before``.

    This is LoRA's whole premise in this project — the claim is that fluency *cannot* regress,
    and that claim is worth exactly as much as this check. Exact equality, not ``allclose``: a
    frozen tensor that moved by one ulp did not stay frozen, and the failure being guarded
    against is "the freeze silently did not apply", not "the freeze was imprecise".
    """
    after = base_parameter_snapshot(model)
    missing = sorted(set(before) - set(after))
    if missing:
        raise RuntimeError(f"parameters vanished between snapshots: {missing}")
    moved = []
    worst = 0.0
    for name, was in before.items():
        now = after[name]
        if was.shape != now.shape:
            raise RuntimeError(f"{name}: shape changed {was.shape} -> {now.shape}")
        delta = float(np.max(np.abs(now - was))) if was.size else 0.0
        if delta != 0.0:
            moved.append((name, delta))
            worst = max(worst, delta)
    if moved:
        raise RuntimeError(
            f"LoRA's freeze did not hold: {len(moved)} of {len(before)} base parameters "
            f"moved (max abs delta {worst:.6e}). The anti-forgetting guarantee this run "
            f"exists to test is void. First few: {sorted(moved)[:5]}"
        )
    return {"frozen_checked": len(before), "moved": 0}


def assert_adapter_moved(model) -> Dict[str, Any]:
    """Raise unless at least one ``lora_B`` has moved off its all-zero initialisation.

    ``_create_lora_B`` initialises to zeros, so an untrained adapter reads as exactly zero and
    contributes nothing. Read NATIVE — this is the check that a FULL read broke.
    """
    b = {
        name: read_native(p.tensor if hasattr(p, "tensor") else p)
        for name, p in model.parameters().items()
        if name.endswith(_LORA_B)
    }
    if not b:
        raise RuntimeError("no lora_B parameters found — injection did not happen")
    moved = sum(1 for v in b.values() if not np.all(v == 0))
    worst = max((float(np.max(np.abs(v))) for v in b.values()), default=0.0)
    if moved == 0:
        raise RuntimeError(
            f"all {len(b)} lora_B tensors are still exactly zero — this run learned nothing"
        )
    return {"lora_B_total": len(b), "lora_B_moved": moved, "max_abs": worst}


def _squeeze_leading(a: np.ndarray) -> np.ndarray:
    """Drop leading size-1 axes, matching ``convert.hf_mapping.squeeze_leading``."""
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a


def merge_lora_state(
    model_state: Dict[str, np.ndarray],
    *,
    rank: int,
    alpha: float,
    use_rslora: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Fold every ``lora_A``/``lora_B`` pair into its base weight; drop the adapter tensors.

    ``LoraLinear.forward`` computes ``linear(x, W) + linear(linear(x, A), B) * scaling``. With
    ``linear(x, M) = x @ M.T`` and shapes ``A = (rank, in)``, ``B = (out, rank)``, that is
    ``x @ (W + scaling * B @ A).T`` — so the merged weight is ``W + scaling * (B @ A)`` and the
    merged model is numerically the LoRA model, not an approximation of it.

    Returns ``(merged_state, report)``. The base weight keeps its original shape, including any
    leading size-1 axes, because ``convert.hf_mapping`` squeezes those itself and a shape change
    here would be an invisible second edit.

    Merging is not optional: ``scripts/eval_improv.sft_checkpoint_to_hf`` raises on tensor names
    it cannot map, so an unmerged LoRA checkpoint fails conversion listing every ``lora_A``/
    ``lora_B`` rather than silently shipping a model with the adapter dropped. That guard is the
    backstop; this function is the fix.
    """
    scaling = lora_scaling(rank=rank, alpha=alpha, use_rslora=use_rslora)
    merged = dict(model_state)
    pairs: List[str] = []
    for name in list(model_state):
        if not name.endswith(_LORA_A):
            continue
        stem = name[: -len(_LORA_A)]
        b_name, w_name = stem + _LORA_B, stem + "/weight"
        if b_name not in model_state:
            raise ValueError(f"{name} has no matching {b_name}")
        if w_name not in model_state:
            raise ValueError(
                f"{name} has no base weight at {w_name} — the adapter cannot be merged into "
                f"a weight that is not in this checkpoint"
            )
        a = _squeeze_leading(np.asarray(model_state[name], dtype=np.float32))
        b = _squeeze_leading(np.asarray(model_state[b_name], dtype=np.float32))
        w_raw = np.asarray(model_state[w_name], dtype=np.float32)
        w = _squeeze_leading(w_raw)
        if a.ndim != 2 or b.ndim != 2 or w.ndim != 2:
            raise ValueError(f"{stem}: expected 2-D after squeeze, got "
                             f"A{a.shape} B{b.shape} W{w.shape}")
        if b.shape[1] != a.shape[0]:
            raise ValueError(f"{stem}: lora_B{b.shape} and lora_A{a.shape} do not compose")
        update = scaling * (b @ a)
        if update.shape != w.shape:
            raise ValueError(f"{stem}: merged update {update.shape} != base weight {w.shape}")
        merged[w_name] = (w + update).reshape(w_raw.shape).astype(np.float32)
        del merged[name], merged[b_name]
        pairs.append(stem)
    if not pairs:
        raise ValueError(
            "no lora_A/lora_B pairs found in this checkpoint — merging a checkpoint that "
            "carries no adapter is almost certainly a mistake (wrong file, or a "
            "full-parameter run)"
        )
    return merged, {
        "merged_pairs": len(pairs),
        "scaling": float(scaling),
        "modules": sorted(pairs),
    }
