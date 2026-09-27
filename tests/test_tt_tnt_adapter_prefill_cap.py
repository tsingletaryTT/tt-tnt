# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Gates for Patch 3 in ``bundle/tt_tnt_adapter.py``: the prefill-bucket clamp.

WHAT BROKE
----------
``tt_transformers`` pads every prefill to a bucket ladder -- 128, then 1024 -- that knows
nothing about the model. tt-tnt-1024 has ``max_seq_len`` 512, so every 129-512-token prompt
was padded to 1024, overran the RoPE tables, hit ``AssertionError: Sequence length 1024
exceeds max seq len 512`` inside the EngineCore and killed the server for every user
(measured 2026-09-27, ``serve2.log`` of the tt-tnt-1024 benchmark).

WHAT THESE PROVE
----------------
* The clamp changes exactly one case (bucket > cap while the prompt fits) and is otherwise
  byte-identical to the stock ladder -- including for tt-tnt's 2048 context, where nothing
  may change at all.
* It is WIRED: the name ``generator.py`` actually calls is rebound, and
  ``initialize_vllm_model`` is what sets the cap. A test of the arithmetic alone would pass
  with the patch never installed (~/CLAUDE.md: "put the guard at the layer that fails").
* It never pads a prompt DOWN (a prompt longer than the cap keeps stock behaviour, so the
  stock assert -- not a silent truncation -- is what fires if vLLM ever lets one through).

Same fake-``models.tt_transformers`` technique as ``test_tt_tnt_adapter_cache.py``: no ttnn,
no device, nothing skipped except the one upstream-source canary.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from test_tt_tnt_adapter_cache import loaded_adapter, make_model_args_cls

_GENERATOR = "models.tt_transformers.tt.generator"


def stock_get_padded_prefill_len(seq_len: int) -> int:
    """Verbatim copy of tt-metal 0.77.0 ``common.py:get_padded_prefill_len``."""
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    return 2 ** (seq_len - 1).bit_length()


@pytest.fixture
def patched():
    """The adapter loaded with a fake ``generator`` module carrying the stock ladder."""
    saved = sys.modules.get(_GENERATOR)
    gen = types.ModuleType(_GENERATOR)
    gen.get_padded_prefill_len = stock_get_padded_prefill_len
    sys.modules[_GENERATOR] = gen
    try:
        with loaded_adapter(make_model_args_cls()) as module:
            yield types.SimpleNamespace(module=module, generator=gen)
            module.restore_patches()
    finally:
        if saved is None:
            sys.modules.pop(_GENERATOR, None)
        else:
            sys.modules[_GENERATOR] = saved


def test_generator_global_is_rebound_at_import(patched):
    """The wiring: the name prefill_forward_text looks up must be the clamp, not the stock."""
    assert patched.generator.get_padded_prefill_len is patched.module._capped_padded_prefill_len


def test_initialize_vllm_model_sets_the_cap_from_max_seq_len(patched):
    """The other half of the wiring: the cap comes from the serve's max_seq_len."""
    patched.module.LlamaForCausalLM.initialize_vllm_model(
        hf_config=None, mesh_device=None, max_batch_size=32, max_seq_len=512
    )
    assert patched.module._PREFILL_CAP == 512
    # ...and the function generator.py calls now clamps the exact crashing case.
    assert patched.generator.get_padded_prefill_len(130) == 512


@pytest.mark.parametrize("n", [1, 64, 128, 129, 130, 200, 384, 511, 512])
def test_512_context_never_pads_past_512(patched, n):
    """tt-tnt-1024: every legal prompt length gets a bucket that fits in 512."""
    patched.module.set_prefill_cap(512)
    got = patched.generator.get_padded_prefill_len(n)
    assert n <= got <= 512
    # Short prompts keep the stock (traced) 128 bucket; only the overshoot is clamped.
    assert got == (128 if n <= 128 else 512)


def test_prompt_longer_than_the_cap_is_never_padded_down(patched):
    """No silent truncation: an over-length prompt keeps stock behaviour."""
    patched.module.set_prefill_cap(512)
    assert patched.generator.get_padded_prefill_len(513) == 1024
    assert patched.generator.get_padded_prefill_len(5000) == stock_get_padded_prefill_len(5000)


def test_2048_context_is_byte_identical_to_stock(patched):
    """tt-tnt (2048 ctx): buckets 128/1024/2048 already fit, so nothing may change."""
    patched.module.set_prefill_cap(2048)
    for n in range(1, 2049):
        assert patched.generator.get_padded_prefill_len(n) == stock_get_padded_prefill_len(n)


def test_uncapped_before_initialize_is_stock(patched):
    for n in (1, 129, 1024, 1025, 4097):
        assert patched.generator.get_padded_prefill_len(n) == stock_get_padded_prefill_len(n)


@pytest.mark.parametrize("bad", [0, -512, 500, None, "x"])
def test_non_tile_aligned_cap_declines(patched, bad):
    """A clamped bucket must itself be legal, so a non-tile cap disables the clamp."""
    assert patched.module.set_prefill_cap(bad) is None
    assert patched.generator.get_padded_prefill_len(130) == 1024


def test_restore_patches_puts_the_stock_function_back(patched):
    patched.module.restore_patches()
    assert patched.generator.get_padded_prefill_len is stock_get_padded_prefill_len


def test_missing_generator_module_declines_without_crashing():
    """No generator module (upstream moved): the adapter still imports, the patch is absent."""
    saved = sys.modules.pop(_GENERATOR, None)
    try:
        with loaded_adapter(make_model_args_cls()) as module:
            assert module._ORIGINAL_GET_PADDED_PREFILL_LEN is None
    finally:
        if saved is not None:
            sys.modules[_GENERATOR] = saved


def _installed_generator_sources():
    """tt_transformers ``generator.py`` / ``common.py`` from any locally installed bundle."""
    roots = [Path(os.environ["TT_METAL_HOME"])] if os.environ.get("TT_METAL_HOME") else []
    cache = Path.home() / ".cache" / "tt-model" / "models" / "episod"
    roots += sorted(cache.glob("tt-tnt*/venv/lib/python3.*/site-packages"))
    roots.append(Path.home() / "tt-metal")
    for root in roots:
        gen = root / "models" / "tt_transformers" / "tt" / "generator.py"
        common = root / "models" / "tt_transformers" / "tt" / "common.py"
        if gen.is_file() and common.is_file():
            return gen.read_text(encoding="utf-8"), common.read_text(encoding="utf-8")
    return None


def test_upstream_prefill_bucket_anchors_still_present():
    """The upstream facts Patch 3 relies on. If one changes, a human should re-check."""
    sources = _installed_generator_sources()
    if sources is None:
        pytest.skip("no tt_transformers source on this machine; the drift canary cannot run")
    generator, common = sources
    # generator.py still calls the ladder by its bare module-global name (what we rebind).
    assert "get_padded_prefill_len(seq_len - num_cached)" in generator
    # The ladder still jumps 128 -> 1024 with no knowledge of max_seq_len.
    assert "if seq_len <= 1024:\n        return 1024" in common


# ---------------------------------------------------------------------------
# Patch 4: the single-chip mesh descriptor for data-parallel ranks
# ---------------------------------------------------------------------------


def _desc_dir(tmp_path):
    (tmp_path / "mesh-1x1.textproto").write_text("mesh_descriptors {}\n", encoding="utf-8")
    return tmp_path


def test_single_visible_chip_gets_the_bundled_descriptor(patched, tmp_path):
    env = {"TT_VISIBLE_DEVICES": "1"}
    got = patched.module._single_chip_mesh_descriptor(env, here=_desc_dir(tmp_path))
    assert got == str(tmp_path / "mesh-1x1.textproto") == env["TT_MESH_GRAPH_DESC_PATH"]


@pytest.mark.parametrize("env", [
    {"TT_VISIBLE_DEVICES": "0000:03:00.0,0000:04:00.0"},   # the DP API server: whole mesh
    {},                                                      # nothing narrowed
    {"TT_VISIBLE_DEVICES": "0", "TT_MESH_GRAPH_DESC_PATH": "/operator/choice"},  # never override
])
def test_descriptor_is_left_alone_otherwise(patched, tmp_path, env):
    before = dict(env)
    assert patched.module._single_chip_mesh_descriptor(env, here=_desc_dir(tmp_path)) is None
    assert env == before


def test_no_bundled_file_means_no_change(patched, tmp_path):
    env = {"TT_VISIBLE_DEVICES": "0"}
    assert patched.module._single_chip_mesh_descriptor(env, here=tmp_path) is None
    assert "TT_MESH_GRAPH_DESC_PATH" not in env


def test_the_repo_ships_the_descriptor_the_patch_looks_for():
    """The bundle copies train/configs/mesh/mesh-1x1.textproto next to the adapter."""
    from pathlib import Path as _P

    desc = _P(__file__).resolve().parents[1] / "train" / "configs" / "mesh" / "mesh-1x1.textproto"
    text = desc.read_text(encoding="utf-8")
    assert "device_topology { dims: [ 1, 1 ] }" in text and "arch: BLACKHOLE" in text
