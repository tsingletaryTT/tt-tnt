# AutoFix Report — 4-chip vLLM serving of `episod/tt-tnt-1024`

> **RESOLVED, after this report was first written.** Everything below through "Final Status"
> describes the investigation as it stood when 4-chip serving looked like a genuine, unfixable
> upstream limitation. It wasn't. Two things were wrong, both on our side of the fence:
>
> 1. **`--additional-config` was silently no-op'ing.** `vllm_tt_plugin/config.py:get_tt_config()`
>    requires the JSON nested under a `"tt"` key: `{"tt": {"fabric_config": "..."}}`. Every
>    `FABRIC_1D_RING` / `FABRIC_2D` attempt below used a flat `{"fabric_config": "..."}` — which
>    `get_tt_config` treats as absent, falling through to the *default* (`FABRIC_1D`). So the
>    "Hypothesis 2" and "Hypothesis 1 experiment 4" conclusions about those two fabric configs
>    making no difference are **wrong** — neither was ever actually applied. Caught only because
>    the user asked for the exact JSON shape they wanted tried and it didn't match what I'd sent.
> 2. **`vllm-tt-plugin` was stale.** Pulling to latest `main` (`bd150c7` -> `6d3bb28`) brought in
>    `ttnn.FabricConfig.__members__.get(...)`-based resolution ("newly added fabrics ... work
>    without a plugin allow-list update") and `FABRIC_2D_TORUS_XY` auto-selection for
>    `BLACKHOLE_GALAXY`.
>
> With both fixed, a **properly re-tested** `FABRIC_1D_RING` (confirmed applied via the
> `Setting fabric config: FabricConfig.FABRIC_1D_RING` log line) still fails identically —
> consistent with the source-level finding that `Topology::Ring` is not 2D. But
> **`FABRIC_2D_TORUS_XY`** (`Topology::Torus`, genuinely 2D) works:
> `multidevice with 4 devices and grid (1, 4) is created`, followed by a clean completion. See
> "Resolution" at the end of this file for the exact working command and what's still open.

## Starting Evidence

- No prior AUTOTRIAGE.md/AUTODEBUG.md existed for this symptom; a fresh AUTODEBUG.md was produced
  by a forked subagent (source-only, no device access) and is referenced below.
- Original failing command: `tt-model serve episod/tt-tnt-1024 --force --instance metal-src-vllm`
  (and equivalent manual `server_example_tt.py` invocations) with `TT_VISIBLE_DEVICES` set to all
  4 chip BDFs and `MESH_DEVICE=P300x2`.
- Failure, identical across every configuration tried:

  ```
  TT_FATAL: Could not find any forwarding direction from src (M0, D0) to dst (M0, D3) (assert.hpp:104)
  ```

  thrown during vLLM's `_initialize_kv_caches -> initialize_from_config -> collective_rpc`, i.e.
  after weight loading completes and before any real prefill/decode.

## Hardware fact established first

Real Ethernet wiring confirmed via `/home/ttuser/tt-metal/build_Release/tools/umd/topology`
(`gozer run --chips all -- ... topology -f cluster_descriptor.yaml`): the four Blackhole chips on
this box's two p300c cards form a genuine 4-cycle — chip0↔chip1, chip1↔chip2, chip2↔chip3,
chip3↔chip0 — with `chip_pci_bdfs` mapping 0-3 to BDFs `0000:01:00.0`..`0000:04:00.0` in order.
So a `dims:[1,4] dim_types:[LINE,RING]` mesh graph descriptor, in that BDF order, is a faithful
model of the real cabling, not a guess.

## Hypothesis Experiments

### Hypothesis 1 — a wrong/missing mesh graph descriptor is the whole problem

**Experiment:** four configurations, escalating:
1. Stock `p300_x2_mesh_graph_descriptor.textproto` (`dims:[2,2]`, no `dim_types`) against a
   `(1,4)` mesh-open request.
2. A hand-authored `train/configs/mesh/mesh-1x4-ring.textproto` with `dims:[1,4]`, no
   `dim_types` (defaults to LINE).
3. Same file with `dim_types:[LINE,RING]`, matching the real cabling above.
4. Same ring descriptor + `--additional-config '{"fabric_config": "FABRIC_1D_RING"}'` (and,
   separately, `"FABRIC_2D"`).

**Result:**
- (1) fails earliest and differently: `Fabric Router Sync: Timeout after 10000 ms on Device 3`
  — a shape/descriptor mismatch (the `(2,2)`-declared board opened as `(1,4)`), matching this
  project's own prior finding from DDP training bring-up that `device_topology.dims` must equal
  the `MeshShape` actually opened.
- (2) fails at mesh-open time with the exact `D0->D3` forwarding TT_FATAL.
- (3) **mesh open succeeds** (real progress — the ring descriptor is accepted), but the
  *identical* TT_FATAL reappears later, at KV-cache/`collective_rpc` time.
- (4) **no change** — identical failure, identical location, regardless of `fabric_config`.

**Verdict:** partially verified, partially refuted. The descriptor mismatch in (1) is real and
distinct. But (2)/(3)/(4) show the descriptor and `fabric_config` are not the deciding factor for
the main failure — something deeper is structurally unable to route `D0->D3` regardless.

### Hypothesis 2 — `FABRIC_1D_RING` still takes the buggy 1D code path

**Evidence, read directly from tt-metal source** (not inferred):

- `tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp:12`:
  `enum class Topology { NeighborExchange = 0, Linear = 1, Ring = 2, Mesh = 3, Torus = 4 };`
  and line 15: `constexpr bool is_2D_topology(Topology t) { return t == Mesh || t == Torus; }`
- `tt_metal/fabric/fabric_context.cpp:162-176` (`get_topology_from_config`):
  `FABRIC_1D -> Linear`, `FABRIC_1D_RING -> Ring`, `FABRIC_2D -> Mesh`,
  `FABRIC_2D_TORUS_{X,Y,XY} -> Torus`.
- `fabric_context.cpp:251`: `this->topology_ = FabricContext::get_topology_from_config(fabric_config);`
  then `:255`: `this->is_2D_routing_enabled_ = is_2D_topology(this->topology_);`

**Verdict: verified.** `FABRIC_1D_RING` maps to `Topology::Ring`, which `is_2D_topology()`
explicitly excludes. So `FABRIC_1D_RING` takes the exact same `!is_2d_fabric` branch as plain
`FABRIC_1D` in `fabric.cpp`'s `append_fabric_connection_rt_args` — the ring flag changes nothing
about which routing algorithm runs. This explains experiment (4)'s null result for
`FABRIC_1D_RING` directly.

`FABRIC_2D` *does* map to `Topology::Mesh` (genuinely 2D, `is_2D_routing_enabled_ = true`), and
should take the different `control_plane.get_forwarding_direction()` branch — yet it produced the
identical error. Two explanations remain open and unresolved: either that 2D routing algorithm
also cannot find a route on a degenerate `(1,4)` shape (only one row — nothing for a "column"
direction to use), or a different bug in the same family. **Not conclusively distinguished; not
pursued further given the root-cause evidence below made it moot for the shape we can actually
request.**

### Hypothesis 3 — the 1D routing-direction lookup is structurally single-hop (the real root cause)

A forked subagent (source-only, no device access; redirected once after an initial non-answer)
traced this precisely:

- `tt_metal/fabric/fabric.cpp:148-169`, the `!is_2d_fabric` branch, loops over 5 compass
  directions calling `control_plane.get_chip_neighbors(src, direction)` and requires `dst` to be
  a **direct, single-hop** neighbor in one of them. The code's own comment:
  > `// TODO: Workaround for #22524 routing tables not having wraparound links for 1D fabric,
  > we loop to match the dst chip since we need to ensure src and dst are on the same line ...
  > remove this once control plane has row/col info/view.`
- `get_chip_neighbors` (`control_plane.cpp:1604-1620`) is backed by direct-edge lookups
  (`get_intra_chip_neighbors`/`get_inter_mesh_connectivity`) — no path traversal, no multi-hop.

**Verdict: verified, and this is the root cause.** For any 1D-fabric topology (`Linear` or
`Ring`), if the destination chip is not an immediate neighbor in one of the fixed compass
directions, `append_fabric_connection_rt_args` cannot find a route and `TT_FATAL`s — regardless
of what the mesh graph descriptor declares, and regardless of `dim_types`. `D0` and `D3` are only
adjacent via the ring's wraparound edge, and the 1D lookup does not do multi-hop and, per the
code's own comment, does not currently model the wraparound edge at all in this path. This is the
tracked upstream defect **#22524** — not fixable from a caller's config, adapter, or manifest.

### Hypothesis 4 — the KV-cache allocation path itself over-eagerly requests all-pairs connections (fixable from our side)

**Experiment:** same forked subagent traced every real caller of `append_fabric_connection_rt_args`
and separately read the actual KV-cache allocation code:
`models/tt_transformers/tt/attention.py:388` (`init_kv_cache`) and
`generator_vllm.py:41` (`allocate_vllm_kv_cache_per_layer`).

**Result:** both use only `torch.zeros` + `ttnn.as_tensor(..., mesh_mapper=ReplicateTensorToMesh(...))`.
**Neither calls `append_fabric_connection_rt_args`, directly or transitively.**

**Verdict: refuted.** There is no fixable KV-cache-setup call site to point at. Whatever collective
runs adjacent in program order to the KV-cache step (most likely a weight all-gather during model
construction, which happens just before) is what actually trips the fatal — but it is still an
instance of Hypothesis 3, not a distinct, fixable bug in our adapter or the vLLM plugin's
KV-cache path.

### Hypothesis 5 — a genuine 2-D mesh shape `(2,2)` (not `(1,4)`) would take the working routing path

**Evidence for feasibility:** `vllm_tt_plugin/utils/dp_discovery.py:parse_mesh_grid` accepts a
literal tuple string via `ast.literal_eval` (its own docstring: `parse_mesh_grid("(2, 4)", ...)`),
so `MESH_DEVICE="(2, 2)"` should bypass the `_MESH_GRID_PRESETS` table (which has no `(2,2)`
entry) and request a genuinely 2-D shape.

**Not run.** This would need `FabricConfig::FABRIC_2D` (which does map to `Topology::Mesh`, a
real 2D topology) *and* a mesh shape with both dimensions > 1, *and* whatever tensor-parallel
sharding logic `tt_transformers`/`vllm_tt_plugin` uses would need to correctly flatten/handle a
non-`(1,N)` mesh for a model whose TP-compatible widths were declared as `{1,2,4}` (a flat 1-D
assumption). This is a materially bigger, less-tested change than anything tried so far, not a
quick config flip. **Left as an open, untried avenue — not attempted on hardware.**

### Hypothesis 6 — other models (Qwen/Llama) already work on 4 chips on this exact box, so there must be a known-good path

**Evidence found:** `~/run-70b-model.sh`, `~/run-qwen32b-direct.sh`, `~/QB2_70B_SUCCESS.md`
(dated 2026-03-07). These use a **structurally different serving stack**: `~/tt-vllm`
(`tenstorrent/vllm`, a full fork with TT support built into `vllm/platforms/tt.py`), not
`vllm_tt_plugin`. Both scripts request `MESH_DEVICE=P150x4`/`P300x2` with plain `FABRIC_1D`
(no ring override at all).

**Why this is not proof of a working path, on inspection:**
1. `QB2_70B_SUCCESS.md`'s own "Next Steps" section admits the API was never tested. A 70B model's
   weight load takes 10-30 minutes *silently*, and per vLLM's own init order, KV-cache
   initialization (where our failure occurs) runs *after* weight loading completes — so this
   note may predate ever reaching the step we are stuck on.
2. Read `~/tt-vllm/vllm/worker/tt_worker.py:545-585` directly: it also calls
   `ttnn.open_mesh_device(ttnn.MeshShape(1,4))` — the exact same mechanism as `vllm_tt_plugin`.
   No MPI-based or otherwise structurally different multi-chip path exists in this fork.
3. Firmware and tt-metal have both moved since March (`19.4.2.0` then vs `19.13.1` measured on
   this box today via the topology tool), so even a genuine March success would not prove
   anything about the current stack.

**Experiment run:** reproduced the exact historical command pattern from `QB2_70B_SUCCESS.md`
against our own (much smaller, much faster to test) model, on real hardware
(`gozer acquire --chips all`), rather than continuing to reason about six-month-old notes.

**Result:** failed before ever opening a device — `ImportError: cannot import name
'layer_type_validation' from 'transformers.configuration_utils'`. The shared `.tenstorrent-venv`
this fork depends on has a `transformers` version newer than what this frozen March-era vLLM
fork checkout expects; some other work sharing that venv has since upgraded it.

**Verdict: inconclusive on the merits, but the claimed evidence does not hold up.** There is no
currently-reproducible, confirmed-working 4-chip example on this machine for either serving
stack. Fixing the shared venv (e.g. pinning an old `transformers`) was not attempted — it risks
breaking other projects sharing `.tenstorrent-venv` and was outside this bug's scope.

## Resolution (supersedes "Final Status" below)

**Fixed.** `episod/tt-tnt-1024` serves correctly across all 4 chips.

**Working command** (from `/home/ttuser/vllm-tt-plugin-standalone/examples`, `vllm-tt-plugin`
pulled to `main` @ `6d3bb28`):

```bash
TT_METAL_HOME=/home/ttuser/tt-metal-src-vllm-home \
PYTHONPATH=/home/ttuser/tt-metal-src-vllm-home \
LD_LIBRARY_PATH=/home/ttuser/tt-metal-src-vllm-home/build/lib \
EXTRA_MODELS_DIR=/home/ttuser/.cache/tt-kernel/bundles \
VLLM_USE_V1=1 \
MESH_DEVICE=P300x2 \
HF_MODEL=episod/tt-tnt-1024 \
TT_VISIBLE_DEVICES=<all 4 chip BDFs> \
TT_MESH_GRAPH_DESC_PATH=/home/ttuser/code/tt-tnt/train/configs/mesh/mesh-1x4-ring.textproto \
vllm serve episod/tt-tnt-1024 --max_model_len 512 --max_num_seqs 32 --port 8000 \
  --additional-config '{"tt":{"fabric_config":"FABRIC_2D_TORUS_XY"}}'
```

Confirmed: `multidevice with 4 devices and grid (1, 4) is created`, followed by a clean
`/v1/completions` response.

**What Hypothesis 2's conclusion got wrong, and why it looked right at the time:** the
`FABRIC_1D_RING`/`FABRIC_2D` experiments under Hypothesis 1 (experiment 4) used
`--additional-config '{"fabric_config": "..."}'` — a flat object. `get_tt_config()`
(`vllm_tt_plugin/config.py:27`) requires the value nested under a `"tt"` key and returns `{}`
otherwise, so both of those "identical failure, no change" results were silently re-testing the
*default* config (`FABRIC_1D`, since this board isn't `is_6u`), not the named fabric at all.
Hypothesis 2's source-level claim about `FABRIC_1D_RING` mapping to `Topology::Ring` (not 2D) is
still correct and was independently reconfirmed with a properly-nested, verified-applied retest
(`Setting fabric config: FabricConfig.FABRIC_1D_RING` logged) — it still fails identically. The
error was specifically in believing `FABRIC_2D` had been tested; it hadn't. A freshly-pulled
`vllm-tt-plugin` main also added `FABRIC_2D_TORUS_XY` auto-selection (for `BLACKHOLE_GALAXY`) and
switched fabric-name resolution to `ttnn.FabricConfig.__members__.get(...)`, which is what made
`FABRIC_2D_TORUS_XY` requestable via `--additional-config` at all on the previously-installed
plugin version.

**Hypothesis 3 (the tt-metal #22524 1D-fabric single-hop limitation) stands, correctly scoped.**
It's real, it's why `FABRIC_1D`/`FABRIC_1D_RING` both fail, and it's still not fixable from our
side for those two fabric configs. It just wasn't the whole story — `FABRIC_2D_TORUS_XY` takes a
structurally different, genuinely-2D routing path (`Topology::Torus`, `is_2D_topology() == true`)
that doesn't hit that code path at all.

**Hypothesis 6 (do Qwen/Llama already work here)** — still stands as written: the six-month-old
`tenstorrent/vllm`-fork evidence remains unconfirmed and that environment is still broken. The
actual working path was found independently, on the current `vllm_tt_plugin` stack, not by
reviving that fork.

**Still open / not yet done:**
- This is not yet the *deployed* configuration — the 2-chip `(1,2)` mesh is still what's live.
  Making 4-chip the default would mean updating `manifests/tt_kernel_manifest-1024.json`'s
  `env`/`resources` and re-pushing, and deciding whether `mesh-1x4-ring.textproto` should be
  vendored permanently (it's currently a manually-passed `TT_MESH_GRAPH_DESC_PATH`, not part of
  the pushed bundle).
- No perf/quality comparison against the 2-chip config has been run (tokens/sec, coherence).
- `_MESH_GRID_PRESETS` in `dp_discovery.py` still has no way to request a genuinely 2-D mesh
  shape like `(2,2)` — irrelevant now since `(1,4)` + `FABRIC_2D_TORUS_XY` works, but worth
  knowing that lever still doesn't exist if a future model needs it.

## Final Status (as originally written, superseded above)

- **Not fixed.** The primary, best-evidenced cause (Hypothesis 3) is a genuine upstream tt-metal
  limitation — tracked as #22524 — in 1D-fabric direction resolution, which is structurally
  single-hop and does not support a ring's wraparound edge. This is not fixable from tt-tnt's
  adapter, manifest, or vLLM serving config; per this project's standing rule, tt-metal itself is
  not edited.
- **Two-chip `(1,2)` serving remains the practical ceiling** on this hardware today and is what
  is currently deployed.
- **Two real, separately-actionable findings surfaced along the way, already handled:**
  - `tt-model-manager` issue #35 / [PR #36](https://github.com/tenstorrent/tt-model-manager/pull/36)
    — `_ActivityTqdm` missing `get_lock`/`set_lock`, blocking every multi-file `tt-model pull`.
    Fixed and PR'd (not auto-closing #35, per instruction — left for its own author/reviewers).
  - `manifests/tt_kernel_manifest-1024.json` is now actually published to the Hub for
    `episod/tt-tnt-1024` (it previously existed only in this repo), and the bundle is properly
    registered in `tt-model`'s local install DB.
- **Commands that prove the final state:**
  ```bash
  # 2-chip serving works (the deployed configuration):
  cd /home/ttuser/vllm-tt-plugin-standalone/examples
  MESH_DEVICE=P300x2 HF_MODEL=episod/tt-tnt-1024 EXTRA_MODELS_DIR=/home/ttuser/.cache/tt-kernel/bundles \
    <env from tt-model serve --print> python3 server_example_tt.py --model episod/tt-tnt-1024 \
    --max_model_len 512 --max_num_seqs 8
  # (gozer grants 2 chips for a 1-chip request on this board; MESH_DEVICE=P300x2 -> (1,4) is
  # overridden to (1,2) by the non-empty TT_VISIBLE_DEVICES, per vllm_tt_plugin/worker.py's
  # documented _resolve_mesh_grid override behavior.)

  # 4-chip reproduction of the root cause (any of these reproduces the TT_FATAL):
  TT_VISIBLE_DEVICES=<all 4 BDFs> TT_MESH_GRAPH_DESC_PATH=train/configs/mesh/mesh-1x4-ring.textproto \
    MESH_DEVICE=P300x2 HF_MODEL=episod/tt-tnt-1024 python3 server_example_tt.py \
    --model episod/tt-tnt-1024 --max_model_len 512 --max_num_seqs 32
  ```
- **Remaining risks / follow-up evidence needed:**
  - Hypothesis 1's `FABRIC_2D` non-result (mesh open succeeds, KV-cache step still fails
    identically) is not fully explained — could be the same degenerate-`(1,4)`-shape limitation,
    or a second, distinct bug. Not chased further once Hypothesis 3 gave a complete, well-evidenced
    answer for the shape we can actually request.
  - Hypothesis 5 (`MESH_DEVICE="(2, 2)"` + `FABRIC_2D`) is untried on hardware and is the only
    remaining lever that could plausibly route around #22524 entirely (by taking the genuinely
    2D `Mesh`/`Torus` code path on a shape that actually has two real dimensions) — but it is a
    materially bigger change (TP-sharding assumptions, `_MESH_GRID_PRESETS` has no such preset)
    than anything else tried, not a quick config flip.
  - Per this session's explicit instruction, no upstream tt-metal issue has been filed for
    #22524's specific manifestation here, and the shared `.tenstorrent-venv` environment-rot
    found in Hypothesis 6 was not repaired.
