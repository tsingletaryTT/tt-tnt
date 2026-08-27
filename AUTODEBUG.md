# AUTODEBUG: `FabricContext::is_2D_routing_enabled` / `append_fabric_connection_rt_args` — multi-hop routing during KV-cache setup

All source citations below are `path:line` against `/home/ttuser/tt-metal` as it exists on this
machine (checked out at the tree recorded elsewhere in this repo's CLAUDE.md as
`620793d898` / `rollback-pre-qwen36-...`; not re-verified for this note).

## 1. `FabricContext::is_2D_routing_enabled` — what determines it

Storage: `tt_metal/fabric/fabric_context.hpp:169` — `bool is_2D_routing_enabled_ = false;`
Accessor: `tt_metal/fabric/fabric_context.hpp:59` — `bool is_2D_routing_enabled() const { return is_2D_routing_enabled_; }`

Set exactly once, in the constructor, `tt_metal/fabric/fabric_context.cpp:255`:

```cpp
this->is_2D_routing_enabled_ = is_2D_topology(this->topology_);
```

`this->topology_` is derived one line earlier (`fabric_context.cpp:251`) from the fabric config:
`this->topology_ = FabricContext::get_topology_from_config(fabric_config);`

`is_2D_topology` itself, `tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp:15`:

```cpp
constexpr bool is_2D_topology(Topology topology) { return topology == Topology::Mesh || topology == Topology::Torus; }
```

**Verdict: it is a pure function of the configured `Topology` enum** (`Mesh`/`Torus` → true,
everything else — in particular `Topology::Linear`/`Ring`, tt-tnt's actual topology on a
single-mesh p300c/QB2 board — → false). It is computed once at `FabricContext` construction and
never touches live neighbor/hop data.

## 2. `append_fabric_connection_rt_args` — the 1D-fabric direction lookup, and whether it is multi-hop

Full function: `tt_metal/fabric/fabric.cpp:118-186` (template `append_fabric_connection_rt_args`).
The load-bearing branch, `fabric.cpp:134-169`:

```cpp
const bool is_2d_fabric = fabric_context.is_2D_routing_enabled();
...
std::optional<RoutingDirection> forwarding_direction;
if (is_2d_fabric) {
    forwarding_direction = control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
} else {
    // TODO: Workaround for #22524 routing tables not having wraparound links
    // for 1D fabric, we loop to match the dst chip since we need to ensure src and dst are on the same line
    // remove this once control plane has row/col info/view
    for (const auto& direction : FabricContext::routing_directions) {
        // This assumes all neighbor chips to the dst mesh are the same
        auto neighbors = control_plane.get_chip_neighbors(src_fabric_node_id, direction);
        auto neighbor_mesh_chips = neighbors.find(dst_fabric_node_id.mesh_id);
        if (neighbor_mesh_chips == neighbors.end() ||
            (std::find(neighbor_mesh_chips->second.begin(), neighbor_mesh_chips->second.end(),
                       dst_fabric_node_id.chip_id) == neighbor_mesh_chips->second.end())) {
            continue;
        }
        forwarding_direction = direction;
        break;
    }
}
TT_FATAL(forwarding_direction.has_value(), "Could not find any forwarding direction from src {} to dst {}", ...);
```

`FabricContext::routing_directions` (`tt_metal/fabric/fabric_context.hpp:37-38`):

```cpp
static constexpr auto routing_directions = {
    RoutingDirection::N, RoutingDirection::S, RoutingDirection::E, RoutingDirection::W, RoutingDirection::Z};
```

`control_plane.get_chip_neighbors` (`tt_metal/fabric/control_plane.cpp:1604-1620`) is a thin
wrapper that unions `get_intra_chip_neighbors` (`connected_chip_ids` of the routing edge whose
`port_direction == routing_direction`, from `mesh_graph_->get_intra_mesh_connectivity()`) with the
inter-mesh connectivity edge in the same direction. **Both are one-edge lookups keyed on
`src_fabric_node_id` directly** — there is no traversal, no path search, no accumulation over
multiple hops. `get_intra_chip_neighbors` returns the *directly wired* neighbor chip id(s) in that
one compass direction from `src`, full stop.

**Verdict: the `!is_2d_fabric` branch does not and cannot support multi-hop.** For each of the 5
compass directions it asks "is `dst` a *direct* neighbor of `src` in this direction?" and picks the
first direction where the answer is yes. If `dst` is two or more hops away from `src` on the 1D
line, `dst_fabric_node_id.chip_id` never appears in any of the 5 direct-neighbor sets, the loop
exhausts all directions with `forwarding_direction` still unset, and the function `TT_FATAL`s with
"Could not find any forwarding direction from src {} to dst {}". This exactly matches the code's
own comment: it is a documented workaround for **upstream issue #22524** ("routing tables not
having wraparound links"), stated in the code as temporary ("remove this once control plane has
row/col info/view"), and it is *by construction* a single-hop-only substitute for the 2D path
(`get_forwarding_direction`), which does do real routing-table lookups.

## 3. Caller trace during KV-cache initialization

Searched the full tree (`grep -rn append_fabric_connection_rt_args`) — ~100 call sites, entirely in
tt_metal CCL/fabric test infra and `ttnn/cpp/ttnn/operations/experimental/ccl/**` /
`ttnn/core/tensor/d2d_stream_service.cpp` (all-gather, reduce-scatter, all-reduce, send/recv-async,
point-to-point, deepseek-prefill dispatch/combine — genuine collective/data-movement ops).

**KV-cache allocation itself does not call this function, directly or transitively, in the
reference paged-attention path.** Traced the actual vLLM-facing KV-cache setup:

- `models/tt_transformers/tt/attention.py:388` `init_kv_cache()` — builds `cache_k`/`cache_v` as
  plain `torch.zeros(...)` and pushes each to device with a single `ttnn.as_tensor(..., mesh_mapper=
  ttnn.ReplicateTensorToMesh(self.mesh_device), memory_config=ttnn.DRAM_MEMORY_CONFIG)` call
  (`attention.py:429-441`). No CCL op, no fabric connection.
- `models/tt_transformers/tt/generator_vllm.py:41` `allocate_vllm_kv_cache_per_layer()` — the vLLM
  plugin's own paged-cache allocator — does the identical thing per layer/submesh: `torch.zeros`
  then `ttnn.as_tensor(..., mesh_mapper=ttnn.ReplicateTensorToMesh(submesh), ...)`
  (`generator_vllm.py:87-103`). Again no CCL op.

`ReplicateTensorToMesh` is a host-side mesh-mapper that shards/replicates a host tensor before
`ttnn.as_tensor` writes it to each device independently — it does not open a fabric worker
connection and never reaches `append_fabric_connection_rt_args`.

**Conclusion on the caller trace: I could not find a real call site in the vLLM/paged-attention
KV-cache-setup path.** If a fabric-connection fatal is actually observed at KV-cache-init time in
this project's own serving flow, it is not coming from the tt_transformers reference
`init_kv_cache`/`allocate_vllm_kv_cache_per_layer` functions as they exist in this checkout — it
would have to come from some other collective op that happens to run at roughly the same point in
program order (e.g. a weight `all_gather`/reduce-scatter during model construction, which runs
interleaved with cache allocation in `generator_vllm.py`), not from the cache tensors themselves.
I did not find and therefore cannot cite a KV-cache-specific call site; this section reports a
negative result rather than a traced call.

## 4. Verdict

**(a) — this is the upstream #22524-class limitation, not a fixable over-eager request in
KV-cache setup.** Evidence:

- `is_2D_routing_enabled()` is a static function of the configured topology, decided once at
  `FabricContext` construction — nothing about KV-cache setup can change it.
- The 1D-fabric fallback in `append_fabric_connection_rt_args` is, by its own explicit source
  comment and by the direct-neighbor-only semantics of `get_chip_neighbors`/
  `get_intra_chip_neighbors`, structurally single-hop. It is not a bug introduced by an eager
  caller asking for a connection it doesn't need — it is the *only* code path 1D fabric has for
  resolving `src → dst` direction, and it fails whenever `dst` is not directly adjacent to `src`.
- No KV-cache-setup call site into this function was found (§3) — hypothesis (b) as stated
  ("fixable KV-cache-setup over-eager connection request") has no code to point at in this
  checkout's reference paged-attention allocator. If tt-tnt's own serving path calls this function
  from somewhere during cache setup, that call site is outside `tt_metal/` and `models/
  tt_transformers/` and was not located by this search — worth grepping tt-tnt's own
  `generator_vllm.py`-equivalent and any custom CCL usage in its serving stack directly, since this
  investigation did not have that file in scope.

**Third explanation, not ruled out:** if the actual failure trace shows this fatal firing at
KV-cache-setup *time* but from a *different* op (a weight-distribution CCL collective running
concurrently/adjacently in program order), the real fix is unrelated to KV cache — it's whatever
non-adjacent chip pair that collective is trying to connect on a 1D-only fabric, which is again an
instance of (a), not (b).
