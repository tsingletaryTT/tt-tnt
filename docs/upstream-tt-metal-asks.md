<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Upstream asks for tt-metal / tt-train

Things this project needs from tt-metal that it cannot fix in its own tree, written up so
that someone with commit rights can act on them without rediscovering the analysis. Each entry
states the defect, the fix, the measurement that justifies it, and what we did instead in the
meantime.

Against tt-metal `620793d898` (`rollback-pre-qwen36-1576-g620793d898`).

---

## 1. Python cannot pass a null attention mask to `CppLlama` / `GPT2Transformer`

Status: open. Worked around in `train/model.py`; see
`.superpowers/attention-mask-fix.md` for the full report and every measurement.

### The defect

`ttml::ops::scaled_dot_product_attention` selects the fused SDPA kernel's mask mode from
*whether a mask object was passed*, not from what is in it
(`tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp:249-255`):

```cpp
ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::Causal;
if (mask.has_value() && mask.value()) {
    mask_tensor = mask.value()->get_value();
    mask_type = ttml::metal::AttentionMaskType::Arbitrary;
}
```

`Arbitrary` does the full S×S attention instead of the triangular half, and additionally loses
the forward program factory's load-balancing path, which is gated on `Causal`
(`tt-train/sources/ttml/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp:305-316`):

```cpp
const bool use_balanced_parallelism =
    (mask_type == AttentionMaskType::Causal) && (St % 2 == 0) && ...
```

A lower-triangular all-ones mask — which is what `ttml.common.utils.build_causal_mask` returns
and what `ttml.common.trainer.train()` passes on every step
(`ttml/common/trainer.py:73-76, 102`) — is exactly what `Causal` already computes. Passing it
buys nothing and roughly doubles the attention work.

tt-train's own example already knows this
(`tt-train/sources/examples/train/train.py:461-467`):

> DeepSeek's composite SDPA needs the mask passed explicitly; gpt2/llama/qwen3 use a fused
> SDPA that materializes its own causal mask internally, so only build it for DeepSeek.

But a caller who follows that advice against `CppLlama` gets a `TypeError`, because the
nanobind binding declares the mask non-optional
(`tt-train/sources/ttml/nanobind/nb_models.cpp:330-337`, and the same at `:237-246` for
GPT-2) — even though the C++ `Llama::operator()` behind it takes
`std::optional<TensorPtr>` (`models/llama.hpp:60-68`) and handles `std::nullopt` correctly all
the way down. There is no alternative Python entry point: all three
`ModuleBase::operator()` overloads `throw std::logic_error` in the base
(`modules/module_base.cpp:114-128`) and only the two non-optional ones are bound
(`nanobind/nb_modules.cpp:78-86`).

### The measurement

One Blackhole p300c, full training step (forward + cross-entropy + backward + AdamW),
s/1000 steps, measured through this project's training entry point over 300 steps at seed 5489:

| shape | explicit causal mask (`Arbitrary`) | `mask=None` (`Causal`) | speedup |
|---|---|---|---|
| 22M params, batch 16, seq 2048 | 503.3 | 356.7 | **1.41x** |
| 123M params, batch 64, seq 512 | 890.0 | 776.7 | **1.15x** |

Correctness, on trained weights: held-out cross-entropy moves by 4.1e-4 nats (4.122742 →
4.122333); a perturbation probe shows both paths are strictly causal (perturbing token *t*
leaves every logit before *t* bit-identical); and against an independent fp32 NumPy reference
the unmasked path's mean absolute error is fractionally *lower* than the masked path's
(0.015445 vs 0.015610, correlation 0.99996 for both).

### The fix

`<nanobind/stl/optional.h>` is already included at `nb_models.cpp:8`, so this needs no new
include. The KV-cache overload (`nb_models.cpp:339-352`) must keep its mask non-optional: there
the mask is not square and `models/llama.cpp` reads
`mask->get_value().logical_shape()[-1]` to size the cache slice.

```diff
--- a/tt-train/sources/ttml/nanobind/nb_models.cpp
+++ b/tt-train/sources/ttml/nanobind/nb_models.cpp
@@ GPT-2, around line 237
         py_gpt2.def(
             "__call__",
             [](models::gpt2::Transformer& self,
                const ttml::autograd::TensorPtr& tensor,
-               const ttml::autograd::TensorPtr& mask) {
-                return self(tensor, std::optional<ttml::autograd::TensorPtr>(mask));
+               const std::optional<ttml::autograd::TensorPtr>& mask) {
+                return self(tensor, mask);
             },
             nb::arg("tensor"),
-            nb::arg("mask"),
+            nb::arg("mask") = std::nullopt,
             "Model forward pass with causal mask.");

@@ Llama, no-KV-cache overload, around line 330
         py_llama.def(
             "__call__",
             [](models::llama::Llama& self,
                const ttml::autograd::TensorPtr& tensor,
-               const ttml::autograd::TensorPtr& mask) { return self(tensor, mask); },
+               const std::optional<ttml::autograd::TensorPtr>& mask) { return self(tensor, mask); },
             nb::arg("tensor"),
-            nb::arg("mask"),
+            nb::arg("mask") = std::nullopt,
             "Model forward pass without KV cache.");
```

A second, separable change belongs in `ttml/common/trainer.py:73-76`: stop building a causal
mask for models whose SDPA materializes its own, matching what `examples/train/train.py`
already does. That one needs the model type plumbed into `train()`'s config, which is a design
call for the maintainer rather than a mechanical edit, so it is described rather than patched
here. Without it, every caller of `ttml.common.trainer.train()` keeps paying the
arbitrary-mask cost even after the binding is fixed.

### What we did instead

`train/model.py` trains ttml's *Python* `Llama` (`ttml.models.llama.Llama`) rather than
`CppLlama`. It is the same architecture over the same fused ops, costs the same per step
(521.7 vs 521.9 s/1000 with the mask still passed), and its `forward` reaches
`ttml.ops.attention.scaled_dot_product_attention`, whose binding is already declared
`nb::arg("mask") = std::nullopt` (`nanobind/nb_ops.cpp:280-293`). The wrapper renames its
parameters to the C++ scheme so checkpoints, HF conversion and `--resume` are unaffected.

**Once the binding is fixed upstream, `train/model.py`'s renaming layer can go away and
`--model-impl cpp` becomes as fast as `python`.**

---

## 2. A mesh graph descriptor whose declared dims disagree with the opened `MeshShape` hangs instead of failing

Status: open. Worked around in this repo by shipping matching descriptors
(`train/configs/mesh/mesh-1x2.textproto`, `mesh-1x4.textproto`); see
`.superpowers/ddp-bringup.md` for the full experiment and every measurement.

Nothing we need is blocked on this — the workaround is complete and costs nothing at
runtime. This is a diagnosability ask, filed because the failure mode is expensive to debug and
will be hit again by anyone bringing up multi-chip training on a box whose physical topology is
not a line.

### The defect

`tt::tt_fabric` accepts a mesh graph descriptor whose `device_topology.dims` differ from the
`MeshShape` the process actually opens, and gives no indication that anything is wrong. Device
open succeeds, the parallelism context initialises and self-reports correctly, the model builds,
the batch shards, and forward/backward/optimizer all run at full speed. The run then **hangs
forever** the first time a CCL collective traverses the mesh axis.

Concretely, on a TT-QuietBox 2 (four Blackhole p300c, physically a 2x2 ring), with tt-metal's
own `tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto` — which
declares `device_topology { dims: [ 2, 2 ] }` and is the correct descriptor for the hardware —
and a process opening `MeshShape([1, 4])`:

```
step 1: batch / forward / backward / loss=10.60938 / SYNC OK / optim.step  (0.3s)
step 2: batch / forward / backward
                                     <- never returns
```

`[1, 4]` is not an exotic request here: `ttml::autograd::ParallelismContext`'s constructor
*requires* a line topology for a DDP-only run — a 2-D mesh `TT_FATAL`s unless the number of
enabled parallelisms equals the number of mesh dimensions
(`tt-train/sources/ttml/autograd/auto_context.cpp:198-204`) — so any DDP-only training job on
this class of hardware must open `[1, N]` while the shipped descriptor declares `[2, 2]`.

Two things make this particularly costly to diagnose:

1. **The host does not block in the collective.** tt-metal enqueues asynchronously, so
   `synchronize_gradients` returns having only queued work and the host stalls at its next
   blocking read — `loss.to_numpy()` in the *following* step. The stack points at the loss read,
   one phase and one iteration away from the actual fault.
2. **Everything that could have reported the mismatch reports success.** `open_device` returns a
   `MeshDevice` of shape `[1, 4]`; `ParallelismContext` inspects that mesh and correctly reports
   a DDP axis of 4 devices. There is no point at which the two shapes are compared.

### The measurement

Four Blackhole p300c, `--size 1024` (123M params), batch 64, seq 512, four training steps,
identical in every respect except the descriptor's declared dims:

| descriptor `device_topology.dims` | opened `MeshShape` | result |
|---|---|---|
| *(none — `enable_fabric` has no default for 4 devices)* | `[1, 4]` | hang before device open; killed at 600 s, no output |
| `[ 2, 2 ]` (tt-metal's `p300_x2`) | `[1, 4]` | opens, trains step 1, **hangs in step 2 forever** |
| `[ 1, 4 ]` (ours) | `[1, 4]` | **works**; 300 steps at 193.4 s/1000, all four replicas bit-identical |
| `[ 1, 2 ]` (tt-metal's `p300`) | `[1, 2]` | works; replicas bit-identical |
| `[ 2, 2 ]` (tt-metal's `p300_x2`) | `[1, 2]` | fails **cleanly** in 10 s: `Fabric Router Sync: Timeout ... on Device 2` |

The last row is the useful contrast: when the descriptor declares *more devices* than are
opened, the failure is caught and the message is accurate. It is only the *shape* disagreement,
at equal device count, that goes undetected.

### The fix

A single equality check where the mesh device is created against the fabric's active mesh
descriptor: if the descriptor's `device_topology.dims` do not match the requested `MeshShape`,
`TT_FATAL` with both shapes named. The error text should say which file is in force (the
descriptor path is already known — `get_mgd_path` sets `TT_MESH_GRAPH_DESC_PATH` when it picks a
default) and that the descriptor must declare the logical mesh being opened, not the physical
cabling.

A second, smaller ask in the same area: `ttml::ttnn_fixed::distributed::enable_fabric` has no
default descriptor for 4 devices (`tt-train/sources/ttml/ttnn_fixed/distributed/tt_metal.cpp:80-88`
handles 8 and 32 only) and silently falls back to a bare `FABRIC_2D` that hangs. Either ship a
4-device default or make the `std::nullopt` path refuse rather than proceed — a fallback that
reliably hangs is worse than an error.

### What we did instead

`train/configs/mesh/mesh-1x2.textproto` and `mesh-1x4.textproto` are vendored in this repo and
selected by device count in `train/run.py`'s `_mesh_graph_descriptor_path`, which exports
`TT_MESH_GRAPH_DESC_PATH` before ttml is imported. `tests/test_run_validation.py` asserts each
file declares the `[1, N]` shape its device count opens, and that an unsupported device count
raises rather than falling back to a mismatched descriptor — because a wrong descriptor hangs
rather than failing, a fallback would be the worst possible default.

---

## 3. A DDP training step re-marks replicated parameters as `Shard(0)`, so checkpoints save every replica

Status: open upstream, but **no longer blocking here** — corrected 2026-08-16 from this
repo, in `train/checkpoint.py:replicated_for_save`. See `.superpowers/ddp-checkpoint-fix.md`.

This is fixable in our own tree;
the correction is the useful part of this update: `ttnn.Tensor.update_tensor_topology` is bound
in Python (`ttnn/cpp/ttnn-nanobind/pytensor.cpp:1611`) and `ttnn.TensorTopology` is constructible
from Python (`ttnn/core/distributed/distributed_nanobind.cpp:734`). The false placement can
therefore be corrected by *any* holder of the tensor, not only where it is written. This repo now
re-marks each parameter `Replicate` immediately before a save and restores the original topology
immediately after, which moves no data at all — a `--ddp 4` save costs exactly what a `--ddp 1`
save costs, and produces a byte-identical-sized file (737,824,624 bytes, verified equal to the
`--ddp 1` figure below).

The defect below is still real and still worth fixing at source: every consumer of a DDP-trained
parameter's topology is currently told something false, and each one has to know to disbelieve
it. What has changed is only that we are no longer waiting on it.

### The defect

Under DDP the weights are replicated and **stay** replicated — verified directly, not assumed:
after training steps on a `[1, 4]` mesh, every chip's copy of every one of the 66 parameter
tensors is bit-identical (`max |replica0 - replica_i| = 0.000000e+00`). The *data* is correct.

The tensor's **topology metadata** is not. Probed on the same parameter
(`llama/llama_block_0/attention/q_linear/weight`, logical shape `[1, 1, 1024, 1024]`) before
and after two DDP training steps:

| when | `Sharding.placements` | `dist_shape` | `is_fully_replicated` | `gather()` returns |
|---|---|---|---|---|
| freshly built model | `[PlacementReplicate()]` | `[4]` | `True` | `(1, 1, 1024, 1024)` |
| after 2 DDP steps | `[PlacementShard(0)]` | `[4]` | `False` | `(4, 1, 1024, 1024)` |

Something in the step — the gradient all-reduce in
`core/distributed/distributed.cpp`, or the output tensor of the fused AdamW kernel — writes back
a parameter whose recorded placement is `Shard(0)` on the DDP axis, even though the value it
wrote is identical on every device.

`ttml.checkpointing.save_checkpoint` then does exactly what the metadata says
(`ttml/checkpointing.py:169`, `Sharding.from_tensor(tensor).gather(tensor)`): a `Shard` axis is
concatenated along its sharded dim. Every saved tensor gains a leading dimension of 4 holding
four identical copies. Measured on this project's 1024 size (123M params), same run, same step
count, differing only in `--ddp`:

| run | checkpoint size |
|---|---|
| `--ddp 1` | 737,824,624 bytes |
| `--ddp 4` | 1,475,602,288 bytes |

`Sharding.gather` is **not** the bug — given `Replicate` it correctly takes a single copy, which
is what the "freshly built model" row shows. It is faithfully honouring wrong metadata.

### Why this matters beyond file size

The resulting checkpoint is wrong in a way that reads as plausible. Every parameter name is
correct and every value is correct; only the shape has an extra leading axis. This project's
`convert/checkpoint_reader.py`, `convert/hf_mapping.py` and `convert/ttml_forward.py` all match
on literal parameter names and assume whole `[1, 1, out, in]` tensors, so the error surfaces (if
at all) far from its cause, during HF conversion or parity checking.

### The fix

Preserve the placement when writing an updated parameter value back. A parameter that was
`Replicate` on a mesh axis before the optimizer step is still `Replicate` after it — the
all-reduce exists precisely to guarantee that. Wherever the post-step tensor is constructed, it
should inherit the parameter's existing `tensor_topology()` rather than defaulting to a sharded
placement.

Failing that, `synchronize_gradients` (which already knows each parameter's placement — it calls
`is_sharded_on_axis` to decide which axes to reduce over,
`core/distributed/distributed.cpp:43-52`) is a natural place to restore it.

### What we did instead

`train/checkpoint.py:replicated_for_save` rewrites each offending parameter's `TensorTopology`
to an otherwise-identical one whose placements are all `Replicate`, runs ttml's saver, and
restores the original topology in a `finally`. ttml's `Sharding.gather` then takes a single copy,
which is what `--ddp 1` writes and what `convert/` expects. Verified on hardware: after 50 DDP
steps, every tensor in the written file is **bitwise equal (max abs difference 0.000000e+00)** to
replica 0 read independently through a `concat_mesh_to_tensor_composer`, which does not consult
the placements being corrected. The restore is what makes this safe to run mid-run at a
`--save-every` boundary: afterwards all 66 parameters carry the topology they carried before, so
a run that checkpoints is the same computation as one that does not.

`assert_saveable_on_mesh` remains a real gate, narrowed rather than removed. It refuses unless
both (a) ttml's live parallelism context is DDP and only DDP — under TP a `Shard` placement
may be the truth, and re-marking it would write a quarter of a model — and (b) the tensor is
distributed over exactly the DDP axis. Both conditions fail closed; any failure to read the
parallelism context is a reason to refuse, never permission.

---

## 4. Stochastic rounding under DDP breaks the replica-identity invariant

Status: open. Not blocking — a `--ddp N` checkpoint records replica 0, which is a complete
and coherent model — but it means "the model" a multi-chip run produces is one of N
non-identical models rather than the single model DDP is supposed to maintain. Measured
2026-08-16; see `.superpowers/ddp-checkpoint-fix.md` §4.

### The defect

Data parallelism's defining invariant is that every replica holds the *same* parameters: the
gradient all-reduce exists precisely to guarantee it. `stochastic_rounding: true` breaks it. Each
device's AdamW chooses its rounding direction from its own RNG, so four replicas that receive a
bit-identical all-reduced gradient still write four different bfloat16 values, and thereafter
perform independent random walks about a common trajectory.

Two `--size 1024` DDP runs on a `[1, 4]` mesh, four steps, identical in every respect except one
optimizer flag:

| `stochastic_rounding` | parameters whose replicas differ | `max |replica0 - replica_i|` |
|---|---|---|
| `false` | **0 / 66** | **0.000000e+00** |
| `true` | **66 / 66** | **2.343750e-02** (`llama/llama_block_5/mlp_norm/gamma`) |

At 50 steps the divergence is 3.125e-02. The RMSNorm gammas are the loudest because they sit at
1.0, where one bfloat16 ulp (0.0078) is an order of magnitude larger than the ~3e-4 update they
receive — which is the same arithmetic that made `stochastic_rounding` necessary in the first
place (see `train/configs/nanollama3_bpe_v2.yaml`). Stochastic rounding is the right fix for
that; it simply was not made DDP-aware.

### Why it matters

1. **It is silent.** Nothing reports it, the loss curve is unaffected, and the natural
   verification — "check the replicas are bit-identical" — is the one this breaks. A checkpoint
   must therefore *choose* a replica, and nothing records which.
2. **It defeats the obvious form of a save-time guard.** This repo wanted to gate its topology
   correction on "the replicas agree, so taking one copy is safe". That gate passes under the
   default config and refuses under the project's recommended one, so the guard had to be built
   on structural facts about the parallelism context instead.

### The fix

Draw the rounding decisions identically across the DDP axis: seed the stochastic-rounding RNG
from a value shared along that axis (the ParallelismContext already knows the axis and the
device's index on it), rather than per-device. A parameter that starts replicated then stays
replicated exactly, and DDP's invariant survives a feature that is otherwise strictly good.

### What we did instead

Nothing that changes the training: this is reported, not worked around. `replicated_for_save`
writes replica 0, and `.superpowers/ddp-checkpoint-fix.md` records that this is a choice among N
rather than a distinction without a difference.

---

## 5. `SFTTrainer`'s `peft_config`/`AdamW` path computes correct LoRA gradients but never applies them

Status: open, blocking. No workaround exists from our side — this is inside `AdamW::step()`,
which our repo does not build or patch. Found 2026-08-27 while designing a LoRA-based approach
to the editor objective (`scripts/train_editor_lora.py`); see CLAUDE.md's matching entry for the
project-side account.

### The defect

`SFTTrainer(model=model, peft_config=LoraConfig(...), ...)` — the exact, documented usage from
`tt-train/docs/SFT_TRAINER.md`'s own LoRA section, no custom code — trains for real steps,
reports a normal-looking loss curve, and never moves a single `lora_A`/`lora_B` tensor from its
initial value. This was isolated to `AdamW.step()` itself through five independent, increasingly
narrow experiments, each confirmed on real hardware before ruling it out:

| test | result |
|---|---|
| ttml's own core autograd: frozen input activation, trainable weight, one `linear` + `backward()` | weight gradient correctly nonzero (0.0339) — ttml's autograd is not the problem |
| standalone `LoraLinear` (no model, no trainer), one forward + `backward()` | `lora_B` gradient correctly nonzero (0.0427); `lora_A` gradient correctly zero (expected: `lora_B` inits to all-zeros, so `d(loss)/d(lora_A)` is mathematically zero until `lora_B` moves) |
| real warm-started model, `LoraModel`-injected, manual forward + `backward()` (no `SFTTrainer`) | `lora_B` gradient correctly nonzero (0.0015); confirms injection genuinely reaches the real forward pass (`type(blocks[2].attention.q_linear) == LoraLinear`, and it is the *same* Python object `.parameters()` returns) |
| manual `zero_grad → forward → backward → clip_grad_norm → optimizer.step()`, replicating `SFTTrainer`'s own sequence by hand | gradient present and nonzero (0.0010) immediately before `.step()`; **`lora_B`'s value is bit-identical before and after** `.step()` |
| `SFTTrainer(..., peft_config=LoraConfig(...))` — the native, undocumented-workaround path, no custom wrapper class at all | same result: `max|delta| == 0.0` after a real training step |

The last two rows are decisive: gradient computation is correct at every point up to and
including immediately before `AdamW.step()` is called, and the parameter is provably unchanged
immediately after. `sft_trainer.py`'s own source (`_save_checkpoint`'s docstring) already flags
this area as unfinished: *"When a `peft_config` is used and the model is wrapped in `LoraModel`,
the default saver currently saves all parameters (base + LoRA)... TODO: filter... to save only
LoRA adapter parameters"* — evidence that PEFT support in this `SFTTrainer`/`AdamW` combination
has not been exercised end-to-end before.

The leading (unconfirmed) hypothesis, from reading `optimizers/adamw.cpp`: `AdamW`'s constructor
populates `m_exp_avg`/`m_exp_avg_sq` only for parameter names with `requires_grad()==true` **at
construction time**, and `step()` looks each parameter's moment buffers up **by name** in those
maps. If the name-to-tensor association `AdamW` captured at construction (from one call to
`model.parameters()`) does not, for a freshly-`LoraModel`-injected parameter, correctly persist
through to the live tensor `backward()` actually accumulates into on a later call, `step()`'s
per-name lookup could silently apply zero effective update — but this was not confirmed at the
C++ source level; it is a plausible mechanism consistent with every symptom observed, not a
proven root cause.

### The measurement

All five rows above, one Blackhole p300c, `tt-tnt-1024` (123M params, 8 blocks), `LoraConfig(rank=8,
alpha=16.0, target_modules=["q_linear","kv_linear","out_linear"])`, warm-started from a real
checkpoint. Every intermediate script is a throwaway diagnostic, not committed to this repo.

### The fix

Needs tt-train maintainer attention inside `AdamW`'s C++ implementation (or wherever the
name-to-tensor binding between `LoraModel`-injected parameters and the optimizer's per-parameter
state is established) — outside what this repo can patch or work around.

### What we did instead

Did not run the real training job. A 3000-step run against a mechanism already proven, on real
hardware, to apply zero update to any LoRA parameter would have been pure waste — the five
diagnostics above are individually cheap (seconds to low minutes each) precisely so a
multi-thousand-step run never has to be gambled on an untested assumption. `scripts/
train_editor_lora.py` is kept in the repo, documented as blocked, ready to resume once this is
fixed upstream — the data-loading, category-building, and stratified-split logic it reuses from
`scripts/train_editor.py` are unaffected by this defect and need no rework.

## 6. Sequential prefill's KV-cache state does not reset between independent requests in a growing multi-turn conversation

### The defect

Serving `episod/tt-tnt-1024` (max_position_embeddings=512) crashes deterministically at the
**4th** turn of ANY growing multi-turn `/v1/chat/completions` conversation (i.e. the 4th HTTP
request in a session where each request resends the full accumulated message history, the
pattern every OpenAI-style chat client — Open WebUI, and this project's own skit turn-by-turn
generation — uses), with:

```
AssertionError: Sequence length 1024 exceeds max seq len 512
```

thrown from `models/tt_transformers/tt/model.py`'s `prepare_inputs_prefill`
(`assert mat_len >= seq_len`), which kills the whole `EngineCore` (`EngineDeadError`) and the
`vLLM` `APIServer` self-shuts-down. The failing request's own prompt, per vLLM's own
`usage.prompt_tokens`, was **~101-107 tokens** — nowhere near 512. `seq_len` is not derived from
the actual request content; it lands at exactly **1024 = 2 × mat_len** every single time,
independent of the real prompt size.

### Reproduction, and what was ruled out

Isolated with `--no-enable-prefix-caching` already set (this project's own earlier
crash-proofing pass), on 2 chips (`MESH_DEVICE=P300x2`), no supervisor/proxy in front:

1. **A single request whose prompt genuinely exceeds 512 tokens is rejected cleanly** with HTTP
   400 ("This model's maximum context length is 512 tokens...") by vLLM's own admission check
   (`vllm/renderers/params.py`) — proves the crash is not simple context overflow.
2. **25 independent, non-growing, single-turn short requests in a row: zero crashes.** Proves it
   is not "N cumulative requests" in the abstract.
3. **A growing multi-turn conversation (each turn appends the prior turn's user+assistant
   messages, matching real chat-client and skit-turn-by-turn behavior): crashes on request #4,
   every time**, with `usage.prompt_tokens` around 101-107 at the failing request — proves it is
   specifically about state carried across turns of ONE conversation, not raw request count.
4. **`--block-size 512` (vs. default 64): identical failure**, same turn, same `seq_len=1024`.
   Rules out page/block-table accounting as the mechanism.
5. **`--max-num-seqs 1` (forces the scheduler to admit exactly one sequence at a time, making
   `batch_size > 1` structurally unreachable — the precondition for `generator.py`'s
   `use_batched_prefill` path): identical failure**, same turn, same `seq_len=1024`. Rules out
   concurrent/batched-prefill slot mixing as the mechanism, despite `seq_len` landing on exactly
   `2 × mat_len` (the value a padded_batch=2 batched-prefill bug would produce) — that value
   apparently arises even on the plain sequential single-user prefill path
   (`prefill_forward_single_user_text`), which computes `last_token_idx = seq_len - 1` from
   `prompt_lens[idx]` and `num_cached_tokens = int(start_pos[idx])` — meaning the leaked state is
   most likely a stale `start_pos`/cached-token count (or a token buffer still sized to a
   previous call's padded width) that survives from one turn to the next instead of resetting,
   even with prefix caching disabled at the request-admission layer.

### Why it matters

Every turn-by-turn generation pattern this project cares about — skit turns
(`train/skit.py`'s five-slot schema), a real editor/dialogue back-and-forth, or simply a human
chatting for more than 3 turns — hits this by the 4th exchange, on any chip count, any block
size, any concurrency setting. It is not a multi-chip fabric issue and not a context-length
issue; it reproduces identically on the plain, single-sequence, single-chip-equivalent path.

### The fix

Lives inside `models/tt_transformers/tt/generator.py`/`model.py`'s KV-cache/`start_pos`
bookkeeping in the tt-metal source tree (`models/tt_transformers`), which this project's standing
rule prohibits patching directly. The state that should reset per independent request (whether
that is `start_pos[idx]`, the paged-attention block table's notion of "computed tokens" for the
reused slot, or a token buffer retained at a previous call's padded width) is not resetting when
`--no-enable-prefix-caching` is set — that flag evidently controls the scheduler's cache-hash
lookup, not whatever internal state this defect actually lives in.

### It is not our model — confirmed on stock `meta-llama/Llama-3.2-1B-Instruct`

Served **`meta-llama/Llama-3.2-1B-Instruct`** (a fully-supported, officially-listed model in
`model_config.py`'s own `MAX_PREFILL_CHUNK_SIZES_DIV1024` table) through the identical stack —
same `server_example_tt.py`, same 2-chip `MESH_DEVICE=P300x2`, same `--no-enable-prefix-caching`
— with `--max_model_len 512` set explicitly to match our model's native context. Ran the
identical growing-multi-turn reproduction (append each turn's user+assistant messages, resend
the full history). **Identical crash, identical signature**, at turn 3 (prompt_tokens=91,
nowhere near 512):

```
AssertionError: Sequence length 1024 exceeds max seq len 512
```

This settles the question of whose bug it is: **it is a generic tt-metal/vLLM serving defect,
not anything in this project's model, adapter, or training.** Real Llama/Qwen deployments never
observe it because they are served at their native context (4096-131072 tokens), where an
ordinary few-turn chat conversation never grows anywhere near double that. Our checkpoint is
uniquely exposed only because `max_position_embeddings=512` is small enough that ordinary usage
trivially reaches the boundary where this latent bug fires — the same defect a production-size
Llama deployment would need a genuinely deep, multi-thousand-token conversation to ever surface.

### What we did instead

Kept serving to single-turn-per-request usage (each skit turn, or each `/v1/completions` call,
submitted as a self-contained prompt with no accumulated chat history) until this is fixed
upstream or root-caused further — confirmed safe by reproduction step 2 above (25 independent
non-growing requests, zero crashes). Interactive multi-turn chat (Open WebUI or otherwise)
against this checkpoint should be treated as unreliable past 3 turns until this is resolved.

**Model-side hardening this finding actually motivates:** since the exposure is proportional to
`(conversation growth per turn) / max_position_embeddings`, not a defect in this project's
training, the two real levers are (a) training a checkpoint with a larger native context (this
project already has 384-at-2048 variants) so ordinary conversations sit far from the boundary
the way production Llama/Qwen deployments do, and (b) a serving-layer guard — independent of the
upstream fix — that server-side truncates or windows conversation history before it reaches
vLLM, so no request ever asks the engine to hold a full unbounded transcript regardless of the
model's declared context.

## 7. `FABRIC_2D_TORUS_XY` on a degenerate 1x4 mesh — a source-level hypothesis, not yet confirmed on hardware

### The defect, as previously measured

`docs/serving-with-tt-kernel.md` §8 already recorded a controlled A/B (same prompt, same
sampling settings, `scripts/story_tools.py`) run back-to-back on 4-chip
(`FABRIC_2D_TORUS_XY`, `mesh-1x4-ring.textproto`: `dims: [1, 4] dim_types: [LINE, RING]`) and
2-chip: 2-chip's candidates were grammatically rough but recognizable English throughout;
4-chip produced invented non-words (`Tryburg`, `Alexandary`, `Higheriq`) across most
candidates — a pattern distinct from ordinary repetition-collapse. Root cause was left as "a
numerical correctness issue in that fabric's collective ops is the leading suspect, untested."
This entry is that follow-up investigation, done at the source level (no hardware was free —
see below), and it narrows the suspect without confirming it.

### What source reading actually narrows down

`tt_metal/fabric/fabric_context.cpp`'s `need_deadlock_avoidance_support()` treats
`FabricType::TORUS_XY` (what `FABRIC_2D_TORUS_XY` maps to via `get_fabric_type()`) as
requiring deadlock-avoidance support in **every** direction unconditionally — the function's
`torus_mismatch` check only special-cases `TORUS_X`/`TORUS_Y` individually, never `TORUS_XY`,
so for `TORUS_XY` the check is always `false` and deadlock avoidance is applied everywhere.
On our actual mesh (`dims: [1, 4]`), only ONE of the two logical dimensions is genuinely
torused (the size-4 `RING` dimension); the size-1 `LINE` dimension has no real wraparound to
protect. Applying deadlock-avoidance logic to a direction that doesn't need it is the *safe*
direction of a mismatch (extra caution, not skipped correctness) and, read in isolation,
looks unlikely to be the mechanism that corrupts *values* rather than *timing/scheduling*.

No CCL op source (`ttnn/.../operations/ccl/**`) references `Torus` by name at all — the
collective-reduce math itself appears topology-agnostic and delegates entirely to the fabric
routing layer for how packets move between participants. That makes the fabric routing layer
(the same `fabric.cpp` area as the already-documented `#22524` 1D workaround-loop bug) the
more plausible place for a value-level defect: if a `TORUS_XY`-configured route enumerates a
degenerate 1-row mesh's participants incorrectly (an extra hop, a missed wraparound, a
double-counted neighbor), a collective's partial sum would be computed over the wrong
participant set — producing a finite, plausible-looking but numerically wrong hidden state,
which is exactly the "fluent but occasionally invents a word" symptom rather than a crash or
gross garbage.

**Absence of evidence, itself noted rather than treated as a null result:** no test file under
`tests/tt_metal/tt_fabric/` and no doc under a fabric-related path in this tt-metal checkout
mentions a degenerate 1xN torus shape by name. Combined with `docs/serving-with-tt-kernel.md`
§8's own note that this project's own 2-chip config has never itself been verified under
`FABRIC_2D_TORUS_XY` either, the honest reading is: a 1x4 "2D torus" is unexercised
configuration space in this tt-metal checkout's own test coverage, not a well-trodden path
this project happened to hit a rare bug in.

### What this is NOT

This is not a confirmed root cause. It is a plausible mechanism (routing-layer participant
miscount under a degenerate torus dimension) inferred from source reading alone, with no
on-device measurement behind it -- the investigation ran while all 4 chips were occupied by
the corrected ctx2048 retrain (see CLAUDE.md's 2026-08-29 entry), so nothing here was tested
live.

### The concrete verification this needs, once hardware is free

The `$autofix`/`ttm-multichip` pattern this project has used before: compare 4-chip TTNN
output to the 2-chip baseline with **identical synthetic weights and inputs** (removes
model-quality confounds entirely), at the component level -- attention output before/after
the all-reduce, specifically -- rather than only comparing generated text. A clean PCC match
at every component would refute this hypothesis outright; a divergence localized to the
all-reduce/all-gather boundary would confirm it and point at the exact op to patch (in our
own model code, not tt-metal, if the fix is a topology/config choice rather than a tt-metal
bug) or file upstream (if it is).

### What we did instead

Kept `docs/serving-with-tt-kernel.md` §8's existing "do not use 4-chip for anything
quality-sensitive yet" caveat in force -- this entry adds a narrowed hypothesis and a concrete
next experiment, not a fix. 2-chip serving (verified stable and correct throughout this
project) remains the config to trust.

## 8. `SFTTrainer._save_checkpoint` writes identical weights to every periodic checkpoint

### The defect

Every `step_*.pkl` written by one `SFTTrainer` run contains **byte-identical model weights**,
despite correct per-file `step` fields and mtimes minutes apart. Measured on
`artifacts/checkpoints-1024-tool-calling/` (3000 steps, `save_interval=1000`):

```
step_1000.pkl  21:24:32
step_2000.pkl  21:27:35
step_3000.pkl  21:30:37
step_1000 vs step_3000 model_state: 66/66 tensors identical, max abs diff 0.000000e+00
```

Not specific to that run: `artifacts/checkpoints-1024-editor/` (a separate run, different data,
2026-08-27) shows the same 0/66. So this has been true for every SFT-path run in this project.

### What is NOT wrong

Two controls bound the blast radius, and both matter:

* **The weights are genuinely trained.** Against the warm-start base, `step_1000.pkl` differs on
  **66/66** tensors (max abs diff 2.3e-2), and the resulting model emits a well-formed tool call
  in 100% of sampled generations where the warm-start base emits 0%.
* **Two different runs produce different weights.** Stage 1 vs stage 2 final checkpoints differ
  on 66/66 tensors, so run-to-run comparisons are unaffected.

### Why it matters

`--save-every` produces duplicates, so **intermediate checkpoint selection is impossible on this
path** — which is exactly what a caller wants when a run overfits. Both tool-calling runs did
overfit (stage 1's best validation loss is step 1000 at 1.642, rising monotonically to 1.747 by
step 3000), and the natural response — evaluate the step-1000 checkpoint — silently evaluates
the same weights again and reports identical numbers. Any past comparison *between steps of a
single SFT run* was comparing identical files.

Which step's weights the duplicates actually hold is not established here.

### The fix

Lives in `SFTTrainer._save_checkpoint` in tt-train, which this project does not build or patch.
The likely shape is a `model_state` captured once (or a stale reference) rather than re-read
from the live model at each save, but that is inferred from the symptom, not confirmed in the
C++/Python source.

### What we did instead

Recorded it and kept using final checkpoints only. Nothing in this project currently depends on
selecting an intermediate SFT checkpoint; the cost so far is wasted training past the overfit
point, not a wrong published number.
