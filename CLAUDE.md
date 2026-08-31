# tt-tnt — project notes for Claude

## What this project is

The reference example of a **Tenstorrent-first model**: tt-tnt (originally built, and
published, as **tt-nanollama3** — see README.md's "Lineage" section for what changed and
why), trained from random init on Blackhole with `ttml` (tt-train), packaged as a
**tt-kernel v4 bundle**, and served through the **Tenstorrent vLLM plugin**. Small model,
complete story — the point is to show end to end what a model built for TT from line one
looks like across train → package → publish → serve.

The log below is kept in the order it happened, under the names used at the time — entries
before the 2026-08-13 rename say "NanoLlama3"/"tt-nanollama3" because that was this project's
name when they were written, not because they refer to something else.

Design: [`docs/superpowers/specs/2026-08-11-tt-nanollama3-design.md`](docs/superpowers/specs/2026-08-11-tt-nanollama3-design.md)

## How we got here (2026-08-11)

The original prompt was a changelog request against `tt-kernel-package-manager`, which turned
into: *"what would it take to make tt-animatediff part of tt-kernel-cache?"* Exploring that
surfaced a blocker, and the plan pivoted twice.

**Pivot 1 — animatediff doesn't fit tt-kernel.** Both tt-kernel serving backends terminate in
`/v1/chat/completions`; a text-to-video model has no chat-shaped output. Then the repo was
updated to `00dba42`, which made it worse: v4 is explicitly "vLLM only, kernels-less" and v3
is "legacy, read-only supported." A model needing a kernel cache has only the deprecating
path; a non-vLLM model has none. Findings recorded in Appendix A of the spec — worth taking
to the tt-kernel maintainers.

**Pivot 2 — use a model we actually own.** The `lfs-00`…`lfs-05` lesson arc in
`tt-vscode-toolkit` builds a Llama-3-style model from scratch, and it had already been
trained: `~/tt-metal/tt-train/checkpoints/nanollama3_char_3k.pkl_final.pkl`. Being an LLM, it
fits v4 perfectly, and the weights are ours outright — no redistribution question.

## Licensing — a relationship to maintain, not a one-time file

This repo is **Apache-2.0**, matching tt-metal and tt-vscode-toolkit. Every source file carries
an SPDX header. That much is settled. What needs *maintaining* is the honesty of the
provenance section in `README.md`, because two upstreams are not simply Apache-2.0:

- **TinyStories is CDLA-Sharing-1.0**, a share-alike *data* license — not permissive. We do not
  redistribute the corpus (it downloads from the Hub at a pinned revision), and we do **not**
  assert that weights trained on it fall outside "Data Derivative". Do not quietly upgrade that
  hedge into a claim.
- **Mini-LLM declares no license** (verified via the GitHub API — `license: None`), so it grants
  no rights. It is credited for architectural *choices*, which come from published papers; our
  implementation derives from tt-train's `nanollama3` config and `ttml`, not its source. Keep
  that distinction sharp — a credit is not a license inheritance.

**Rules for future work:**
- New source files get the SPDX header pair. No exceptions.
- Adding a dependency, corpus, or checkpoint source means adding its license to the README's
  provenance section *in the same change* — not later.
- **When weights are finally published**, the model card must name the corpus and its license
  explicitly, and describe the model as a demonstration rather than a capable general model.
  At ~22M parameters over a fraction of an epoch, any other framing would be false.
- If a future corpus is more permissive (or more restrictive) than TinyStories, the provenance
  section changes with it. It is not boilerplate.

## Key decisions

- **~22M params, 32K BPE, small real corpus.** Uses the existing `nanollama3.yaml` unchanged.
  Deliberately *not* Mini-LLM's ~80M/361M-token/~5h-A100 run, which would need a new model
  config plus a data pipeline and would block packaging behind a multi-day job.
- **Ship both altitudes.** ttnn path as the portable default, the lesson's TT-Lang kernels as
  the tuned path, bound by parity tests against `reference_gpt.py`. This is what makes it an
  exemplar rather than a packaging demo.
- **Real adapter, not a disguise.** Validate conversion via a throwaway `LlamaForCausalLM`
  spike, then write `TTNanoLlama3` registering its own `arch_name`.
- **Reuse policy:** never reimplement what ttnn or ttml already provide.

## Gotchas learned during design

- `nanollama3.yaml` vs `nanollama3_char.yaml` differ **only** by `vocab_size: 32000` — same
  6 heads / 3 groups / 384 dim / 6 blocks / seq 256 / θ=500000. Switching to BPE does not
  make the model meaningfully larger; it adds the embedding table.
- Only `shakespeare.txt` (1.1 MB) is present in `tt-train/data/`. Far too small for a 32K
  vocab — a real corpus is required.
- Upstream tt-metal runs **no CI for training ops on Blackhole** (`tt-train` `GTEST_SKIP`s
  softmax, cross-entropy, rmsnorm, SDPA on p100/p150). The lesson arc's run is our own
  verification, at v0.73. `~/tt-metal` is currently on `rollback-pre-qwen36-1576-g620793d898`.
- Let `ttml` close the device — bypassing its `finally` triggers a teardown abort in
  `MetalContext::destroy_all_instances`.
- If the board times out on device open, `tt-smi -r` first. Common on p300c/QB2.
- `max_sequence_length` is 256, so the manifest's `max_model_len` must be 256. Don't promise
  more than the model was trained for.

## `feat/tokenizer-and-corpus` (2026-08-11)

Built the corpus-prep and tokenizer pipeline the model needs before any training run:
`train/data.py` (fetch + normalize TinyStories), `convert/tokenizer.py` (32K byte-level
BPE, exported for both ttml load paths), and `scripts/build_tokenizer.py` (the script that
actually produces the shipped artifacts). Real numbers from the production run: the 2.2 GB
raw download (2,227,753,162 bytes) prepares down to 536,870,821 bytes / 3,548,279 lines at
the 512 MB cap, and the tokenizer trained on that reaches exactly 32000 tokens — no
shortfall. (Line count is higher than a naive pre-fix estimate would suggest: `</s>` is 9
bytes shorter than `<|endoftext|>`, so once separator lines are rewritten, more whole
lines fit under the same 512 MB cap before truncation kicks in.)

A whole-branch review caught three things worth not rediscovering:

- **`vocab_size` is a ceiling, not a promise.** `BpeTrainer` stops merging once the corpus
  runs out of pairs worth learning — a small corpus (or a small `--corpus-mb`) silently
  under-shoots the target. `scripts/build_tokenizer.py` now reloads the export via
  `load_exported()` and hard-fails (non-zero exit) if the achieved vocabulary doesn't
  match the requested one, rather than printing "Done" over a mismatch that would only
  surface later as an embedding-shape failure in tt-train.
- **`PreTrainedTokenizerFast` overrides `add_prefix_space` on wrapping.** Setting
  `pre_tokenizers.ByteLevel(add_prefix_space=True)` on the backend tokenizer is not
  enough — `PreTrainedTokenizerFast.__init__` (transformers 4.52.4) applies its own
  `add_prefix_space=False` default onto the wrapped tokenizer, silently discarding it.
  Merges were being *learned* with prefix-space on but *applied* with it off. Fix: pass
  `add_prefix_space=True` to the `PreTrainedTokenizerFast(...)` constructor itself, and
  verify by reading the exported `tokenizer.json` back — don't assume the flag survived
  wrapping. One consequence: decode() now legitimately produces a leading space on text
  that doesn't already start with whitespace — that's the injected prefix space coming
  back out, not data loss.
- **TinyStories' `<|endoftext|>` separator must be mapped to `</s>`, not left as prose.**
  Passed through unmodified, it's four ordinary subword tokens per occurrence (662,878
  occurrences in the production corpus — 18.7% of all lines), a wasted vocabulary slot on
  the subword `endoftext`, and — critically — zero real appearances of `</s>` in training
  data, so the model never learns a stop token. `prepare_corpus` now rewrites lines that
  are *exactly* the separator (after stripping) to the literal text `</s>`, which the
  tokenizer maps to the eos special id; lines that merely mention the separator inside
  other text are left alone, since a substring replace would corrupt real prose.

## `feat/training-entrypoint` (2026-08-11)

Wrote `train/run.py`, the hardware entrypoint that actually trains NanoLlama3 on Blackhole.
tt-train's own Python trainer (`examples/python/transformers/training.py`) doesn't run
against the current tree — three independent breaks (stale `trainer` module import, an
extra `val_ids` argument `train()` doesn't accept, a `TrainingConfig` missing the `seq_len`
`train()` reads) plus a hardcoded `shakespeare.txt` data path — so this entrypoint reuses
`TransformerModelFactory`, `create_optimizer`, `initialize_device`, `set_seed`, and `train()`
itself, and supplies our corpus, our tokenizer, `seq_len`, and a real `evaluate()`. That last
part matters: ttml's `train()` fills `val_losses` by copying the last training loss under a
comment calling it placeholder behavior, so a val number straight out of `train()` means
nothing — `evaluate()` runs the model in eval mode over held-out tokens with its own
`get_batch_ttml`/`build_causal_mask` calls and averages properly.

**Tokenizing the real corpus** (Task 1's pipeline, run against the actual TinyStories
corpus): `TokenStats(total_tokens=127635889, train_tokens=114872301, val_tokens=12763588,
vocab_size=32000)` — inside the 1.3–1.6×10^8 estimate, vocab exactly on target.

**A real 20-step run on p300c/Blackhole**, batch size 64, seq_len 256:

- First train loss: `10.6875` — matches the freshly-initialized-model expectation of
  `ln(32000) ≈ 10.4` closely enough to confirm the vocab/model setup is right.
- Last train loss: `7.4688`, monotonically non-increasing across all 20 steps (one flat
  step at 8.0625→8.0625, otherwise strictly down).
- Real validation loss (our `evaluate()`, 10 sampled batches, not `train()`'s placeholder):
  `7.0281` — distinct from the last train loss, so the loop was not bypassed.
- Step 1 took 18.73s (kernel compilation, not model performance); steps 2–20 averaged
  ~0.12–0.14 s/step (climbing to ~7.15 it/s by the end) — that second number is the one
  that reflects the model.

No checkpoint is written (`model_save_interval` is 0 in `RunConfig`/`build_yaml_config`) —
that's Stage 3's job, once we've read `ttml/checkpointing.py`'s format.

## `feat/checkpointing` (2026-08-11)

Added a validated checkpoint header schema over `ttml.checkpointing` (`train/checkpoint.py`)
and periodic save/resume to the training entrypoint (`train/run.py`, calling `train()` in
`--save-every`-sized chunks so the same `optimizer` object persists across calls — AdamW's
moments carry over rather than resetting each chunk). Then ran the first real training run
worth keeping.

**The real run**, p300c/Blackhole, `--steps 3000 --save-every 500 --batch-size 64`:

- First train loss `10.6875` — consistent with a near-uniform initial distribution (uniform
  over a 32000-token vocabulary would be `ln(32000) ≈ 10.37`; the measured value is 0.31 nats
  above that, still the expected ballpark for a freshly-initialized model, not an exact
  match), last train loss `1.9219`, real validation loss (our own `evaluate()`, 10 sampled
  batches, not `train()`'s placeholder) `1.8781` — perplexity `≈ e^1.8781 ≈ 6.5`. Validation
  coming in *below* that last train figure is expected, not a labeling bug: at 0.43 of an
  epoch there's no repeated exposure to the data to overfit on, dropout is 0.0 so there's no
  train-time regularization noise either, and the train figure is a single noisy mini-batch
  while the validation figure averages ten — comparing a lone sample to a ten-batch average
  will show sampling noise in either direction. That the two numbers differ at all (rather
  than being identical) is itself the evidence that `evaluate()` genuinely ran, instead of
  silently falling back to `train()`'s placeholder copy.
- **Windowed-mean loss curve** (300-step windows, reconstructed from all 3000 per-step
  readings): a clean log shape — steep for the first ~300 steps, then a steadily decelerating
  decline into a noisy 1.9–2.1 band for the back half — not a plateau, not divergence.

  | Steps       | Mean   | Min    | Max     |
  |-------------|--------|--------|---------|
  | 1–300       | 4.4747 | 3.0625 | 10.6875 |
  | 301–600     | 2.8236 | 2.5469 | 3.1875  |
  | 601–900     | 2.4894 | 2.2656 | 2.7344  |
  | 901–1200    | 2.3149 | 2.1250 | 2.4844  |
  | 1201–1500   | 2.1941 | 2.0469 | 2.3438  |
  | 1501–1800   | 2.1187 | 1.9688 | 2.2969  |
  | 1801–2100   | 2.0543 | 1.9141 | 2.2188  |
  | 2101–2400   | 2.0060 | 1.8750 | 2.2031  |
  | 2401–2700   | 1.9621 | 1.8594 | 2.1250  |
  | 2701–3000   | 1.9345 | 1.8125 | 2.0938  |

  As expected for a raw per-batch metric, the curve is *not* step-to-step monotonic: of 2999
  step-to-step transitions, 1359 went up and 1417 went down (223 unchanged), while every
  windowed average above kept falling.
- Steady state `~0.134 s/step` (7.44–7.50 it/s), matching Plan 2's 0.12–0.14 s/step. Unlike
  Plan 2, this run's first step showed no visible compiler-warmup stall — most likely because
  Task 2's same-shapes hardware runs earlier today had already warmed the on-disk kernel
  cache; not confirmed by inspecting the cache directly, so treat as a plausible explanation,
  not a verified one.
- Total wall clock `~6 min 47 s` (process start to final printed loss), including all six
  checkpoint writes — the writes are bounded below ~0.5 s per write. Measurement resolution
  here (~±0.25 s per 500-step chunk, from checkpoint-file mtime deltas) exceeds the actual
  effect size (~0.16 s implied by the chunk-to-chunk variance), so "no visible cost" would
  overstate what was measured; every 500-step chunk took the same ~67.1 s regardless, which is
  the tighter, honest claim.
- Six checkpoints, `nanollama3_step00000500.pkl` through `nanollama3_step00003000.pkl`
  (`artifacts/checkpoints/`, gitignored), 132,185,963 bytes each at the time of this run,
  final one at step 3000 as requested. (A later header back-fill — see the checkpoint-header
  fix wave below — added 339 bytes to each file's header record; tensor data is untouched.)

**This is a demonstration, not a capable model.** `3000 × 64 × 256 ≈ 49.2M tokens` is about
**0.43 of one epoch** over the 114.9M-token training split — this run never saw even half the
corpus once. TinyStories is also a synthetic, deliberately simple corpus (short children's
stories, small effective vocabulary, regular grammar, built so small models can fit it), so a
low loss here is expected and does not indicate general language competence — no text was
decoded from this checkpoint to check.

## `feat/checkpointing` — final whole-branch review fix wave (2026-08-11)

A final review before merge found the header schema above missing exactly what it existed
to prevent: architecture facts a converter can't recover without guessing. Fixed, no
training re-run (the box is shared; the six checkpoints above stand as the run's evidence):

- **Header now carries `intermediate_dim=1024`, `weight_tying=True`, `rms_norm_eps=1e-5`,
  `weights_dtype="bfloat16"`, and the full `transformer_config`.** The first three exist only
  as ttml C++ defaults (`modules/llama_block.cpp`, `models/llama.hpp`,
  `modules/rms_norm_module.hpp`) — nanollama3.yaml never sets them. `weight_tying` is the one
  that actually matters: because it's on, these checkpoints have no `llama/tok_emb/weight`
  tensor at all (confirmed against the manifest — 50 model tensors, none named `tok_emb`); a
  converter that didn't know would produce a model with a randomly-initialized embedding
  table and raise no error.
- **All six existing checkpoints were back-filled in place** with
  `scripts/backfill_checkpoint_headers.py` — a pure-CPU, stdlib-`pickle`-only rewrite of each
  file's header record, tensor bytes copied through unchanged. Verified byte-for-byte against
  a pre-backfill backup: the tail (everything after record 0) is bit-identical; only the
  header record grew, by exactly 339 bytes per file (132,185,963 → 132,186,302).
- **`total_tokens` (the whole corpus, 127,635,889) renamed to `corpus_tokens`**, with a new
  `batch_size` field and a derived `tokens_seen = step * batch_size * seq_len` — the number
  that actually describes training volume (49,152,000 at step 3000), which `total_tokens`
  was silently overstating by ~2.6x for anyone reading the header as a model-card source.
- **`latest_checkpoint()`'s docstring corrected** from "newest" to "highest-step" (it sorts
  by zero-padded step, not mtime) and `--resume` now prints the loaded header's `created_at`
  alongside the step, so an operator sharing `artifacts/checkpoints/` across runs can see
  which run's weights they actually got.
- **`checkpoint.load()` now validates the header before restoring any tensor**, not after —
  a bad or future-format header fails fast instead of first mutating the live model.
- **Added `convert/checkpoint_reader.py`**, a pure-CPU (`pickle`-only, no ttml/ttnn) reader
  for a checkpoint's header and tensor manifest — what the back-fill script needed, and the
  first piece of the CPU-side conversion path referenced in the design spec's Known Risks.

(Per-fix commands and the full post-back-fill verification log for this wave live in the
review's own working notes, not linked here — see this section for the numbers that matter
to anyone reading this file without that tree checked out.)

## `feat/hf-conversion` Task 3 — numerical verification finds a real bug (2026-08-11)

Task 3 added `tests/test_hf_parity.py` (4 tests, skip-guarded on `artifacts/hf/config.json`
existing) and ran the brief's Step 2/3 checks by hand. Structural checks and generation look
fine; the perplexity cross-check does not, and the gap traces to exactly the failure mode the
plan's Known Risks called out in advance.

**Structural re-derivation (independent of the controller's numbers, matched exactly):**
22.025088M params, embed/lm_head tied, `model.norm.weight` shape `(384,)`, next-token entropy
on "...a little girl named" = **4.7509 nats** (uniform ceiling 10.37), top-5
`[' Lily',' Lucy',' Jane',' Sue',' Sarah']`.

**Generated sample** (brief's exact command, `do_sample=True, temperature=0.8, top_p=0.95`,
no seed set, reported verbatim, not cherry-picked):

> Once upon a time, there was a little girl named Lily. Lily had a pretty flower. She loved
> to dance. She loved to dance. One day, she danced every day, she found a big, blue flower.
> The flower was very pretty flower in the sun, her dress, and it had a big,

Locally fluent TinyStories-flavoured English, globally drifting and repetitive ("She loved to
dance. She loved to dance.") — the expected shape for 0.43 of an epoch on a 22M model, not a
sign of a layout trap on its own.

**Perplexity cross-check — the brief's own Step 3 command has a bug.** Its literal code
(`m(x[:, :-1], labels=x[:, 1:])`) pre-shifts both `input_ids` and `labels` by hand, but
`LlamaForCausalLM`'s internal loss function (`ForCausalLMLoss` in
`transformers.loss.loss_utils`) *also* shifts `labels` internally before computing
cross-entropy. Passing already-shifted tensors through `labels=` double-shifts, comparing
each prediction against the token two positions ahead instead of one. Run literally, it
reports **8.53 nats** — worse than doing nothing. Verified against two independent correct
formulations (`model(x, labels=x)`, which lets HF do its one intended shift, and manual
`cross_entropy` on `logits` vs. `x[:, 1:]` with no `labels=` kwarg at all): both agree at
~3.19–3.20 nats on the same data, confirming the double shift as the reason 8.53 differs from
the corrected number, not a second bug.

With the shift bug fixed, and sampling matched to how the training run's own `evaluate()`
computes 1.8781 (`ttml.common.data.get_batch`: 10 batches of 32 random 256-token windows
drawn uniformly across the *full* 12.76M-token validation set, not one contiguous block) —
**HF-side val loss = 3.20 nats** (range 3.13–3.27 across the 10 batches). That is **1.32 nats
above 1.8781** — a fail by the brief's own "1+ nats means the conversion is wrong somewhere
the entropy check didn't catch" threshold. The entropy and generation checks above did not
catch this; that is exactly why Task 3's Step 3 exists.

**Root-caused via the brief's own Known Risks, without touching `artifacts/hf/` or any
tracked file** (all work done against scratch copies under
`/tmp/.../scratchpad`, using `convert.to_hf.convert_checkpoint` called directly):
- *Gate/up swap, ruled out.* Swapping `w1`/`w3` in `MLP_ROLES` in-process and reconverting
  made loss **worse** (3.63 nats), not better — the current `w1=gate_proj`/`w3=up_proj`
  assignment in `convert/hf_mapping.py` is correct.
- *RoPE interleaved-vs-split-halves, confirmed as the cause.* `convert/to_hf.py` copies
  `q_proj`/`k_proj` weight rows straight through with no permutation. Applying the classic
  Meta-Llama interleaved→split-halves permutation (reshape each head's rows as
  `(head_dim/2, 2)`, transpose, flatten — the same operation Meta's own
  `convert_llama_weights_to_hf.py` applies) to `q_proj` and `k_proj` in a scratch copy of the
  converted weights, with everything else unchanged, brought the same random-batch loss
  measurement down to **1.927 nats** (range 1.83–2.00) — within 0.05 nats of 1.8781, a clean
  pass. No source file or artifact was modified to get this number; it is a diagnostic
  reconversion in `/tmp` only.

**Conclusion at this point in the investigation: `artifacts/hf/` was measurably wrong.** It
loaded without error, tied weights correctly, produced finite non-uniform logits, and
generated plausible-looking text — every check Tasks 1–2 could have run would pass — but its
RoPE layout did not match ttml's convention, which silently degrades attention quality without
producing garbage output. Task 1/2's test suites (`test_hf_mapping.py`, `test_to_hf.py`) had
no test that would catch this; `rope_theta` is checked, the row layout within each head was
not. This was reported to the controller rather than patched immediately, since fixing it
meant writing to `artifacts/hf/`, which was off-limits under Task 3's constraints. The
controller authorized the fix; see the next section for what shipped.

## `feat/hf-conversion` Task 3 fix — RoPE row permutation, `artifacts/hf/` regenerated (2026-08-11)

Fixed for real, with the controller's authorization to write `artifacts/hf/` (the earlier
prohibition was to protect a directory believed validated; Task 3 showed it wasn't).

**The fix: `convert/hf_mapping.permute_rope_qk`.** ttml's `q_linear`/`k_linear` rows are
ordered for RoPE's *interleaved* pairing (row `2i` pairs with row `2i+1`); HF Llama's
`rotate_half` expects *split-halves* pairing (row `i` pairs with row `i + head_dim // 2`). A
weight matrix carries no signal of which convention its author assumed, so this was invisible
to every shape/name check — the tensor was the right shape, in the right place, with the
right name. The permutation (`reshape(num_heads, head_dim//2, 2, in_features).transpose(0, 2,
1, 3).reshape(out_features, in_features)`) is the same row reordering Meta's own
`convert_llama_weights_to_hf.py` applies when converting original-format (interleaved) Llama
checkpoints — not invented for this project. `num_heads` and `head_dim` come from the
checkpoint header's `transformer_config` (`num_heads` for `q_proj`, `num_groups` for
`k_proj`, both times through `config["num_attention_heads"]`/`int(tc["num_groups"])` in
`convert/to_hf.py`) — never hardcoded — so a differently-shaped future model gets the right
block size automatically. `v_proj` is untouched: RoPE rotates queries and keys before the
attention dot product, values pass through unrotated. 5 new tests in `test_hf_mapping.py`
pin the permutation's shape, that it's a true row permutation (no row dropped or duplicated —
checked by comparing row sets, not just `.shape`), a hand-verified example of which rows move
where, and that it uses whatever head count it's given rather than an assumed 6/3/64.

**`artifacts/hf/` regenerated** via `python scripts/convert_checkpoint.py` from the same
`nanollama3_step00003000.pkl` checkpoint (untouched — `artifacts/checkpoints/` remained
off-limits throughout). `model.safetensors` is a fresh file; everything else about the
pipeline (tokenizer files, config assembly) is unchanged.

**Re-verified end to end, same methodology as before, same checkpoint:**

| Check | Before fix | After fix | Target |
|---|---|---|---|
| Entropy, "...a little girl named" | 4.7509 nats | 4.9765 nats | < 7.0 (uniform 10.37) |
| Top-5 | `Lily,Lucy,Jane,Sue,Sarah` | `Lily,Lucy,Jane,Sue,Mia` | — |
| HF-side val loss (10×32×256, matched sampling) | 3.20 nats | **1.927 nats** | 1.8781 ± 0.2 |

The loss lands 0.049 nats from target — a clean pass, and it exactly reproduces the 1.927
measured in the earlier scratch-copy diagnostic (same checkpoint, same fix, same code path,
so this is confirmation the regeneration applied the fix correctly, not a new independent
result). Entropy moved a little (4.75 → 4.98 nats) but stayed far below the 7.0 test threshold
and the 10.37 uniform ceiling — a properly-rotated attention mechanism sharpens the
prediction slightly further, as expected, though this single number was never going to be
the thing that caught the bug.

**New sample, same command, no seed, verbatim, not cherry-picked:**

> Once upon a time, there was a little dog named Max. Max loved to play with his ball. One
> day, Max saw a big ball in the park. Max wanted to play with the ball, but he was very
> dirty. Max had an idea. He would push the ball with his paws to clean it.

**Does it look better, or just different?** Read honestly: this sample keeps one character
and one throughline for its whole length (dog wants to play with a dirty ball, forms a plan
to clean it) with no verbatim-repeated sentences, where the earlier sample looped ("She loved
to dance. She loved to dance.") and lost its thread in the last clause. On this single
comparison it reads as more coherent, not merely different — but it's one temperature-0.8
sample against one other temperature-0.8 sample, and generation is stochastic, so this is a
data point, not proof that every sample from the fixed model beats every sample from the
broken one. The loss number (3.20 → 1.927 nats, a real and repeatable difference on 320
held-out windows) is the reliable evidence; the prose is corroborating, not dispositive. This
matches the general shape of the lesson regardless of which single sample happened to land
better: structural checks and even a read of the generated text are not enough on their own
to confirm a conversion is right, which is the entire reason Task 3's numerical comparison
exists.

**Regression test added:** `test_hf_parity.py::test_validation_loss_matches_the_training_run`
computes the same 10×32×256 random-window loss and asserts it's within 0.2 nats of 1.8781,
skip-guarded (separately from the module's `artifacts/hf/`-existence guard) on
`artifacts/tokens/val_ids.npy` existing. This is the test that would have caught the RoPE bug
before it ever reached a report — nothing in the suite pinned this number before now.

**The brief's own Step 3 example command was also fixed**, in
`docs/superpowers/plans/2026-08-11-hf-conversion.md`, to remove the double-shift bug found
while executing this task (see the section above) and to match the training run's random
sampling, so the next person to read the plan doesn't inherit either defect.

Test suite: **108 passed** (103 from the numerical-verification commit + 4 new in
`test_hf_mapping.py` for `permute_rope_qk` + 1 new regression test in `test_hf_parity.py`),
0 skipped (converted model and validation tokens both present on this machine), 0 failed.

## `feat/hf-conversion` — pre-merge whole-branch review fix wave (2026-08-11)

A final review before merge (verdict: ready to merge) found five cheap, high-value items.
None required a training re-run; `artifacts/checkpoints/` was never touched.

**The 0.049-nat residual gap's documented cause was wrong.** Both `test_hf_parity.py`'s
`LOSS_TOLERANCE` comment and this file's Task 3 write-up (above) left the residual gap
between 1.9271 (converted model) and 1.8781 (training run) attributed to, or open to being
read as, fp32-CPU-vs-bf16-device precision. Measured directly instead: same seed, same
windows, bf16 gives 1.9315 and fp32 gives 1.9314 — dtype accounts for roughly 1e-4 nats, not
0.049. The real driver is sampling: seeds 0, 1, and 2 against the same model give 1.9314,
1.9208, and 1.8856 nats respectively — a seed-to-seed standard deviation of 0.024 nats, which
puts the 0.049-nat gap at roughly z ≈ 1.2. That's unremarkable noise, not a signal, and
correcting the *reason* matters even though the pass/fail verdict doesn't change: attributing
a real, measured 0.024-nat seed-to-seed spread to a nonexistent precision effect would send
the next person chasing bf16/fp32 numerics instead of understanding that the regression
test's fixed seed (`np.random.default_rng(0)`) is deliberately pinned for exactly this
reason — an unpinned seed would make the 0.2-nat gate flakier for no benefit. `LOSS_TOLERANCE`
stays at 0.2; only the explanation changed, in `tests/test_hf_parity.py` and in
`.superpowers/sdd/2026-08-11-hf-conversion/task-3-report.md`'s concern #4.

**Four other fixes, in brief:**
- **README's Status section was publicly stale.** It said conversion and packaging were both
  still pending and "no text has been decoded" — both false since the Task 3 fix wave above.
  Updated to state the conversion is numerically verified (1.9271 vs. 1.8781) and weights
  remain unpublished with tt-kernel packaging as the sole remaining stage, keeping the
  existing honest capability framing and citing the Max-the-dog sample as one data point.
- **`convert_checkpoint` now raises on unmapped ttml tensors** instead of silently
  `continue`-ing past them, and **`llama/fc/weight`'s fan-out to both HF embedding slots is
  now conditional on `header["weight_tying"]`** rather than unconditional. Untied models
  (`llama/tok_emb/weight` present alongside `llama/fc/weight`, per
  `ttml/models/llama.cpp:466`) were the real risk this closes: before this fix, the real
  embedding table would be silently dropped as "unmapped" while `fc/weight` was written to
  *both* `model.embed_tokens.weight` and `lm_head.weight`, producing a model that loads
  cleanly, reports `tie_word_embeddings: false`, and is numerically wrong with no error at
  any stage. Every real checkpoint produced so far has `weight_tying=True`, so this path was
  untested until now — new tests use a synthetic untied manifest rather than attempting to
  produce a real untied checkpoint.
- **`convert_checkpoint` now verifies the emitted HF key set is exactly what the config
  implies** (`9 × num_hidden_layers + 3`) before writing, raising with the missing/unexpected
  names. Previously a truncated manifest would silently produce a safetensors file missing
  keys, and `transformers` would randomly initialize them with only a warning.
- **`build_config`'s hardcoded `bos/eos/pad = 1/2/3` is now checked, not just trusted.**
  `convert_checkpoint` cross-references `tokenizer_dir`'s `special_tokens_map.json` /
  `tokenizer_config.json` and raises if the resolved ids disagree with the hardcoded values.
  Verified correct against `artifacts/tokenizer/` today; this is a guard against silent drift
  if the tokenizer is ever regenerated with different special-token ids, not a refactor.

Re-ran `scripts/convert_checkpoint.py` against the same `nanollama3_step00003000.pkl`
checkpoint (Fixes 3-5 touch the write path); `AutoModelForCausalLM.from_pretrained` still
loads `artifacts/hf/` and `test_validation_loss_matches_the_training_run` still passes.

## `feat/numpy-parity` — an independent NumPy reference, and a sharper gate (2026-08-12)

**Why this plan exists.** The HF-conversion loss gate (`test_hf_parity.py`, 0.2-nat
tolerance) caught the RoPE bug above, but two things about it are uncomfortable: its 2σ
floor is ~0.22 nats (sampling sd 0.024 × ~9), so anything cheaper than that is invisible; and
**all 13 RMSNorm gammas in the trained checkpoint are exactly 1.0** (an upstream
`stochastic_rounding` issue — see `docs/superpowers/specs/2026-08-11-followups.md`), so
swapping two norms' destinations changes loss by exactly `0.0000`. 23% of the conversion's
mapping decisions were validated by nothing.

Three tasks: **Task 1** derived ttml's forward pass straight from its C++ source into
`docs/ttml-forward-reference.md`, deliberately never reading `convert/hf_mapping.py` or
`convert/to_hf.py` — a NumPy path built from that converter would just agree with its own
misunderstandings. **Task 2** implemented `convert/ttml_forward.py` from that doc and
validated it independently by reproducing the training run's own held-out cross-entropy
(1.8488 measured, vs. training's 1.8781 — see the Task 2 report). **Task 3** (this section)
built the actual instrument: `tests/test_numpy_parity.py`, comparing the NumPy path's logits
against `artifacts/hf/`'s logits directly, at a tolerance measured from data rather than
picked in advance.

### What the parity gate measures, and the tolerance

Both paths run on the host in **float32** (`AutoModelForCausalLM.from_pretrained(...,
torch_dtype=torch.float32)`), from the same bfloat16-stored checkpoint weights, on a fixed
seeded window of `val_ids.npy`. This is a **NumPy-vs-HF** comparison, not NumPy-vs-device —
the earlier (and wrong) worry that a bf16 RMSNorm divisor makes ~1e-3 unreachable bounds a
device comparison, not this host-vs-host one.

Measured across six seeds/windows before picking a number: max absolute logit difference
**5.2e-6 to 8.5e-6**; max relative difference (restricted to `|logit| > 0.01` — unrestricted
relative error is dominated by meaningless blowups near zero-crossings, e.g. two logits of
-1.05e-5 and -1.15e-5 differ by "8%" while being numerically indistinguishable) **1.4e-4 to
4.7e-4**; correlation indistinguishable from 1.0 (`1 - corr ≈ 1e-13`). A NumPy-vs-NumPy
control (float32 throughout vs. bf16-rounded activations at every sub-layer boundary) showed
this precision effect alone would cost ~0.03 absolute and ~3-4 orders of magnitude more than
the actual NumPy-vs-HF gap — confirming the tight agreement is real, not an artifact of both
sides sharing rounding.

Tolerances set from that (all in `tests/test_numpy_parity.py`): `MAX_ABS_TOLERANCE = 1e-3`
(~100-200x the measured worst case), `MAX_REL_TOLERANCE = 5e-3` floored at `|logit| > 0.01`
(~10x the measured worst case, and tighter than the plan's own ~1e-2 "something is wrong"
ballpark by 2x), `MIN_CORRELATION = 0.9999`. Wide margin above measurement noise, and — per
the not-hollow proof below — many orders of magnitude below what an actual bug produces.

**Proof the gate is not hollow.** Monkeypatching `permute_rope_qk` to the identity function
(Plan 4's exact historical bug — straight-copied RoPE rows) and reconverting into a scratch
directory (never touching `artifacts/hf/`) produced max_abs = **4.60**, correlation =
**0.972** — ~4600x over the abs tolerance's budget. Plan 4's reviewer measured the same bug
at the loss level as 3.2015 nats against a 1.8781 target; this gate catches the identical
defect at the logit level, by a much larger margin relative to its own tolerance than the
loss gate had relative to its.

### What this gate still cannot see

1. **The norm-mapping blind spot this plan exists to close.** On the real checkpoint (all
   gammas exactly 1.0), swapping two RMSNorm gammas' HF destinations is a no-op — verified
   directly: swapping block 0's and block 1's `input_layernorm.weight` mapping and
   reconverting gives max_abs = 1.23e-5, identical (to the precision measured) to the no-swap
   baseline on this test's seed/window, because the swap moves nothing between two gammas
   that are both exactly 1.0. **The parity gate is exactly as blind to this as the loss gate
   is, on this checkpoint, for the same reason** (`test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint`
   measures and confirms this rather than assuming it). Closed *structurally* instead:
   `test_convert_checkpoint_places_each_rmsnorm_gamma_at_its_correct_destination` builds a
   synthetic checkpoint with distinct non-unit gammas and asserts each lands at its correct
   HF tensor name — a test that runs unconditionally (no `artifacts/` dependency) and stays
   meaningful regardless of whether the real checkpoint's gammas ever stop being degenerate.
2. **What neither path implements is invisible to both.** If both the NumPy reference and
   the converter drop RoPE scaling (the checkpoint header records no `scaling_factor`), they
   agree with each other and are both wrong relative to ttml's actual runtime behaviour. This
   harness validates the *conversion* — does `convert/` correctly translate ttml's checkpoint
   into HF's format — not the checkpoint's *completeness*.
3. **ttnn's own accumulation/output dtype on real hardware is untraced.** Both paths compared
   here run entirely on the host; neither touches a Tenstorrent device. This says nothing
   about NumPy-vs-device agreement, which needs its own (looser) tolerance — Task 1's finding
   that ttml's RMSNorm kernel packs its mean divisor as bfloat16 bounds *that* comparison, not
   this one.
4. **This compares two implementations of the same architecture, not the checkpoint against
   ground truth.** If ttml's own forward pass itself has a bug relative to what the training
   run's loss curve implies, nothing here catches it — that anchor is Task 2 Step 3's
   independent cross-entropy check against the training run's own held-out figure, not this
   gate.
5. **`convert/checkpoint_reader.py` is a shared dependency of both paths.** The NumPy path
   (`convert.ttml_forward.forward`) and the HF path (`convert.to_hf.convert_checkpoint`) both
   call `convert.checkpoint_reader.read_tensors`, which owns the name↔tensor association and
   the declaration-order stream walk over the checkpoint's pickle records. A misassignment or
   a stream-order error there is **common-mode**, not independent: both paths would read the
   same wrong tensor under the same name and agree perfectly while both being wrong. This is
   the one piece of "independence" this plan does not actually have — the two paths are
   independently *derived* downstream of the reader, not independently *reading the
   checkpoint*. It is anchored only by the coarser CE test (Task 2's held-out cross-entropy
   check, floor ≈0.22 nats) and by `test_checkpoint_reader.py`'s own ordering tests, not by
   the parity gate, which cannot see it by construction.

### The standing skip-guard gap, partially addressed

Task 2's review noted the decisive tests are all `skipif`-guarded on `artifacts/` (gitignored),
so a CI run can report "N passed" while nothing load-bearing executed. The synthetic
gamma-mapping test above is deliberately **not** guarded — it needs no real artifacts and
runs every time — but every other test in `test_numpy_parity.py` still needs the real
checkpoint, tokenizer, converted `artifacts/hf/`, and `val_ids.npy`, and is still skip-guarded
on them. The gap is not closed, just narrowed by one genuinely-unconditional test.

Test suite: **151 passed** (146 pre-existing + 5 new in `test_numpy_parity.py`), 0 skipped, 0
failed on this machine (all `artifacts/` fixtures present); the synthetic gamma test alone
would still pass, unconditionally, on a machine with none of them.

## `feat/numpy-parity` — pre-merge whole-branch review fix wave (2026-08-12)

A final review before merge (verdict: ready to merge, with fixes) found seven small items —
none changing a measured number, several requiring one. No training re-run;
`artifacts/checkpoints/` and `artifacts/hf/` were never touched.

- **The parity window was widened from 64 to 256 tokens** — the model's full
  `max_position_embeddings`. At 64, positions 64-255 were covered only by the CE check's
  0.22-nat floor; a position-dependent defect (e.g. a RoPE angle that drifts with sequence
  length) could have hidden there. Re-measured at 256 tokens across seven seeds: max_abs
  ranges ~8.3e-6 to ~1.6e-5, max_rel ~3.0e-4 to ~5.6e-4, correlation indistinguishable from 1
  (1 - corr ~1.2 to 1.3e-13) — still ~60-120x inside the gate's tolerances. The whole file
  still runs in ~5.8s.
- **Every measured figure in `test_numpy_parity.py`'s docstrings now matches what the
  committed test configuration actually produces** — the previous headline table quoted a
  256-token measurement (8.46e-6) that the committed 64-token test could not itself produce
  (its own number was 5.26e-6/2.13e-4). In a branch whose thesis is "measured, not assumed,"
  the number a future debugger compares a failure against has to be reproducible by running
  the test, not a number from a related but different configuration.
- **The independence claim was one dependency too strong.** Both the NumPy path and the HF
  path call `convert.checkpoint_reader.read_tensors`, which owns the name↔tensor association
  and the declaration-order stream walk — a bug there would be common-mode, producing two
  identically-wrong paths that agree perfectly. `docs/ttml-forward-reference.md` was already
  honest about this; CLAUDE.md's "What this gate still cannot see" list and the plan's
  "different routes" framing were not, so both now name `checkpoint_reader` as a shared
  dependency explicitly (see item 5 in the list above and the plan's independence section).
- **The Global Constraint "`convert/` must NOT import `ttnn` or `ttml`" had no test for two
  of its four target modules.** `convert.checkpoint_reader`, `convert.tokenizer`, and
  `scripts/backfill_checkpoint_headers.py` all had the subprocess-probe test; `convert.ttml_forward`
  and `convert.to_hf` did not, despite `ttml_forward.py`'s own docstring claiming (in a
  garbled sentence that conceded as much on close reading) that something checked it.
  Added `test_ttml_forward.py::test_ttml_forward_module_imports_no_tenstorrent` and
  `test_to_hf.py::test_convert_to_hf_module_imports_no_tenstorrent`, same pattern, and fixed
  the docstring.
- **Added a fourth not-hollow proof: the epsilon-placement probe.** Task 1 found epsilon
  moved outside the sqrt the *one* perturbation invisible to the CE check (Δ = -0.0002 nats).
  `test_parity_gate_is_not_hollow_it_catches_epsilon_moved_outside_the_sqrt` monkeypatches
  `rms_norm` for one `forward()` call (no reconversion needed) and measures max_abs = 0.0370
  — 37x over the 1e-3 budget. Notably, correlation stays at 0.9999985, *above*
  `MIN_CORRELATION`'s 0.9999 floor: this defect is loud in absolute/relative terms while
  leaving the logit *shape* almost undisturbed, unlike the RoPE bug (corr ~0.93). Documented
  in `convert/ttml_forward.py` that `RMS_NORM_EPS` being a plain hardcoded module constant
  (not threaded from the header) is what makes this probe constructible — a future "read it
  from the header" cleanup would silently remove this coverage.
- **Strengthened the norm-swap blindness test with a byte-identical hash check.** The
  previous version passed identically whether its monkeypatch fired or silently didn't — no
  non-vacuity guard. Now converts both the unpatched and norm-swapped checkpoints into
  separate throwaway directories and asserts their `model.safetensors` files are
  SHA-256-identical (measured: both hash to `3a85bb08e1d2...490462200d`) — proof that *no*
  instrument could see the swap, not just that this one didn't. All three logit metrics
  (`max_abs`, `max_rel`, `corr`) are now asserted, making the "same tolerance" comment
  literally true.
- **Two documentation corrections:** renamed `attention()`'s local `embedding_dim` (q's
  *out-features*, not the model's embedding dim — numerically equal only because
  `head_dim == hidden/num_heads` on this architecture) to `q_out_features`; and
  `docs/ttml-forward-reference.md` §10's summary row and closing line no longer read as if
  the NumPy-vs-HF numerical tolerance were still an open problem — Task 3 resolved it at
  ~5e-6 to ~1.6e-5, two orders of magnitude inside 1e-3. Q1's own body was already correctly
  scoped to the NumPy-vs-*device* comparison and did not need to change.

Test suite: **154 passed** (151 pre-existing + 3 new: the epsilon-probe test and the two
import-purity tests), 0 skipped, 0 failed.

## `feat/packaging` Task 1 — repairing the HF artifact before publication (2026-08-12)

**Why this task exists.** Publication (Task 2+) is gated on the artifact being clean —
three defects, all cheap to fix now and expensive after weights are public, were found
during packaging-plan review of `artifacts/hf/`.

**Fix 1 — `generation_config.json` was missing entirely.** `transformers` logs its absence
and falls back to `config.json`'s token ids. `convert_checkpoint` now builds one via
`transformers.GenerationConfig(bos_token_id=..., eos_token_id=..., pad_token_id=...)` and
calls `.save_pretrained(out_dir)` — using the library's own class rather than a hand-rolled
dict so the on-disk shape matches whatever this environment's `transformers` considers
standard. The three ids are read from `config` (the exact dict `build_config` returned,
already written to `config.json`), not re-derived from `_BOS_TOKEN_ID` et al. directly, so
the two files are structurally incapable of disagreeing.

**Fix 2 — `tokenizer_config.json` declared the wrong class.** `convert/tokenizer.py` exports
via `PreTrainedTokenizerFast.save_pretrained()`, which writes `tokenizer_class:
"PreTrainedTokenizer"` — `transformers` strips the `Fast` suffix on save, an upstream quirk,
not a bug in this project's export code. The tokenizer actually loads as
`PreTrainedTokenizerFast`. Corrected **on the copy in `out_dir` only**, after
`convert_checkpoint`'s existing `shutil.copy2` — `artifacts/tokenizer/` is a separate
artifact on its own publication schedule, and patching it there would invalidate its own
tests. Verified the source is untouched: `artifacts/tokenizer/tokenizer_config.json` still
reads `tokenizer_class: "PreTrainedTokenizer"` after conversion.

**Fix 3 — no guard tied `max_position_embeddings` to the checkpoint's trained sequence
length.** The real trap: `tokenizer_config.json` advertises `model_max_length:
1000000000000000019884624838656` (transformers' "no limit" sentinel), so a serving stack
that derives `max_model_len` from the tokenizer instead of `config.json` would silently
accept ~4k-token contexts from a model trained to a 256-token window — degraded output, no
error (`scripts/chat.py` already carried a comment about exactly this). `build_config`
already derives `max_position_embeddings` from `header["seq_len"]`, so in normal operation
the two can't disagree; the new check in `convert_checkpoint` raises `ValueError` (not a
bare `assert` — see the project's global guard convention) if they ever do, so a future
change to `build_config` that breaks that derivation fails loudly at conversion time rather
than silently at serving time. `test_convert_checkpoint_raises_if_max_position_embeddings_disagrees_with_header`
proves the check is reachable by monkeypatching `build_config` to tamper with its own
output.

**The duplicate-embedding question (plan Task 1 Step 3) — settled empirically, not
re-litigated.** `model.safetensors` was 68.6 MB for a 44 MB model because
`lm_head.weight` duplicated `embed_tokens.weight` under `tie_word_embeddings: true`. Measured
directly before implementing (see `.superpowers/sdd/2026-08-12-packaging/progress.md`):
dropping `lm_head.weight` takes the file from 57 tensors / 68,632,400 B to 56 tensors /
44,056,336 B (36% smaller), `AutoModelForCausalLM.from_pretrained` still loads with **no**
warnings, `torch.equal(embed_tokens.weight, lm_head.weight)` is still `True` after load
(`transformers` reconstructs `lm_head` from the tied embedding), and logits are
bit-identical (max diff 0.0). Implemented as unconditional-on-tying: the tied-embedding
branch in `convert_checkpoint`'s tensor-assembly loop now writes only
`model.embed_tokens.weight`; the untied path (`tok_emb` → `embed_tokens`, `fc` → `lm_head`,
two genuinely distinct tensors) is unchanged. The completeness post-condition's expected-key
count is now conditional on `weight_tying` — `9 × num_hidden_layers + 2` (embed_tokens,
norm) when tied, `+ 3` (adding `lm_head`) when untied — rather than a single hardcoded `+ 3`
that would have made a correct tied conversion fail its own completeness check.

**Verification.** Full suite: **164 passed** (154 pre-existing + 10 new in
`tests/test_to_hf.py`), 0 failed. `tests/test_numpy_parity.py` (the gate that actually proves
numerical correctness, independent of everything else in this task) still passes after
regeneration. Regenerated `artifacts/hf/` via `scripts/convert_checkpoint.py`:
`model.safetensors` 44,056,304 bytes (56 tensors, no `lm_head.weight`), new
`generation_config.json` (`{bos,eos,pad}_token_id` = 1/2/3), `tokenizer_config.json`'s
`tokenizer_class` now `PreTrainedTokenizerFast`, `config.json`'s `max_position_embeddings`
still 256. `scripts/chat.py` smoke-tested against the regenerated artifact: loads with no
warnings, reports `context 256`, generates coherent completions.

## `feat/real-training` — the multi-epoch run: gammas fixed, model measurably better (2026-08-12)

Three tasks. Task 1 fixed the frozen-gamma bug found in `feat/numpy-parity`'s postmortem
(`stochastic_rounding: true`, `train/configs/nanollama3_bpe_v2.yaml`). Task 2 added periodic
real validation (`--val-every`) and an unconditional startup warning when
`stochastic_rounding` is disabled. Task 3 ran the real thing. **Fix round 1 (below) corrects
two overstated interpretations an independent review caught — see
`task-3-review.md` for the full evidence; every underlying measured number was reproduced
exactly and none of them changed.**

**Before the run:** disk was at 98% / 93 GB free (down from the ~140 GB the brief cited at
dispatch). First checkpoint measured (not assumed) at 132,186,302 bytes; `AdamWFullPrecision`
was never needed (Task 1's fix keeps the format unchanged), so all 11 checkpoints from this
run are that same size — 1.454 GB total, in line with the brief's ~1.3 GB estimate.

**The run:** `python train/run.py --config train/configs/nanollama3_bpe_v2.yaml --steps
21034 --save-every 2000 --val-every 1000 --batch-size 64 --checkpoint-dir
artifacts/checkpoints-v2` — 21,034 steps (3.000036 epochs over the 114.9M-token training
split), one p300c, 47m13s wall clock, ~7.42 steps/s. `stochastic_rounding: True` confirmed
at startup before trusting the run; the step-2000 checkpoint's gammas were checked for
degeneracy (sd range 2.02e-2..8.21e-2, all nonzero) before letting the remaining ~43 minutes
proceed. **`artifacts/checkpoints/` was never touched by this task** (still `2026-08-11
17:57`, unchanged). **`artifacts/hf/` was also not written by this task**, but its files do
carry today's date (08:21) — that's Plan 6 Task 1's own deliberate regeneration, an hour
before this run started, not something Task 3 did.

**The curve:** train loss 10.6875 → 1.375. Validation fell steeply for ~8000 steps
(2.1969 → 1.5695) then flattened for the remaining 13,000 steps (62% of the run) into a
1.46–1.59-nat band (one outlier at step 9000, 1.59375; excluding it, the rest sit in
1.45–1.53) while train loss kept falling — a mild overfitting signature, but not a clean
"turn": the best val value (1.4563 @ step 17,000) and the final one (1.4602 @ step 21,034)
differ by only 0.004 nats, well inside noise. Full curve (22 points) in
`artifacts/checkpoints-v2/val_losses.jsonl` and `.superpowers/sdd/2026-08-12-real-training-run/task-3-report.md`.
Read plainly: this corpus/architecture pair has largely exhausted what steps 8000–21,034 had
left to teach it about held-out loss — evidence for Plan 8's dataset-blend rationale, not for
training longer on the same mix.

**Paired comparison (the number that matters):** `convert/ttml_forward.py`'s pure-NumPy
forward pass, 32 seed-0 256-token windows, baseline (`nanollama3_step00003000.pkl`) vs. new
final checkpoint. Baseline mean CE 1.8733 (sd 0.3242, reproducing the brief's cited
1.8781/0.315); new mean CE 1.4228 (sd 0.2908). **Paired diff (baseline − new): +0.4505 nats,
sd of the paired differences 0.0878, SE 0.0155 — every one of 32 windows favors the new
checkpoint.** The two models' per-window losses correlate at r = 0.9651 (hard windows are
hard for both), which is *why* the paired sd (0.0878) is the right yardstick here and not
either model's own unpaired sd (~0.30–0.32) — comparing the paired difference against an
unpaired sd is exactly the mistake this brief warned about, and an earlier draft of this
entry made it (0.27 does not exceed 0.315). Measured correctly: even the **smallest**
per-window improvement (0.2709 nats) is **3.1 paired sds** above zero, and the mean is
**29 SE** from zero (0.4505 / 0.0155). Not noise, by a wide margin.

**Norm-swap ablation, re-measured — two swaps, and the result is real but far below what
this project's tests can detect.** `docs/model-development-troubleshooting.md`'s "+0.0000 ←
blind spot" row and the pinned `test_parity_gate_is_blind_to_a_norm_swap_on_the_real_checkpoint`
both refer to the **canonical** swap — block 0 ↔ block 1 `attention_norm`/`input_layernorm`
gammas, confirmed here still exactly 0.000000 on the baseline. On the new checkpoint that
canonical swap now costs **+0.00652 nats** (sd 0.0066, t = 5.6, 25/32 windows worse). A
second, gentler within-layer swap (block 3's `attention_norm` ↔ `mlp_norm`, the one this
task originally measured) costs **+0.0018 nats** (sd 0.0040, t = 2.5, only 22/32 windows
worse — 10 of 32 actually get *better*, an inconsistent sign that a small sample has a real
chance of reading backwards). **Neither swap is anywhere close to something this project's
loss-based checks would actually catch**: even the larger, canonical effect (0.0065) is ~45×
below either model's per-window sd (~0.29–0.32) and ~31× below the project's own 0.2-nat
detection floor; the smaller swap is ~163× and ~113× below those same floors respectively.
Correctly stated, the Plan-4 blind spot is closed only in the narrow sense that the number is
no longer *identically* zero — **a norm mis-mapping of this kind still slips past every
loss-based gate this project ships**, at the sample sizes those gates actually use (8–32
windows). This ablation alone remains a weak-to-useless instrument for this error class; the
structural/permutation tests (`test_hf_mapping.py`, `test_numpy_parity.py`'s per-destination
gamma checks) stay the actually-reliable defense. (The companion HF-parity-gate figure for
the canonical swap was not re-measured on this checkpoint; the previously-cited `5.86e-6` is
inherited from the plan and does not match the parity test's own docstring number for the
same swap, so it is dropped here rather than repeated unverified.)

**Generated samples, same prompt (`"Once upon a time, there was a little"`), verbatim, not
cherry-picked, unseeded (`do_sample=True, temperature=0.8, top_p=0.95`, no seed — not
reproducible by construction):**

> Baseline (`artifacts/hf`, step 3000): Once upon a time, there was a little girl named Lucy.
> Lucy loved to play with her toys. One day, Lucy saw a big, thick, pretty toy in the box.
> Lucy wanted to play with the toy, so she went to the box and pushed it with her hands. The
> toy made a loud noise and stopped working. Lucy

> New (step 21034), sampled from `artifacts/hf-v2-scratch/` — a scratch conversion made only
> for this comparison, which has **not** been through the parity gate (that gate is pinned to
> the baseline checkpoint): Once upon a time, there was a little boy named Tim. Tim loved to
> play with his toys. One day, Tim saw a big, high chair in the store. He wanted to ride the
> chair, but it was too high for him. Tim saw a tall man named Bob. He asked, "Bob, can you
> help me get

**Visibly better, or only numerically better?** Mostly the latter. Both samples are fluent,
loop-free, single-character TinyStories prose — the new one has a slightly clearer causal
chain on this one draw, but it is a difference of degree, not a qualitative leap. The ~24%
relative reduction in held-out cross-entropy (every window improved) is real, repeatable, and
verified straight from the `.pkl` checkpoints (unaffected by the sample's own unverified
conversion); the prose improvement is real but easy to miss without the paired numbers. Both
are honest findings, not a contradiction.

Test suite: **176 passed, 0 skipped** on this machine (`test_checkpoint_gammas_are_not_degenerate[checkpoints-v2]`
stops skipping once `artifacts/checkpoints-v2/` exists locally — no test was added by this
task, so a fresh clone without `artifacts/` will still show skips). Full detail:
`.superpowers/sdd/2026-08-12-real-training-run/task-3-report.md`.

## `feat/corpus-assembly` — nine sources, a token budget, and a manifest that has to be true (2026-08-13)

The corpus stopped being "TinyStories" and became a blend: nine licence-audited sources
mixed to a 400M-token budget, with a provenance manifest whose entire purpose is to make
*"what was this model trained on"* exactly answerable. Eight tasks: broaden `spine`, strip
residual Gutenberg front matter, measure and settle the shares, blend, retrain the
tokenizer, re-measure and generate the licensing document, freeze an evaluation prompt set,
triage the minors.

**The shares were settled twice, against two different tokenizers.** `scripts/measure_corpus.py`
is a gate, not a report: it counts what each source can actually supply and exits non-zero
when a slice cannot reach its target share within the upsample cap. The first settle moved
`flavour` from 2.00% to its arithmetic ceiling (0.5% — 2.00% needed 12.8x against a 4x cap,
i.e. it was impossible, not merely tight) and gave the freed 1.5 points to `spine`. Then
Task 5 retrained the tokenizer on the blend, which changed what a "token" *is*: measured
availability fell 6–24% for every domain except tinystories (−0.5%, because the old
vocabulary was tinystories-specialised to begin with). That pushed `procedural` over the 4x
working limit, so Task 6 re-settled — 13% → 12% for `procedural`, the point to `tinystories`,
two `upsample` factors raised. **Lesson: a token budget is denominated in a unit the
tokenizer defines. Retraining the tokenizer silently re-denominates every measurement taken
before it.**

**The circularity is real and has to be cut.** tokenizer → availability → shares → blend →
tokenizer. It does not converge on its own; whichever arrow you cut, the tokenizer ends up
one revision behind the corpus it will be used on. We cut it after Task 6 and wrote the
consequence down (`docs/corpus_blend.md`) rather than chasing it. Chasing it is an infinite
loop that produces a new "one revision behind" statement each time round.

### The two content-loss regex bugs — the reusable lesson

Both were the same mistake, found in consecutive review rounds, in the same pattern
(`_FRONT_MATTER` in `scripts/prepare_corpus.py`, which strips Project Gutenberg packaging
from the head of each document).

1. **Blanket `re.IGNORECASE` makes `[A-Z]` match lowercase.** `produced\s+by\s+[A-Z]` under
   `IGNORECASE` matches `produced by nature`, so any word-wrapped prose line beginning
   "produced by …" was classified as a producer credit and deleted. Not hypothetical: 12
   real prose lines were stripped from `poetry.txt`, and in 12 cases that line WAS the whole
   document, so the document vanished. Fixed with a scoped-flag group, `(?-i:...)`.
2. **The scoped fix was scoped too narrowly.** `(?-i:[Pp]roduced\s+by\s+[A-Z])` turned the
   flag off but kept `[Pp]` matching either case, so a lowercase "produced by" still matched
   whenever the NEXT word was capitalised — which 19th-century prose does constantly
   ("produced by Nature herself", "produced by God's providence"). Same failure mode, now
   gated on the next word's case instead of closed. Fixed by requiring the literal capital:
   `(?-i:Produced\s+by\s+[A-Z])`. A genuine PG credit is always line-initial and capitalised.

Carry these forward:

* **`re.IGNORECASE` is not scoped to the literals you were thinking about.** It applies to
  every character class in the pattern, including the `[A-Z]` you wrote precisely *because*
  you wanted case to matter. If one alternative in a case-insensitive pattern needs case
  sensitivity, use `(?-i:...)` — and note that it scopes the **entire group**, not just the
  class next to it. The round-3 comment claimed it covered "only `[A-Z]`", which is wrong
  about the regex engine, and the wrong comment is what let round 4 exist.
* **A deletion rule needs a measured blast radius, not an argument.** Both bugs were found
  by counting documents before and after, not by reading the pattern. `poetry`'s kept-doc
  count going 3,085,102 → 3,085,114 is what proved bug 1 was real, and every source's count
  being unchanged is what proved bug 2 had not yet reached the shipped corpus.
* **A pattern that eats real prose is far worse than one that leaves packaging behind.**
  Front-matter stripping is asymmetric: leftover "Produced by David Price" costs a few
  tokens; a deleted document is gone and nothing downstream can tell.
* **Regenerate the artifact after fixing a content bug.** Both fixes rebuilt
  `artifacts/corpus/*.txt` from the untouched `artifacts/raw/` sources. The raw copies exist
  for exactly this.

### Pre-merge whole-branch review fix wave

Nine findings; the artifact did not match what the manifest claimed.

* **C2 — the blend was not the blend the manifest described.** `_emit` sized its emission
  with a flat `tokens_per_word=1.3` while `plan_blend` gated on tokenizer-MEASURED
  availability. Real tokens/word runs 1.194 (`tinystories`) to 1.559 (`wikipedia_simple`)
  across the nine — a 30% spread — so the emitter over-emitted for eight of them by exactly
  `real_ratio / 1.3`. `wikipedia_simple` declared `upsample=1` and made 1.058 passes,
  duplicating ~5.8% of Simple Wikipedia undeclared; `procedural` made 4.034 passes against
  the 4x limit Task 6 moved a whole share point to stay under; the blend was 425,024,350 real
  tokens against a 400M budget with shares up to ~3 points off — while the manifest reported
  every `achieved_share` as exactly its target to 15 decimal places.

  Fixed by deriving each source's ratio as `available_tokens / file_word_count`. The
  satisfying part: with the ratio right, repetition collapses to `want / available`, which
  the planner's gate ALREADY holds at or below the declared `upsample` — the emitter
  structurally cannot exceed a source's declared repetition any more. **Lesson: when a gate
  and the thing it gates measure in different units, the gate is decorative.** The 1.3 was
  honest where it came from (`measure_corpus.py`'s no-tokenizer fallback, a deliberate
  slight over-estimate so the gate errs toward reporting more supply); it became a bug when
  it was copied into code that had a real measurement available.

  The manifest now records the tokenizer's own count of exactly the text emitted per source,
  chunked the way `measure_corpus.py` chunks it so the two numbers are comparable. Real
  total: **399,594,747 tokens (−0.101% of budget)**, every slice within 0.065 points.
* **C1 — the legacy path wrote TinyStories into `blend.txt`.** `build_tokenizer.py`
  defaulted `--corpus` to the blend, and when that file was absent fetched TinyStories and
  wrote it *into that path*. On a fresh clone the README sequence therefore produced a 512 MB
  TinyStories file named `blend.txt`, and every later run found it, skipped the fetch, and
  trained on TinyStories forever. **A corpus is just a text file; its name is the only claim
  anyone makes about its contents, so the name has to be defended in code.** The legacy path
  may now never create that name. `train/tokenization.py` still defaulted to
  `corpus.txt`, so the documented quickstart crashed at step 2 — both defaults now agree and
  a test holds them equal.
* **I3 — `:.0%` rendered `flavour`'s 0.5% share as `0%`.** In the GENERATED licensing
  document, whose banner promises it cannot go stale. Generation protects against drift, not
  against a format string. `train.corpus.format_share` keeps fractions.
* **I1/I2, I4** — five rationales still quoted the pre-retrain measurement, `spine` claimed
  an upsample computed at a share it no longer holds and a "largest drop of any slice" that
  belonged to `wikipedia_simple`, and `prepare_corpus.py` justified its email rule with
  "every source here is a pre-1929 public-domain text" — false, since `tinystories` is 2023
  GPT-generated and `wikipedia_simple` is a live encyclopedia. The rule is empirically
  harmless and was left alone; only the reason changed, to the measurement it rests on.
  **A false reason is worse than no reason: it tells the next maintainer the wrong thing is
  safe.**
* **I6 — "FROZEN" prompt set that wasn't.** Rewriting every prompt's `text` to garbage left
  the suite green; only ids, probes and count were pinned. Now digested over sorted
  `(id, text)`. **Pinning the labels of a fixture is not pinning the fixture**, and a prompt
  set whose ids are stable while its text drifts is worse than an unfrozen one, because the
  results still look comparable.
* **I7/I8/I9** — tests for `build_tokenizer.py` (it had none and the highest blast radius on
  the branch), the tokenizer-ordering note above, and this section.

Test suite: **419 passed, 1 skipped**.

## The corpus had no document boundaries at all (2026-08-14)

The prompt: `artifacts/corpus/blend.txt` contains **zero** document separators, while the old
TinyStories-only `corpus.txt` — the corpus the *published* model trained on — contains
**662,878**. Find where document identity is lost between the raw jsonl and the prepared
`.txt`, fix it at the right layer, rebuild every artifact, and set up a 2048-context run.

**Where it was lost: `scripts/prepare_corpus.py` wrote each document as `text + "\n\n"`.** A
document boundary was spelled exactly the way a paragraph break *inside* a document is
spelled, so nothing downstream could distinguish them. `train/tokenization.py` then finished
the job — it encodes the corpus one line at a time and drops the newline, so blank lines
contribute no tokens whatsoever. Zero `</s>` in the corpus, zero id 2 in the token arrays
(still verifiable: `artifacts/tokens/` and `artifacts/tokens-stratified/` are kept, and both
have zero in their first 20M tokens).

That is not a tidiness bug. A position-wise loss probe showed per-token loss flat from
position ~64 to 511, on books as much as on short items — with boundaries unmarked, distant
context genuinely *is* unpredictable, so the model was right to ignore it. The mid-generation
topic collapse in the samples is the same fact from the other side. **Lesson: a delimiter that
is indistinguishable from ordinary formatting is not a delimiter.** The old pipeline got this
right by accident, because TinyStories shipped an explicit `<|endoftext|>` line; the
nine-source rewrite dropped the idea along with the format.

* **The separator belongs in `prepare_corpus.py`, and nowhere else.** It is the only stage
  that can see a document at all: `fetch_corpus.py` writes one JSON object per document,
  `prepare_corpus.py` consumes them one at a time, and `blend_corpus.py` sees only
  concatenated text which it repeats and truncates. Putting the boundary anywhere later would
  have meant guessing at it.
* **`</s>` was already the right token, and this was checked rather than assumed.** Id 2, an
  *added* token (so byte-level BPE can neither split it nor absorb a neighbour),
  `special_tokens_map.json`'s `eos_token`, and already written as `eos_token_id` into
  `config.json` *and* `generation_config.json` by `convert/to_hf.py`. The serving path was
  waiting for a token the training data never contained.
* **The fix nearly introduced a worse bug.** `biglam/gutenberg-poetry-corpus` has one row per
  **line** of verse — 3,085,117 rows of ~7 words. A per-row separator would have fired an
  end-of-document token every seven words and, at poetry's 1% share, put roughly a *third* of
  every `</s>` in the blend inside that slice: a seven-word prior for "stop". Caught by asking
  what a "document" is for each source before writing any. `CorpusSource.rows_per_document`
  (64 for poetry, 1 elsewhere) makes it 48,205 documents and 6,002 separators instead.
  **Lesson: "one row = one document" is an assumption about the upstream dataset, not a
  property of jsonl.**
* **The truncated tail is closed deliberately.** `_emit` truncates each source's final pass at
  word level, mid-document. Left open, source A's half-sentence would run into source B's
  first document at each of the nine seams — the same defect, just rarer. The closing
  separator is counted in `emitted_words`, because that number is what the stratified split
  uses to locate each source's boundary in the finished corpus.
* **`train/tokenization.py`'s `add_special_tokens=False` comment was wrong in both halves.**
  It claimed the corpus already carried separators (true of the legacy path, false of this
  one) and that `True` would double them (false outright — this tokenizer's post-processor is
  a plain ByteLevel, so `True` injects nothing; measured). The flag stays `False` for the real
  reason: that function is called once per **line**, so a tokenizer that ever gained a
  template post-processor would wrap every line rather than every document. **A comment that
  survives the code it describes becomes a trap.**

**Verified empirically, not declared.** `blend.txt` holds **798,771** `</s>` lines against
zero before. `artifacts/tokens-v3/` holds 734,978 + 63,793 = **798,771** occurrences of id 2 —
equal to the token, so none were added, lost, or split. Decoded windows put the separator
exactly where documents end: a TinyStories story closing before "Once upon a time"; the last
line of an Oz book before the *next* book's title page; the Vatican City article's category
tail before "Velocity is a measure of how fast something moves".

**Shares did not need re-settling.** Availability rose by exactly **2 tokens per document**
for every source (the separator plus its newline), which can only loosen the scarcity gate.
The blend totals 399,508,203 tokens against the 400M budget (−0.123%), every slice within
0.083 points of target. The expected "+1.4% of tokens" did not happen either: the budget is
fixed, so the separators *displace* text rather than adding to the total. And ~800k, not
~5.5M — most of the document-heavy sources are used fractionally (`tinystories` 0.28x,
`wikipedia_simple` 0.88x), so most of their documents never enter the blend.

**2048 context.** Raised in `train/configs/model/tt-tnt-384.yaml` and `train/sizes.py`
together (the anti-drift test holds them equal). `tt-tnt-1024` deliberately stays at 512 — the
evidence for 2048 is a measurement on the 384 shape, and that size has never been trained.
That divergence is what forced `--seq-len` to default to the selected size's own
`max_sequence_length` instead of a fixed 512: `build_yaml_config` enforces
`seq_len == max_sequence_length`, so a fixed default had become a guaranteed error for the
default size. **Lesson: a constant shared by two things that are allowed to differ is a bug
waiting for the day they do.** No training was started.

Test suite: **511 passed, 1 skipped** for this change (500 before it). A concurrent,
still-untracked `tests/test_probe_context_use.py` — the position-wise loss probe whose
measurement motivated all of the above — adds 36 more, for **547 passed, 1 skipped** when the
whole `tests/` directory is collected.

**A registry rationale is an artifact too.** Re-measuring availability broke
`test_a_rationale_that_cites_availability_cites_the_CURRENT_availability`, exactly as designed:
seven rationales still quoted the pre-separator numbers. That gate has now caught stale prose
on three separate occasions, which is the argument for having it.

## The weight cache that lied (2026-08-14)

**Prompt:** "Design and implement a fix for a stale-cache bug, carried in this model's own
adapter." Hardware paused by the owner — source reading, unit tests, CPU-only work; any
device verification deferred with an exact command.

The bug, from `.superpowers/serve-tt-tnt-v3.md` F7: `tt_transformers` caches converted
weights at `model_cache/<repo_id>/<device>/tensor_cache_bfp8/` and decides to reuse them
with a bare existence check. tt-tnt was retrained, republished to the **same** repo id, and
served — the server came up clean, reported the correct `max_model_len: 2048`, and ran the
**previous** model's weights, logged as an ordinary warm start. Because that model could not
emit EOS by construction, the headline number would have been a confident "0% termination"
and would have read as a real regression in the very fix being tested. Caught only because
someone noticed a directory mtime predated the publish.

**Where it actually lives** — three facts, all now pinned by a canary test:

| what | where |
|---|---|
| cache key computed | `model_config.py:577` — `os.path.join("model_cache", HF_MODEL, self.device_name)` |
| the single funnel every weight path flows through | `model_config.py:3017` — `weight_cache_path(self, dtype)` |
| reuse-vs-reconvert decided | `ttnn/ttnn/operations/core.py:719` — `if not cache_path.exists() or not cache_path.is_file()` |

Only a *deserialisation* failure (`core.py:725`) ever triggers a re-conversion. Nothing
compares the cache against the weights it came from.

**The fix: content-address the cache key**, by appending one component to
`weight_cache_path` —
`…/tensor_cache_bfp8/src-rev-a3c85ec799fe/`. The fingerprint is the HF commit sha
(`hf_config._commit_hash`, which `transformers` stamps on the config at
`configuration_utils.py:812` and `ModelArgs.__init__` has already loaded by
`model_config.py:616`), falling back for local checkpoint directories to a sha256 over
`(name, size, mtime_ns)` of `config.json` and the weight files.

**Why not the alternatives**, all of which were on the table:
* *Validate before use.* To know a cached tensor is wrong you must produce the right one —
  i.e. pay the conversion the cache exists to avoid — unless you keep a side manifest, which
  is this fix with more moving parts. It also overwrites, so flipping back to the previous
  revision pays again. Fingerprinting keeps every revision warm.
* *Refuse a cache older than the weights (mtime).* There is no local file whose mtime means
  "publish time" — the source is a Hub repo id and HF blob mtimes are *download* times, whose
  order against the conversion is arbitrary within one session. Reading a timestamp is what
  the human had to do; it is not a thing to automate.
* *Disable the cache.* Correct and slow, so it gets switched back off the first busy
  afternoon, restoring the bug. **A fix that gets disabled is not a fix.**

**The failure mode was silence, so the fix must not introduce a different silence.** Every
state in which the guarantee does not hold is audible: a new fingerprint beside an existing
one WARNs and names the superseded revision (the log line whose absence made this invisible);
leftover un-fingerprinted `.tensorbin` files WARN that they are now dead (and are *not*
deleted — that is a human's call); an unavailable fingerprint or `TT_TNT_CACHE_FINGERPRINT=0`
WARNs that stale weights are possible again; and if `weight_cache_path` has moved or its
first two parameters are no longer `(self, dtype)`, the patch **declines to install** and
says so rather than crashing the serve or silently no-opping. Each fingerprinted directory
also gets a `tt_tnt_cache_source.json` stamp, so `ls` answers the question that cost an
afternoon.

**Local checkpoints need `mtime_ns`, not just size.** A retrain of the same architecture
writes byte-identical file *sizes* — a size-only digest would reproduce this exact bug. The
cost is a false miss (one redundant conversion, logged) after a re-download. False misses
cost minutes; false hits cost a published measurement. `test_local_checkpoint_retrain_…`
asserts the premise (`len(v1) == len(v2)`) before asserting the behaviour, so the test cannot
pass for the wrong reason.

**Tests run without tt-metal and without hardware** by installing a fake
`models.tt_transformers` into `sys.modules` before loading the adapter — so 18 of the 19 gates
always execute rather than skipping into vacuity. The 19th, the upstream-drift canary, reads
tt-metal's *source text* (never imports it, so no ttnn, no device) and asserts all three
anchors above still exist; it skips with a reason when the tree is absent. **Mutation-checked
three ways**: reverting the patch fails 13, keying on a constant fails 9, dropping `mtime_ns`
fails the retrain test specifically.

**F8 is deliberately not in the adapter.** `~/.cache/tt-kernel/bundles/` is consumed by
`tt-model serve` *before* the vLLM process exists: the stale bundle's `vllm_metadata.json`
supplies the launch command (`--max_model_len 512`) that starts the process that would import
this adapter. The adapter cannot reach backwards past its own `argv`. It belongs in tt-kernel
(`cli.py:1276 _serve_vllm` → `_ensure_vllm_pulled`), as a `--refresh` flag plus a revision
comparison against the Hub on serve.

Suite: **600 passed, 1 skipped** (581 + 19 before this change, baseline held).

## A behavioural metric, and the correction it needed (2026-08-14)

**The prompt.** "Build a behavioural quality metric... it is the binding constraint on the
project." We could measure loss (per-source, stratified), context use, termination and
genre-collapse rate — but the actual goal is qualitative prose, and that was assessed by a human
reading 15 greedy completions. Fifteen deterministic completions cannot separate an improvement
from noise and cannot be run in a loop. `scripts/score_behaviour.py` is the numeric version:
many sampled completions per frozen prompt, five signals, standard errors everywhere.

**Power comes from samples, not prompts.** `docs/evaluation_prompts.json` is digest-pinned for
cross-checkpoint comparability and stays frozen; 32 completions per prompt at T=0.8 give the
variance. The aggregate is the mean over the 15 *per-prompt* means with the SEM taken over
prompts — completions of one prompt are not independent observations of model behaviour, the
same "what is the exchangeable sampling unit" rule `probe_context_use.py` applies to windows.
Comparisons are **paired by prompt**, which removes between-prompt variance and is what makes 15
prompts enough to see anything at all.

**Every detector is calibrated in-run, not asserted.** The collapse markers were chosen by
measuring each candidate's rate per million words in `tinystories.txt` against the eight other
prepared corpora (nothing under ~45x lift is included), and every run re-measures the whole
detector on held-out corpus text at completion length: 48.5% sensitivity on real TinyStories,
0.0–1.6% on the other eight sources. So the reported collapse rate is a **lower bound**, usable
as a comparator, never as an absolute prevalence — the report says so in those words. The
register signal (per-source interpolated unigram+bigram LMs, deliberately simple enough to check
by hand) reports its own 9-way accuracy the same way: 99.9% on `tinystories`, and much lower on
the narrative trio `folklore`/`gutenberg_children`/`weird`, which is stated so nobody reads a
claim the model cannot support.

**The correction, which is the interesting part.** Built as a single union of eleven markers, the
collapse detector said v1 and v3 collapse at indistinguishable rates — contradicting the 9/15 vs
1/15 hand count. The per-marker breakdown said why: `once_upon_a_time` went 10.8% → **0.0%** and
`little_X_named` 8.3% → **0.4%**, while `one_day_comma` (14.8% → 12.5%) and `so_very_happy`
(7.9% → 10.2%) did not move. The union averaged a large real effect against a null one. Split
into **story-frame** vs **lexical-habit** collapse — by what each marker *is*, with the original
union retained beside them — and the frame signal reproduces the hand count. The split was made
*after* seeing the disagreement; the per-marker control table exists so that can be audited
rather than trusted.

**Where it still disagrees, and why we believe it.** The register signal says v1 and v3 write in
the same register (`nearest == tinystories` 0.525 for both, difference 0.000 ± 0.038). We believe
it: no prior finding actually claimed a register improvement (the hand count counted frame
phrases, which this metric agrees about); an independent detector — the hand-written lexical
marker list — says the same thing; and the register signal demonstrably has the dynamic range to
see a change (per-prompt values span −0.77 to +2.03 nats/word). **v3 stopped writing fairy tales
but did not stop using fairy-tale words.** That is a narrower win than "9/15 → 1/15" implies, and
it is the target for the next run.

**It can say "worse."** Verdicts are per-signal against a declared direction, and a difference
whose 95% interval spans zero is "no change", never a small improvement. Three of nine signals
moved the wrong way on this pair (repeat rate, longest repeated span, prompt engagement); none
crosses significance, and the report names them anyway under the verdict. The repetition
regression is if anything understated — v3's completions are *shorter* (it terminates), so it had
fewer chances to repeat and still repeated more.

**Power is capped by the prompt set, not the sample count.** Decomposing the paired SEM: for
story-frame collapse, within-prompt sampling noise is only 0.015 of the observed 0.042, so
doubling to 64 completions per prompt would buy ~3%. The 15 frozen prompts are the constraint.
The right way to buy power is a *second* frozen set with new ids, reported separately — never by
editing this one.

**Mutation-checked, eleven ways.** Reducing the collapse detector to its single most obvious
marker fails 8 tests; inverting the repeat rate fails 5; making termination check only the last
token (batched `generate` pads after `</s>`, so that reads 0%) fails 2; pooling completions
instead of aggregating over prompts fails 6; making the verdict unable to say "worse" fails 4.
Also verified: normalising the corpus reader's case/punctuation — which would silently disarm two
markers *and* misreport the detector's own accuracy — fails its test.

Suite: **667 passed, 1 skipped** (600 + 67 before this change, baseline held). CPU-only
throughout: no ttnn, no ttml, no device, nothing written under `artifacts/`.

## A second frozen prompt set, and an honest answer about what it buys (2026-08-14)

**The prompt.** "Build a SECOND frozen evaluation prompt set... power is capped by the PROMPT
count, not the sample count." The previous section ends by saying the right way to buy power is a
second frozen set with new ids, reported separately. This is that set:
`docs/evaluation_prompts_b.json`, 45 prompts, pinned by `tests/test_evaluation_prompts_b.py`.
`docs/evaluation_prompts.json` (set A) was not touched — its digest is unchanged, and a test in
set B's suite asserts that, so "set A moved" can never be a silent side effect of a set B edit.

**How the prompts were derived, and what that rules out.** From the corpus design — `train/corpus.py`'s
per-source rationales and `docs/corpus_blend.md` — and explicitly *not* from any model's observed
output. Not one prompt was chosen because a checkpoint failed on something like it. An instrument
reverse-engineered from the last bug is tuned to the last bug and cannot find the next one.
Two orthogonal axes, both readable off the file: the id prefix names the corpus **register** the
opening leans toward (`b-spine-*`, `b-proc-*`, `b-weird-*`, `b-folk-*`, `b-child-*`, `b-wiki-*`,
`b-poem-*`, `b-flav-*`, `b-tiny-*`, `b-null-*`), the `probe` field names the **behaviour** it
stresses (set A's seven tags plus `default-register`). Registers: spine 6, procedural 7, weird 4,
folklore 4, gutenberg_children 3, wikipedia_simple 4, poetry 2, flavour 5, tinystories 2, neutral 8.
Probes: target-voice 7, agentic 7, coherence 7, grounding 5, perpendicular 5, oracular 3, stutter 3,
default-register 8. `procedural` gets the largest non-neutral block because it is the slice that
measurably benefits most from long context (+0.130 at 3× SEM), so a context change should show
there first.

**The neutral block is the part set A could not do.** Eight openings (`b-null-*`) that lean toward
no slice at all, so the model's *default* register is measurable and not only its steerability.
Every set A prompt hands the model a register; that is why "v3 stopped writing fairy tales but kept
using fairy-tale words" took a separate detector to see. Checked rather than asserted: scoring each
prompt's own text under this repo's nine per-source register models, set B's 45 prompts spread over
all nine sources with no source above 22%, against set A's 15 which touch six sources with
tinystories at 33%. The neutral eight land on six different sources — dispersion is what
slice-neutrality looks like under a weak detector. The check is weak (8–14 words is little
evidence) and was deliberately **not** used to tune the prompts; tuning against it would be
overfitting an instrument to another instrument.

**The power it buys, stated conditionally, because the honest answer is conditional.**
On the same paired-SEM basis as the v1-vs-v3 comparison, story-frame collapse:

| | n | effect | SEM | min. detectable | \|t\| |
|---|---:|---:|---:|---:|---:|
| set A, as measured | 15 | −0.0833 | 0.0416 | **0.0814** | 2.01 |
| set B, if per-prompt effects have set A's spread (sd 0.161) | 45 | −0.0833 | 0.0240 | **0.0470** | 3.47 |
| set B, if it has the same *live* prompts as set A and 30 dead ones | 45 | −0.0278 | 0.0148 | 0.0290 | 1.88 |

**So the number is 0.047 — conditional on set B's prompts differing between checkpoints as much as
set A's did.** The third row is the warning, and it is not hypothetical: set A's whole −0.083 comes
from three prompts, 71% of it from two, and v3 sits at exactly 0.000 frame collapse on 12 of 15.
Adding prompts on which both models score zero shrinks the effect and the SEM at the same rate, so
|t| does not move — 45 prompts would settle nothing. What buys power is prompts on which the two
models *differ*, not prompts. Set B is if anything more register-dispersed than set A, so for this
one floor-limited signal the third row is a live possibility and the 0.047 should be read as the
optimistic end. Its own per-prompt table will say which world we are in, and that is the right time
to find out — a frozen set is frozen *before* it judges anything.

For the signals that are not floor-limited the √3 improvement is unconditional, because every
prompt varies: termination MDE 0.104 → 0.060, `nearest == tinystories` 0.075 → 0.043, prompt
engagement 0.069 → 0.040, 4-gram repeat 0.0098 → 0.0057. The general lesson is the one the metric
keeps teaching: against a model that has already eliminated a behaviour, no prompt count rescues a
signal at its floor. That needs a signal with headroom — which is what the lexical-habit split and
the register margin are for.

**The sets are never pooled.** `--prompt-set {a,b}` defaults to `a`, so every existing invocation
and every committed measurement stays reproducible — verified by re-running the committed v1-vs-v3
comparison and diffing: all nine paired numbers identical, only the prose that now names the set
changed. Set B's outputs carry `-setB` in the filename, the JSON records `prompt_set`, and
`--compare` *refuses* a cross-set pair rather than computing one (tested with two payloads that
share every prompt id, so the refusal is not an accident of disjoint ids). A JSON with no
`prompt_set` key is read as set A — a fact about this repo's history, not a guess, since set B did
not exist when those files were written.

**Mutation-checked, on the file, not in the abstract.** Changing one word of one prompt in
`docs/evaluation_prompts_b.json` ("measures" → "gauges" in `b-wiki-01`) fails 2 tests
(`test_prompt_text_is_frozen_not_just_the_ids`, `test_the_digest_does_not_depend_on_file_order`);
reverted, 19 pass. The digest also detects a prompt dropped, a prompt added, and two prompts' texts
*swapped* — the last is the case a digest over ids and texts separately would miss.

**One correction carried forward.** The old docstring line "power comes from more samples per
prompt, never from more prompts" was the design intent and is wrong; it is corrected in place
rather than left standing, with the decomposition that disproved it.

**No set B scores are reported here, deliberately.** A 2-sample × 24-token run against
`artifacts/hf-tt-tnt-v3` proved the plumbing end to end (45 prompts, report written, JSON written)
and its output went to a scratch directory, not to `docs/measurements/`. Interpreting a smoke test
as a finding is exactly what freezing a set beforehand is meant to prevent.

Suite: **697 passed, 1 skipped** (667 + 30 before this change, baseline held). CPU-only: no ttnn,
no ttml, no device touched, nothing written under `artifacts/`.

---

## The first run of the `1024` size (tt-tnt-1024a) — and what it says about register

**Prompt.** Train `1024` for the first time ever: seed 5489, 10,764 steps, batch 64, seq 512
(the size's own `max_sequence_length`), `artifacts/tokens-v3`, `nanollama3_bpe_v2.yaml`. The
brief was explicit that this was unexercised territory — "treat failure as a likely and
reportable outcome, not something to force past" — and that register, unmoved across four runs,
was the last open quality axis. Full report: `.superpowers/1024-first-run.md`.

**It ran, first time, with nothing tuned.** 123.0M parameters, 2h42m on one p300c at
**903 s/1000 steps** (the 40-step smoke run measured 873 and projected 2.61 h; actual 2h42m, the
3.4% gap being 22 validation passes and 11 checkpoint writes). Mesh auto-discovered as `(1,1)`;
`TT_VISIBLE_DEVICES` deliberately never exported. Memory was never close to binding.

**The smoke run paid for itself immediately.** Its 40-step checkpoint would not convert:
`config.json` said `intermediate_size: 1024` while the weights were 2816 wide.
`train/run.py` wrote the header's `intermediate_dim` as a **literal 1024** — the value ttml
*derives* for the 384 model, so the constant was invisibly correct for the only size ever
trained and silently wrong for this one. Training never read the field (the weights were always
right), but `convert/to_hf.py` copies it into `config.json`, so every 1024 checkpoint would have
produced an unloadable model. Fixed in `bd9a2d2` by moving the four ttml-C++-only header facts
into `ttml_cxx_header_fields(size)`, which derives from the size registry. The long run was
restarted 4 minutes in, so all 11 checkpoints are correct by construction.

**Loss: no capacity effect at the end.** Final validation 2.9281 vs v3's 2.9391 — a 0.011-nat
difference against a 0.1944 seed-only noise floor. 5.6x the parameters bought nothing
measurable at convergence. It *did* descend far faster early (−0.81 nats at step 1000, decaying
to −0.01 by the end): the signature of a **data-bound** loss, consistent with ct8's ~80M
binding-constraint estimate, which this run does not refute.

⚠️ **Those two losses are not on the same scale.** `evaluate()` windows at `cfg.seq_len`, so
v3's number is over 2048-token windows and 1024a's over 512 — a harder problem. 1024a matched v3
under a worse evaluation condition, but the size of any real advantage is unrecoverable from
these numbers.

**Register moved — the first thing in this project that has.** On set B, against v3:

| signal | delta | seed-only delta | ratio |
|---|---:|---:|---:|
| tinystories margin | −0.2613 | −0.0745 | **3.51x** |
| nearest source == tinystories | −0.1368 | −0.0368 | **3.72x** |

Both clear their paired CIs by a wide margin *and* are multiples of the seed-only floor.
Register is not immovable and is not purely a corpus-mix problem.

**The trap sprang exactly as the brief predicted.** The collapse-rate signals came back "better"
from the paired test while moving **1.01x and 1.05x** what the *seed alone* moves them
(−0.0549 vs −0.0542; −0.0590 vs −0.0563) — not near the noise floor, numerically identical to
it. Reported as not interpretable. `engagement` is the mirror-image error: a 2.99x ratio over a
tiny denominator that fails its own paired minimum-detectable (+0.0198 vs 0.0275). A finding has
to clear **both** gates.

⚠️ **The confound, not papered over.** This run changed capacity (22M → 123M) **and** context
(2048 → 512). It cannot attribute the register effect to capacity alone. The clean experiment is
a **384-at-512 run on `artifacts/tokens-v3` at seed 5489** — identical to v3 but for sequence
length. It does not exist, it costs about one v3-length run, and it is the obvious next step:
context is the cheaper variable to rule out. A seed replicate of 1024a would further harden the
claim, which currently rests on one run per arm.

**Multi-chip is still unexercised.** The whole reason `num_groups=4` was chosen is mesh widths
{1,2,4}; this run used `(1,1)`. What it delivers is the first *weights* those paths can be
tested with.

Suite: **735 passed, 1 skipped** (729 + 6 header-derivation regression tests, parametrized over
every registered size). Nothing deleted; nothing written under the protected v3/v4/v5, tokenizer,
tokens, or corpus paths. Not pushed.

---

## One evaluation entry point, and a designated current model (2026-08-15)

**Prompt.** "Build a single evaluation entry point... this is not a convenience wrapper, it is
an error-prevention tool. The project has excellent instruments but no way to run them as one
benchmark. Every comparison so far has been hand-assembled, and **every significant error made
in this project came from the joining, not the measuring**." The brief named the three: a
512-window loss compared against a 2048-window loss; a trajectory *average* reported where the
*endpoint* was the number, inflating an effect 9.7x; and deltas quoted against SAMPLING error
when RUN-TO-RUN error was the floor, which produced the "LR decay improves register" finding a
seed-only control later refuted.

`scripts/evaluate.py` is that entry point. It measures nothing itself — it invokes
`score_behaviour.py` / `probe_context_use.py` / `eval_per_source.py` as **subprocesses** and
joins their JSON. Subprocess rather than import on purpose: the argv *is* the provenance
record, so every report states the exact command that produced each number.

### The two guards, and why they are where they are

**The window guard.** The eval window defaults to a **constant** (`DEFAULT_WINDOW = 512`), never
to the model's own `max_position_embeddings` — that is the exact mechanism that made the window
ride along with the model. For training-time trajectories the window is *read*, not assumed:
`convert/to_hf.py` sets `max_position_embeddings = int(header["seq_len"])` and verifies it, and
`evaluate()` windows validation at that same `seq_len`, so the converted config's field IS the
units of `val_losses.jsonl`. `require_matched_window()` refuses a mismatch, names each model
beside its own window, and the CLI exits **2 before loading any model** — the refusal has to be
cheap to hit. `--skip-trajectory` is the only way past it, and the report records the refusal
and its reason rather than leaving a blank.

**The floor labelling.** `FLOOR_RATIO_MIN = 1.2`, and a delta at or below it is
`NOT INTERPRETABLE` *whatever its confidence interval says*. The threshold is not a round number
chosen for comfort — it is where this project's own history puts it: the refuted LR-decay
register delta sat at **1.03x**, and the 1024 run's two collapse signals came back "better" from
the paired test at **1.01x** and **1.05x**. The mirror-image error has its own label:
`below paired detection`, for a large ratio over a tiny denominator that cannot clear its own
minimum-detectable difference (the 1024 run's engagement, 2.99x but +0.0198 against a 0.0275
MDE). **Both gates must pass.**

### The floor is derived, and refuses to be invented

Derived at runtime the way `render_licensing.py` derives its table from the registry:
per-signal behavioural floors from `docs/measurements/behaviour-tt-tnt-v3-vs-tt-tnt-v5-setB.json`
(committed), and the loss floor from `artifacts/checkpoints-tt-tnt-{v3,v5}/val_losses.jsonl`.
Reproduced exactly: **sd 0.1944**, mean 0.0413, **8/22 negative**; behavioural
`tinystories_margin` 0.0745, `register_tinystories_share` 0.0368.

⚠️ **`artifacts/` is not committed** (`artifacts/.gitignore` is `*`), so a fresh clone has no
loss floor. Hence `--refresh-floor`, which renders the committable
`docs/measurements/seed-noise-floor.json` from the raw sources. Runs prefer live derivation,
fall back to the snapshot, and record which they used. With neither, **no ratios are printed at
all** and the report says why — including no "no verdict" summary quoting the threshold, because
that would put a number in a report that nothing in it was measured against.

**The loss floor is the sd, not the mean.** The seed control's *mean* paired delta is +0.041 and
its sign wanders (8/22), so the spread is what a candidate has to beat. The report prints the
floor's own sign split beside the candidate's, which is what makes 22/22 legible.

### The sign test, preferred and not exclusive

For trajectories the report leads with the exact two-sided sign test (`math.comb`, no new deps):
capacity is 22/22 negative, p = 4.8e-7, against a floor that changes sign at 8/22. It also
prints mean-vs-sd, and says why both: the sign test says *how consistently* the difference
pointed one way, mean-vs-sd *how large* it was. And it prints the **endpoint** and the
**trajectory average** as separate rows, labelling the endpoint the headline — the 9.7x mistake
in one table row.

⚠️ One caveat kept in view rather than hidden: the seed floor was measured at a **2048** window
and is applied to **512**-window deltas. That is the same floor the committed 384s512 analysis
used, so the ratio is still printed — with a ⚠️ note saying it is a floor borrowed across
windows.

### The ad-hoc escape valve

`--try "text"` / `--try-file`. Generates at greedy/0.8/1.0 and writes to
`scratch/adhoc-prompts/ADHOC-<utc>-<slug>.md` — **outside `docs/`, outside git** (`scratch/`
added to `.gitignore`), banner-marked top and bottom. `assert_scratch_path()` refuses anything
else, including the frozen prompt JSONs, `docs/measurements/`, a `behaviour-*` filename even
inside scratch, a `..` escape, and a same-prefix sibling (`relative_to`, not `startswith` —
`adhoc-promptsEVIL` is not inside `adhoc-prompts`). `--help` and the file header both say a
genuinely diagnostic prompt gets promoted into a **new** set with **new ids** in a deliberate
commit, never by editing an existing one.

`--out-dir` moves a whole run elsewhere, so a 2-sample plumbing check cannot be mistaken for a
finding — the convention this project already followed by hand. `assert_writable_out_dir()`
refuses anything under `artifacts/`.

### The designation

`docs/current_model.json` designates **`artifacts/hf-tt-tnt-1024a`**, with `reason`,
`evidence` (six paths), and `qualification` all **required** by `load_designation()` — a bare
"the current model is X" is how a designation goes stale unnoticed. The reason is the matched-
window result: −0.2994 nats on average against the 384s512 control, negative at 22/22
checkpoints. The qualification is honest about what that does not mean: 1024a is trained at a
512 context while v3 is at 2048, so it serves a quarter of the context, its loss advantage is
established **only at 512**, it is 123.0M parameters against v3's 22M, and its register gain
over v3 was substantially **context, not capacity** (the clean capacity leg moves the
tinystories margin 1.03x the floor — a null by the standing rule). All five checkpoints are
listed as `candidates` with one line each, so a reader can see what was not chosen. Nothing in
the repo writes this file.

### Mutation-checked, on the source, not in the abstract

Ten mutations applied to `scripts/evaluate.py`, suite re-run, then reverted:

| mutation | result |
|---|---|
| window refusal never raises | **7 failed** |
| refusal says "different windows" without saying which is which | **1 failed** |
| default window = the model's own context (the original bug) | **2 failed** |
| floor gate removed; CI verdict alone decides | **4 failed** |
| `FLOOR_RATIO_MIN` 1.2 → 1.0 (lets the refuted 1.03x through) | **3 failed** |
| boundary made exclusive (`<` for `<=`) | **1 failed** |
| missing floor treated as passing rather than unknown | **1 failed** |
| second gate removed (ratio alone decides) | **1 failed** |
| floor hardcoded to today's numbers instead of derived | **1 failed** |
| loss floor uses the mean instead of the spread | **2 failed** |
| ratio (or seed-floor) column dropped from the table | **1 failed** each |
| scratch-path guard removed | **4 failed** |
| trajectory report loses the endpoint/average distinction | **1 failed** |

Two mutations survived the first pass and the *tests* were fixed, not the mutations excused:
the "names both windows" test passed against a vague message because the message also quotes
the historical 2048/512 precedent — retested with 777/333 and a proximity regex requiring each
window to sit beside its own model's name; and "ratio column dropped" passed because the
verdict bullets further down still printed ratios — retested per table row.

### Tests, and what they do not require

`tests/test_evaluate.py`, **72 tests, no model needed**: a "converted model" here is a directory
containing `config.json`, which is exactly what `read_model_facts` reads. The two tests that
genuinely need this machine's checkpoints (`artifacts/` is gitignored) **skip with an explicit
reason** — verified by pointing them at nonexistent paths and confirming they skip rather than
pass vacuously.

Suite: **808 passed, 1 skipped** (736 + 72, baseline held). CPU only, no device touched, no
lease held. Nothing deleted; nothing written under `artifacts/`; neither frozen prompt JSON
modified. Smoke runs (2 samples × 16 tokens, 4 windows) went to `scratch/`, not
`docs/measurements/` — and reproduced the committed capacity result end to end: endpoint
−0.2656 at 1.37x the floor, sign 22/22, matched-window pooled loss delta −0.2913.

## `perf/attention-mask` (2026-08-16) — a 1.41x training speedup for free

**The prompt:** land a measured ~1.4x training speedup, from an opportunity that had already
been identified and verified: `ttml.common.trainer.train()` always passes an explicit causal
mask, which forces `AttentionMaskType::Arbitrary` in the SDPA program factory — roughly 2x the
attention FLOPs, with load balancing disabled. Mid-task the constraint changed: **tt-metal may
not be edited**, which took the obvious fix (two lines of nanobind) off the table.

**The result:** 503.3 → **356.7 s/1000 steps** at the 384 shape (**1.41x**) and 890.0 → **776.7**
at 1024 (**1.15x**), measured through `train/run.py` itself. Full report, with all five
correctness checks, in `.superpowers/attention-mask-fix.md`; the upstream ask is tracked at
`docs/upstream-tt-metal-asks.md`.

**What made it possible without touching tt-metal:** ttml ships *two* Llamas. The C++
`CppLlama` cannot be handed a null mask from Python — `nb_models.cpp:330-337` binds the mask
as a non-optional `TensorPtr`, so `model(x, None)` is a `TypeError`, and there is no back door
(`ModuleBase`'s three `operator()` overloads all throw in the base, and only the two
non-optional ones are bound). But `ttml.models.llama.Llama` is a pure-Python implementation of
the same architecture, and its `forward` reaches
`ttml.ops.attention.scaled_dot_product_attention`, whose binding *is* declared
`nb::arg("mask") = std::nullopt`. `train/model.py` wraps it; `--model-impl {python,cpp}`
selects.

**The thing that had to be checked before believing any of it:** that the Python model is not a
slow reference implementation. It is not — with the mask still passed, C++ and Python cost
521.7 vs 521.9 s/1000 at 384 and 896.9 vs 893.6 at 1024, i.e. within 0.4%. The entire
difference is the mask, not the language. A 1.4x win paid for with a slower model would have
been no win, and this was measured rather than assumed.

**Three traps worth not rediscovering:**

- **`weight_tying` defaults *oppositely* in the two configs.** C++ `LlamaConfig`
  (`models/llama.hpp:35`) defaults to `Enabled`; Python `LlamaConfig` defaults to `Disabled`.
  Our YAMLs set no key, so every checkpoint this project has written is tied. Built with Python
  defaults the model silently gains 12.3M parameters at the 384 shape while `run.py`'s header
  keeps stamping `weight_tying: True` — a `config.json` claiming `tie_word_embeddings: true`
  over untied weights. Caught by the parameter count not matching (34,313,088 vs 22,025,088).
- **The two implementations name parameters differently, in exactly two path segments.** Root
  (`Llama/…` — the Python base names itself after its class — vs `llama/…`) and block
  (`blocks/0/…` from a `ModuleList` attribute + index vs `llama_block_0/…`). Everything below
  the block is already identical. Because *every* consumer here goes through
  `model.parameters()`, fixing it there (plus one `create_name("llama")`) keeps checkpoints, HF
  conversion, `convert/ttml_forward.py`, and `--resume` working with **no change anywhere
  else** — verified by loading `checkpoints-tt-tnt-v5` (model + optimizer) into the Python model
  and by converting a freshly-written one all the way to a HuggingFace directory.
- **The mask is dropped only when *verified* causal**, not whenever one is passed. It is pulled
  to host once and compared against `np.tril(np.ones(...))`, verdict cached by object identity
  (two slots, so a stale mask is not pinned on the device). The KV-cache path never drops it:
  there the mask is not square and `forward_kv` reads `mask.shape()[-1]` to size its cache
  slice.

**Correctness, five ways** (the speedup is meaningless without it): perturbing token *t* leaves
every logit before *t* bit-identical on both paths and changes positions ≥ *t* identically
(strictly causal, no leak); held-out cross-entropy on trained weights moves 4.1e-4 nats; against
the fp32 NumPy reference the unmasked path's mean error is *lower* than the masked path's
(0.015445 vs 0.015610); a 300-step same-init trajectory differs by at most one to two bf16 ULP;
and the end-to-end `run.py` curves at both shapes deviate by at most 0.119 nats against a
0.194-nat seed-noise floor.

Suite: **831 passed, 1 skipped** (808 baseline + 23 new in `tests/test_model.py`, all host-only).
tt-metal left untouched at `620793d898`. Nothing deleted; nothing written under `artifacts/`
(timing runs checkpointed to a scratch directory outside the repo).

## Four-chip DDP — a 3.98x wall-clock win, and the descriptor that silently hid it (2026-08-16)

Prompt: bring up multi-chip data-parallel training and produce a real 4-chip benchmark, with an
explicit warning that `synchronize_gradients` silently early-returns when the parallelism
context is uninitialised, so a run can be 4x faster and silently wrong. Full report and every
measurement: `.superpowers/ddp-bringup.md`.

**Result.** `--size 1024`, batch 64, seed 5489, 300 steps, s/1000: `[1,1]` **770.2** →
`[1,4]` **193.4** (**3.98x**) with the Python model; 892.9 → 223.2 (**4.00x**) with cpp. Both
`[1,1]` figures reproduce the recorded baselines (776.7 / 890.0) to under 1%. **`--model-impl`
and `--ddp` are independent levers** — 892.9 → 193.4 is 4.62x over this morning's start.

**The blocker was the mesh graph descriptor's *shape*, and it fails silently.** tt-metal's
`p300_x2_mesh_graph_descriptor.textproto` correctly declares the physical `[2,2]` wiring, but a
DDP-only run must open `[1,4]` (`ParallelismContext` TT_FATALs on a 2-D mesh unless two
parallelisms are enabled). That mismatch is **not rejected**: the mesh opens, the model trains at
full speed, step 1 completes, and step 2 hangs forever in the first gradient all-reduce.
`.superpowers/seqlen-ddp-investigation.md` predicted this conflict as the likely first failure
but expected a rejection — the reality is worse. Fix needs no tt-metal change: vendored
`train/configs/mesh/mesh-1x{2,4}.textproto` declaring the *logical* shape, exported via
`TT_MESH_GRAPH_DESC_PATH`. **Rule: `device_topology.dims` must equal the `MeshShape` opened.**

**Gradients provably synchronise — two independent proofs plus a negative control.** (1) After
DDP steps all four replicas are **bit-identical** (max `|replica0 - replica_i|` = **0.0** over
66 tensors); deliberately skipping the context init gives **2.44e-3** — so the instrument
detects the exact silent failure, and that divergent run produced a perfectly ordinary loss
curve. (2) `[1,4]` vs `[1,1]` val-loss at the same seed differs by at most **0.048** against the
**0.194**-nat noise floor, tracking at every point.

**Two traps worth not rediscovering:**
- **A TT hang stalls at the first *blocking read after* the fault, not the fault.** Ops are
  enqueued asynchronously, so `synchronize_gradients` returned fine and the host wedged one
  phase and one step later in `loss.to_numpy()`. Instrument with an explicit
  `ttnn.synchronize_device()` before believing any phase attribution.
- **The mesh-width rule does not apply to DDP.** `ModelSize.servable_mesh_widths` mirrors
  `tt_transformers`' *tensor-parallel serving* assertion; DDP replicates the model and shards
  only the batch, so head counts are irrelevant. The only divisibility DDP needs is
  `batch_size % N == 0`. `--size 384` is **not** excluded from 4-chip DDP.

**The one thing that does not work: saving a checkpoint under DDP** — and the repo's own note
saying it "appears already fixed" was refuted by hardware. The optimizer step re-marks each
parameter's topology from `Replicate` to `Shard(0)` while the data stays genuinely replicated,
so ttml's saver writes every replica concatenated on dim 0 (1,475,602,288 bytes vs 737,824,624;
tensors `(4,1,out,in)` not `(1,1,out,in)`). `Sharding.gather` is *not* the bug — it faithfully
honours wrong metadata. `train/checkpoint.py:assert_saveable_on_mesh` now refuses to write such a
file; produce publishable weights from `--ddp 1`, which costs no fidelity. This also means the
parity gate against a DDP checkpoint **remains outstanding** — there is no valid one to run.
*(Fixed the same day — see the next section. The `--ddp 1` advice and the outstanding parity
gate no longer apply.)*

## DDP checkpointing, fixed (2026-08-16) — `.superpowers/ddp-checkpoint-fix.md`

**Prompt:** make `--ddp 4` write a valid checkpoint, keep the guard meaningful, and prove the
result end to end — or prove it cannot be fixed from our side.

**It could.** The previous section's conclusion that this needed an upstream change was the one
thing it got wrong. **`ttnn.Tensor.update_tensor_topology` is bound in Python**
(`pytensor.cpp:1611`), and `ttnn.TensorTopology` is constructible from Python
(`distributed_nanobind.cpp:734`) — so the false `Shard(0)` marking can be corrected by any holder
of the tensor, not only where it was written. `train/checkpoint.py:replicated_for_save` re-marks
each parameter `Replicate`, saves, and restores the original topology in a `finally`. **No data
moves**; a `--ddp 4` save now costs what a `--ddp 1` save costs. Result: **737,824,624 bytes**,
byte-for-byte the single-chip size, every tensor `(1,1,out,in)`.

**Rejected alternatives:** extracting a replica ourselves (would mean reimplementing ttml's
atomic streaming writer — a second checkpoint writer free to drift); assigning a
freshly-replicated tensor back into each parameter (maximum data movement, for a problem whose
own diagnosis says the data is already correct).

**Why the restore, when `Replicate` is the *truthful* marking:** saves happen mid-run at
`--save-every` boundaries, so a save must leave the training state exactly as it found it.
Afterwards all 66 params read `Shard(0)` again and training continues normally.

**The guard is narrowed, not deleted.** `assert_saveable_on_mesh` now asks "is this sharding
*explainable*?" and refuses unless both (1) ttml's parallelism context is **DDP and only DDP** —
under TP a `Shard` may be the truth, and re-marking would write a quarter of a model — and (2)
the tensor is distributed over **exactly the DDP axis**. Both fail closed. Deliberately *not*
gated on the replicas agreeing numerically; see the stochastic-rounding finding below for why
that obvious check would have been wrong.

**Decisive proof of the writer:** after 50 DDP steps, every tensor in the file is bitwise equal
(**max abs 0.000000e+00**, 0 shape mismatches) to replica 0 read independently through a
`concat_mesh_to_tensor_composer`, which does not consult the placements being corrected. True
both with and without stochastic rounding.

**The strict acceptance test — `--ddp 4` bit-identical to `--ddp 1` — cannot pass, and the reason
is not checkpointing.** Measured max abs difference at 50 steps: **9.155e-03** (SR off), 3.52e-02
(SR on); shapes all correct, 0 tensors with a spurious leading axis. Its premise ("the replicas
are bit-identical, so the extracted weights should be too") conflates *within-run* replica
identity with *across-run* equality. Three measurements settle it: (i) at **step 1**, identical
weights and identical data give losses of 10.6094 (`ddp 4`) vs 10.6875 (`ddp 1`) — 1.25 bf16 ulp,
before any optimizer or checkpoint code runs, because `ddp 1` means over 64 sequences while
`ddp 4` all-reduces four means over 16; (ii) the step-1 parameter difference is **6.103516e-04 =
exactly 2 × lr**, the largest a single Adam step can produce, because Adam's update is ±lr
regardless of gradient magnitude, so a last-bit gradient difference yields a full-sized step of
opposite sign; (iii) it grows ~15x over 50 steps as an amplified random walk. **Val loss agrees
where it counts:** 7.0523 vs 7.0875, against the 0.1944-nat seed-noise floor.

**Stochastic rounding breaks DDP's replica-identity invariant** — unexpected, and it corrects the
previous section. Four steps, identical but for one flag: `stochastic_rounding: false` → **0/66**
params' replicas differ (0.0); `true` (i.e. `nanollama3_bpe_v2.yaml`, what real runs use) →
**66/66** differ, max 2.34e-2. Each device rounds from its own RNG, so replicas random-walk apart
despite identical all-reduced gradients. Consequences: it is why the guard is structural rather
than numeric (a bit-identity check passes on the default config and refuses on the recommended
one), and a `--ddp N` checkpoint is *replica 0's* weights — one of four coherent models, not the
only one. Upstream ask 4.

**End to end:** the `--ddp 4` checkpoint converts to HF, loads as `LlamaForCausalLM`
(122,962,944 params), and generates. **The parity gate ran against a DDP checkpoint** — the
verification listed as outstanding — at **max_abs 2.56e-06**, *tighter* than the baseline's own
8.3e-6–1.6e-5. `tests/test_numpy_parity.py` now takes four optional env overrides
(`TT_TNT_PARITY_CHECKPOINT_DIR` etc.); **defaults unchanged**, committed gate still 6/6. Two
baseline-calibrated meta-tests fail under an override and both are information: the norm-swap
test fails **because the SR checkpoint's gammas are real (0.977–1.031) and the blind spot is
gone** — the first checkpoint this repo has produced against which that gate is not blind — and
the not-hollow test still catches the RoPE bug at max_abs 2.93 (~2900x over budget), failing only
a sharpness constant measured on a 60x-longer-trained model.

Suite: **852 passed, 1 skipped** (845 + 7 new). tt-metal untouched at `620793d898`. Nothing
deleted; every run checkpointed to a scratch dir outside the repo.

Two new entries in `docs/upstream-tt-metal-asks.md` (descriptor-shape mismatch hangs instead of
failing — diagnosability only, nothing blocked; and the topology re-marking above — this one
does block DDP checkpointing).

Suite: **845 passed, 1 skipped** (831 baseline + 14 new). tt-metal left untouched at
`620793d898`. Nothing deleted; all benchmark runs checkpointed to a scratch dir outside the repo.

## Embedding geography — is a token-to-Tensix-grid layout discovered or imposed? (2026-08-16)

One question, no hardware: *does the embedding space have a geography that corresponds to the
corpus?* It matters because a proposal wanted to lay the 32,000-token vocabulary onto
Blackhole's Tensix grid and sample by spatial neighbourhood, claiming *direction = corpus
register*. If sources already occupy distinct embedding regions the layout is **discovered**;
if not it is **imposed** and the claim is decoration. Full report:
[`.superpowers/embedding-geography.md`](.superpowers/embedding-geography.md).

**Answer: discovered, at 139 sigma above its own noise floor.** k-NN purity over cosine
neighbours of 1,350 source-labelled tokens is **0.5458** against a label-permutation floor of
0.1103 ± 0.0031 and a chance baseline of 0.1111; a linear probe recovers which of nine sources
a token belongs to **78%** of the time. Characteristicness is **log-odds with an informative
Dirichlet prior, z-scored** (Monroe et al. 2008) — chosen over tf-idf because it divides by its
own standard error, so a token must be skewed *and* well-attested. That statistic favours
frequent tokens, so the result is reported against a **frequency-only control** (log count +
embedding norm: 0.135 purity, 0.21 probe) and re-run with the 500 commonest tokens excluded,
where it goes **up**, not down.

**The prediction was directionally right and specifically wrong, and the specifics are the
finding.** Narrative sources really are more entangled: as matched 4-way problems (chance
0.25), `folklore`/`gutenberg_children`/`spine`/`weird` reach 0.5813 while
`flavour`/`poetry`/`procedural`/`wikipedia_simple` reach 0.7865 — 44% vs 72% of the headroom
above chance. But (a) **`tinystories` does not separate cleanest** (0.544, mid-table, confused
with `gutenberg_children`) — `score_behaviour.py`'s 99.9% register control classifies a
*passage* by syntax, while a sampler picks *tokens*, and children's-narrative vocabulary is
children's-narrative vocabulary; and (b) **`spine` is the cleanest of the four narrative
sources** (0.699), because its vocabulary is naturalist/taxonomic, not narrative. The real
entangled cluster is `folklore` ↔ `weird` ↔ `gutenberg_children`. Consequence for the pitch:
**four to five clean directions, not six** — and `weird`'s top tokens (`Ġletter`, `Ġsupper`,
`ĠMadame`, `Ġcarriage`) read as period social prose, which is a question about that slice's
contents worth asking separately.

New: `scripts/probe_embedding_geography.py` (reads one tensor from `model.safetensors`, never
the model; numpy-only k-NN/silhouette/probe/PCA — sklearn and matplotlib are importable in the
venv but are not declared dependencies, and none was added), `tests/test_probe_embedding_
geography.py` (41 tests, none needing the model — including a planted-null test asserting the
tool returns NOT INTERPRETABLE when there is nothing there), and
`docs/measurements/embedding-geography-tt-tnt-1024a.{md,json}`. Nothing built beyond the
measurement: no sampler, no layout, no kernel. Suite: **982 passed, 1 skipped** (941 + 41 mine;
the 941 is the 939 baseline plus 2 from a concurrent agent's uncommitted `benchmark_external`
work, which this change does not touch).

## The grid-distance gate — the 2-D squash keeps the cells and loses the directions (2026-08-17)

The gate the embedding-geography report proposed for itself, run: *assign the vocabulary to an
11x10 grid and re-run the same purity statistic with grid distance substituted for cosine
distance.* Full report:
[`.superpowers/grid-distance-gate.md`](.superpowers/grid-distance-gate.md).

**The gate as written passes — and the pitch fails anyway, because the aggregate hides which
half of the idea works.** Grid-distance k-NN purity is **0.3996 ± 0.0101** against a
label-permutation floor of 0.1105 ± 0.0033 (88 sigma), which is 0.73 of the 0.5458 cosine
baseline and **66%** of that baseline's headroom above its own floor. No collapse.

**But almost all of it is which tokens share a core, and almost none is which cores are
adjacent.** Three independent ways of seeing the same thing: (a) randomly placing the *same*
clusters scores **0.3943** against the annealed layout's 0.3996 — the layout is worth half a
purity point; (b) with cell-mates excluded, purity is 0.1702 against 0.1029 for random
placement, i.e. adjacency carries **15%** of the cosine headroom; (c) the hop profile —
0.4836 within a core, 0.1681 at one hop, 0.1132 at four, against a 0.1111 baseline. **The
register correlation reaches about one hop.** A *label-cheating* layout that groups same-source
cells into continents reaches only 35%, so the ceiling is the squash, not the annealer (and
across 9 restarts, QAP cost and off-cell purity correlate at r = +0.13 — the objective the
annealer minimises is not the one the pitch cares about).

**What survives is the useful half.** A core's contents are register-coherent — within-cell
purity 0.4836, **86%** as coherent as a cosine 10-NN ball, rising to 0.6805 on 816 cells where a
core beats a cosine neighbourhood outright — and six *neighbouring* cores return **4.22** of a
possible 4.56 distinct sources (90% of the way from "one register" to chance). So neighbourhood
sampling really does hand back divergent, coherent registers. It just is not because of
direction: random placement scores 4.55 on the same statistic. Consequence: **build the sampler,
drop the story, and place clusters wherever the NoC prefers.**

**Four grids, and more room does not rescue direction.** 11x10 = 110 (this box's harvested
p300c), 17x12 = 204 (full die), and 2x2 tilings at 440 and 816 treated as one flat torus (an
optimistic idealisation — an inter-die Ethernet hop is not a NoC hop, stated as such). Adjacency
headroom goes 15% → 16% → 23% → 21%: it plateaus. What a bigger grid buys is cell coherence,
0.484 → 0.681. **36 layouts** were built (4 grids x 3 clusterings x 3 annealing restarts) and
every table reports the distribution; not one passed the direction clause.

New: `scripts/probe_grid_layout.py` (balanced spherical k-means over all 32,000 tokens, spectral
init, simulated-annealing QAP on torus Manhattan distance; numpy only, and it imports
`probe_embedding_geography`'s `permuted_purity`/`purity_from_neighbours` rather than
reimplementing them, so the two measurements' floors are the same computation),
`tests/test_probe_grid_layout.py` (45 tests, none needing the model — including planted nulls
that must come back REFUTED, and an identity test pinning the annealer's O(cells) swap delta to
a full cost recompute), and `docs/measurements/grid-layout-gate-tt-tnt-1024a.{md,json}`. Nothing
built beyond the measurement: no sampler, no kernel.

Suite at the time of this commit: **1028 passed, 7 failed, 1 skipped** — 1035 collected, which
is the 990 baseline plus my 45. (One of mine skips explicitly once the scratch corpus profile is
gone: the test that checks this script's cosine baseline still reproduces 0.5458 needs a token
profile, and re-tokenising nine million words is not a unit test. It ran, and it passed, before
the cache was removed.) The 7 failures are **not this change**: a concurrent agent's
uncommitted TinyStories-reduction experiment had just re-cut `train/corpus.py`'s target shares
(tinystories 31% → 10%) and re-blended `artifacts/corpus/blend.txt` without regenerating the
docs those tests check, so `test_corpus_blend_doc` (4), `test_render_licensing` (2) and
`test_measure_corpus` (1) fail on the mismatch. Verified by checking out HEAD into a throwaway
worktree with only my two new files added: all 7 pass there, along with all 45 of mine. Nothing
of that agent's work was staged, stashed, reverted or touched — this commit stages five explicit
paths. My measurement is unaffected by the re-blend either way: it reads a prefix of each
**per-source** file with an equal word budget, not the blend.

---

## Reducing TinyStories — the fifth intervention, and the first that moved register

**Prompt.** "Does reducing TinyStories move register?" Four interventions had failed to move
it (sampling, the stratified-split mixture correction, an LR decay tail refuted by a seed
control, and capacity — a null at 1.03x the floor). The only two things that ever moved this
model were corpus changes; TinyStories was 31% of training tokens and is literally what the
register metric measures similarity to. Two arms, three seeds each, at `1024`'s exact shape.
Full report: `.superpowers/tinystories-reduction.md`.

**Design.** Arm A = the shipped 31% blend (`artifacts/tokens-v3`), reusing
`checkpoints-tt-tnt-1024a` as seed 5489 plus two new seeds. Arm B = TinyStories at **10%**,
the freed 21 points redistributed **proportionally** across the six sources with headroom, so
the ratio of every non-TinyStories source to every other is preserved — that is what makes it
a single-variable change. `procedural` and `flavour` could not be scaled (a proportional
share needs 5.10x and 4.53x against the 4x cap) and were pinned; four sources needed a higher
`upsample` (gc 2→3, wiki 1→2, folklore 2→3, weird 3→4). Shares summed to exactly 1.0,
strange slices held at 35.5%. Seeds 5489 / 20260815 / 8191, `--ddp 4`, `--model-impl python`,
10,764 steps — five new runs, ~47 min each.

**Register moved. 1.79x and 1.36x the seed floor, 3/3 seeds, both gates.**

| signal | arm A (sd) | arm B (sd) | delta | ×floor |
|---|---:|---:|---:|---:|
| nearest source == tinystories | 0.2192 (0.0149) | 0.1535 (0.0308) | −0.0657 | **1.79x** |
| tinystories margin | −0.5076 (0.0431) | −0.6090 (0.0449) | −0.1014 | **1.36x** |
| — lexical-habit collapse | 0.0516 (0.0059) | 0.0352 (0.0142) | −0.0164 | 0.29x — not interpretable |

**It is not a loss trade.** Loss rose +0.3857 nats, but **+0.4288 was predicted from the
validation-mixture change alone** — the two arms hold out per-source tails of *different*
blends, and arm B's val set is 10.2% tinystories against arm A's 31%, so a model of identical
quality already scores ~0.43 worse. The residual is **−0.0431 = 0.22x the loss floor**. The
null hypothesis for loss in a corpus experiment is not zero, and computing it is what turned
an uninterpretable number into an interpretable one.

**It IS a termination trade, mechanistically explained.** `termination rate` regresses 1.37x
the floor (3/3 in direction) and generations run longer (`n_tokens` +0.59, 1.79x).
`tinystories` supplied **73%** of the blend's document separators, so cutting it left arm B
with **41.6% fewer `</s>`** — one per 836 tokens against one per 491, counted directly on both
token arrays. Predicting that count from the share arithmetic gives 468,001 against 466,191
measured, agreeing to 0.4%. Separator density is a property of document length, not of
register, so this is separately fixable.

⚠️ **One seed would have got this wrong.** The per-seed effect is monotone — 0.92x, 1.74x,
2.70x. Seed 5489 alone, the seed every previous single run used, lands inside the floor on
every signal and would have been written up as a null. The mirror-image error is in the same
data: the seed-5489 pair reports `4-gram repeat` **worse at 5.32x** and `longest repeat`
**worse at 6.28x**, which at the arm level are **0.24x** and **0.90x** — noise. Single pairs
lie in both directions.

⚠️ **"Less like TinyStories" is not "more like spine."** The mass that left tinystories
(−0.066) went to `gutenberg_children` (+0.031) and `procedural` (+0.024); `spine` gained
**+0.007**. The model moved onto the *other* simple-narrative backbone. The obvious next
experiment is therefore not "try 5%" but holding the backbone total fixed and moving share
into `spine`.

**The confound the brief called the most likely source of a spurious positive is structurally
absent.** The register profile is fit from `artifacts/corpus/<name>.txt` and the *names* in
`SOURCES` — never `target_share`, never `upsample`; `blend_corpus.py` writes only `blend.txt`.
Three proofs: the nine per-source sha256s are identical across the re-blend; re-measured
availability is byte-identical; and re-scoring `hf-tt-tnt-1024a` **with arm B's registry
loaded** reproduced the committed numbers **bit-for-bit**, including per-source nearest
counts. (That last one also means the concurrent commit landing mid-experiment could not have
perturbed a measurement.)

**Deliberately not adopted.** The corpus was restored to the shipped 31% blend afterwards —
`blend.txt`'s sha256 reproduces `24f3d112…` exactly, the blend being deterministic. Adopting
arm B would rewrite `docs/corpus_blend.md` (digest-pinned provenance for the *published*
model's corpus) and edit three share-pinning test files, as a side effect of an experiment,
for a corpus no published model trained on. Measuring and shipping are different acts.
Everything needed for adoption is kept:
`.superpowers/tinystories-reduction-armB-registry.patch`, `artifacts/tokens-lowts/`, and the
three arm B models.

Suite: **1034 passed, 2 skipped** — 1036 collected, the same 1036 as the entry above, whose
7 failures were this experiment's uncommitted re-cut of `train/corpus.py` and are green again
now that the corpus is restored. The measurement sets add no tests. The second skip is not
mine either (`test_probe_grid_layout` skips without a cached corpus profile; the other is
`test_sizes` without `TT_METAL_HOME`). No new dependencies, nothing published to the Hub.

---

## 2026-08-20/21 — improv thinking, and four instruments that lied

**Prompt.** "What would it take to make it a thinking model too?" → a design conversation that
landed on improv rather than reasoning: give the model a five-slot think-block
(`offer / accept / add / stakes / handback`) before each story continuation, one slot per named
failure mode — escalating to the worst place, blocking with the dullest next step, drifting so
far out nobody can follow. Then: spec, plan, and execution via subagent-driven development.
Merged as `8cab579`.

**The result.** Stage 1 is **PARTIAL**. Schema adherence **98%** (784/800; the no-think control
emits blocks 0% of the time, which is the control that shows adherence comes from training and
not from the harness). Substituting another story's block changes **100%** of continuations, so
the block steers rather than decorates. And it moves **none** of the four failure-mode scores at
α = 0.01. The mechanism works end to end and buys nothing on the metrics we chose.

**The headline reversed once, and that is the most useful thing here.** An earlier pass reported
**0% adherence** with a supporting diagnostic (the token opening `<think>` at rank 126, nll
10.34 under teacher forcing). Both were measured on a run in which all 17 RMSNorm gammas were
*provably frozen*: `stochastic_rounding` defaults off on the `SFTTrainer` path, which bypasses
the unconditional warning `train/run.py:909` prints. Proof was a tensor diff, not an inference —
think vs nothink `step_3000.pkl` had exactly 17 bit-identical tensors (precisely the gammas) and
49 differing, and those 17 were bit-identical to the warm-start checkpoint. With the gammas free,
0% became 98%. This is the same frozen-gamma bug fixed in the pretraining path that morning,
recurring in a code path that skips the guard.

**Four instruments that lied, all of them mine.**

1. **`--eval-every` is not `--val-every`.** The first four-arm run logged no validation curve at
   all (`--val-every` defaults to 0 = disabled), so `compare_runs.py` had nothing to compare.
   Discovered after the runs had burned hardware time.
2. **Splitting the corpus on `\n\n` measures paragraphs, not stories.** The separator is `</s>`.
   Blank-line splitting reports a median of 40 tokens where the real figure is 199, which would
   have silently wrecked cut-point selection. Caught by re-measuring, having already written the
   wrong number into a spec table.
3. **HF-style unshifted labels against ttml's pre-shifted convention.** ttml expects
   `labels[t] = token at t+1`; the plan wrote `labels[t] == input_ids[t]`. Two full arms trained
   against effectively wrong targets. Invisible to two task reviews because every test asserted
   label STRUCTURE and none asserted label SEMANTICS or loss values — and because the smoke
   trained from random weights, where a right and a wrong convention look equally bad. Fixed by
   `labels = [-100]*(len(p_ids)-1) + c_ids + [-100]`, which keeps the last prompt position
   supervised (its target is the first completion token — the transition being trained).
4. **`git check-ignore -v`'s exit status does not mean "ignored".** With `-v` it also prints
   negation matches, so I read an un-ignore rule as proof of ignoring and told an implementer
   their correct fix had failed. `git add -n` is the check that answers the actual question.

**Tests that could not fail, four of them.** A truncating `s[:20]` mock making a `<=` assertion
vacuous. A `with_think` guarantee relocated onto a helper production no longer calls. Structural
assertions blind to the label-shift convention. And `groundedness`, which passed a discrimination
test on constructed extremes (grounded 1.000 vs "Gorthax and Vermilion argued about the Treaty of
Blunn" 0.333) while being *dead on real data* — mean 0.998, 99.25% of scores exactly 1.0, because
the corpus has 641 hub words above 2,000 neighbours and 80.1% of prefix words are hubs, so
"connects to any prefix word" is almost always true. Replaced with normalised PMI (`870c9b4`):
0.00% at ceiling, sd 0.089. **My first replacement test was also hollow** — asserted on a
5-document synthetic table, which the boolean scorer passed, because a small table has no hubs
and so cannot exhibit the bug. The guard had to be artifact-gated against the real table.

**Paired versus unpaired, twice now.** Two dense runs differing only in seed show mean |delta|
0.0495 — larger than the 0.0481 MoE-vs-dense effect. That is not a refutation: the effect is a
*paired* delta with per-step oscillation cancelled, the 0.0495 is *unpaired* and still contains
it. Comparing them is comparing a quantity with its noise floor removed against one that keeps
it. The mirror-image error (applying an unpaired floor to a paired design) once stamped a t≈6
result NOT INTERPRETABLE.

**Process notes.** 20 controller rulings recorded in the SDD ledger, three of which a subagent
corrected and two of which I corrected myself. A subagent deleted ~2.8 GB of superseded
checkpoints at 98% disk because I propagated hardware-safety rules into every dispatch and never
the standing "don't delete without asking" rule. I then destroyed `artifacts/improv/` myself with
`git worktree remove --force` — git was clean and pushed, but untracked artifacts lived in the
doomed copy. Recoverable only because derivation is deterministic: regenerating reproduced
18,791 traces and a 6.05% drop rate exactly. Three separate agents plus the final reviewer
self-reported running a bare `import ttml` without a lease; the hazard is documented in a
docstring nobody reads before their first command.

Suite: **1139 passed, 2 skipped.**

## 2026-08-21 — skits, tasks 1-2: the gates hold, the yield does not

Stage 2 turns each think-block slot from a description into a falsifiable prediction about a
*later* turn: five turns (model/partner/model/partner/model), partner turns being real corpus
text rather than generated, so `accept` has an actual offer to accept and `handback` has an
actual next turn to be tested against. Tasks 1-2 built the schema (`train/skit.py`) and the
derivation (`scripts/derive_skits.py`). Suite **1148 passed, 2 skipped**.

**A shared module nearly ate a published measurement.** Task 1's brief shipped a fixture story
that does not derive — `split_sentences` over-splits dialogue-with-attribution, so
`'"It catches the light!" said her friend.'` becomes two sentences and the turn pair shares no
content word. The implementer fixed it by changing the splitter in `train/improv.py`, which is
upstream of `derive_traces.py` and therefore upstream of the *committed* `improv-stage1.json`.
The change moved stage-1's inputs from 18,791 kept / 6.05% drop to 18,954 / 5.23%, and it was
not even a fix: the old pattern over-splits dialogue attributions, the new one under-splits any
sentence ending in a quoted period, which in a dialogue-heavy corpus is at least as common.
Reverted; the limitation is now documented in `train/skit.py` where the next reader meets it.
The green suite (1144) was no protection — no test pinned the derivation counts. Re-deriving and
comparing against the recorded manifest was the check that settled it. Task 2's implementer hit
the same fixture independently and correctly replaced the story instead of the module.

**Three of five tests passed against the bugs they were named for.** A reviewer mutation-tested
Task 1 and found the `offer`-sourcing test survived sourcing `offer` from the whole scene, and
the `stakes` test survived diffing against the wrong turn — because the replacement fixture
threaded the word "cat" through nearly every sentence, so any span overlapped any other and the
overlap assertions were vacuous. Rewritten with turn-unique vocabulary and identity assertions.
Confirmed by re-running the mutants myself: each now goes red, including one the reviewer did not
use (leaking a partner turn into the supervised region), which only the segments test catches.
Same failure shape as the four hollow tests in stage 1, recurred despite an explicit warning in
the brief. A third finding was real but benign: the `MIN_SENTENCES` gate can never fire uniquely,
since `len(sents[2:7]) != 5` always catches a short story downstream. Kept it anyway — an
explicit precondition is less fragile than an emergent property of a slice width — and renamed
the test to what it actually verifies.

**The gates compound, and that is a yield problem, not a quality problem.** Real-corpus
derivation keeps **1,921 of 20,000 stories — a 90.4% drop rate**, against stage 1's 18,791.
Each of the three model turns independently requires both a word carried from its single
preceding sentence and a fresh word; three such gates multiply rather than add
(~0.46³ = 0.096, matching 1921/20000). 73.3% of drops are ordinary accept/add failures and only
17.5% carry the dialogue-splitter fingerprint, so this is inherent strictness, not a bug. The
corpus holds 2,119,489 stories and `--limit 20000` only ever mirrored stage 1's comparison
budget, so `--limit 200000` yields ~19,210 skits — stage-1-comparable at zero semantic cost.
Relaxing the accept/add gates to raise yield would buy examples by destroying the measurement:
those gates *are* the falsifiable prediction that makes stage 2 different from stage 1.

**Resolved.** Both were done: derivation re-ran at `--limit 200000` (18,610 skits, within 1% of
stage 1's 18,791 — the earlier "~19,210" was a projection), and the `stochastic_rounding` guard
reads `trainer.optimizer.get_state_dict()` and was proven by deliberately breaking it on hardware,
where it raised before step 1.

## 2026-08-23 — skits, tasks 3-5: the result, and the control that cut it by 4.5×

Stage 2 shipped. Suite **1212 passed, 2 skipped**. Verdict **PARTIAL**, published in
`docs/measurements/skits-stage2.json`.

**The headline is `add`, not `accept`.** Stage 1 moved 0 of 4 scorers. Stage 2's think arm beats
its shuffled control on all four — but the shuffled control breaks two links at once, the
block↔turn link *and* the shared-context link, so its gap conflates plan-following with the model
echoing a scene it can already see. The implementer added a second control I had not asked for —
score the same block against the *no-think* arm's turn for the same skit — and it cut `accept`'s
effect from **+0.603 to +0.134**. Reviewing that control turned up a confound its author could not
see: the arms are separately trained models, so the contrast is plan-presence ⊕ arm-identity.
Bounding the arm term with the cross-arm reference test gives `accept` **+0.083**, `stakes`
**+0.017** (withdrawn — barely above its own arm term), and `add` **+0.348** — where the confound
carries the *opposite* sign, so `add` is biased downward. There is a mechanism: the `add` word is
already visible in the context only ~40% of the time versus **92.8%** for `accept`, which is mostly
the protagonist's name. The slot that looked weakest is the only one measuring what we claimed.
The artifact says outright "THE CLAIM IS NOT 'all four slots move'", and the withdrawal rule is
code (`CONFOUND_WITHDRAW_SHARE`), not a judgement applied by hand.

**A story that felt true and wasn't.** "The mechanism that gives plan-following also gives looping"
explained a real failure — the think arm repeats itself more (0.058 vs 0.023, t=4.234). It is
wrong. Correlation with plan-following is **+0.039**; correlation with the block repeating the
*same* word across all three turns is **+0.283**. Those 32 of 256 skits average 0.147 against
0.045, and removing them drops t to 2.699 — below threshold, NOT INTERPRETABLE. A model following a
*varied* plan does not loop. The fix proposed under the wrong mechanism (score blocks for novelty
across turns) happens to be the right fix for the real one.

**Ten hollow tests, and the tenth changed the diagnosis.** Nine were value-level: assertions that
held against both correct and incorrect code. The tenth is a class, not an instance — whole
*decision functions* that appear in no test file and are reached only through a driver. Four
mutations in that class each rewrote a published claim while all 1,212 tests passed: inverting one
word at the degeneration call site flipped the artifact's verdict from PARTIAL to "STAGE 2
SUCCESS"; swapping two arguments flipped every arm-quality proxy and restored the withdrawn slot to
headline; reversing `score_pair`'s positional args manufactured a significant finding. Fixing
instances one at a time had not converged — the verdict wiring was pulled into a tested function
and the *withdrawal machinery added in the same commit* got no test at all.

**The threshold ruling paid for itself.** Requiring this eval to define its own Bonferroni
constants (α=0.05/11, t=2.843) rather than import stage 1's (0.01/2.576) was defensive when
written. Cross-arm `accept` landed at t=2.698 and `escalation` at 2.594 — both *between* the two
thresholds. Importing stage 1's would have manufactured two findings, one of them
"the think arm accepts offers better".

**What the eval population actually is.** 90.7% of stories drop. Three sequential gates each
demand a carried word and a fresh word, and they multiply (~0.46³). The corpus is 54.6% dialogue by
story and the kept skits are 31.0% — a **43% relative loss**, because the sentence splitter
fragments dialogue-with-attribution and those skits then fail the accept gate. We are measuring the
monologic residue. Reading a derived skit shows the deeper problem: the "partner turn" is the same
narrator's next sentence, not a second voice, which is why `handback_anticipation` tops out at
0.119 on ground truth. A probe of alternation-by-position (five or more quoted utterances, roles by
index parity) finds **18.8% of stories usable → ~80,000 skits**, four times what we trained on, and
produces real yes-and structure. That is skits-v2.

**Process.** An unleased `import ttml` happened a fifth time, always the same way — importing the
package to check it exists (`importlib.util.find_spec` is the tool; the lesson is now in the global
CLAUDE.md). A test-fix agent held a mutant on disk for 2m29s while an eval ran concurrently; it was
harmless only because the mutant touched `labels` and the reader consumed `input_ids` — luck, not
design, so mutations are now serialised against live runs. `scripts/eval_skits.py --rescore-from`
re-derives every published number from stored generations with no model, tokenizer, or device,
verified under an import blocker that raises on `torch`/`transformers`/`ttnn`/`ttml`.

## Skits v2, task 1 — the splitter cutover and real dialogue turns (2026-08-23)

Design: [`docs/superpowers/specs/2026-08-23-reach-dial-design.md`](docs/superpowers/specs/2026-08-23-reach-dial-design.md).
Task 1 built the foundation: one dialogue-aware splitter, turn extraction by alternation, a
dialogue derivation script, and the stage-1 republication the cutover obliged.

**One splitter, and it needed to scan rather than match.** Both historical variants are wrong
in opposite directions on the same two strings, and no lookbehind can fix that: the two cases
differ only in what FOLLOWS the closing quote. `train/dialogue.py` tracks quote state and asks
one named question — `continues_as_attribution` — at the single ambiguous position. Also worth
recording: **1,224 tests, and not one of them called `split_sentences`.** Changing it broke
nothing, which is exactly why it had drifted this long.

**The dialogue loss was stage 2's, not stage 1's.** Measured, not assumed. Stage 1's random-cut
population already retained 93% of the corpus's dialogue density (95% now). The stage-2 skit
population carried **0.68×** the corpus rate under the old splitter and **1.38×** under the new
one, with kept rising 9.6% → 13.2%. So the 43% relative loss was the accept gate eating
dialogue-with-attribution, and it is now reversed. Stage 1's republished derivation moved only
6.05% → 5.15% drop — but **40% of the shared examples changed text**, because the cut point is
`randint(2, len(sents) - 2)` and `len(sents)` moved. A republication that reported only the
drop rate would have read as a rounding correction.

**Attribution really is unnecessary, and dropping it drops the bug.** No name is read anywhere.
The cost is measurable and is published rather than hidden: 39.6% of adjacent turn pairs are
separated by nothing but a tag (`"…," he said. "…"`), which is an **upper bound** on how often
two adjacent turns are the same voice. Telling `"Don't cry," he said. "I can help."` (one voice)
from `"Hello," said Amy. "Goodbye," said Ben.` (two) needs the subject of the tag — i.e. exactly
the attribution that got the probe's speakers backwards. So it is a manifest field, not a gate.

**Yield is 4× short of the projection, and the gates were not touched.** 18.89% of stories clear
five utterances (the probe's 18.8%, confirmed at 400k). But only 11.3% of those survive
accept/add, so 400,000 stories yield **8,543 skits**, not 80,000; the whole corpus projects to
~45,000. The drop table says why and says it per gate: `no_accept` at some model turn is 61,781
of the 66,770 derivation failures. Real exchanges are short — `"Ouch!"`, `"Thank you, ghost!"` —
and share no content word with what preceded them. The spec's own sample skit drops at turn 4.

**A tenth-class test caught in the act.** `classify_turn_failure`'s four-row table passed against
a mutant that checked the `add` gate before `accept`, because every row failed exactly one gate.
Ordering needed a row where BOTH fail. Same shape as the ten before it.

## Skits v2, task 4 — the EUREKA measurement, and the dial moves (2026-08-23)

Design: [`docs/superpowers/specs/2026-08-23-reach-dial-design.md`](docs/superpowers/specs/2026-08-23-reach-dial-design.md).
Result: [`docs/measurements/reach-dial.json`](docs/measurements/reach-dial.json).

**The dial works.** Within-model, scene-paired, 826 scenes, one model turn each, everything
teacher-forced except the four characters of the dial value. Realised distance runs
**0.6497 → 0.7336 → 0.7792** across `near`/`mid`/`far`, all three steps significant at
`CRITICAL_T = 2.843` (t = 16.2 / 13.9 / 23.3) and **all three still significant after
residualising on log `add_df`** (t = 7.4 / 9.0 / 12.5). So it is a REACH dial, not a frequency
dial. Coherence held: groundedness fell 0.030 at `far` against a 0.05 margin declared before
the data existed.

**The negative control needed the strict threshold to stay negative.** `nodial`'s three steps
came in at t = 2.36 / −0.30 / 2.35. Two of those sit *between* stage 1's 2.576 and this eval's
2.843. Importing the looser constant — the thing the spec forbade in writing — would have turned
the negative control positive and destroyed the headline. That is not a hypothetical any more.

**EUREKA is still "not met", on one gate, and the gate failed in the wrong direction.** The
`add` slot-hit rate is non-monotone: near 0.477, mid 0.567, far 0.507. "Hold at every setting"
(worst-vs-best, declared) fails by 0.090 — but the worst setting is **`near`**, and `far` is 3
points *above* `near`, so the spec's own stated worry ("stops fulfilling an ambitious plan")
passes. Reported all three ways with the declared reading named as the gate; a mutant that
swaps the gate to the flattering reading is RED in the suite. Moving a pre-declared threshold
after seeing which side of it you landed on is the one thing this project cannot do.

**A meaningless dial value buys 38% of the range.** `reach: blue` — same arm, same well-formed
six-slot schema, off-vocabulary value — is distinguishable from `far` (t = −17.8) so the primary
control passes, but it sits 0.379 of the way from `near` to `far`. Part of what the dial does is
"a token is here", not "the token means far". A control reported as a boolean would have hidden
that; the fraction is a field.

**The 34-minute association table, in 81 seconds.** `train.reach.reach_distance` only *looks
up* the pairs it is asked about, so the eval counts one streaming corpus pass restricted to the
~3,200 unigrams and ~69,000 pairs it needs. Proved equal to the full table two ways: on a
fixture corpus built both ways, and by re-deriving all 1,000 gold rows' stored
`reach_distances` at full scale — **max abs error 0.0, `add_df` mismatches 0**, a hard refusal
if not. The leave-one-out had to be made *exact* (subtract the story only from pairs it really
contributes to) for that to work, because a generated `add` word need not be in the story at all.

**`artifacts/` is git-ignored, so a side file cannot make a measurement reproducible.** All
1,000 scenes and 7,000 generations are embedded in the published artifact, and
`--rescore-from docs/measurements/reach-dial.json` round-trips byte-identical.

**Two more hollow tests, caught by running the mutant rather than reading the test.** One
asserted the first `add:` line wins using a *second think-block* — the `</think>` truncation was
doing all the work, so dropping the `key not in found` guard passed. The other checked the gold
reproduction using only an unscorable row, which emptied the error list and made `matches` False
for the wrong reason. Both fixed by making the fixture able to fail. Also: stage 2's 4-gram
degeneration metric scored **exactly 0.0000 on every setting** here — a skit turn is one
sentence and rarely has four content words — while `"Snowmen are snowmen and snowmen"` sat in the
sample. A metric that cannot fire is worse than no metric; replaced with type/token.

### Task 4, review round 2 — the control that measured its own filter (2026-08-24)

**A frequency control that was actually a null-enrichment filter, and I published its reading
as a finding.** The df-matched subsample keeps pairs whose two `add` words have similar
document frequency. When the dial emits the SAME word under both settings the frequencies are
identical, so the pair is *always* kept — and its distance difference is **exactly 0.0 by
construction**. 78–92% of each matched subsample was such a pair. Averaging a real effect over
a pool padded with structural zeros shrinks it by the padding ratio, and I read that shrinkage
as "a large share of the raw movement IS frequency" and as a 3–5× disagreement between the two
controls. Over the *informative* pairs the two controls agree to three decimal places
(+0.0630 vs +0.0604). **The instrument, not the subject, produced the finding** — the exact
failure the global notes warn about, in a control I wrote specifically to be careful.

**A hard-coded `not_monotone: true` under a key named `frequency_confound_here`.** It described
the derivation's gold buckets, not this run; the run's realised `add_df` is strictly monotone
(10.79 → 11.72 → 12.03, spearman +0.419). Any literal that describes a *different population*
under a key claiming to describe *this* one is the same class of bug. Computed now, and the
derivation's numbers are labelled as the derivation's.

**Leaf-only test suites have now failed to catch this class eleven times.** Every decision
function had a fixture test; the 340-line composer that wires them had none, and three
one-line mutations inside it each rewrote a published claim while 1,505 tests passed — one of
them (`add`-hit scored against the raw generation, which always contains `add: <word>`) flipped
`eureka_criterion_met` from false to TRUE. `analyse` is now driven end to end on a synthetic
corpus. **Both versions of that fixture were themselves vacuous first**: the first saturated
(NPMI clamped to 0, every distance exactly 1.0) and the second let the control conditions emit
a single word, tilting the nuisance fit so a pure frequency dial read as a reach dial. A
fixture built to test a property of scale has to be checked for that property before anything
is asserted on it.

**A key name is a contract.** Renaming `cleaner_contrast` to something more honest broke
`tests/test_reach_cli.py`, because `scripts/reach.py --about` quotes the field. The key is back
with its provenance attached — and the CLI still states the spec's framing as a finding when
this run does not support it. Flagged rather than edited: both files were out of scope.

### Task 4, the 9000-step rerun — a plateau, not a floor (2026-08-24)

Result: [`docs/measurements/reach-dial-9000.json`](docs/measurements/reach-dial-9000.json).
Same eval, same 1,000 held-out scenes, same locally-defined thresholds; one variable, the
checkpoint.

**Both pre-declared questions came back negative, and that made the earlier result stronger.**
The dial is unchanged at 3x the training budget: residualised `near<far` went **+0.060438 →
+0.060392, a change of −0.000046**, and every one of the six step-deltas moved by less than
0.001. The `add` adherence gate **still fails** (shortfall 0.0896 → 0.1062, worst setting still
`dial:near`, `far` still above `near`). So **undertraining is refuted** as the explanation for
that gate, and my own claim that the 3000-step effects were "a FLOOR, measured below the
model's trained ceiling" is **falsified** — they are a plateau. Corrected in the report rather
than left standing.

**When two runs agree to four decimal places, suspect the instrument first.** My first move was
not to write the finding up but to ask whether I had evaluated the same checkpoint twice —
a wrong `--arm-root`, a stale HF work-dir, a copied store all produce exactly this. Measured
from the stored text: **38.8% of the 7,000 continuations differ**. That check is now a
permanent field (`generation_churn`), because a near-zero churn beside "stable effects" would
have been a fabricated finding, and the provenance strings would have looked perfect either way.

**"Not converged" was the wrong frame, and it was mine.** The learning rate is CONSTANT 1e-5
with no decay and no warmup. A constant-LR run cannot converge; it asymptotes, and it keeps
producing small monotone val improvements indefinitely because the step size never anneals. So
"val loss still falling" and "early stopping never fired" are what the RECIPE does, not evidence
that the budget binds — 3x the steps bought 12.7% val improvement and zero effect change.
`convergence_framing()` computes this from the manifests now. The consequence: a null here is no
longer "undertrained", it is "this recipe has plateaued in practice".

**A default path can destroy a published measurement, silently, with every test green.** Reusing
the default `--out` for the longer-trained rerun would have overwritten the reviewed 3000-step
artifact that `scripts/reach.py` quotes. `default_paths()` derives a suffixed path per
(arm_root, step), `main` refuses to overwrite an artifact whose recorded step differs, and both
are tested. The step/manifest provenance check was written in `main()` first and a mutation that
deleted it survived — driver-only guards are unreachable from tests, the same hole as round 2's
composer mutants. Extracted and tested.

### Task 7, the `add` slot — a POS filter that would have worked for the wrong reason (2026-08-24)

Report: `.superpowers/sdd/2026-08-23-reach-dial/task-7-report.md`. New artifact
`artifacts/reach-content/` (24,376 skits); `artifacts/reach-skits/` untouched.

The reach dial was statistically fine and read as nothing, because `add` is the highest-IDF
*fresh* word and a TinyStories turn is three words long — so the slot's 25 commonest values
(18.5% of it) were `look please hi hello love what wow why come yes thank ok okay …`. Now they
are `look want love come give go doing friend need stop name help see try …`.

**The filter I was about to ship would have been right by accident.** Tagged as written, the
particles pile onto `NNP` — `hi` 120/120, `hello` 118/120, `wow` 115/120 — because they open a
quoted utterance and are therefore *capitalised*. An NN/NNS-excluding-NNP filter scores well on
this corpus and inverts the first time a particle appears mid-utterance. Worse, the same
artifact tags `look` NNP 93/120 and `come` 84/120, and those are verbs that must be **kept**:
the right answer and the wrong one came out of the same accident. `pos_tags_in_turn` lowercases
every token before tagging, which removes the cue in both directions — and doing that
**destroys the POS signal entirely**: `hi`, `hello`, `wow` all become `NN`, indistinguishable
from `comet`. Which is the honest finding. *Part of speech cannot separate a greeting from a
noun*, because `hello` in "say hello" genuinely is one. Grammar was never the axis; **where the
word lives** is. `narration_rate` — the fraction of a word's corpus occurrences outside
quotation marks — separates `give .584 come .509 look .503 love .293` from `okay .237 thank .182
hi .070 please .066 wow .033 let's .014`, and the POS gate is demoted to what it is actually
good at: rejecting adjectives, per instance.

**Isolated-word tagging carries no signal at all here** — `nltk.pos_tag([w])` returns `NN` for
every one of `please wow okay thank hey comet hello look love dragon kite hi`. The brief
reported a *mixed* set of wrong tags; I could not reproduce that and the difference matters,
because "backwards" and "uninformative" imply different fallbacks.

**A threshold chosen on the thing you then measure is not measured.** `NARRATION_FLOOR = 0.20`
was fitted against the brief's own 25-word ground truth; precision/recall are reported against
250 observations hand-labelled *before the classifier ran* (0.9493 / 0.9424, majority floor
0.556). Both sets are in the manifest, and the top-25 one is labelled **FIT, not a test**.

**Report which gate is SOLELY responsible, not which fired.** Five gates, and `content_add_reasons`
returns all of them rather than the first, precisely so the manifest can say that `clitic` fired
4,384 times and was solely responsible **zero** times — redundant on this corpus, kept as a rule
only because it generalises. The narration floor's 1,013 sole rejections are the real payoff
(`ow`, `mmm`, `yuck`, `sir`, `dear`, `whee`) *and* its real cost in the same column (`belongs`,
`deserve` — verbs the narrator never says).

**The fix made the confound worse, and that is the headline finding, not a footnote.**
`spearman(add_df, distance)` +0.2078 → **+0.2334**; `far`'s median `add_df` +32%. Removing the
particles promoted the *common verbs* that were second choice in the same turns, so the slot got
**more** concentrated (top-25 share 0.1846 → 0.2181, distinct 6,442 → 4,846). A slot can read
better and measure worse at the same time.

**A fixture of `comet` vs `hello` cannot test this classifier** — every candidate design,
including two provably wrong ones, gets that pair right. The parametrisations are built on
`look/love/come/want/give/doing/mean` and on real turns from the artifact, and the
`SpeechProfile` fixtures carry the **real measured corpus counts**, because the floor's
justification is where real words fall relative to it. Eleven mutants, each reverted: the ones
that matter are `choose_add_word` ignoring its filter (4 red), `pos_tags_in_turn` no longer
lowercasing (6 red, including `come`/`give` wrongly rejected), and a content run **defaulting
onto `artifacts/reach-skits/`** — which is why `resolve_out_path` is a named function instead of
two lines in `main`. The artifact test is the satisfying one: pointed at the old artifact it
names **311 distinct particles** in the `add` slot; pointed at the new one the set is empty.

## 2026-08-23/24 — the reach dial: a real control, on a corpus with nothing to reach for

**Shelved after this entry.** The dial works, it is small, and the binding constraint turned out
to be the corpus rather than the model. Suite **1655 passed, 2 skipped**.

**The bet.** Stage 2's one publishable slot was `add`, and the reason was mechanical: its value is
absent from the visible context ~40-44% of the time against **92.8% for `accept`**, which is mostly
the protagonist's name. `add` is where the model names something not in front of it and then uses
it. So: make that reach *controllable*. A `reach` slot — `near`/`mid`/`far` — derived as the tercile
of the NPMI distance between the `add` word and what is already in play, rendered **before** `add`
in the block because the block generates left to right and a dial declared after the word it
governs cannot govern it.

**It works.** Forcing the dial moves realised distance monotonically, scene-paired over 826
held-out scenes, surviving frequency control: residualised `near`<`far` **+0.0604 (t 12.5)**, with
~53% of the raw effect attributable to word frequency and ~47% surviving. A control arm that never
saw the slot shows nothing. **The pre-declared EUREKA criterion was not met** on one gate — the
`add`-hit rate — and I declined to loosen it, having earlier accepted an implementer's refusal to
*tighten* a pre-declared gate after seeing data. That constraint is symmetric or it is worthless.

**Three eliminations, each costing real compute.** More steps: at 3× the budget the effect moved
0.08% — a plateau, not a floor, which **falsified my own published claim**. Undertraining: the
adherence gate got *worse* at 9000 steps, refuting it. Vocabulary: a validated content-word filter
(precision 0.949) removed every discourse particle and made the slot **more** concentrated with a
**worse** frequency confound, because the particle mass collapsed onto common verbs. A slot can read
better and measure worse.

**The corpus is the ceiling.** TinyStories is 13,777 distinct words with the top 1,000 covering
90.9% of tokens. `cathedral` 0, `submarine` 0, `meteor` 1. `far` collapsing to 88 distinct words
against `near`'s 265 is a model faithfully reflecting a world with a thousand usable words in it.

**Instrument problems, five of them, each found before it reached a conclusion.** NPMI's zero
conflates "unrelated" with "never co-occurred". A story is a document of its own association table,
so the zero-evidence rate collapsed to a structural 0.0000 until leave-one-out fixed it. The
frequency confound runs *opposite* to the obvious worry — NPMI rewards rare pairs, so `far` selects
*commoner* words. The df-matched "second control" was a **null-enrichment filter**: it retained 100%
of scenes where the dial did nothing and ~7% where it changed the word, so its +0.0140 was a mean
over a sample 78% definitional zeros — read correctly, the two controls **agree**. And POS tagging
on isolated words is uninformative (everything returns `NN`); my own probe reported a `NNS`/`VBD`/`JJ`
mix only because I passed all twelve words in one call and nltk tagged them as a sentence.

**Twelve hollow tests now.** The eleventh and twelfth were the same shape and the shape is no longer
a surprise: whole *composition* functions — `analyse`, `build_observations`, `per_setting_table`, a
step/manifest guard — imported by no test, where every leaf underneath was fixture-tested. Three
independent mutants each rewrote a published claim while all 1,505 tests passed, one flipping
`eureka_criterion_met` from false to **true**. A leaf-only suite has failed to catch this every
single time. The next spec should test composition by default rather than treat each instance as an
oversight.

**What is worth keeping.** A controllable-generation mechanism that measurably works on a 123M model
with a null control. Three eliminated hypotheses with evidence. And a measurement apparatus better
than the result it measured: `scripts/eval_reach.py --rescore-from` re-derives every published number
from stored generations with no model, tokenizer, or device — verified byte-identical under an import
blocker that raises on `torch`/`transformers`/`ttnn`/`ttml`, with gold distances at max absolute
error 0.0.

**Unfinished.** A noun-preferring rank key — the only untested reason `far` picks `look` over
`comet` — was started and stopped when this was shelved. 93.4% of model turns offer 2+ candidates,
so it would have acted nearly always; `comet`/`volcano`/`dragon` tag `NN` and survive the gate,
while `unicorn`/`teddy` tag `JJ` and are rejected. Resume there if the corpus changes.

## 2026-08-27 — 4-chip serving fixed; a story-writing toolchain finds two real limits

**4-chip serving works.** Root cause of the `TT_FATAL: Could not find any forwarding direction
from src (M0, D0) to dst (M0, D3)` from the earlier 4-chip investigation: `FABRIC_1D_RING`
maps to `Topology::Ring`, which `is_2D_topology()` excludes -- so it silently takes the same
single-hop-only routing branch as plain `FABRIC_1D` (the tracked upstream `#22524` limitation),
regardless of the ring flag. `FABRIC_2D_TORUS_XY` maps to `Topology::Torus` (genuinely 2D) and
works: `multidevice with 4 devices and grid (1, 4) is created`, real generation. Two of my own
earlier "no difference" conclusions about `FABRIC_1D_RING`/`FABRIC_2D` were wrong for a boring
reason -- `--additional-config` needs the JSON nested under a `"tt"` key
(`vllm_tt_plugin/config.py:get_tt_config()`), and I'd sent it flat, so it silently no-op'd both
times. Full account, including the dead ends: `AUTODEBUG.md`, `AUTOFIX.md`,
`docs/serving-with-tt-kernel.md` §8.

**But 4-chip output is measurably worse than 2-chip, on the exact same prompts.** A controlled
A/B (`scripts/story_tools.py`, same prompt, same slot, same sampling settings) found 4-chip
producing invented non-words ("Tryburg", "Alexandary", "Higheriq") across most candidates, where
2-chip's candidates were grammatically rough but recognizable English throughout. This is a real
quality regression in `FABRIC_2D_TORUS_XY`'s collective ops (numerical correctness is the leading
suspect, unconfirmed) -- not this checkpoint's ordinary ceiling. Documented as a "do not use for
anything quality-sensitive yet" caveat in the serving doc. 2-chip remains the config to trust.

**A tool-calling storytelling harness (`scripts/story_tools.py`) surfaced two more real limits,
both worth keeping.** Built to drive `tt-tnt-1024-dialogue` one short, judged turn at a time --
the turn structure deliberately reuses the *skits* checkpoints' five-slot schema
(`offer/accept/add/stakes/handback`, `train/skit.py`) as a prompting-and-judging discipline, not
as something this checkpoint was trained on. Two findings:

1. **Chaining short restarts does not obviously beat one long unguided generation.** 24
   candidates across three parameter settings for a single `accept` turn came back almost
   uniformly broken (`"she had"`/`"one day"`/`"alone"` filler loops). Each fresh short call
   re-rolls this model's fragile sentence-onset behavior from a cold start; a long generation
   only has to survive that onset once. The earlier-looking "success" of long unguided
   completions may partly be exactly this -- fewer rolls of the same weak die, not better
   underlying quality.
2. **Prompting the model to "edit" its own draft turn does nothing.** Framed explicitly as an
   editing task (`self_edit()` in `story_tools.py`: "Draft: ... / Better version:"), the same
   base model produced *worse*, unrelated fragments, not a cleaned-up draft. tt-tnt-1024-dialogue
   was never trained to critique or revise text, and prompting alone does not manufacture that
   capability.

**Follow-up worth funding, not just noting: train the model to actually be an editor.** Given
(2)'s clean negative result, the honest next lever isn't a better prompt -- it's a training
objective. A concrete shape: fine-tune (or continue-train) on paired (draft, better) text where
"better" is a real edit -- grammar-fixed, de-repeated, or otherwise improved -- so the model
learns "revise this" as a task the way `tt-tnt-1024-dialogue`'s thin 2% Q&A slice taught it
"answer this." This is also where the standing "skits and dialogue aren't orthogonal" goal
connects: an editor objective, a dialogue objective, and the skits turn-structure objective are
three lenses on the same underlying question -- can this model do something with text beyond
raw next-token continuation -- and a future checkpoint that trains on more than one of them
together is the actual integration point, not three permanently-separate checkpoints.

## `feat/editor-training` Task 5 — the real run, and a shape bug the corpus's own variety exposed (2026-08-27)

Following through on "skits and dialogue aren't orthogonal": `scripts/train_editor.py`
(subagent-driven, Tasks 1-4) blends editor pairs (`train/corrupt.py`'s four corruptors over
real corpus sentences), poetry-instructions pairs (a from-scratch dataset, HF's
`isaacrehg/poetry-instructions` being unusable on licensing grounds), and the existing skits
corpus into one continued-training stage, warm-started from `tt-tnt-1024-dialogue`.

**First attempt crashed at step 75/3000** with a real device-side `TT_FATAL` inside ttml's
SDPA backward kernel: `u_scaler shape mismatch: expected (*, 63360, 32), got Shape([1, 1,
65536, 32])`. Root cause: `sft_collate_fn` pads each batch only to that batch's own longest
example (capped at `MAX_SEQ_LEN=512`), so batch shape varies step to step on this dataset's
wide length distribution (min 12, median 473, max 512 tokens — far more varied than the
skits/dialogue corpora that exercised this same collate function without ever tripping this).
ttml sizes an internal backward-pass scratch tensor from the *first* batch's shape and asserts
rather than resizing when a later batch is larger — confirmed by the direction of the mismatch
(expected/cached value smaller than the actual one). Fixed on our side, no tt-metal edit: pad
every example to the fixed `MAX_SEQ_LEN` cap *before* it reaches the dataloader
(`_pad_to_max_seq_len`), so every batch is exactly `(batch_size, MAX_SEQ_LEN)` from step 1
onward and no later batch can ever exceed the first one's shape. Two tests added; the CPU-only
`--dry-run` reproduces Task 4's exact committed example-construction numbers unchanged — the
fix touches only what reaches the device.

**The real run**, `--steps 3000 --save-every 1000`, one leased Blackhole chip, `(1,1)` mesh
only: **9m11s wall clock**, ~5.5–5.6 it/s steady state. Both pre-flight/post-flight guards
(reused from `scripts/train_skits.py`) passed: `stochastic_rounding` read back `True` from the
optimizer state before training started, and 17 RMSNorm gammas moved by 32.7% of their
elements (5,695/17,408, max|Δ| up to 0.0234) against the warm-start base — not frozen, not
degenerate. Train loss (noisy, mixed editor/poetry/skit per-batch) fell from **7.156 → 3.688**
across the run; the real signal is validation loss (256 held-out examples, evaluated every 250
steps), which fell smoothly and monotonically at every checkpoint: **1.599 → 1.597 → 1.596 →
1.591 → 1.589 → 1.587 → 1.583 → 1.583 → 1.580 → 1.578 → 1.576 → 1.574** (steps 250 through
3000). Three checkpoints written (`step_1000.pkl`, `step_2000.pkl`, `step_3000.pkl`,
491,857,839 bytes each — an `SFTTrainer`-format `{"step","model_state"}` pickle, not the ttml
multi-record format `train/run.py` writes) under `artifacts/checkpoints-1024-editor/`
(gitignored). `artifacts/checkpoints-1024-dialogue/` was not touched.

Not yet evaluated for the thing that actually matters — whether `step_3000.pkl` can edit a
draft turn better than the clean negative result `story_tools.py::self_edit()` found on the
un-trained base checkpoint, and whether it regresses base/dialogue capability. That is Task 6.

## `feat/editor-training` Task 6 — the before/after evaluation, and a real regression (2026-08-27)

Converted `step_3000.pkl` to `artifacts/hf-tt-tnt-1024-editor` (CPU-only, `sft_checkpoint_to_hf`)
and ran the three checks Task 6's plan asked for. Two are on-device (vLLM/ttnn); one is
CPU-only (`scripts/evaluate.py`, transformers, no device). **Verdict: the editor objective did
not fix the negative result, and introduced a real, confirmed regression on repetition and
termination.**

**Serving hit two real bugs on the way, both fixed; a third, pre-existing one was found and
deliberately not chased further.** (1) `gozer run --chips all` auto-exports
`TT_VISIBLE_DEVICES`, which `docs/serving-with-tt-kernel.md` §4.2 already documents as
breaking single-device serving — fixed by wrapping the actual server invocation in
`env -u TT_VISIBLE_DEVICES`. (2) `scripts/eval_editor.py`'s held-out draft/better pairs are
raw, uncapped real corpus sentences (unlike training examples, which are truncated to
`MAX_SEQ_LEN`); one long draft crashed the *entire* vLLM engine (a fatal `AssertionError`, not
a per-request error) — fixed with a client-side tokenizer length guard before any request is
sent (commit `e837f4d`). (3) A genuinely new crash surfaced after ~18 sequential, *independent*
single-turn completion requests (`AssertionError: Sequence length 1024 exceeds max seq len
512`, killing the engine) — every individual prompt measured well under 200 tokens, so this is
position/slot state carrying over across unrelated requests, not one long request. This
matches §7's already-standing, already-investigated (nine hypotheses refuted) "on-device
generation is degenerate" limitation, specifically the "position handling per decode step"
surface it names as still worth attacking. `--max-num-batched-tokens 512` and
`--no-enable-prefix-caching` changed the failure point but did not fix it. Recorded as new
evidence in §7, not chased further — a pre-existing, already-extensively-investigated serving
defect is not this task's job to root-cause.

**A whole-branch review caught a real spec gap this write-up first missed: the run never
implements spec §4's mandated base-blend anti-forgetting slice.** §4 calls for four sources
per step — a *majority* base-blend refresh (resampled from the existing nine-source
`artifacts/corpus/blend.txt`, at the settled shares, specifically to prevent forgetting) plus
minority editor/skits shares. `build_combined_examples` has only three: editor pairs (22.8% of
the realized mix), skits (21.6%), and poetry-instructions — which is not in the spec at all and
ended up the *majority*, 56.0%. There is no base-blend slice at 0%, no `--blend` argument, and
no ledger ruling that deliberately dropped it — the plan's own Self-Review claims §4 coverage
that was never implemented. Combined with the plan's `--resume` → `--warm-start` switch (a
separate, documented, and independently fine decision), the run is 3000 steps of pure
fresh-optimizer SFT on 100% task data with zero base-corpus refresh — exactly the failure mode
spec §Risks names by name ("too large an editor/skits share risks displacing base fluency").
This is the leading mechanistic explanation for the regression measured below, not merely "the
mixing ratio" as an earlier draft of this entry put it, and it means the editor *objective*
itself has not actually been tested under the design the spec called for.

**On-device checks, bounded by that crash.** Held-out corruption recovery
(`docs/measurements/editor-eval.json`, n=15 — kept below the ~18-request crash threshold
rather than force a larger n through repeated server restarts against a decode path already
documented as unreliable): mean word-overlap 0.463 — and, once the review caught that the
check never computed the comparison the spec and this module's own docstring actually specify
("closer to real English than the draft was"), the draft baseline itself: mean draft-vs-better
overlap **0.859**. Edited output is closer to target than the uncorrected draft on **0 of 15**
pairs, and worse on 12 of 15. Fake-word rate 0.400. Re-running
`story_tools.py::self_edit()` on the exact 2026-08-27 negative-result draft ("The girl wished
she had no one night and she was always had been very special.") against the new checkpoint:
all 3 candidates still flagged `internal_repeat` — the same failure shape as before training.
Given §7's decode defect predates and is independent of this training work, on-device
generation quality cannot currently distinguish "the editor objective didn't help" from "the
serving path can't show it either" for a checkpoint this size.

**The CPU-only comparison is not confounded by that defect, and it is unambiguous.**
`scripts/evaluate.py --model hf-tt-tnt-1024-editor --against hf-tt-tnt-1024-dialogue
--skip-trajectory` runs entirely through `transformers` on CPU — no ttnn, no device — so its
result is a real behavioural comparison, not a serving artifact. Matched-window loss improved
slightly (2.7726 → 2.7092, −0.0634; no seed floor exists for this instrument, so read
alongside the signals below rather than alone). Three of ten behavioural signals clear both
the seed floor and their own paired interval as real, confirmed regressions: **termination
rate worse** (2.14× the floor), **4-gram repeat rate much worse** (21.76× the floor), **longest
repeated span much worse** (30.17× the floor). Five signals are `NOT INTERPRETABLE` (at or
below the 1.2× floor rule); `prompt engagement` clears the floor but not its own paired
minimum-detectable difference. Full table: `docs/measurements/evaluation-tt-tnt-1024-dialogue-
vs-tt-tnt-1024-editor.md`.

**One correction to the earlier draft of this entry.** The training run's own validation
curve (1.599 → 1.574 across the run, quoted in Task 5's entry above as "the real signal") is
**not** a signal for the editor or poetry objectives at all. `train_editor.py` holds out the
literal tail of a list concatenated as editor → poetry → skits; because the skits block
(18,610 examples) is larger than the 256-example val slice, every held-out example is a skit.
The number is real and correctly computed — it is just a skits-only validation curve, not
evidence about editor or poetry training quality, the way `train_skits.py`'s own tail-split
precedent assumed a *homogeneous* list where the tail is representative.

**Read plainly: this run made the model repeat itself more and terminate less, while
delivering only a small, floor-less loss improvement and no confirmed editor-capability gain.**
The intended objective (fix `self_edit()`'s negative result) did not measurably succeed on the
one check that could show it — now confirmed by a real comparison, not just an absent gain, per
the corrected numbers above — and the one check clean of the serving confound shows a real
cost. The leading candidate cause is no longer a shrug: the missing base-blend slice (above)
means this run trained on 100% task data with no anti-forgetting refresh, which is exactly the
failure mode the spec's own risk section predicted for too large a task-data share. A follow-up
run implementing that missing slice, before drawing any further conclusion about the editor
objective itself, is the obvious next step — not "try a different mixing ratio blind."
`tt-tnt-1024-editor` is a documented negative result, not a checkpoint to promote.

**Two more caveats from the same review, both real, neither changing the verdict.** (1)
`build_editor_pairs.py`'s `better` targets are corpus *lines*, not sentences — measured on the
shipped `artifacts/editor-pairs/pairs.jsonl`: only 57.4% end in terminal punctuation, 33.6% do
not start with a capital letter, 8.8% are under 4 words (hard-wrapped Gutenberg fragments and
wiki table rows like `"Sparsbach (67475)"`). The module's docstring claim that `better` is
"guaranteed grammatical English by construction" does not hold at the line level, and training
on lowercase/unterminated fragments is a second plausible contributor to the measured
repetition/termination regression, on top of the missing base-blend slice above. (2) The
`self_edit()` before/after comparison spans a detector change: commit `73b6f53` (recovered
earlier this session) fixed `story_tools.py::score_candidates` to exclude stopwords from its
repeat count, so the "before" negative result (pre-training) and the "after" result quoted
above were scored by two different, though closely related, detectors. The direction is
conservative here — the stricter post-fix detector is the one that still flags the editor
checkpoint's output, so the conclusion holds — but the two runs were not scored identically.

## The base-blend follow-up's first attempt trained a second, worse-broken checkpoint (2026-08-27)

Implemented `scripts/train_editor.py`'s missing base-blend anti-forgetting slice (spec sec.4)
-- `build_base_blend_example`/`sample_base_blend_examples` sample random windows directly
from `artifacts/tokens-v3/train_ids.npy` (the pre-tokenized stream tt-tnt-1024-dialogue
itself trained on, so no re-tokenization or share computation needed), sized at
`--base-blend-ratio 0.6` (a majority, per spec) against the existing editor+poetry+skits mix,
plus a real fix for the val-split bug the final review had only documented before (it was
100% skits): `stratified_split` now holds out an even share from every category's own tail.
Real dry-run: base_blend=129,043 (60.0%), total 215,072 examples (was 86,029).

**The first real training run on this code produced a catastrophically broken checkpoint,
not an improvement.** Ran clean (3000 steps, ~10 min, gammas moved 34.6%, both quality
guards passed) and looked fine by every SIGNAL the run itself prints -- but the CPU-only,
device-independent no-regression check against `tt-tnt-1024-dialogue` was unambiguous:
**matched-window held-out loss went from 2.7726 to 15.0296 nats** -- worse than the
`ln(32000) ≈ 10.37` uniform-random ceiling -- alongside `4-gram repeat rate` at **420.29x**
the seed floor and `longest repeated span` at **849.98x**, both far beyond even the first
(already-regressed) run's 21.76x/30.17x. Direct generation confirmed it: an ordinary
TinyStories-style prompt ("Once upon a time, there was a little...") still produced fluent,
coherent text, but every one of the six "spine"-register frozen prompts tested collapsed
into repeating a single word from the very first generated token --
`"to to to to..."`/`",,,,..."`/`"but but but..."`/`"one one one..."`/`"and and and..."`/
`"that that that..."`, a different word each time, always immediate and total. This is not
"didn't help" -- it is a real, severe, reproducible break, and the CPU-only instrument caught
it before any hardware time was spent trying to serve or further evaluate the checkpoint.

**Root cause: a labels-shift bug in the base-blend example builder, the same bug class the
2026-08-20 improv-thinking entry already documented once, in a different function.**
`build_base_blend_example` shipped as `labels = list(ids)` (unshifted, HF-style) --
identical to `input_ids`. ttml's own convention, which `build_editor_example`/
`build_poetry_example` already implement correctly a few functions above it in the SAME
file, is that `labels[t]` is the token that should be predicted GIVEN inputs up to and
including position `t`, i.e. `labels[t] == input_ids[t+1]`, not `input_ids[t]`. Training a
causal transformer to predict "the token that is already at this exact position" from a
hidden state that was computed FROM that same token is a trivial, causally self-referential
shortcut -- and at a 60% majority share, that shortcut competing against real next-token
prediction on the same shared weights is sufficient to catastrophically corrupt the model's
actual language-modeling ability, which is exactly what the 15.03-nat held-out loss and the
prompt-collapse pattern show.

**Fixed.** `sample_base_blend_examples` now fetches windows of length `seq_len + 1` (one
extra token) and `build_base_blend_example` splits them as `input_ids = window[:-1]`,
`labels = window[1:]` -- correctly shifted, still full supervision, no `-100` anywhere. Two
tests that had asserted the WRONG (unshifted) behavior as correct were rewritten to check the
shift explicitly (`labels[t] == input_ids[t+1]` for every `t`, and a case where
`input_ids != labels` on the same window) -- the old tests would have passed against either
implementation, which is exactly the "hollow test" pattern this project has hit many times
before: a test that only checks shape/type, never the actual semantic content the bug lived
in. `artifacts/checkpoints-1024-editor-blend` (the broken run) and
`artifacts/hf-tt-tnt-1024-editor-blend` are left on disk as the broken-run artifact, not
promoted, not registered as a candidate to build on -- the corrected run gets a fresh
`--out-root`. Real hardware re-run with the fix is the next step.

## The corrected base-blend run: a real, partial fix (2026-08-27)

Re-ran with the labels-shift fix, `artifacts/checkpoints-1024-editor-blend-v2`, otherwise
identical settings (3000 steps, `--base-blend-ratio 0.6`). Both quality guards passed
(gammas moved 31.2%, not frozen), val loss fell smoothly 2.637 -> 2.390. Generation on the
exact "spine" prompts that catastrophically collapsed in the broken run now produces real,
grammatical sentences again -- the specific failure mode (immediate single-word repetition
from token one) is gone, confirmed by direct generation before trusting any aggregate number.

**The three-way comparison against `tt-tnt-1024-dialogue`, all from the same CPU-only,
device-independent instrument (`scripts/evaluate.py`, matched 512-token window, prompt set
B):**

| signal | run 1 (no base-blend) | run 2 (broken, unshifted labels) | run 3 (fixed) |
|---|---:|---:|---:|
| matched-window loss delta | -0.0634 | **+12.2570** | -0.0730 |
| termination rate | worse, 2.14x floor | worse, 3.19x floor | **NOT INTERPRETABLE, 0.10x** |
| 4-gram repeat rate | worse, 21.76x floor | worse, 420.29x floor | **better, 3.00x floor** |
| longest repeated span | worse, 30.17x floor | worse, 849.98x floor | below paired detection, 4.13x |
| prompt engagement | below paired detection, 1.88x | worse, 26.31x floor | **worse, 3.87x floor** |

**Read plainly: the base-blend slice, correctly implemented, fixed the two most severe
regressions from the first run outright.** Termination rate is no longer distinguishable
from the dialogue checkpoint at all (was a confirmed regression at 2.14x the floor);
4-gram repeat rate is now measurably *better* than the dialogue checkpoint (a real,
floor-clearing finding in the right direction, not just "no longer worse"); longest
repeated span no longer clears its own paired-detection threshold. This is the design
spec's anti-forgetting hypothesis holding up under a real, corrected test -- the first
run's regression really was substantially the missing base-blend slice, not an inherent
property of the editor/poetry/skits objectives themselves.

**Not a clean sweep.** `prompt engagement` is a new, real, floor-clearing regression
(3.87x) that wasn't confirmed in run 1 (only "below paired detection" there, 1.88x) --
smaller than what it fixed, but real, and worth naming rather than glossing over. And this
comparison does not re-verify the actual editor/self-edit capability the whole objective
was for -- `self_edit()` and the held-out corruption-recovery check were not re-run against
this corrected checkpoint (the earlier `--n=15` bound and the section-7 on-device decode
defect both still apply if that's attempted). What's confirmed here is CPU-side behavioural
non-regression-turned-improvement on the axes that broke in run 1, not that the editor
objective itself now measurably works.

**Registered as a candidate (`tt-tnt-1024-editor-blend-v2`), not promoted to `current`.**
Promotion is a deliberate, evidence-cited decision in this project's convention, and the
evidence here is incomplete in one specific way: the editor capability itself (the thing
the whole `feat/editor-training` line exists for) has not been re-verified on this
checkpoint. The right next step, if this line continues, is the on-device editor-specific
checks -- not a promotion decision made on the CPU-only regression numbers alone.

## The on-device editor checks, run against the corrected checkpoint: still a negative result

Served `artifacts/hf-tt-tnt-1024-editor-blend-v2` (2-chip, `TT_VISIBLE_DEVICES` unset,
`--max-num-batched-tokens 512 --no-enable-prefix-caching` -- the same recipe already
worked out for `tt-tnt-1024-editor`) and ran the same two on-device checks.

**Held-out corruption recovery** (`docs/measurements/editor-eval-blend-v2.json`, n=15,
same bound as before): mean word-overlap **0.318** against the same 0.859 draft baseline
-- **0/15 improved, 13/15 worse** (was 0/15 improved, 12/15 worse on `tt-tnt-1024-editor`).
Fake-word rate 0.467 (was 0.400). Flat-to-slightly-worse, not better.

**`story_tools.py::self_edit()`** on the identical 2026-08-27 negative-result draft: all 3
candidates still flagged `internal_repeat`, and the actual text reads if anything more
broken than before -- e.g. `"The girl had a girl thought she wished she wished she wished
she wished she had a friend: The girl had a friendles: The girl had to."` -- word-salad
repetition rather than the earlier run's more legible (still looping) sentences.

**Read plainly: the base-blend fix repaired the general-fluency regression it was meant to
fix, but it did not improve -- and if anything slightly worsened -- the actual editor
capability.** This separates two things that were previously confounded in one number: the
CPU-only comparison in the section above shows the checkpoint stopped actively damaging
base behaviour (termination, repetition), which is a real and useful result on its own. But
the editor objective itself -- teaching the model to take a garbled draft and produce a
better version -- has not worked in either the original run or this corrected one. Adding a
majority anti-forgetting slice was necessary to stop the regression; it was never going to
be sufficient to create a capability the training signal itself hasn't demonstrated it can
teach at this scale (19,593 editor pairs, now a smaller relative share of a 215K-example
mix). `tt-tnt-1024-editor-blend-v2`'s `docs/current_model.json` note is unchanged by this
(still correctly "not promoted") -- this closes the open question it left, rather than
reversing its verdict.

## LoRA for the editor objective: blocked by a real upstream bug, no training run attempted (2026-08-27)

LoRA is structurally a different lever than the base-blend slice for the SAME anti-forgetting
problem: it freezes 100% of the base weights, so general fluency cannot regress by construction,
regardless of the task-data share. `ttml.modules.lora` ships a complete implementation (`LoraLinear`,
`LoraColumnParallelLinear`/`LoraRowParallelLinear`, `LoraModel`, plus a native `SFTTrainer(...,
peft_config=LoraConfig(...))` integration) -- no third-party tool needed for this.

**A real naming trap, found and fixed before writing training-critical code.** `LoraModel` does not
override `.parameters()`, so calling it on the wrapper returns names walked from the wrapper's own
root (`LoraModel/model/blocks/N/...`), not the canonical `llama/llama_block_N/...` this project's
checkpoint format and HF conversion expect. `TtTntLoraModel`, a two-line subclass delegating
`.parameters()` to the inner model, fixes this -- the same pattern `train.model.TtTntLlama` already
uses for its own name translation, confirmed correct by direct inspection (printing both the wrapper's
and the inner model's parameter names on real hardware before trusting either).

**The training run never happened, because five independent hardware diagnostics proved it would
have wasted the whole budget.** `scripts/train_editor_lora.py`, built and dry-run tested, was ready
to launch a real 3000-step run when a first attempt (with the naming fix) trained for the full 3000
steps and reported `lora_B` moved from zero: 0/24 -- every single LoRA parameter stayed at EXACTLY
its initial value across the entire run, despite a normal-looking loss curve. Rather than assume a
bug in my own code and iterate blindly, each link in the chain was tested in isolation, on real
hardware, cheapest-first:

1. ttml's own core autograd (frozen input, trainable weight, one `linear` + `backward()`) --
   correctly computes a nonzero weight gradient. Not an autograd bug.
2. A standalone `LoraLinear` (no model, no trainer) -- `lora_B`'s gradient is correctly nonzero;
   `lora_A`'s is correctly zero at step 1 (mathematically expected: `lora_B` inits to all-zeros, so
   `d(loss)/d(lora_A)` is exactly zero until `lora_B` moves away from it). Not a `LoraLinear` bug.
3. The REAL warm-started, LoRA-injected model, manual forward + `backward()` (no `SFTTrainer`) --
   `type(blocks[2].attention.q_linear) is LoraLinear`, it is the SAME object `.parameters()`
   returns, and it produces a real nonzero `lora_B` gradient. Injection genuinely reaches the real
   forward pass; not a wiring bug.
4. Manually replicating `SFTTrainer`'s own `zero_grad -> forward -> backward -> clip_grad_norm ->
   optimizer.step()` sequence by hand -- gradient present and nonzero (0.0010) immediately before
   `.step()`. **`lora_B`'s value is bit-identical before and after `.step()`.**
5. `SFTTrainer(model=model, peft_config=LoraConfig(...))` -- the exact, documented, un-customized
   usage from `tt-train/docs/SFT_TRAINER.md`'s own LoRA section, bypassing my `TtTntLoraModel`
   entirely -- same result: `max|delta| == 0.0`.

**This isolates the defect to `AdamW.step()` itself, inside `ttml`'s compiled C++ -- something this
repo does not build or patch.** `sft_trainer.py`'s own source independently corroborates that this
path is unfinished: its checkpoint-save docstring already has a `TODO: filter... to save only LoRA
adapter parameters`, evidence nobody has exercised `peft_config` end-to-end through a real optimizer
step before. Full writeup, every measurement, and the leading (unconfirmed) hypothesis about the
exact C++ mechanism: `docs/upstream-tt-metal-asks.md` entry 5.

**No tt-metal issue filed, per this project's standing "document, don't file yet" instruction.**
`scripts/train_editor_lora.py` is kept, clearly marked blocked in its own module docstring -- the
data-loading, category-building, and stratified-split logic it reuses from `scripts/train_editor.py`
needs no rework once the upstream fix lands; only the actual parameter-update path is broken. Five
cheap diagnostics (seconds to low minutes of real device time each) is what made it possible to
prove this BEFORE gambling a 3000-step run on an untested assumption, rather than after.

## Growing-conversation KV-cache crash: confirmed upstream, hardened, then the variant sprawl cleaned up (2026-08-28/29)

**The bug.** Serving `episod/tt-tnt-1024` through the TT vLLM plugin crashed reliably a few
turns into any growing multi-turn conversation (Open WebUI chat, or turn-by-turn skit
generation), with `AssertionError: Sequence length {seq_len} exceeds max seq len {mat_len}` --
the whole `EngineCore` dying (`EngineDeadError`). Reproduced directly and narrowed by
elimination, each with a controlled experiment on real hardware: NOT simple context overflow
(a genuinely-too-long single prompt gets a clean HTTP 400 from vLLM's own admission check);
NOT request volume (25 independent, non-growing short requests: zero crashes); NOT block-size
accounting (`--block-size 512` vs. default 64: identical failure); NOT concurrent/batched-
prefill slot mixing (`--max-num-seqs 1`, which makes batching structurally impossible: identical
failure). It IS specifically about a growing conversation's accumulated messages -- crashed
deterministically at request #4 with the real prompt only ~100 tokens, `seq_len` always landing
at exactly `2 x mat_len`.

**It is not our model.** Served stock `meta-llama/Llama-3.2-1B-Instruct` -- no custom adapter --
on the identical stack at the same `--max_model_len 512`, ran the identical reproduction:
**identical crash, identical signature.** This is a generic tt-metal/vLLM KV-cache defect.
Real Llama/Qwen deployments never see it because they run at native context (4k-131k tokens),
where an ordinary chat conversation never approaches double that; our checkpoint's honest
512-token context is what exposed a bug that was always there. Full reproduction and the
control experiment: `docs/upstream-tt-metal-asks.md` entry 6.

**Two hardening fixes, both in this repo alone, no tt-metal edit.**
1. **A chat template now ships baked into the model's own `tokenizer_config.json`**
   (`convert/to_hf.py`'s `_CHAT_TEMPLATE`/`MAX_CHAT_TEMPLATE_MESSAGES`), capping rendered
   history to the last 5 messages -- the exact proven-safe boundary from direct reproduction
   (5 messages served fine, 7 crashed). No server needs a `--chat-template` CLI flag pointed
   at an external file anymore; the old throwaway `scratch/minimal_chat_template.jinja` and
   its reference in `scripts/serve_supervisor.py` are gone. A test pins the cap number itself
   as a tripwire (`test_chat_template_cap_is_at_or_below_the_reproduced_failure_point`) so a
   future edit can't silently raise it past the proven-safe point.
2. **`tt-tnt-1024`'s registered context raised 512 -> 2048** (matching `384`'s precedent),
   trained for real: DDP-4-chip, 10,764 steps (same as the original 1024a run for
   comparability), seed 5489, `artifacts/checkpoints-tt-tnt-1024-ctx2048`. First mesh-open
   attempt hit a transient fabric-router-sync timeout (unrelated flaky ethernet handshake);
   the retry trained cleanly, real val loss **2.4504** (train loss 10.63 -> 2.54). At a
   matched 512-token window against the prior checkpoint: **2.7726 -> 2.5408, -0.2318 nats,
   a real improvement**. Behaviourally: every signal in the paired comparison is either at or
   below the 1.2x seed-floor rule or below its own paired minimum-detectable difference --
   a clean null, not a regression, at n=1 seed. Full report:
   `docs/measurements/evaluation-tt-tnt-1024-dialogue-vs-tt-tnt-1024-ctx2048.md`.

**Then: the artifact sprawl this whole project had been quietly accumulating got named and
fixed.** Before this, `artifacts/hf-tt-tnt-1024-*` held a dozen-plus directories --
`-1024a`, `-dialogue`, `-2ep`, `-editor`, `-editor-blend`, `-editor-blend-v2`, `-v077`,
`-ctx2048` -- 4.6GB, several of them documented negative results left sitting alongside the
one actually in use, and `docs/current_model.json` had grown a `supersedes` chain plus a
`candidates` array re-describing every one of them. Direct instruction: stop creating a new
suffixed sibling per idea; there should be ONE directory, always overwritten. Every HF
conversion directory is a cheap, regeneratable derivative of its source `.pkl` checkpoint --
verified each still had one before deleting anything -- so nothing irreplaceable was at risk.
Consolidated to **`artifacts/hf-tt-tnt-1024`** (matching the already-clean, never-suffixed
Hub repo name `episod/tt-tnt-1024`, and the `384` line's own long-established one-fixed-name
convention, `artifacts/hf-tt-tnt-v3`). Reclaimed 3.2GB deleting `-1024a`, `-2ep`, `-editor`,
`-editor-blend`, `-editor-blend-v2`, `-v077`, then also `-dialogue` once its ctx2048
replacement was renamed into the canonical slot. 18 scripts/tests defaulting to the old
`-dialogue` path were bulk-updated; `scripts/publish_to_hub.py`'s `episod/tt-tnt-1024` target
entry updated to the new path and `max_position_embeddings: 2048` (caught, before committing,
that its `hf_dir` must NOT be `None` the way `episod/tt-tnt`'s is -- `None` resolves to the
module-level `HF_DIR` constant, which is the *384 line's* canonical path; the two sizes need
their own fixed paths, not a shared sentinel). `docs/current_model.json`'s `supersedes`/
`candidates` apparatus was removed outright -- historical checkpoints are project history,
recorded here in CLAUDE.md's log, not a growing list this file has to keep in sync with every
artifact directory that ever existed; `tests/test_evaluate.py`'s
`test_the_shipped_designation_file_is_valid_json_with_every_candidate_listed` (which enforced
the very museum being retired) was replaced with a test for the opposite invariant --
no `candidates`, no `supersedes`, just the one required designation. Raw training checkpoints
under `artifacts/checkpoints-*` were left untouched throughout: they are the actual
irreplaceable evidence, not what was asked to change. `docs/model-card-1024.md` updated with
the real context/loss facts and an explicit note that its MoE/die-routing/thinking/skits/
reach-dial sections describe experiments run against the *prior* 512-context checkpoint, not
re-verified against these retrained weights. Full suite: 1631 passed, 5 skipped (all
pre-existing, correctly-guarded skips for now-genuinely-absent optional local artifacts).

## The ctx2048 retrain used the wrong corpus generation, and published a real capability regression (2026-08-29)

Published `episod/tt-tnt-1024` (the ctx2048 checkpoint above), ran a real vLLM throughput
benchmark on hardware, then began the two follow-ups this project's own convention calls
for on any retrain: a seed replicate, and a check that the retrain didn't silently lose the
capability the checkpoint it replaces was designated for. The second one caught a real
mistake in the FIRST one.

**The bug.** The ctx2048 training run used `--tokens-dir artifacts/tokens-v3`. `tokens-v3` is
the corpus generation from *before* the `dialogue` slice (`databricks-dolly-15k`) was added —
`docs/corpus_blend.md` says this explicitly, and it is directly checkable: the dialogue
checkpoint's own header records `corpus_tokens: 391823393`, which matches `tokens-v4`'s
total token count exactly (391,823,393) and does not match `tokens-v3`'s (391,921,555).
`tokens-v3` was the *right* choice for the original `tt-tnt-1024a` run (which predates the
dialogue slice) and the *wrong* one for a checkpoint meant to continue the dialogue lineage —
copying the earlier recipe without checking which corpus generation the checkpoint being
"raised in context" had actually been trained on.

**Verified, not assumed.** `Q: What is the capital of France?\nAnswer:` under greedy decoding
through `artifacts/hf-tt-tnt-1024` (the wrong-corpus checkpoint, CPU-only, no device needed):
`Cantons, Cantons, Cantons, Cantons, ...` — a repetition loop, not merely a wrong-but-fluent
answer. The dialogue checkpoint this replaced answered `Paris` correctly under the identical
prompt and decoding. `Q: What is the capital of Italy?` similarly degrades to a circular
non-answer (`The capital of the Italian region of Italy...`). This is not a subtle
regression: the retrain trained on a corpus that never contained the dialogue slice at all,
so there was nothing for the capability to survive on.

**What was already in flight when this was found, and what happened to it.** A seed
replicate (`--seed 20260815`) had been launched on the same wrong corpus and was ~18 steps in
when the Q&A check came back negative; killed immediately (negligible compute lost) and its
partial checkpoint deleted rather than left to confuse a later run. A corrected run
(`artifacts/checkpoints-tt-tnt-1024-ctx2048-v4corpus`, same seed 5489, `--tokens-dir
artifacts/tokens-v4`, otherwise identical) was launched in its place. The already-published
Hub weights (`episod/tt-tnt-1024`) and the local `artifacts/hf-tt-tnt-1024`/
`artifacts/checkpoints-tt-tnt-1024-ctx2048` were **not** reverted or deleted -- both
`docs/current_model.json` and `docs/model-card-1024.md` were updated in place with an
explicit, dated regression notice instead, so a reader hits the correction before hitting a
stale claim, and the Hub-published copy will be corrected by re-publishing once the
`tokens-v4` retrain passes the same Q&A check that caught this. `docs/current_model.json`'s
`_readme` and `current.qualification` both say, in these words: treat the dialogue capability
as **known false**, not merely unverified, until that lands.

## Tool calling: the mechanism is a clean success, the voice is not (2026-08-29)

**The ask.** "When someone asks me a question, I use the witty response tool. Or the absurdist
tool. Or the factual tool. Or the misunderstood-the-question tool." Real tool calls, not a
mode tag — tools as *formalised roles* carrying structured metadata. Voice brief: Brautigan,
Kerouac, Ginsberg, Stein, Steve Martin, Douglas Adams, Tom Robbins, Borges, Zork, AD&D,
interactive fiction. "We need wildcards. We need boring. We need ingenious. We need I had no
idea it could go there."

**What shipped.** `train/tool_calling.py`: four tools through the hermes protocol already wired
into serving (`factual_response(answer, confidence)`, `witty_response(answer, technique)`,
`absurdist_response(answer, logic)`, `misunderstood_question(interpreted_as, answer)`), 100
hand-authored seeds in that register, plus real Q&A pairs mined from the corpus's own dialogue
slice and mechanically expanded. `scripts/train_tool_calling.py` (SFT, 60% base-blend
anti-forgetting), `scripts/eval_tool_calling.py` (three gates + warm-start control).

**The mechanism works, measured against a control that makes the number mean something:**

| gate | stage 1 | stage 2 | control (warm-start base) |
|---|---:|---:|---:|
| emission | 100.0% | 100.0% | **0.0%** |
| parse | 85.9% | 82.8% | 0.0% |
| schema-valid | 75.0% | 75.0% | 0.0% |
| distinct tools used | 4/4 | 4/4 | 0/4 |

The base never emits a tool call, so the format came from training and not the harness — the
same control shape that made improv-stage1's 98% adherence meaningful. **Unseen questions
scored HIGHER than seen** (78% vs 72% schema-valid), so it learned the format rather than
memorising rows.

**The voice did not arrive, and the first reason was my own design error.** Stage 1's corpus was
100 hand-authored against 800 derived — the good material was **11%** — and the model learned
the mechanical templates almost exclusively, quoting `derive_templated_variants` verbatim
("Officially, … several committees are still reviewing the paperwork"; "You're welcome, that one
was free"). Fixed by repeating the seeds 8x to parity (800/800), with a test pinning the share
in 40–60% so it cannot drift back. That fix **worked at what it targeted**: mechanical-template
phrases fell from **7/8 to 3/8** sampled completions, and the two most distinctive ones went to
**zero** (n=8, so suggestive rather than conclusive).

**But the wit still did not appear** — stage 2's content is incoherent rather than templated.
"The capital of France is the capital of France." Portugal answered "Madrid." confidently. The
one faint echo ("The sky blue is gray, it looks like it's in a maze") is closer to word salad
than to Zork. **The honest conclusion: the tool-calling MECHANISM is a success and the CONTENT
is bounded by the base model.** A 123M model that cannot reliably name the capital of Portugal
cannot construct "Mount Cleverest" or "F is the capital letter of France" — those need the
joke's mechanism understood, not imitated. This is the same shape as this project's own
dialogue-slice finding: a small curated slice buys **form, not knowledge**, and 100 hand-written
examples cannot lift a model into a register its base has no purchase on.

**Not published, not designated.** A PARTIAL with good structure and bad content is not a
checkpoint to promote. Kept as `artifacts/checkpoints-1024-tool-calling{,-s2}` with both
measurements committed.

**The end-to-end loop closed, after a joining error that cost the whole capability.** The point
of training on the literal `<tool_call>` text was for vLLM's hermes parser to return a REAL
structured `tool_calls` object. Served with `--enable-auto-tool-choice --tool-call-parser
hermes` and given OpenAI-format tool schemas, the first attempt returned **0/3** tool calls —
while the *same weights on the same server* emitted a well-formed call in **100%** of raw
`/v1/completions` requests. Only the rendered prompt differed: my chat template emitted
`user: …\nassistant:`, and **no checkpoint in this project has ever been trained on that
shape.** Both question-answering formats here are Q/Answer — the dialogue slice writes
`Question: … Answer: …`, and `build_training_text` writes `Q: {q}\nAnswer:{call}`. The
template was rendering an interface the model had never seen.

Corrected to render `Q:`/`Answer:` (multi-turn: `Q: A\nAnswer: B\nQ: C\nAnswer:`), and both
properties now hold together, measured: **3/5 structured `tool_calls` through
`/v1/chat/completions`** (the 2 misses are truncation and malformed JSON — content, not
format), and the windowing guard still plateaus at ~101 prompt tokens across 14 growing turns
with zero crashes. Republished. Two tests pin it: one asserting the template renders Q/Answer
and never `user:`/`assistant:`, one asserting the rendered prefix is a literal prefix of
`build_training_text`'s output, so the two constructions cannot drift apart again.

**A chat template is not cosmetic** — it is the interface between what a client sends and what
the model was trained to continue, and a mismatched one silently costs the entire capability
while every offline check still passes. This is the same class as the errors
`scripts/evaluate.py` exists to prevent: the failure was in the *joining*, not the measuring.

**The two conversion paths now share one function.** `convert/to_hf.py::convert_checkpoint`
installed the chat template and the `tokenizer_class` correction;
`scripts/eval_improv.py::sft_checkpoint_to_hf` (the SFT path, used for every
skits/improv/tool-calling checkpoint) did neither — which is how the tool-calling artifact
reached a served vLLM with no chat template at all and returned a bare HTTP 400 on every chat
request. Extracted as `convert.to_hf.apply_tokenizer_fixups(out_dir)` and called from both,
with a test asserting both call sites exist so they cannot drift apart again.

**A guard caught me, correctly.** I passed `--eval-every 0` to stage 2 to save wall clock;
`assert_eval_wired` refused to start, because holding out 100 examples the trainer would never
evaluate means a silently empty validation curve — the exact `--eval-every` vs `--val-every`
confusion recorded in the 2026-08-20 entry.

**A claim of mine that was simply wrong, corrected.** I wrote here that `LossRecorder` records
train only and that stage 1's eval curve was empty. False. The eval rows were there all along
(12 in stage 1, 6 in stage 2) — my query looked for `split == "eval"` and a key `eval_loss`,
while the schema is `split == "val"` with the loss under `loss`. **I diagnosed a bug in the
instrument from a bug in my own query**, which is the same shape as everything else in this
entry: check what the artifact actually contains before describing it.

**And reading it properly surfaced a real, long-standing bug.** Both tool-calling runs overfit
(stage 1's best val is step 1000 at 1.642, rising monotonically to 1.747 by step 3000), so the
obvious move was to evaluate an earlier checkpoint. It scored *identically* — same parse rate,
same schema-valid rate, same tool distribution, same per-question counts. Against this project's
own rule that identical numbers mean suspect the instrument first, the cause is concrete:
**`step_1000.pkl` and `step_3000.pkl` are byte-identical in all 66 tensors** (`max abs diff
0.0`), despite different `step` fields and mtimes three minutes apart. The **earlier editor run
shows exactly the same thing** (0/66 differ), so this is not new — **every SFT run in this
project has been writing the same weights to every checkpoint file.**

Bounded carefully, because two things are true at once: the saved weights ARE genuinely trained
(66/66 tensors differ from the warm-start base, max abs diff 2.3e-2), and two *different runs*
produce genuinely different weights (stage 1 vs stage 2 final: 66/66 differ) — so the
stage-1-vs-stage-2 comparison above stands. What is destroyed is **intermediate checkpoint
selection on the SFT path**: `--save-every` produces duplicates, early stopping is impossible,
and any past comparison *between steps of one SFT run* would have been comparing identical
files. Which step's weights the duplicates hold is not yet established. Not fixed tonight —
it lives in `SFTTrainer._save_checkpoint`, which is tt-metal's, not this repo's.

## The 2048 context raise, reverted — and a hypothesis of mine that the data refuted (2026-08-29)

The corrected `tokens-v4` retrain finished and **still** would not answer questions: greedy
"capital of France" came back circular ("the capital of France is the capital of the Republic
of France"), Italy answered "Cologne", sky answered "the sky is blue" six times. Right corpus,
same failure. So the corpus was necessary and not sufficient, and something about the 2048
retrain itself was the problem.

**The arithmetic mistake, found by checking rather than assuming.** I had held step count at
10,764 "for comparability" while quadrupling `seq_len`. Step count is not the comparable
quantity when sequence length moves: `10,764 x 64 x 512 = 352.7M tokens = 1.00 epoch`, but
`10,764 x 64 x 2048 = 1,410.9M = 4.00 epochs`. Both 2048 runs silently did four epochs. And
this project had *already measured* what extra epochs do — `docs/measurements/evaluation-tt-tnt-1024a-vs-tt-tnt-1024-2ep.md`
records termination rate 0.0542 -> 0.0250, **worse at 2.00x the seed floor**, the only signal
clearing both gates, with episod-log.md's gloss: "a second pass ... taught the model to **stop
less**." Repetition loops are exactly *stops less*.

**That hypothesis was then tested, and the data did not support it.** The failed 4-epoch run
left six checkpoints spanning 0.74–4.00 epochs at fixed seed/corpus/context/LR/batch — an
accidental controlled epoch sweep, free to evaluate, per the standing rule to ask whether the
answer is already on disk. Termination came back **non-monotone**: 0.031, 0.094, 0.250, 0.062,
0.031, 0.062. Against the 512 control (0.156), **every** pairwise comparison is NOT SIGNIFICANT
at n=32 (|t| 0.75–1.76; binomial SE ~0.05 at these rates). Type/token ratio is flat across
everything, control included (0.707–0.741). **The metrics I chose to test my hypothesis cannot
tell these models apart at all** — a designed instrument that failed to discriminate, which is
the honest result and not a detail to bury. What actually separates them is Q&A coherence,
which I *read* rather than scored. The mechanism remains unisolated, and the 512-vs-2048
comparison is confounded anyway (seq_len and epochs moved together). Recorded in
`docs/measurements/ctx2048-regression-and-guard-efficacy.json` with its own limitations
section, so the next person inherits the null rather than re-deriving it.

**What made the decision anyway: questioning the premise instead of the parameter.** The
context raise existed only to dodge the growing-conversation KV-cache crash. But the same
change shipped a chat-template windowing guard (5-message cap). If the guard alone sufficed,
2048 was never needed. Tested directly at **512 context — the exact crash condition**: 14
growing turns, **zero crashes**, `prompt_tokens` plateauing at ~102 while messages sent grew to
27, against turn-0/1/2 = 23/65/106 then **engine crash at turn 3** before. The plateau lands
precisely on the 5-message boundary, which is the guard working and not a coincidence.

So: **reverted to the proven 512 dialogue weights, kept the guard, republished** (Hub sha
`038d6c6a`, `--verify` green on all seven checks, bundle manifest and a fresh `tt-model` pull
both rendering `--max_model_len 512`). `train/sizes.py`, the YAML, the manifest and
`_PUBLISHED_CONTEXT` all went back to 512 — that constant moved twice in one day without
either value having been wrong when set, which is the argument for it existing.

**Two lessons worth carrying.** *When you change one training parameter, recompute every
derived quantity before claiming comparability* — "same steps" silently became "4x the epochs",
and the tell was arithmetic I could have done in ten seconds at dispatch time rather than after
two 4.5-hour runs. And *before optimising a parameter, check whether the reason for the
parameter still exists* — two full retrains chased a context length whose entire justification
had already been satisfied by a one-line chat template shipped in the same change.

**The lesson.** A corpus generation is not implied by a model size or a training recipe --
`train/paths.py`'s own docstring already warns about exactly this class of mistake for
tokenizer/tokens regeneration ("retraining the tokenizer... produces numerically different
ids under the same filenames, with nothing on disk to tell the two generations apart"), and
this is the same failure mode one level up: reusing a *specific* tokens directory by copying
a prior run's recipe, without checking whether the checkpoint actually being extended had
moved to a newer one. The check that catches it is cheap and CPU-only -- decode a handful of
prompts the model was previously known to answer, greedy, before trusting a training run's
loss curve to mean the capability came along for free.

## 2026-08-31 — one stale cache behind three findings, and the v0.78.0 adoption plan

Prompt: analyse the model's state against recent work, the latest tt-model-manager, and the
tt-metal releasing today; propose five things to rescue it. Then: "start with 5 and 3" — the
disk/checkpoint-hygiene item and the release item.

### The state of the model, in one number

`docs/measurements/external-*.json` already had it: **1.458 bits/byte on WikiText against
GPT-2's 0.977**, at 123.0M parameters versus 124.4M. Same size, and MMLU/ARC-Challenge sit at
or below chance. The cause is not architecture — it is **352.7M training tokens for 123M
parameters, 2.87 tokens/param against Chinchilla's 20**. Compute is not the constraint (4-chip
DDP measured 169.4k tok/s, so 2.5B tokens is ~4h and 10B is ~16h); the corpus is: the nine
sources hold **651.5M unique tokens and 447.9M of that is TinyStories**. Every fine-tune this
project has run has been polishing a base that has not read enough. Recorded here because it
reframes the backlog, not because it was acted on today.

### One root cause, three findings

`SFTTrainer._save_checkpoint` reads each parameter with `to_numpy(FLOAT32, composer=...)` and
no `precision=`, taking the binding's `FULL` default. Our parameters are bf16, so
`AutocastTensor` keeps them in the half slot and leaves the full slot empty; AdamW mutates that
tensor **in place** and never calls `set_tensor()`, so the first FULL read's bf16→fp32 typecast
is **cached and returned forever after**. tt-metal names the hazard in
`autograd/autocast_tensor.cpp` (#41657). Every other serialization site in ttml already reads
`NATIVE` — `checkpointing.py:38`, `sharding.py:78,90` — which is exactly why the pretrain path
is fine (**50/50** tensors differ across a run) and the SFT path is not (**0/66**).

1. **The duplicate-checkpoint bug is fixed.** `train.checkpoint.save_sft_checkpoint`, passed via
   upstream's own `checkpoint_saver` hook, so no tt-metal edit. It reads through
   `ttml.sharding.Sharding.gather` (ttml's NATIVE read) rather than reimplementing precision
   handling, and refuses a sharded parameter instead of concatenating replicas. Wired into all
   three SFT entry points. Proven on hardware **with a negative control in the same session**,
   model rebuilt per arm so the caches are independent: default saver **66/66 identical**
   (max diff 0.0), ours **0/66 identical** (max diff 1.5625e-02). The unit test is
   mutation-checked — reverting the read to FULL turns it red.
2. **The LoRA blocker was the instrument, and is withdrawn.** `docs/upstream-tt-metal-asks.md`
   entry 5 reported that AdamW computed correct LoRA gradients but never applied them. That
   rested on comparing a `to_numpy()` read taken *before* training against one taken after —
   and `lora_B` is created BFLOAT16 (`ttml/modules/lora.py:69`), so **taking the baseline is
   what guaranteed the result**. Re-measured on the same 24 tensors in one session: NATIVE
   **24/24 moved** (max |Δ| 8.117676e-03), FULL **0/24**. LoRA works. The three gradient
   diagnostics stand; only the two value comparisons are withdrawn. This matters beyond LoRA:
   freezing 100% of base weights is the *structural* answer to the anti-forgetting problem
   `scripts/train_editor.py` had to counterweight with a 60% base-blend slice.
3. **What it cost while live.** Intermediate checkpoint selection on the SFT path. Both
   tool-calling runs overfit (best val at step 1000, rising to step 3000) and evaluating the
   earlier checkpoint returned identical numbers, because it held identical weights. Any past
   comparison *between steps of one SFT run* compared identical files. **Checkpoints written
   before today remain duplicates — the fix does not repair them.**

**The lesson, which this project has now hit in both directions:** a read can have a side
effect. Taking a baseline is an action, not an observation, and here the baseline read was the
entire mechanism of the bug it appeared to detect.

### The cache had already corrupted a published measurement

Found while listing the prune's deletions for approval — reading the list is what surfaced it,
which is the argument for listing deletions rather than describing them.

`docs/measurements/reach-dial-9000.json` is the "3x the training budget" rerun, and its
contrast is invalid. Every checkpoint inside one SFT run holds that run's **first-save**
weights, so:

| comparison | result |
|---|---|
| `reach-conv/ckpt-dial` step_1000 vs step_9000 | **66/66 identical**, max abs diff 0.0 |
| `reach-conv/ckpt-dial` step_1000 vs step_5000 | **66/66 identical**, max abs diff 0.0 |
| `reach/ckpt-dial` step_1000 vs step_3000 | **66/66 identical**, max abs diff 0.0 |
| `reach/ckpt-dial` step_3000 vs `reach-conv/ckpt-dial` step_9000 | 0/66 identical, max 2.3438e-02 |

So the rerun compared **1000-step weights against 1000-step weights from two different runs**,
while the `step` fields read 3000 and 9000. **"Undertraining is refuted" is withdrawn** — no
larger budget was ever evaluated — and with it my own correction that the 3000-step effects
were "a plateau, not a floor". Both revert to unresolved.

The uncomfortable part: that entry *did* suspect the instrument, and asked exactly the right
question — "had I evaluated the same checkpoint twice?" — then cleared it by measuring
`generation_churn` at 38.8%. That check was correctly computed and could not have caught this,
because the two artifacts genuinely are different runs. **The check tested file identity when
the defect was step identity.** A guard aimed one concept to the left of the failure.

What survives: the validation curves (computed on the live model during training, so the 12.7%
val improvement is real), and every effect measured *within* one artifact — only the step label
is wrong. What generalises: **every SFT-derived measurement in this project** (filenames
`step_N.pkl` — improv, skits, reach, reach-conv, editor\*, tool-calling\*) was made on its
run's step-1000 weights. Arm-vs-arm comparisons survive as like-for-like at 1000 steps; any
claim of the form "at N steps" or "more steps did not help" does not. Pretrain checkpoints
(`tt_tnt_step*.pkl`, `nanollama3_step*.pkl`) are unaffected — `ttml.checkpointing` already read
NATIVE. The filename scheme is the discriminator.

This also means the SFT runs have been paying for 3000 steps and keeping 1000. Re-running the
ones whose conclusions matter is now cheap and, for the first time, will actually produce
different checkpoints at different steps.

### Disk: the prune is a precondition, not housekeeping

`/` is 100% full with ~31G free, and **146.4G of `artifacts/` is `.pkl` checkpoints across 47
run directories**. Keeping only the final checkpoint per run reclaims **119.4G** (31G → 150G)
and loses nothing the repo depends on, verified rather than assumed:

* **Trajectories and the seed-noise floor read `val_losses.jsonl`, never weights**
  (`scripts/evaluate.py:120-121`). `scripts/prune_checkpoints.py` touches nothing but `.pkl`.
* Published measurements are already in `docs/measurements/`; nothing under `docs/` is touched.
* **No code references an intermediate checkpoint.** The only `.pkl` paths in
  `tests/`/`scripts/`/`train/`/`convert/` are finals; every other match is a `tmp_path` fixture
  or a comment. `latest_checkpoint()` and `test_numpy_parity` both select the highest step,
  which is what is kept — spot-checked per directory (`checkpoints` keeps the exact file the
  parity gate picks; `reach-conv` keeps the step_9000 that `reach-dial-9000.json` cites).
* For SFT runs the intermediates were duplicate bytes anyway, per the bug above.

The trade, stated once: an intermediate cannot be recovered later except by retraining. The
tool is dry-run by default and writes a manifest of exactly what it removed.

**Run 2026-08-31: 225 files, 112.5G freed, 31G -> 144G (100% -> 96%).** Record:
`docs/checkpoint-prune-20260831.json` (copied out of `artifacts/`, which is gitignored, so the
record of a destructive operation survives a fresh clone). **Both** `checkpoints-tt-tnt-1024-
ctx2048` and `-ctx2048-v4corpus` were excluded: the epoch sweep in
`ctx2048-regression-and-guard-efficacy.json` records step labels but no directory, and its
epoch arithmetic is identical under either corpus (2000 x 64 x 2048 = 262.1M tokens is 0.743
of both `tokens-v3`'s and `tokens-v4`'s train split), so which directory it used is not
recoverable from the artifact. Excluding both cost 3.4G against a 112.5G reclaim — the cheap
side of an unresolvable ambiguity. **Lesson for the next measurement: record the checkpoint
PATH, not just the step label.** A step label was not enough to identify the inputs three days
later, and the same artifact turned out not to be at the step it claimed.

Verified after: suite unchanged at 1754 passed / 5 skipped; all 66 metadata files intact; both
floor trajectories still 22 points; the three finals code actually opens
(`checkpoints-v077-beta2-control`, `checkpoints-1024-dialogue`, and the parity gate's
`checkpoints/nanollama3_step00003000.pkl`) all present.

### v0.78.0: what to take, what it buys, and what does not land

**Not tagged yet** at time of writing (tip is `v0.78.0-dev20260830`; stable cadence Jul 28 →
Aug 5 → Aug 18). 485 commits since `v0.77.0`, 23 in `tt-train`, 9 in `models/tt_transformers`.

**The prize is `352245bc0d`** — "GPT-OSS: minimal fix for garbage output on multi-device
Blackhole (#52176)". On-device sampling intermittently returned an out-of-range token id on
multi-device Blackhole while a decode trace was live (logits correct at every step; only the
sampled token corrupt; one bad token poisons the context, presenting as gradual degradation),
because decode-trace inputs and the sampling pre-compile were allocated behind a live prefill
trace. **This is our code path, not a demo's**: `tt_sampling.py:901-909` is the
`ttnn.sampling(..., output_tensor=tt_out_tok)` call, `generator.py:1452,1473` thread
`skip_precompile`, and 633 of the fix's changed lines are in
`models/tt_transformers/tt/generator.py`. It is a much stronger candidate for our 4-chip
`FABRIC_2D_TORUS_XY` quality regression than the fabric-routing hypothesis in
`docs/upstream-tt-metal-asks.md` entry 7 — recorded there with the ways the match is imperfect
(GPT-OSS saw an out-of-range id; we saw plausible invented words, which is the *stale-buffer*
rather than *never-written* reading of the same defect) and with the cheap decisive test:
compare host argmax against the device-sampled token per decode step.

**Also worth having:** `ade9c254f2` "[tt-train] Fix sdpa_fw mask=None hang, WH scale truncation,
sdpa_bw validation gaps" — which touches the exact `mask=None` path our 1.41x training speedup
rides on, so its five correctness checks must be re-run rather than assumed. Plus tt-train
kernel fixes in ops every step goes through (cross-entropy target page overrun, `layernorm_bw`
dx double-subtract, AdamW state-restore beta ordering, `moe_group` NOC race).

**What does NOT land, so don't hope:** `nb_models.cpp` still binds the attention mask
non-optional (entry 1 open), and `sft_trainer.py` has **no commits at all** in the range — the
duplicate-checkpoint bug and the precision cache are both still there upstream, which is why
we fixed them on our side.

**Adoption must not be a flag day, and today it would be one.** `~/tt-metal-src-vllm-home` is a
**symlink to `~/tt-metal`**, so serving and training share one v0.77.0 build and an in-place
upgrade moves both at once. The reversible path uses tt-model-manager's instance registry
(`00dba42`), which already holds two entries (`metal-0.75-vllm` at ttnn 0.75.0, `metal-src-vllm`
at 0.77.0): build v0.78.0 into a **separate tree**, register it as a third instance, and point
serving at it while `~/tt-metal` stays at 0.77.0 for training. Rollback is then a selection
change, not a rebuild. Our manifests already declare `platform.ttnn: ">=0.65,<0.80"`, so 0.78.0
is in range with no manifest edit. A second tree plus build costs ~14G — **which the prune above
has to happen first to afford.**
