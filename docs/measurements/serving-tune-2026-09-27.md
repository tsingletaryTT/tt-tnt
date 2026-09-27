# Serving tune, 2026-09-27: chat template, context, batch size, 1-chip start

Both published v6 thin bundles, re-served from a copy of the installed bundle with this
branch's adapter (1.1.0), run.sh, and tokenizer_config.json swapped in. Stack: ttnn 0.77.0,
vLLM 0.25.1 (empty target), vllm-tt-plugin 0.1.0, tt-tnt-models-closure 0.77.0. QuietBox 2
(2 x p300c), every serve under a gozer lease. Weights: `episod/tt-tnt` @ `fd37c60`,
`episod/tt-tnt-1024` @ `8874d97` (Stage B). The full working notes, logs and per-run JSON are
in the orchestrator's scratchpad (`hf/tt-tnt/TUNE.md`); this file keeps the numbers.

## Method

`vllm bench serve --backend vllm --dataset-name random --ignore-eos --seed 0 --num-warmups 2`,
streaming `/v1/completions`, greedy (`--temperature 0`) unless marked "default" (no
temperature sent, so vLLM's 1.0 applies). **Every row is the second of two identical runs**:
the first run of a new concurrency compiles new batch shapes, and a single 5 s compile
inside a 7 s run is what made an earlier pass read c32 = 2,239 tok/s for a configuration
that does 11,926 warm (see "Instrument notes"). N = 16 at c1, 64 at c8, 128 at c32, 256 at
c64, 512 at c128.

## The sweep: why max_num_seqs stays 32

* `max_num_seqs` 64 and 128 on one model instance **do not start**: tt_transformers 0.77.0
  hard-rejects them (`ModelArgs.__init__`: `supported_batches = {1, 2, 4, 8, 16, 32}` ->
  `ValueError: Batch size 64 not supported`). 32 is the ceiling per model instance.
* The only way past 32 is vLLM data parallelism (N independent replicas). That needed a new
  adapter fix (Patch 4) to start at all on a p300c, and then measured worse where it matters:

### tt-tnt, 1 chip (the bundle), max_num_seqs 32
| run | conc | N ok/fail | TTFT med (p99) ms | TPOT med (p99) ms | tok/s/user | aggregate out tok/s |
|---|---|---|---|---|---|---|
| tp1_c1 | 1 | 16/0 | 4.98 (5.4) | 1.80 (1.82) | 556 | 549 |
| tp1_c8 | 8 | 64/0 | 14.67 (19.6) | 1.95 (2.42) | 512 | 3,833 |
| tp1_c32 | 32 | 128/0 | 38.20 (51.1) | 2.40 (2.65) | 417 | 11,926 |
| tp1_c64 | 64 | 256/0 | 369.00 (412.8) | 2.46 (2.70) | 407 | 11,942 |
| tp1_c1_after | 1 | 16/0 | 4.67 (5.5) | 1.78 (1.87) | 563 | 551 |
| tp1_long_c1 | 1 | 8/0 | 12.46 (14.4) | 2.94 (3.01) | 340 | 331 |
| tp1_c8_default | 8 | 64/0 | 19.97 (24.5) | 6.37 (6.63) | 157 | 1,229 |

### tt-tnt, DP2 (both chips of one p300c), 2 x 32 seqs
| run | conc | N ok/fail | TTFT med (p99) ms | TPOT med (p99) ms | tok/s/user | aggregate out tok/s |
|---|---|---|---|---|---|---|
| dp2_c1 | 1 | 16/0 | 5.96 (6.3) | 2.08 (2.14) | 480 | 472 |
| dp2_c8 | 8 | 64/0 | 12.04 (14.7) | 3.29 (3.39) | 304 | 2,378 |
| dp2_c32 | 32 | 128/0 | 21.82 (44.3) | 4.05 (4.37) | 247 | 7,528 |
| dp2_c64 | 64 | 256/0 | 46.63 (74.0) | 4.59 (4.96) | 218 | 12,854 |

### tt-tnt-1024, TP4 (the bundle), max_num_seqs 32
| run | conc | N ok/fail | TTFT med (p99) ms | TPOT med (p99) ms | tok/s/user | aggregate out tok/s |
|---|---|---|---|---|---|---|
| tp4_c1 | 1 | 16/0 | 6.18 (9.0) | 2.88 (7.16) | 347 | 296 |
| tp4_c8 | 8 | 64/0 | 19.03 (32.5) | 3.04 (3.34) | 329 | 2,487 |
| tp4_c32 | 32 | 128/0 | 48.14 (93.6) | 3.77 (4.15) | 265 | 7,779 |
| tp4_c64 | 64 | 256/0 | 574.59 (683.4) | 3.88 (4.69) | 258 | 7,567 |
| tp4_c1_after | 1 | 16/0 | 5.92 (6.2) | 2.88 (3.00) | 348 | 344 |
| tp4_long_c1 | 1 | 8/0 | 12.82 (13.5) | 2.87 (3.32) | 349 | 334 |
| tp4_c8_default | 8 | 64/0 | 79.74 (116.1) | 15.33 (17.63) | 65 | 500 |

### tt-tnt-1024, DP4 (4 x 1-chip replicas), 4 x 32 seqs
| run | conc | N ok/fail | TTFT med (p99) ms | TPOT med (p99) ms | tok/s/user | aggregate out tok/s |
|---|---|---|---|---|---|---|
| dp4_c1 | 1 | 16/0 | 8.03 (8.4) | 3.25 (3.31) | 308 | 305 |
| dp4_c8 | 8 | 64/0 | 14.38 (43.2) | 7.75 (8.08) | 129 | 1,018 |
| dp4_c32 | 32 | 128/0 | 22.29 (117.7) | 8.23 (9.23) | 122 | 3,713 |
| dp4_c64 | 64 | 256/0 | 54.77 (143.4) | 8.67 (10.21) | 115 | 6,662 |
| dp4_c128 | 128 | 512/0 | 126.41 (172.7) | 9.45 (10.04) | 106 | 12,366 |
| dp4_long_c1 | 1 | 8/0 | 10.21 (10.8) | 3.46 (3.54) | 289 | 285 |
| dp4_c8_default | 8 | 64/0 | 15.97 (24.0) | 10.07 (10.43) | 99 | 784 |

(The tt-tnt-1024 "default" TP4 row is a single pass from the earlier sweep.)

**Decision (criterion: best aggregate at c32/c64 with c1 TPOT within 10%):**

| model | chosen | why |
|---|---|---|
| tt-tnt | 1 chip, `max_num_seqs 32` | DP2 is worse at c32 (7,528 vs 11,926 tok/s), only +8% at c64, and costs +16% c1 TPOT and a second chip |
| tt-tnt-1024 | TP4, `max_num_seqs 32` | DP4 is +13% c1 TPOT and loses at c8/c32/c64 (3,713 vs 7,779 at c32); it only wins at c128 (12,366 vs ~7,600) |

DP replicas run far slower per sequence than the same model alone (tt-tnt-1024 TPOT 8-9 ms
per replica at 2-32 users each vs 2.9-3.8 ms for TP4), which is what sinks it. The DP profile
works and is documented for batch-heavy (c >= 128) workloads, but is not shipped.

## Accepted prompt size

| model | max_model_len | long prompt (c1, greedy) | max_model_len + 1 | server afterwards |
|---|---|---|---|---|
| tt-tnt | 2048 (explicit) | ISL 1,900: TTFT 12.5 ms, TPOT 2.94 ms, 8/8 ok; 2,032-token prompt ok | 2,049 tokens -> **HTTP 400** | alive (200) |
| tt-tnt-1024 | 512 (explicit) | ISL 384: TTFT 12.8 ms, TPOT 2.87 ms, 8/8 ok; 496-token prompt ok | 513 tokens -> **HTTP 400** | alive (200) |

Before this branch any tt-tnt-1024 prompt of 129-512 tokens killed the engine; the random
ISL-128 set (which draws a 130-token prompt) and the ISL-384 set now complete with 0 failures.
tt-tnt's buckets (128 / 1024 / 2048) never exceeded its 2048 context, so the clamp is a no-op
there (pinned by `tests/test_tt_tnt_adapter_prefill_cap.py`).

## Chat

Every `/v1/chat/completions` request's server-side prompt ids (`return_token_ids`) equal the
local render of the shipped template (tt-tnt 3/3, tt-tnt-1024 7/7 conversations, including a
system message and a multi-turn history). The templates themselves are proven against the
training pipeline in `chat-template-proof.json`.

## 1-chip vs 4-chip output (tt-tnt-1024, greedy, 10 prompts x 64 tokens)

| run | mean tokens matching CPU fp32 before first divergence | prompts identical to CPU |
|---|---|---|
| TP4 (4 chips, FABRIC_2D_TORUS_XY) | 17.4 | 4/10 |
| TP1 (1 chip) | 15.1 | 3/10 |
| DP4 (four 1-chip replicas) | 15.1 (identical to TP1 on 10/10) | 3/10 |

TP4 and TP1 agree token-for-token on 4/10 prompts. Every divergence is ordinary bf16-vs-fp32
drift into different but fluent English. **No invented non-words on either topology** -- the
documented 4-chip regression ("Tryburg", "Alexandary") did not reproduce on the Stage B
weights with these prompts. Ten prompts cannot rule it out; they do mean 4-chip is not
measurably worse than 1-chip here.

## Instrument notes

* Cold vs warm: first-run numbers at a new concurrency include compiles (p99 TTFT up to 5.7 s).
  Only the second pass is reported above.
* Chip-to-chip: a 1-chip tt-tnt serve on chip `0000:03:00.0` measured c1 TPOT 2.53 ms, against
  1.77-1.80 ms on `0000:01:00.0` in two separate serves. Not investigated further; the final
  tt-tnt numbers are from `0000:01:00.0`.
