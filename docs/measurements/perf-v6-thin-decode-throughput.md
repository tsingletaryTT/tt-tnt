<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# v6 thin-package decode throughput (2026-09-04)

First benchmark of both editions served as real **v6 thin** `tt-model` bundles (previously only
packaged as the old host-provisioned v4-style bundle). Batch=1, greedy, `max_tokens=128`, 5 timed
runs after 1 warmup, decode-only throughput derived from each response's own
`usage.completion_tokens` divided by wall-clock request time (not counted from text).

| model | arch | mean tok/s | min | max |
|---|---|---|---|---|
| `episod/tt-tnt` (22M) | 1-chip (P150) | **618.16** | 605.42 | 632.43 |
| `episod/tt-tnt-1024` (123M) | 4-chip (P300x2, `FABRIC_2D_TORUS_XY`) | **324.81** | 285.80 | 345.80 |

The larger, 4-chip model is slower in tok/s, not faster — expected at batch=1: tensor-parallel
serving adds cross-chip communication per decode step with nothing to amortize it against (no
concurrent requests), while the extra chips buy headroom for throughput *under load*
(concurrent requests), not single-stream latency. Not a regression or a packaging defect.

Raw data: [`perf-v6-thin-decode-throughput-tt-tnt.json`](perf-v6-thin-decode-throughput-tt-tnt.json),
[`perf-v6-thin-decode-throughput-tt-tnt-1024.json`](perf-v6-thin-decode-throughput-tt-tnt-1024.json).

Bundle provenance: both served from a v6 thin package (`tt-model package-thin`) built with a
locally cherry-picked `tt-metal-models` wheel (tenstorrent/tt-metal#54478's packaging code
applied onto the `v0.77.0` tag ahead of its PyPI publish), `vllm-tt-plugin` @ `6d3bb28` (last
commit still on vLLM 0.25.1), and a prebuilt empty-target vLLM 0.25.1 wheel. Two tt-model-manager
bugs were found and fixed getting here (draft PRs `tenstorrent/tt-model-manager#70` and `#71`).
Not a quality/correctness measurement — see `docs/tt-kernel-conformance.md` and
`docs/serving-with-tt-kernel.md` for those; `tt-tnt-1024`'s 4-chip path still carries the
documented, unresolved correctness regression relative to 2-chip.
