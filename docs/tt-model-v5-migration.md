<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# tt-model dropped v3/v4 — what it breaks here, measured

**2026-09-01.** `tt-model-manager` main moved 16 commits ahead of the checkout this box was
using. `dbf0ce5` ("Drop legacy v3/v4 manifest schemas — keep only v5 (fat) + v6 (thin)")
is a declared hard break. `tt-model`/`tt-kernel` are installed **editable** from
`~/code/tt-kernel-package-manager`, so updating that repo changed the live CLI immediately.

## What is actually broken

**The published bundle.** Verified against the Hub, not inferred:

```
$ tt-model info episod/tt-tnt-1024
Unsupported bundle schema_version '4'; this tt-model reads schema(s) 5, 5.1, 6.
Re-publish the bundle with a current tt-model.
```

**The local manifests**, for a different reason than the schema check. `manifests/tt_kernel_
manifest-{384,1024}.json` carry no `schema_version` at all, so they take the new default `"5"`
and *pass* that gate — then fail validation on three now-required fields. Measured by running
them through the new `Manifest.from_json`:

| | |
|---|---|
| required now | `arch`, `name`, `producer`, `tt_metal_version` |
| we have | `name` ✅ — plus `mesh`, `entrypoint`, `weights`, `resources`, `env` (all still valid) |
| **missing** | **`arch`, `producer`, `tt_metal_version`** |
| now ignored | `platform`, `runtime`, `target`, `description` |

So the migration is **add three fields**, not a rewrite — much smaller than "re-publish
everything" suggests. `producer` is a nested `Producer` type, not a string.

## What is NOT broken, checked rather than assumed

The removed CLI surface is `start`, `install`, `push`, `run`, `doctor`, `clean`, `instances*`,
`dev*`. Grepping this repo finds `tt-model push`, `tt-model instances` and `tt-model doctor` in
16, 2 and 2 places — **all of them prose**. Verified:

- `scripts/publish_to_hub.py` makes **zero** subprocess calls; it uses `huggingface_hub`
  directly, and its `tt-model push` mentions are docstrings and `--help` text.
- `scripts/stack_probe.py` builds exactly one `tt-model` argv — `["tt-model", "version"]`,
  which still exists and still returns `0.1.0`. Its `instances`/`doctor` mentions are
  docstrings recording past observed failures.

**No live code path here calls a removed command.** The stale prose is worth correcting, but
nothing executes it.

## What this invalidates elsewhere

⚠️ **The v0.78.0 adoption plan's reversible path is gone.** CLAUDE.md's 2026-08-31 entry
proposes building v0.78.0 into a separate tree, registering it as a third tt-model *instance*,
and pointing serving at it while `~/tt-metal` stays at 0.77.0 for training — so rollback is a
selection change rather than a rebuild. `src/tt_kernel/instances.py` is **deleted** on main
(present at the old HEAD, absent at `origin/main`). That plan needs a new mechanism before it
can be executed; it is not merely out of date.

## What landed that we asked for

`8eefade` — `feat(serve): add --refresh to re-pull a republished revision before serving`.
This is **F8**, the ask recorded in `docs/upstream-tt-metal-asks.md`: the stale-bundle bug
whose `vllm_metadata.json` supplied a `--max_model_len 512` launch command from a superseded
publish, costing an afternoon and nearly a fabricated "0% termination" measurement. Confirmed
present in `tt-model serve --help`. Our own local branch `fix/activitytqdm-lock-issue-35`
also landed upstream as `29f8fd4`; the branch is kept but is now redundant.

## Other repos, same sweep

- **vllm-tt-plugin** (`~/vllm-tt-plugin-standalone`): was on a detached HEAD, now on `main`,
  +2 commits — including `d2ccebc [deps] Update to vLLM 0.26.0`, a version bump worth
  re-verifying the serving path against before trusting it.
- **tt-cli** (`~/code/tt-cli`): `origin` is the fork `tsingletaryTT/tt-cli`, whose `main` is
  from **2026-02-19**. Real upstream `tenstorrent/tt-cli` is active (pushed today). Added as
  the `upstream` remote and fetched; the working branch is **12 ahead, 64 behind**. Left
  alone — 36 uncommitted files live there and reconciling them is not a mechanical call.
