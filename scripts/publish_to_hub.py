#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Publish (or re-publish) the tt-tnt HF artifact to the Hugging Face Hub.

Uploads the current HF artifact directory (``HF_DIR``, see below -- it is *not*
``artifacts/hf`` any more) and applies ``docs/model-card.md`` as the model card to
``episod/tt-tnt``. Re-runnable by design, because it has to be run more than once:

* Once, to do the initial publish (default action, below).
* Every time ``tt-model push`` runs against this repo, because tt-kernel's ``tag_repo``
  (``src/tt_kernel/hub.py:56-66``) replaces the card's front matter with
  ``ModelCardData(tags=...)`` and *nothing else* -- it destroys ``license``,
  ``pipeline_tag``, ``library_name``, and ``datasets`` on every push. ``--restore-card``
  re-applies ``docs/model-card.md`` after that damage (see plan Task 4 Step 2).

Safety rules baked into this script, not left to the caller's discipline:

* The repo is created private and is NEVER flipped public here. There is no ``--public``
  flag. Flipping visibility is a separate, explicitly-confirmed action (plan Task 4 Step 5)
  and does not belong in a re-runnable publish script.

  That flip has since happened, out-of-band, with explicit authorization: ``episod/tt-tnt``
  and ``episod/tt-tnt-corpus`` were both made public on 2026-08-14. This script's behavior
  did not change and does not need to -- ``create_repo(..., private=True, exist_ok=True)``
  only applies ``private=True`` when it actually creates a repo; per ``huggingface_hub``'s
  own docs, "this value is ignored if the repo already exists," so re-running the initial
  publish path against the now-public repo cannot silently flip it back. ``EXPECTED_PRIVATE``
  below records the current expectation for ``--verify`` rather than leaving it as a
  hardcoded assumption that would go stale the way this docstring almost did.
* Any action that writes to the Hub (initial publish, ``--restore-card``) requires ``--yes``.
  ``--dry-run`` never touches the Hub, regardless of ``--yes``.
* ``--verify`` is read-only: it round-trips the *published* copy through ``transformers``,
  not local state, so it actually proves what a downstream user would get.

A note on the "repo-level license" the packaging plan asks for: this script also calls
``huggingface_hub.metadata_update()`` right after repo creation, before any card exists, so
the license is set via a dedicated metadata API rather than living solely inside the prose
file this script uploads. Measured directly against a disposable scratch repo before writing
this: that call is NOT independent of card front matter under the hood -- the Hub stores
license only as part of the README's YAML block, so a `tag_repo`-style full front-matter
replacement wipes it exactly like everything else. The real defense is `--restore-card`
after every tt-kernel operation, not the order tags are set in. This script sets the license
early anyway (belt-and-suspenders, and it does make the repo well-formed before the full
card exists) but does not claim it survives what only ``--restore-card`` can fix.

Usage:

    python scripts/publish_to_hub.py --dry-run              # preview the initial publish
    python scripts/publish_to_hub.py --yes                  # do the initial publish
    python scripts/publish_to_hub.py --restore-card --dry-run
    python scripts/publish_to_hub.py --restore-card --yes   # re-apply the card after tt-model push
    python scripts/publish_to_hub.py --verify                # round-trip check against the Hub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: There is deliberately NO default publication target. ``--repo-id`` is required.
#:
#: This used to default to ``episod/tt-tnt`` -- which TARGETS itself describes as "the
#: protected baseline". So a bare ``publish_to_hub.py --yes``, run while working on the 1024
#: line, would publish the OTHER model: the one nobody had touched, using whatever happened to
#: be in its artifact directory. Publishing is outward-facing and awkward to walk back, and
#: this repo has already published a wrong-corpus checkpoint once. A publish must therefore
#: name its target out loud; there is no invocation that guesses.
LICENSE = "apache-2.0"

#: The local HF artifact this script uploads -- i.e. the directory whose contents are
#: supposed to *be* ``episod/tt-tnt``.
#:
#: This was ``artifacts/hf`` up to and including the v2/256 publish, and
#: ``artifacts/hf-tt-tnt-v1`` for the 512 publish. It is deliberately neither now.
#:
#: Pointing it back at ``artifacts/hf`` would be a silent downgrade: that is the protected,
#: unregeneratable v2 baseline (``train/paths.py::PROTECTED_RELATIVE``), still holding
#: ``max_position_embeddings: 256`` and the pre-blend tokenizer. Pointing it back at
#: ``artifacts/hf-tt-tnt-v1`` is a subtler downgrade and the one that is live right now: v1
#: is a *loadable, current-tokenizer, correctly-shaped* artifact that differs from v3 only
#: in its weights and one config field, so nothing about a mis-aimed upload would look
#: wrong. It would keep the repo id, the card, the tokenizer and the architecture, and
#: quietly replace 2048-context weights trained on an EOS-carrying corpus with
#: 512-context weights that can never emit a document boundary.
#:
#: ``_assert_local_artifact_is_publishable`` below refuses any directory whose context
#: length is not ``EXPECTED_MAX_POSITION_EMBEDDINGS``, so this constant is checked rather
#: than merely believed. That check is what makes the v1/v3 confusion detectable at all:
#: the two directories are otherwise byte-identical in every file except
#: ``config.json`` and ``model.safetensors`` (verified: ``tokenizer.json``,
#: ``tokenizer_config.json`` and ``special_tokens_map.json`` all compare equal).
HF_DIR = ROOT / "artifacts" / "hf-tt-tnt-v3"
CARD_PATH = ROOT / "docs" / "model-card.md"

# What the round trip in --verify must find, and (for the context length) what the local
# artifact must be before it may be uploaded at all. These are not arbitrary: they are the
# properties the packaging plan calls out as the ones a serving stack would get wrong
# silently if the artifact were malformed (context length, weight tying, vocab, and the
# exact parameter count as a coarse "did the whole state dict actually load" check).
#
# CONTEXT LENGTH, 256 -> 512 -> 2048. The standing rule has never changed: this constant
# describes the artifact that is ACTUALLY published, and it moves only once a model has
# actually been retrained and republished at the new context. It moved to 512 on
# 2026-08-14 for tt-tnt-v1, and to 2048 later the same day for tt-tnt-v3, which was
# trained at seq_len 2048 (``artifacts/checkpoints-tt-tnt-v3``, 10,764 steps, final
# val_loss 2.939) and whose ``artifacts/hf-tt-tnt-v3/config.json`` reads
# ``max_position_embeddings: 2048``.
#
# Note this is not a loosening: the value moved but the rule did not, and it guards in
# BOTH directions -- ``--verify`` fails if the Hub ever stops being 2048, and
# ``_assert_local_artifact_is_publishable`` refuses to upload a local directory that is
# not 2048, which is exactly the v1-over-v3 downgrade the HF_DIR note above describes.
#
# 2048 is a power of two, which ``resources.max_model_len`` in the manifests separately
# requires (``model_config.py:1150-1152``, capped_warmup_seq_len); see
# ``tests/test_manifests.py``.
EXPECTED_MAX_POSITION_EMBEDDINGS = 2048
EXPECTED_TIE_WORD_EMBEDDINGS = True
EXPECTED_VOCAB_SIZE = 32000
EXPECTED_PARAM_COUNT = 22_025_088
PROMPT = "Once upon a time, there was a little"

# ---------------------------------------------------------------------------
# PUBLICATION TARGETS
#
# The constants above describe ``episod/tt-tnt`` and exist to stop a downgrade of
# that LIVE, PUBLIC repo -- they are not a general policy. A second model with a
# different shape needs its own invariants, not a loosening of these, so targets
# are declared per repo and ``--repo-id`` selects one. The default is unchanged.
#
# Adding a target is a deliberate act: state the artifact and the four properties
# a serving stack would otherwise get wrong silently, and the same assertions run
# against it as against v3.
# ---------------------------------------------------------------------------

TARGETS = {
    "episod/tt-tnt": {
        # None means "whatever HF_DIR is at call time" -- resolved in target_for. A
        # snapshot here would silently defeat the wrong-context-length guard, whose test
        # patches HF_DIR to point at a bad artifact.
        "hf_dir": None,
        "max_position_embeddings": EXPECTED_MAX_POSITION_EMBEDDINGS,
        "tie_word_embeddings": EXPECTED_TIE_WORD_EMBEDDINGS,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "param_count": EXPECTED_PARAM_COUNT,
        "card": CARD_PATH,
        # Explicitly authorized out-of-band on 2026-08-14 (see module docstring); --verify
        # checks the Hub against this recorded expectation rather than a hardcoded True/False
        # that would go stale the day visibility legitimately changes again.
        "expected_private": False,
        "manifest": ROOT / "manifests" / "tt_kernel_manifest-384.json",
        "note": "tt-tnt-v3, 384-dim at a 2048 context. The protected baseline.",
    },
    "episod/tt-tnt-1024": {
        # NOT None -- None resolves to module-level HF_DIR above, which is the 384 line's
        # canonical path (hf-tt-tnt-v3). artifacts/hf-tt-tnt-1024 is this size's OWN ONE
        # canonical directory, always overwritten with the current best idea rather than
        # accumulating a new -dialogue/-editor/-ctx2048-suffixed sibling per change (see
        # CLAUDE.md's 2026-08-29 consolidation entry for why this stopped being a
        # per-experiment directory), but it is a size-specific fixed path, not the shared
        # HF_DIR constant.
        "hf_dir": ROOT / "artifacts" / "hf-tt-tnt-1024",
        "max_position_embeddings": 512,
        "tie_word_embeddings": True,
        "vocab_size": 32000,
        "param_count": 122962944,
        # Never explicitly authorized public the way episod/tt-tnt was (see that target's
        # comment and the module docstring) -- stays private until a deliberate, separate
        # decision flips it, the same way this script itself never flips visibility.
        "expected_private": True,
        "card": ROOT / "docs" / "model-card-1024.md",
        "manifest": ROOT / "manifests" / "tt_kernel_manifest-1024.json",
        "note": (
            "tt-tnt-1024, raised to a 2048-token context (from 512) to push the "
            "growing-conversation KV-cache crash (docs/upstream-tt-metal-asks.md "
            "entry 6 -- confirmed a generic tt-metal/vLLM defect, not this "
            "project's model) far out of ordinary reach, plus a chat template "
            "shipped in tokenizer_config.json that caps rendered history to the "
            "last 5 messages as a backstop. Matched-window (512) loss improved "
            "-0.2318 nats against the prior dialogue-trained checkpoint; no "
            "behavioural signal clears both the seed floor and its own paired "
            "interval in either direction (n=1 seed). See "
            "docs/measurements/evaluation-tt-tnt-1024-dialogue-vs-tt-tnt-1024-ctx2048.md."
        ),
    },
}


def target_for(repo_id):
    """Invariants for a publication target. Unknown repos are refused, not guessed."""
    try:
        target = dict(TARGETS[repo_id])
    except KeyError:
        known = ", ".join(sorted(TARGETS))
        raise SystemExit(
            f"no publication target declared for {repo_id!r}. Declare one in TARGETS "
            f"with the artifact and its invariants -- this script does not guess what "
            f"it is uploading. Known targets: {known}"
        )
    if target["hf_dir"] is None:
        target["hf_dir"] = HF_DIR
    return target


# PRIVACY, False since 2026-08-14. The repo was created private (as this script still does
# for any repo it has to create fresh) and was later flipped public out-of-band, with
# explicit authorization -- not through this script, which has no code path that can do
# that (see test_publish_to_hub.py::test_source_never_sets_private_false and the module
# docstring). ``--verify`` checks the Hub against this constant so the expectation lives in
# one edited place rather than as a hardcoded ``True`` that would silently start failing --
# or worse, stop meaning anything -- the day visibility legitimately changed again.
EXPECTED_PRIVATE = False


def _artifact_files(hf_dir=None) -> list[Path]:
    """Files ``upload_folder`` would send, in a stable order for printing and testing."""
    hf_dir = HF_DIR if hf_dir is None else hf_dir
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"{hf_dir} does not exist -- run scripts/convert_checkpoint.py first")
    return sorted(p for p in hf_dir.iterdir() if p.is_file())


def _assert_local_artifact_is_publishable(hf_dir=None, expected=None) -> None:
    """Refuse to upload a local artifact whose context length isn't what we claim to ship.

    ``--verify`` checks the artifact *after* it is on the Hub, which is too late to prevent
    a bad publish -- it only tells you one happened. This is the same constant applied
    before the write, so ``EXPECTED_MAX_POSITION_EMBEDDINGS`` guards the upload rather than
    only describing it.

    The failure it exists to stop is concrete and has a live trigger: several local
    directories in ``artifacts/`` are loadable HF models of this architecture
    (``artifacts/hf``, ``artifacts/384/hf``, ``artifacts/hf-v2-scratch``), they differ only
    in weights and a config field, and all of them would upload perfectly happily. Pointing
    ``HF_DIR`` at the wrong one produces a Hub repo that still has the right name, the
    right card, and the right shape -- and half the trained context.
    """
    import json

    hf_dir = HF_DIR if hf_dir is None else hf_dir
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} does not exist -- {hf_dir} is not an HF model dir")

    want = (EXPECTED_MAX_POSITION_EMBEDDINGS if expected is None
            else expected["max_position_embeddings"])
    config = json.loads(config_path.read_text())
    actual = config.get("max_position_embeddings")
    if actual != want:
        raise ValueError(
            f"{config_path} has max_position_embeddings={actual!r}, but this target "
            f"publishes {want}-context weights. Refusing to "
            f"upload: this is how a shorter-context model silently replaces a longer one "
            f"under the same repo id. Point HF_DIR at the right artifact, or update "
            f"EXPECTED_MAX_POSITION_EMBEDDINGS if the published context really is changing."
        )


def _print_upload_plan(repo_id: str) -> int:
    """Print the file list and total size that would be uploaded. Returns the total bytes."""
    target = target_for(repo_id)
    files = _artifact_files(target["hf_dir"])
    total = 0
    print(f"repo:    {repo_id} (private=True if newly created -- per huggingface_hub, "
          f"ignored if it already exists, so this cannot silently re-privatize an existing "
          f"public repo; license={LICENSE})")
    print(f"card:    {target["card"].relative_to(ROOT)}")
    print(f"source:  {target["hf_dir"].relative_to(ROOT)}")
    print("files:")
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"  {f.name:30s} {size:>12,} B")
    print(f"total:   {total:,} B ({total / 1e6:.2f} MB)")
    return total


def _load_card_for_hub(card_path=None):
    """Load ``docs/model-card.md`` as a ``ModelCard`` fit to push to the Hub.

    ``docs/model-card.md`` intentionally leads with an HTML-comment explanation (SPDX
    headers, and a note on why the file exists) *before* the YAML front-matter fence, so a
    maintainer opening the file in an editor sees the explanation first. That trips a real
    gotcha: ``huggingface_hub.ModelCard``'s front-matter regex is anchored to the very start
    of the string (``^\\s*---``, no ``re.MULTILINE``), so ``ModelCard.load()`` on the raw
    file finds no metadata block -- and, worse, does not raise. It logs a warning and
    silently returns an EMPTY ``CardData`` (confirmed directly: ``card.data.license`` comes
    back ``None`` from the unmodified file). Pushing that would be worse than doing nothing.

    The fix: find the first front-matter fence and construct the card from that point
    onward, matching what the Hub actually needs (front matter must lead the README there
    too). Fail loudly, not silently, if that fence is missing or license didn't parse.
    """
    from huggingface_hub import ModelCard

    card_path = CARD_PATH if card_path is None else card_path
    raw = card_path.read_text()
    stripped = raw.lstrip()
    if stripped.startswith("---"):
        content = stripped
    else:
        idx = raw.find("\n---")
        if idx == -1:
            raise ValueError(f"{card_path}: no YAML front-matter fence ('---') found; "
                              "refusing to push a card with no metadata")
        content = raw[idx + 1:]

    card = ModelCard(content)
    if card.data.license is None:
        raise ValueError(f"{card_path}: parsed card has no `license` in front matter after "
                          "stripping leading comments -- refusing to push what looks like an "
                          "empty card")
    return card


def _set_license(repo_id: str) -> None:
    """Set the repo license via a dedicated metadata call, not by hoping the card sticks."""
    from huggingface_hub import metadata_update

    metadata_update(repo_id, {"license": LICENSE}, repo_type="model", overwrite=True)


def _push_card(repo_id: str, card_path=None) -> None:
    card = _load_card_for_hub(card_path)
    card.push_to_hub(repo_id, repo_type="model")


def _report_card_state(repo_id: str) -> None:
    """Print what front-matter fields are actually present on the Hub right now.

    This is the check the packaging plan asks for after every tt-kernel operation:
    "verify front matter after every tt-kernel operation, and restore what was lost."
    """
    from huggingface_hub import ModelCard

    card = ModelCard.load(repo_id, repo_type="model")
    print("current card front matter on the Hub:")
    for field in ("license", "library_name", "pipeline_tag", "datasets", "tags"):
        print(f"  {field}: {getattr(card.data, field, None)!r}")


def cmd_publish(repo_id: str, dry_run: bool, yes: bool) -> int:
    # Resolve the target FIRST: an undeclared repo is refused here rather than
    # uploaded with whatever the module constants happen to say.
    target = target_for(repo_id)
    """Create the repo (private), set the license, upload the artifact, apply the card."""
    # Before the plan is even printed, so a wrong HF_DIR is reported by --dry-run too --
    # the preview is worth nothing if it happily previews an upload that must not happen.
    _assert_local_artifact_is_publishable(target["hf_dir"], target)
    _print_upload_plan(repo_id)

    if dry_run:
        print("[dry-run] no repo created, nothing uploaded, no card pushed.")
        return 0

    if not yes:
        print("refusing to publish without --yes (use --dry-run to preview safely)",
              file=sys.stderr)
        return 2

    from huggingface_hub import HfApi

    api = HfApi()

    print(f"creating (or reusing) repo {repo_id} (private=True only takes effect if this "
          f"call actually creates it) ...")
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    print(f"setting repo-level license={LICENSE} ...")
    _set_license(repo_id)

    print(f"uploading {target["hf_dir"].relative_to(ROOT)} -> {repo_id} ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(target["hf_dir"]),
        commit_message="Upload tt-tnt HF artifact (config, weights, tokenizer)",
    )

    print(f"applying model card from {target["card"].relative_to(ROOT)} ...")
    _push_card(repo_id, target["card"])

    print("re-asserting repo-level license (belt-and-suspenders after the card push) ...")
    _set_license(repo_id)

    _report_card_state(repo_id)
    print("done. Visibility is never changed by this script, in either direction.")
    return 0


#: Filename the bundle manifest takes in the Hub repo. tt-model reads exactly this name.
HUB_MANIFEST_NAME = "tt_kernel_manifest.json"


def cmd_publish_manifest(repo_id: str, dry_run: bool, yes: bool) -> int:
    """Upload only the bundle manifest, replacing the one in the Hub repo.

    Separate from the weight publish on purpose. The manifest is small, it changes for
    reasons the weights do not (a schema migration, a corrected resource limit), and
    re-uploading 246 MB of unchanged safetensors to fix a JSON field is how an artifact
    acquires revisions nobody can explain later.

    The manifest is validated through the INSTALLED tt-model's own ``Manifest.from_json``
    before anything is uploaded. That is the whole point: the repo shipped a schema-4
    manifest that the current tt-model refuses outright, and the way that happened was a
    file nobody had run through the reader. Publishing one the local tooling cannot read is
    now impossible rather than merely discouraged.
    """
    target = target_for(repo_id)
    src = target.get("manifest")
    if src is None:
        print(f"error: target {repo_id!r} declares no manifest", file=sys.stderr)
        return 2
    print(f"repo:     {repo_id}")
    # relative_to() raises for any path outside the repo, which a test fixture (and an
    # operator pointing at a staged file) legitimately is. Display must never be the thing
    # that fails a publish.
    try:
        shown = src.relative_to(ROOT)
    except ValueError:
        shown = src
    print(f"manifest: {shown} -> {HUB_MANIFEST_NAME}")

    # Write intent is checked BEFORE validation, and deliberately. Refusing a missing --yes
    # is an argument error: it must not depend on tt-model being importable, on a network,
    # or on anything else. CI has no tt-model, so the original order made this guard report
    # an unrelated ImportError there -- a guard that says the wrong thing in the one
    # environment that runs it automatically is not a guard.
    if not dry_run and not yes:
        print("\nrefusing to write without --yes.", file=sys.stderr)
        return 2

    try:
        from tt_kernel.manifest import SUPPORTED_SCHEMAS, Manifest
    except ImportError:
        print("error: tt-model (tt_kernel) is not importable; cannot validate before "
              "publishing, and publishing unvalidated is what caused the schema-4 break.",
              file=sys.stderr)
        return 2

    try:
        m = Manifest.from_json(src.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"error: {src.name} is not readable by the installed tt-model: {exc}",
              file=sys.stderr)
        return 2

    # A manifest naming another repo's weights would serve the wrong model from the right
    # URL -- silent, and exactly the class of mistake --repo-id being required guards against.
    if m.weights is not None and m.weights.repo_id != repo_id:
        print(f"error: {src.name} points weights at {m.weights.repo_id!r}, not {repo_id!r}",
              file=sys.stderr)
        return 2

    print(f"schema:   {m.schema_version} (tt-model reads {', '.join(sorted(SUPPORTED_SCHEMAS))})")
    print(f"arch:     {m.arch}   device_count: {m.device_count}   "
          f"tt_metal: {m.tt_metal_version}")
    print(f"weights:  {m.weights.repo_id}@{(m.weights.revision or '')[:12]}")

    if dry_run:
        print("\n--dry-run: nothing uploaded.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=HUB_MANIFEST_NAME,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Publish bundle manifest at schema v{m.schema_version}",
    )
    print(f"\nuploaded {HUB_MANIFEST_NAME} to {repo_id}")
    return 0


def cmd_restore_card(repo_id: str, dry_run: bool, yes: bool) -> int:
    """Re-apply the target's card, for use after a tt-model push damages front matter."""
    target = target_for(repo_id)
    print(f"repo:    {repo_id}")
    print(f"card:    {target["card"].relative_to(ROOT)}")

    if dry_run:
        # Print the TARGET's card, not the module-level default. This message used to name
        # CARD_PATH (docs/model-card.md, the 384-dim tt-tnt card) no matter which repo was
        # selected, while the line above printed the correct per-target card — so a --dry-run
        # against episod/tt-tnt-1024 contradicted itself and looked like a display quirk.
        print(f"[dry-run] would push card from {target["card"].relative_to(ROOT)} to {repo_id}, "
              f"then re-set license={LICENSE}. No changes made.")
        return 0

    if not yes:
        print("refusing to push without --yes (use --dry-run to preview safely)",
              file=sys.stderr)
        return 2

    # Pass the target's card explicitly. It was not a display quirk: this call omitted the
    # argument, so _push_card fell back to its card_path=None default (docs/model-card.md) and
    # --restore-card would publish the 384-dim model's card onto whichever repo was named.
    # cmd_publish above has always passed target["card"]; only this path was wrong.
    _push_card(repo_id, target["card"])
    _set_license(repo_id)
    _report_card_state(repo_id)
    print("card restored.")
    return 0


def cmd_verify(repo_id: str) -> int:
    """Round-trip verification from the Hub, not local state. Read-only.

    Reads its expectations from ``target_for(repo_id)``, not the bare module-level
    ``EXPECTED_*`` constants -- those describe ``episod/tt-tnt`` only (see the constants'
    own comment). Before this fix, ``--verify`` for ANY OTHER target silently checked it
    against the wrong repo's shape and privacy expectation: verifying
    ``episod/tt-tnt-1024`` (122,962,944 params, expected private) reported false failures
    for "parameter count == 22,025,088" and "repo private == False", because both were the
    384 line's numbers applied to a different model entirely.
    """
    from huggingface_hub import HfApi, ModelCard
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = target_for(repo_id)
    checks: list[bool] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}{('  ' + detail) if detail else ''}")
        checks.append(bool(cond))

    print(f"loading {repo_id} fresh from the Hub (not from {target['hf_dir'].name}/) ...")
    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    model.eval()
    cfg = model.config

    # Labels are interpolated from the target dict, never spelled out. A hardcoded
    # "max_position_embeddings == 256" next to a comparison against a value that says 512
    # is a check that lies in its own output -- and this file had exactly that until the
    # 512 bump, which is precisely when a reader most needs the label to be true.
    expected_max_pos = target["max_position_embeddings"]
    check(f"max_position_embeddings == {expected_max_pos}",
          cfg.max_position_embeddings == expected_max_pos,
          f"(got {cfg.max_position_embeddings})")
    expected_tied = target["tie_word_embeddings"]
    check(f"tie_word_embeddings is {expected_tied}",
          cfg.tie_word_embeddings is expected_tied,
          f"(got {cfg.tie_word_embeddings})")
    expected_vocab = target["vocab_size"]
    check(f"vocab_size == {expected_vocab}", cfg.vocab_size == expected_vocab,
          f"(got {cfg.vocab_size})")

    expected_params = target["param_count"]
    n_params = sum(p.numel() for p in model.parameters())
    check(f"parameter count == {expected_params:,}", n_params == expected_params,
          f"(got {n_params:,})")

    import torch

    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.95)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"sample generation: {text!r}")
    # .strip() before comparing: the tokenizer's BPE decode adds a leading space before the
    # first token (a normal artifact of space-prefixed byte-level BPE, not a defect), so the
    # raw decoded string is " Once upon..." rather than "Once upon...". Stripping avoids a
    # false failure on that whitespace while still requiring the model to have reproduced
    # the prompt and appended new tokens after it.
    check("generation extended the prompt",
          len(text) > len(PROMPT) and text.strip().startswith(PROMPT[:10]))

    api = HfApi()
    info = api.model_info(repo_id)
    expected_private = target["expected_private"]
    check(f"repo private == {expected_private}", info.private is expected_private,
          f"(got {info.private})")

    card = ModelCard.load(repo_id, repo_type="model")
    check("card front matter has license == apache-2.0", getattr(card.data, "license", None) == LICENSE,
          f"(got {getattr(card.data, 'license', None)!r})")

    if not all(checks):
        print("one or more checks FAILED -- stopping. Do not re-upload blindly; diagnose first.",
              file=sys.stderr)
        return 1
    print("all checks passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, choices=sorted(TARGETS),
                   help="Target model repo. REQUIRED and restricted to one declared target, "
                        "so a single invocation publishes exactly one model and never the "
                        "wrong one by default.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would happen; never contacts the Hub for writes.")
    p.add_argument("--yes", action="store_true",
                   help="Required to actually write to the Hub (publish or --restore-card).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--restore-card", action="store_true",
                       help="Only re-apply docs/model-card.md as the repo card (use after "
                            "`tt-model push` damages front matter). Skips repo creation "
                            "and weight upload.")
    mode.add_argument("--manifest", action="store_true",
                       help="Publish ONLY the bundle manifest (manifests/tt_kernel_manifest-"
                            "<size>.json) to the repo, validated against the installed "
                            "tt-model first. Does not touch weights, tokenizer or card.")
    mode.add_argument("--verify", action="store_true",
                       help="Read-only round-trip check: load the published model+tokenizer "
                            "fresh from the Hub via transformers and assert key fields.")
    args = p.parse_args(argv)

    if args.verify:
        if args.dry_run or args.yes or args.restore_card:
            print("--verify is read-only and takes no other flags", file=sys.stderr)
            return 2
        return cmd_verify(args.repo_id)

    if args.restore_card:
        return cmd_restore_card(args.repo_id, dry_run=args.dry_run, yes=args.yes)

    if args.manifest:
        return cmd_publish_manifest(args.repo_id, dry_run=args.dry_run, yes=args.yes)

    return cmd_publish(args.repo_id, dry_run=args.dry_run, yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
