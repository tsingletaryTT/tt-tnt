#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Write a ``tokenizer_config.json`` carrying one of this project's chat templates.

For fixing an ALREADY-PUBLISHED artifact without re-converting its weights: e.g. the live
``episod/tt-tnt`` (no template at all -> every chat request is HTTP 400) and
``episod/tt-tnt-1024`` (the old ``Q:\\nAnswer:`` template). Every field of ``--src`` is kept;
only ``chat_template`` (and the ``tokenizer_class`` fixup) change -- the same
``apply_tokenizer_fixups`` a fresh conversion runs, so the two paths cannot drift.

    python scripts/write_tokenizer_config.py --chat-format dolly_qa \\
        --src artifacts/hf-tt-tnt-1024/tokenizer_config.json --out /tmp/upload/tokenizer_config.json

The format for each published repo is ``scripts/publish_to_hub.py``'s ``TARGETS[...]
["chat_format"]``; ``--repo-id`` looks it up so the name need not be retyped.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from convert.chat_templates import CHAT_TEMPLATES  # noqa: E402
from convert.to_hf import apply_tokenizer_fixups  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    fmt = ap.add_mutually_exclusive_group(required=True)
    fmt.add_argument("--chat-format", choices=sorted(CHAT_TEMPLATES))
    fmt.add_argument("--repo-id", help="take the format from publish_to_hub.TARGETS")
    ap.add_argument("--src", type=Path, required=True, help="existing tokenizer_config.json")
    ap.add_argument("--out", type=Path, required=True, help="where to write the result")
    args = ap.parse_args(argv)

    chat_format = args.chat_format
    if args.repo_id:
        from scripts.publish_to_hub import target_for

        chat_format = target_for(args.repo_id)["chat_format"]

    # apply_tokenizer_fixups edits a directory's tokenizer_config.json in place; run it on a
    # scratch copy so --src is never modified (it may be the canonical artifact).
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "tokenizer_config.json"
        shutil.copy2(args.src, work)
        apply_tokenizer_fixups(Path(tmp), chat_format)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work, args.out)
    print(f"wrote {args.out} (chat_format={chat_format})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
