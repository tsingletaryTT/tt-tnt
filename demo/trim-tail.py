#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Cut everything after a timestamp off an asciicast.

`tt-demo compress` idle-trims — it drops long gaps *within* a recording. It has no way to
drop the tail, and the tail is where a terminal recording ends badly: the scene's duration
expires, the recorder sends `q`, the TUI resets the terminal, and the last frames are a
mode-reset escape and a literal `[exited]`. None of that is the demo.

Truncating rather than re-recording is the right move here because the footage before the
cut is already correct — re-shooting to lose two seconds would spend nine minutes of
hardware and produce a *different* run, which is worse evidence, not better.

The original is preserved beside the trimmed file as `<name>.raw.cast`, which
`demo/.gitignore` already excludes, so the uncut capture stays on disk without entering
the repo.

    python demo/trim-tail.py demo/assets/editor-lora.cast --at 397.8
    python demo/trim-tail.py demo/assets/editor-lora.cast --before "SCANNING FOR TRAINING"

`--before` is usually what you want: it finds the first frame containing a marker and cuts
just ahead of it, so the cut point is defined by what is on screen rather than by a number
that only matches one particular take.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

#: SGR, cursor and charset escapes, stripped before searching for a marker. A TUI splits
#: text across styling runs, so a marker rarely appears contiguously in the raw bytes.
_ESC = re.compile(r"\x1b\[[0-9;=?]*[A-Za-z]|\x1b\([AB0]|\x1b[>=]")


def load(path: Path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise SystemExit(f"{path}: empty")
    header, events = lines[0], []
    for line in lines[1:]:
        if not line.startswith("["):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return header, events


def find_marker(events, marker: str):
    """Timestamp of the first event whose visible text contains ``marker``."""
    for t, _kind, data in events:
        if marker in _ESC.sub("", data):
            return t
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cast", type=Path)
    ap.add_argument("--at", type=float, help="cut at this timestamp (seconds)")
    ap.add_argument("--before", help="cut just before the first frame containing this text")
    ap.add_argument("--keep-raw", action="store_true", default=True,
                    help="preserve the uncut capture as <name>.raw.cast (default)")
    args = ap.parse_args()

    if (args.at is None) == (args.before is None):
        ap.error("give exactly one of --at or --before")

    header, events = load(args.cast)
    if not events:
        raise SystemExit(f"{args.cast}: no events")
    total = events[-1][0]

    if args.before is not None:
        found = find_marker(events, args.before)
        if found is None:
            raise SystemExit(f"marker {args.before!r} never appears — nothing trimmed")
        cut = found
    else:
        cut = args.at

    kept = [e for e in events if e[0] < cut]
    if not kept:
        raise SystemExit(f"cut at {cut} would remove everything — refusing")

    raw = args.cast.with_suffix(".raw.cast")
    if args.keep_raw and not raw.exists():
        shutil.copy2(args.cast, raw)

    with args.cast.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for e in kept:
            # asciinema's own encoding: compact separators and real UTF-8. The defaults
            # (spaces after commas, ensure_ascii) rewrite every box-drawing glyph as a
            # 6-byte \uXXXX escape -- which showed up as a TRIMMED file 10% LARGER than
            # the capture it was cut from.
            fh.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"{args.cast.name}: {total:.2f}s -> {kept[-1][0]:.2f}s "
          f"({len(events) - len(kept)} of {len(events)} events removed)")
    if args.keep_raw:
        print(f"uncut capture preserved at {raw.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
