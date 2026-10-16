#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Prove, without importing anything, that every in-bundle import resolves.

``packaging/package-thin.sh`` ships the model's own code in ``extra_code/`` and the adapter
at the bundle root. Nothing else supplies a ``models`` package at serve time, so an import
inside that code that points at a file which is not there fails only after a multi-minute
vLLM boot. This script finds that failure in a second, with no ttnn, no torch and no
hardware: it only parses source files with ``ast``.

What it checks
--------------
Every ``import x.y`` and ``from x.y import z`` in the scanned files (relative imports
included, at any depth, including lazy imports inside functions).

* A name whose top-level package exists in ``--root`` (for example ``models``) is
  *in-bundle*. Its module path must exist there as a ``.py`` file, a package directory or a
  namespace directory. ``from x.y import z`` also requires ``x.y`` to exist. It cannot
  check that ``z`` exists inside it, because ``z`` may be an attribute set at import time.
* Any other top-level name (``torch``, ``vllm``, ``ttnn``, the standard library ...) is
  *external*. It is listed in the summary so a human can compare it with the pinned
  requirements, and it is never an error here.

What it does not check
----------------------
* Imports built at run time (``importlib.import_module(name)``, ``__import__``). They are
  reported as warnings, so the reviewer can confirm each one by hand.
* Attribute names. A resolved module can still lack the attribute a ``from`` import asks for.

Exit status: 0 when every in-bundle import resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable

__all__ = ["check", "main"]


def _module_exists(roots: list[Path], dotted: str) -> bool:
    """True when ``dotted`` is a module, package or namespace directory under any root."""
    parts = dotted.split(".")
    for root in roots:
        base = root.joinpath(*parts)
        if base.is_dir() or base.with_suffix(".py").is_file():
            return True
    return False


def _package_of(path: Path, roots: list[Path]) -> list[str] | None:
    """Dotted package parts of the module at ``path``, or None if it is outside every root.

    A file ``root/a/b/c.py`` is module ``a.b.c`` in package ``a.b``; ``root/a/b/__init__.py``
    is package ``a.b`` itself. Relative imports resolve against the package.
    """
    for root in roots:
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        parts = list(rel.with_suffix("").parts)
        return parts[:-1]  # both root/a/b/c.py and root/a/b/__init__.py live in package a.b
    return None


def _imports(tree: ast.AST) -> Iterable[tuple[int, int, str | None, list[str]]]:
    """Yield (lineno, level, module, names) for every import statement, lazy ones included."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, 0, alias.name, []
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, node.level, node.module, [a.name for a in node.names]


def _dynamic_import_lines(tree: ast.AST) -> list[int]:
    """Line numbers of ``importlib.import_module(...)`` and ``__import__(...)`` calls."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name in ("import_module", "__import__"):
                lines.append(node.lineno)
    return lines


def check(files: list[Path], roots: list[Path]):
    """Scan ``files``; return (unresolved, external, dynamic).

    ``unresolved`` is a list of ``"file:line: message"`` strings (errors). ``external`` is the
    set of external top-level names seen. ``dynamic`` is a list of ``"file:line"`` warnings.
    """
    # Top-level names that live in a root are in-bundle; everything else is external.
    local_tops = {p.name.removesuffix(".py") for r in roots for p in r.iterdir()
                  if p.is_dir() or p.suffix == ".py"}
    unresolved: list[str] = []
    external: set[str] = set()
    dynamic: list[str] = []
    for f in files:
        tree = ast.parse(f.read_text(), filename=str(f))
        pkg = _package_of(f, roots)
        for line in _dynamic_import_lines(tree):
            dynamic.append(f"{f}:{line}")
        for lineno, level, module, names in _imports(tree):
            where = f"{f}:{lineno}"
            if level:
                # Relative import: level 1 is this file's package, level 2 its parent, ...
                if pkg is None:
                    continue  # a file outside every root (the adapter) has no package context
                if level - 1 > len(pkg):
                    unresolved.append(f"{where}: relative import climbs above the root")
                    continue
                base = pkg[: len(pkg) - (level - 1)]
                target = ".".join(base + ([module] if module else []))
                if target and not _module_exists(roots, target):
                    unresolved.append(f"{where}: relative import resolves to {target!r}, which does not exist")
                continue
            if module is None:
                continue
            if module.split(".")[0] not in local_tops:
                external.add(module.split(".")[0])
            elif not _module_exists(roots, module):
                unresolved.append(f"{where}: {module!r} does not exist under the shipped code")
    return unresolved, external, dynamic


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", action="append", type=Path, required=True,
                    help="a directory that is on PYTHONPATH at serve time (repeatable)")
    ap.add_argument("--file", action="append", type=Path, default=[],
                    help="an extra .py file to scan (the adapter), outside any root (repeatable)")
    args = ap.parse_args(argv)
    roots = [r for r in args.root]
    files = [p for r in roots for p in sorted(r.rglob("*.py"))] + list(args.file)
    unresolved, external, dynamic = check(files, roots)
    print(f"scanned {len(files)} files under {len(roots)} root(s)")
    print("external top-level imports (not checked here): " + ", ".join(sorted(external)))
    for d in dynamic:
        print(f"WARNING dynamic import, confirm by hand: {d}")
    for u in unresolved:
        print(f"UNRESOLVED {u}")
    print(f"{len(unresolved)} unresolved in-bundle import(s)")
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
