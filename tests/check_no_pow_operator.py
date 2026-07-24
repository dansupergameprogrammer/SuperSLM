#!/usr/bin/env python3
"""AST ban on the `**` operator in the two fixture generators that write
committed golden bytes (S-HARDEN-5 M1 fold; Claude/Vitruvius/
SuperSLM_SHARDEN678_Bundle_Design-2026-07-23.md Sec5.2(c), Sec5.3, Sec5.4).

`tests/gen_intmath_fixtures.py:484`'s `scale ** 2` compiled to a libm
`pow(scale, 2.0)` call on the emitted golden-fixture path -- exactly what the
design's own "no sin/cos/log/exp/pow call executes anywhere on the
regeneration path" claim was supposed to guarantee was already gone (M1). A
grep for the two-character operator would also flag the file's two
pre-existing, unrelated, exact-integer `**` uses (`10 ** 12` at :428,
`4 ** m` at :436) -- an AST walk over `ast.BinOp(op=ast.Pow)` is precise
regardless of whether either operand is a literal, a float, or an int
expression, so this check does not need to distinguish them: after M1's fix
rewrites all three sites to eliminate `**` entirely (Sec5.2(c)), the ban is
total and unconditional, with zero surviving exceptions to special-case.

This does NOT walk tests/reference/ (the vendored, hash-pinned reference
implementation, which is allowed to contain `**` internally -- it is pinned
material this project does not author or extend, not code whose job is
"compute a value and emit it as a golden byte").

Exit code 0 iff neither generator contains any `**` use; 1 otherwise, naming
every file:line hit.
"""
from __future__ import annotations

import ast
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# The two generators that write committed fixture bytes (design Sec5.2(c)).
# Relative to this file's directory.
_CHECKED_FILES = (
    "gen_intmath_fixtures.py",
    "gen_matmul_fixtures.py",
)


class _PowFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[int] = []

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802 (ast.NodeVisitor convention)
        if isinstance(node.op, ast.Pow):
            self.hits.append(node.lineno)
        self.generic_visit(node)


def find_pow_uses(path: str) -> list[int]:
    """Every source line in `path` containing an `ast.BinOp(op=ast.Pow)`,
    ascending. Empty if none."""
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    finder = _PowFinder()
    finder.visit(tree)
    return sorted(finder.hits)


def main() -> int:
    failures: list[str] = []
    for rel_path in _CHECKED_FILES:
        abs_path = os.path.join(_THIS_DIR, rel_path)
        if not os.path.isfile(abs_path):
            failures.append(f"{rel_path}: file not found at {abs_path}")
            continue
        for line in find_pow_uses(abs_path):
            failures.append(f"{rel_path}:{line}: uses the ** operator")

    if failures:
        print("check_no_pow_operator.py: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"check_no_pow_operator.py: OK -- zero ** uses across {len(_CHECKED_FILES)} generators")
    return 0


if __name__ == "__main__":
    sys.exit(main())
