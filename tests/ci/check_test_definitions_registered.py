#!/usr/bin/env python3
"""CI/local structural check (T-1574 test suite split, §3, §7.2, D-SLM592).
Closes the T-1627 Loki fracture: every check derived from the `SSLM_TEST`
macro is structurally blind to a test-shaped function that was never given
the macro at all, because both of its compared operands are themselves
functions of the macro's own presence. This check's two operands are not:
the "defined" side is every niladic, `void`-returning, file/anonymous-
namespace-scope function definition in the checked source files -- read as
raw `[static] void Name() { ... }` text, never as `SSLM_TEST(` -- and the
"registered" side is the running binary's own `--list-tests` report. Neither
operand is a function of whether the macro was used, which is exactly why
this check, and only this check, can detect its absence.

Deliberately does NOT require the body to contain a CHECK/CHECK_MSG token:
measured against the real pre-migration tree, 132 of the suite's 461 known
tests delegate their assertion to a shared helper and contain no CHECK token
in their own body -- a body filter would have hidden exactly the population
this check exists to catch.

The boundary in the unsafe direction, stated because it is not zero: "test-
shaped" means niladic and void-returning, so a hand-written function that
returns bool or takes a parameter is outside the defined set this check
covers, and SSLM_TEST's own void(*)() contract could never have registered it
either -- the residual is a function someone intends as a test, writes in a
shape the registry could never accept, and never notices is not running.

Exit 0 iff every test-shaped definition is present in the binary's own
--list-tests report. Exit 1 otherwise, naming every definition that is not.
"""
from __future__ import annotations

import os
import subprocess
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import _source_scan as scan  # noqa: E402


def test_shaped_definitions() -> dict[str, tuple[str, int]]:
    """name -> (file, line) for every niladic, void-returning,
    file/anonymous-namespace-scope function definition across the checked
    source files. A definition that also carries SSLM_TEST does not match
    this scan at all (its raw text reads `SSLM_TEST(Name, Order) { ... }`,
    not `void Name() { ... }`) -- this population is exactly the complement
    of what the macro produced."""
    result: dict[str, tuple[str, int]] = {}
    for path in scan.checked_source_files():
        text = open(path, encoding="utf-8").read()
        lines = scan.line_of(text)
        for fn in scan.find_function_definitions(text):
            if not fn["is_niladic_void"]:
                continue
            rel = os.path.relpath(path, scan.REPO_ROOT)
            line = lines[fn["open"]]
            # First definition wins for reporting; a duplicate name is a
            # different fault (RegisterTest's own duplicate-name abort would
            # catch it if both were registered) and out of this check's scope.
            result.setdefault(fn["name"], (rel, line))
    return result


def registered_test_names(binary: str) -> set[str]:
    proc = subprocess.run([binary, "--list-tests"], cwd=scan.REPO_ROOT, capture_output=True, text=True,
                           check=True)
    names = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _order = line.rsplit(",", 1)
        names.add(name)
    return names


def main() -> int:
    defined = test_shaped_definitions()
    # Currency: the binary must be at least as new as every file this scan
    # just read, or "registered" is a claim about a stale build rather than
    # the tree the "defined" side above was actually computed from (T-1632
    # Significant 2).
    binary = scan.locate_or_build_binary(scan.checked_source_files())
    registered = registered_test_names(binary)

    missing = sorted(set(defined) - registered)
    if missing:
        print("check_test_definitions_registered.py: FAILED", file=sys.stderr)
        print(
            f"  {len(missing)} test-shaped definition(s) exist in source but are NOT registered "
            f"(never given SSLM_TEST):",
            file=sys.stderr,
        )
        for name in missing:
            rel, line = defined[name]
            print(f"  - {name} ({rel}:{line})", file=sys.stderr)
        return 1

    print(
        f"check_test_definitions_registered.py: OK -- all {len(defined)} test-shaped definitions "
        f"are registered ({len(registered)} total registered)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
