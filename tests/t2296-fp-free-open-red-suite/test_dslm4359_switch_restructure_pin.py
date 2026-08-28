"""T-2367 (Brunel), D-SLM5001 item (3) -- pin for the seven switch-jump-table
symbols' own restructure (D-SLM4359, D-SLM4988).

WHY THIS FILE EXISTS. `test_check_fp_free_scan.py`'s own
`test_dslm4359_seven_switch_jump_table_symbols_must_not_block_gate`
(T-2366) scans the real, already-built corpus at
`D:/SuperSLM/.worktrees/optb-build` -- a read-only reference build this
ticket's own environment section names as "read-only to you, so configure
your own." That corpus is compiled from engine commit `a1df129`, which
predates this build round's own source changes entirely; it can never
reflect this remedy until whoever owns that shared directory rebuilds it
from a commit that includes this round's work, and rebuilding it is outside
this ticket's writable scope. The remedy itself is verified directly this
session against a freshly configured build
(`D:/SuperSLM/.worktrees/t2367-bld`, `cmake -S ... -B ... && cmake --build
... --target superslm --config Release`): 0 REJECT, 0 REFUSE under checks
(A)/(B) for all seventeen objects, where the pre-remedy build read 264
REJECT / 1 REFUSE (`Claude/Brunel/t2367-fp-gate-build-round-2026-08-28.md`).

This file pins the remedy at the one level that does not depend on any
external build directory staying current: the SOURCE ITSELF. Per
`StandardsDocument.md`'s standing law that a fix round's newest production
change is structurally the one nobody pins, this is a NEW cell for a remedy
this round actually landed (the seven-symbol switch-to-if-chain
restructure), not a re-statement of the existing, environment-dependent
seven-symbol sweep above.
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_ENGINE_ROOT, "src")

# (source-relative path, function name, expected named branches, fallback string
# or None for a function that returns a bool/enum default rather than "?"/"Unknown")
_RESTRUCTURED = [
    ("artifact.cpp", "IsKnownSectionType", 21, None),
    ("artifact.cpp", "ExpectedDtype", 7, None),
    (os.path.join("forward", "checked_chain_funnel.cpp"), "SslmForwardStatusName", 32, '"?"'),
    ("model.cpp", "SslmModelStatusName", 61, '"Unknown"'),
    ("model.cpp", "ValidateConfigGeometryJoin", None, None),
    ("proof_manifest.cpp", "ConfigGeometryStatusName", 6, '"?"'),
    ("proof_manifest.cpp", "SectionTypeName", 17, '"Unknown"'),
]


def _read(rel_path: str) -> str:
    with open(os.path.join(_SRC, rel_path), encoding="utf-8") as f:
        return f.read()


def _function_body(text: str, name: str) -> str:
    """Returns the brace-balanced body of the first top-level function
    definition named `name` found in `text` (a free function or a function
    with a qualified return type on the same line -- every one of the seven
    below is declared this way; no method body, template, or lambda needs
    to be excluded from this narrow, single-purpose scan)."""
    m = re.search(r"^\S.*[ *&]" + re.escape(name) + r"\([^;]*?\)\s*(?:noexcept\s*)?\{",
                  text, re.MULTILINE)
    assert m, "could not locate a definition of {!r} in the given source text".format(name)
    start = m.end() - 1  # index of the opening '{'
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces scanning {!r}'s own body".format(name))


def test_seven_switch_jump_table_symbols_contain_no_switch_statement():
    """Must-accept, source-level: none of D-SLM4359's own seven symbols'
    function bodies contain a `switch` statement any more -- the class of
    construct whose compiler-emitted jump-table dispatch is what carried an
    unaccountable byte range (REFUSE, `?ExpectedDtype`/`?IsKnownSectionType`)
    or an unvetted indirect edge (REJECT via check (C), the other five) into
    these symbols' own compiled extents (design Sec4.1, D-SLM4359/D-SLM4988).
    A `switch` keyword reappearing in any of these seven bodies is exactly
    the regression this remedy exists to prevent, whether reintroduced by a
    future edit or by a merge that reverts this round's own change.
    """
    offenders = []
    for rel_path, name, _branches, _fallback in _RESTRUCTURED:
        text = _read(rel_path)
        body = _function_body(text, name)
        if re.search(r"\bswitch\s*\(", body):
            offenders.append("{}::{}".format(rel_path, name))
    assert offenders == [], (
        "the following restructured symbols still contain a `switch` "
        "statement in their own body -- the exact construct D-SLM4359's "
        "restructure removes, whose compiler-emitted jump-table dispatch is "
        "what caused these symbols to REFUSE/REJECT the ship gate: "
        "{}".format(offenders)
    )


def test_seven_switch_jump_table_symbols_preserve_every_named_branch():
    """Must-accept: the if-chain each restructured function was rewritten to
    still names every one of the enumerators/statuses the switch it replaces
    named, in the same count -- a restructure that silently dropped a branch
    would be a behavior change this ticket's own contract forbids ("must not
    change any observable behaviour"). Counted by the number of `==`
    equality comparisons against a qualified enumerator name in each
    function's own body, which is exactly one per branch for every function
    in this list except `ValidateConfigGeometryJoin` (a control-flow
    restructure rather than a name table, checked by its own narrower cell
    below instead).
    """
    for rel_path, name, expected_branches, _fallback in _RESTRUCTURED:
        if expected_branches is None:
            continue
        text = _read(rel_path)
        body = _function_body(text, name)
        comparisons = re.findall(r"==\s*\w+::\w+", body)
        assert len(comparisons) == expected_branches, (
            "{}::{} -- expected {} named-enumerator comparisons (one per "
            "branch the original switch named), found {}: the restructure "
            "may have dropped or duplicated a branch".format(
                rel_path, name, expected_branches, len(comparisons))
        )


def test_validate_config_geometry_join_preserves_every_status_mapping():
    """Must-accept: `ValidateConfigGeometryJoin`'s own restructured if-chain
    still maps all four distinct outcomes (Ok, KvHeadsExceedsHeads,
    HeadsNotDivisibleByKv, HiddenSizeGeometryMismatch) plus the
    ZeroAttentionHeads/ZeroKeyValueHeads fallthrough and the unrecognized-
    status fallback -- six ConfigGeometryStatus values and one default,
    exactly as the switch it replaces did.
    """
    text = _read("model.cpp")
    body = _function_body(text, "ValidateConfigGeometryJoin")
    for status in ("Ok", "KvHeadsExceedsHeads", "HeadsNotDivisibleByKv",
                   "HiddenSizeGeometryMismatch", "ZeroAttentionHeads", "ZeroKeyValueHeads"):
        assert "ConfigGeometryStatus::{}".format(status) in body, (
            "ValidateConfigGeometryJoin no longer names ConfigGeometryStatus::{} "
            "-- the restructure may have dropped a branch".format(status)
        )
    assert "unrecognized ConfigGeometryStatus" in body, (
        "ValidateConfigGeometryJoin no longer carries its own "
        "unrecognized-status fallback"
    )
