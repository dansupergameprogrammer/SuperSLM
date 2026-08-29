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

T-2371 (Brunel), D-SLM5017/D-SLM5021/D-SLM5018 S1. The original form of
`test_seven_switch_jump_table_symbols_preserve_every_named_branch` asserted
each function's branch count against a hardcoded literal (21, 7, 32, 61, 6,
17) parsed only from the function's own body -- so a header gaining an
enumerator with no matching arm left the literal (and the body's own count)
unchanged, and the pin stayed green through exactly the regression it
existed to catch. Five of the six restructured functions below were
genuinely exhaustive switches before the restructure (verified at source
against each enum's own header: `SslmForwardStatus` 32/32,
`SslmModelStatus` 61/61, `ConfigGeometryStatus` 6/6, `SslmSectionType`
21/21 for `IsKnownSectionType`) -- the state in which a compiler's own
`-Wswitch` warns on a newly added enumerator. The sixth, `SectionTypeName`,
was exhaustive only over the seventeen-enumerator `SslmSectionType` that
existed when it was authored; four enumerators (`CalibrationBand`,
`DeltaFoldScales`, `UFoldScales`, `DampedGreedyConstants`) were added to
the header afterward with no matching arm here, a real, pre-existing gap
this round closes in `src/proof_manifest.cpp` itself (D-SLM5018 S1) rather
than leaving unexhaustive. `ExpectedDtype`'s restructure similarly replaced
fourteen explicit case labels (all mapped to `Raw`) with an implicit
fallthrough default; this round restores all twenty-one as explicit
comparisons in `src/artifact.cpp` (same output for every input) so its own
text states the same completeness the switch it replaced did. All six are
therefore now genuinely exhaustive over their own enum, and this file
derives each one's EXPECTED set of named enumerators from the enum's own
header instead of a literal, asserting SET equality (not just a count, so
a duplicated comparison cannot silently stand in for a missing one) --
restoring exactly the property `-Wswitch` provided, checkable on every
platform including the ones that never had `-Wswitch` (MSVC's own `C4062`
is off at `/W4`), and proven both ways below: green against the real,
unmutated headers, and provably red against a header carrying one
unhandled enumerator, by construction.
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_ENGINE_ROOT, "src")
_INCLUDE = os.path.join(_ENGINE_ROOT, "include", "superslm")

# (source-relative path, function name, header-relative path or None for a
# control-flow restructure with no name-table of its own, enum name or None)
_RESTRUCTURED = [
    ("artifact.cpp", "IsKnownSectionType", "artifact.h", "SslmSectionType"),
    ("artifact.cpp", "ExpectedDtype", "artifact.h", "SslmSectionType"),
    (os.path.join("forward", "checked_chain_funnel.cpp"), "SslmForwardStatusName",
     "checked_chain_funnel.h", "SslmForwardStatus"),
    ("model.cpp", "SslmModelStatusName", "model.h", "SslmModelStatus"),
    ("model.cpp", "ValidateConfigGeometryJoin", None, None),
    ("proof_manifest.cpp", "ConfigGeometryStatusName", "proof_manifest.h", "ConfigGeometryStatus"),
    ("proof_manifest.cpp", "SectionTypeName", "artifact.h", "SslmSectionType"),
]

_SYNTHETIC_ENUMERATOR = "T2371SyntheticMutationEnumerator"


def _read(rel_path: str) -> str:
    with open(os.path.join(_SRC, rel_path), encoding="utf-8") as f:
        return f.read()


def _read_header(rel_path: str) -> str:
    with open(os.path.join(_INCLUDE, rel_path), encoding="utf-8") as f:
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


def _enum_body_text(header_text: str, enum_name: str) -> str:
    """Returns the brace-balanced body text (braces included) of
    `enum class enum_name { ... };` inside header_text."""
    m = re.search(r"enum class\s+" + re.escape(enum_name) + r"\b[^{]*\{", header_text)
    assert m, "could not locate `enum class {}` in the given header text".format(enum_name)
    start = m.end() - 1  # index of the opening '{'
    depth = 0
    i = start
    while i < len(header_text):
        if header_text[i] == "{":
            depth += 1
        elif header_text[i] == "}":
            depth -= 1
            if depth == 0:
                return header_text[start:i + 1]
        i += 1
    raise AssertionError("unbalanced braces scanning enum {!r}'s own body".format(enum_name))


def _enumerator_names(enum_body: str) -> list:
    """Every enumerator identifier declared inside a `{ ... }` enum body
    (braces included), comments stripped, one name per line -- the
    convention every enum this file reads is written in (confirmed at
    source for all four headers below)."""
    inner = enum_body[1:-1]
    inner = re.sub(r"//.*", "", inner)
    inner = re.sub(r"/\*.*?\*/", "", inner, flags=re.DOTALL)
    names = re.findall(r"^\s*([A-Za-z_]\w*)\s*(?:=\s*[^,]+)?,?\s*$", inner, re.MULTILINE)
    return [n for n in names if n]


def _header_enumerator_set(header_rel: str, enum_name: str) -> set:
    return set(_enumerator_names(_enum_body_text(_read_header(header_rel), enum_name)))


def _function_named_enumerator_set(rel_path: str, name: str) -> set:
    """Every bare enumerator name this function's own body compares against
    with `==` (`EnumType::Name` reduced to `Name`) -- the set this
    function's own text demonstrates it considers, independent of how many
    times any one of them is compared."""
    text = _read(rel_path)
    body = _function_body(text, name)
    qualified = re.findall(r"==\s*(\w+::\w+)", body)
    return {q.split("::", 1)[1] for q in qualified}


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
    for rel_path, name, _header_rel, _enum_name in _RESTRUCTURED:
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
    still names exactly the set of enumerators the switch it replaces named
    -- a restructure that silently dropped or duplicated a branch would be a
    behavior change this ticket's own contract forbids ("must not change
    any observable behaviour"). The EXPECTED set is derived from each
    enum's own header (D-SLM5021), never a literal: `IsKnownSectionType`,
    `ExpectedDtype`, and `SectionTypeName` are checked against
    `SslmSectionType`'s own full enumerator set (`artifact.h`);
    `SslmForwardStatusName` against `SslmForwardStatus`
    (`checked_chain_funnel.h`); `SslmModelStatusName` against
    `SslmModelStatus` (`model.h`); `ConfigGeometryStatusName` against
    `ConfigGeometryStatus` (`proof_manifest.h`). `ValidateConfigGeometryJoin`
    is a control-flow restructure rather than a name table and is checked by
    its own narrower cell below instead. A header gaining an enumerator with
    no matching arm here changes the header side of this comparison and not
    the body side, which is exactly what makes this comparison catch it --
    proven by construction in
    `test_enumerator_pin_is_mutation_provable` below.
    """
    for rel_path, name, header_rel, enum_name in _RESTRUCTURED:
        if header_rel is None:
            continue
        function_set = _function_named_enumerator_set(rel_path, name)
        header_set = _header_enumerator_set(header_rel, enum_name)
        missing = header_set - function_set
        extra = function_set - header_set
        assert not missing and not extra, (
            "{}::{} -- named enumerator set does not match {}'s own "
            "definition of {} enumerators: missing {}, unexpected {} -- "
            "the restructure may have dropped a branch, or {} gained an "
            "enumerator with no matching arm here".format(
                rel_path, name, header_rel, enum_name,
                sorted(missing), sorted(extra), enum_name)
        )


def test_enumerator_pin_is_mutation_provable():
    """Must-reject, by construction (D-SLM5021: "mutation-prove it: add an
    enumerator without an arm and show the cell goes red"). For each of the
    four enums the cell above reads, this test builds a MUTATED COPY of the
    real header text -- in memory only, nothing on disk is written or the
    real source tree touched -- carrying one synthetic enumerator with no
    matching arm anywhere, and confirms the exact comparison the cell above
    performs would report that enumerator as missing for every function
    indexed over that enum. This is the demonstration that the cell above
    is a live regression guard rather than a comparison that happens to
    hold today: it proves the pin goes red on the shape S1 warns about,
    not merely that it is currently green.
    """
    mutated_sets_by_enum = {}
    checked_at_least_one = False
    for rel_path, name, header_rel, enum_name in _RESTRUCTURED:
        if header_rel is None:
            continue
        key = (header_rel, enum_name)
        if key not in mutated_sets_by_enum:
            real_header_text = _read_header(header_rel)
            real_enum_body = _enum_body_text(real_header_text, enum_name)
            real_set = set(_enumerator_names(real_enum_body))
            mutated_enum_body = (
                real_enum_body[:-1].rstrip().rstrip(",")
                + ",\n\t" + _SYNTHETIC_ENUMERATOR + ",\n}"
            )
            mutated_header_text = real_header_text.replace(
                real_enum_body, mutated_enum_body, 1)
            mutated_set = set(_enumerator_names(
                _enum_body_text(mutated_header_text, enum_name)))
            assert mutated_set - real_set == {_SYNTHETIC_ENUMERATOR}, (
                "mutation fixture FAILED to add exactly one synthetic "
                "enumerator to {}'s own body -- cannot prove the pin "
                "catches this shape until the fixture itself is trusted"
                .format(enum_name)
            )
            mutated_sets_by_enum[key] = mutated_set

        mutated_set = mutated_sets_by_enum[key]
        function_set = _function_named_enumerator_set(rel_path, name)
        missing_against_mutated = mutated_set - function_set
        assert _SYNTHETIC_ENUMERATOR in missing_against_mutated, (
            "{}::{}'s own comparison against a {} carrying one unhandled "
            "enumerator did NOT flag it as missing -- the pin above would "
            "not catch this exact regression".format(
                rel_path, name, enum_name)
        )
        checked_at_least_one = True
    assert checked_at_least_one, (
        "internal inconsistency: no entry in _RESTRUCTURED carries a "
        "header/enum pair -- this mutation proof never ran"
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
