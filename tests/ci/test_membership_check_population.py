"""Curie's population-validation suite for S-HARDEN-7's `membership-check` CI
job (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_Design-2026-07-23.md Sec3.2,
Sec3.3; T-411).

Per the commission: the adversary's four strike probes (Claude/Loki/
superslm-sharden678-bundle-strike-2026-07-23-probe-*) ARE the
independently-found validation population StandardsDocument Sec4 requires --
they derived the PRIOR rule's population three ways and proved it collapsed
to 12/13/14 depending on scan strategy, which is what forced the rule's
correction (condition 2, condition 4(b)) in the first place. This suite
reuses that same three-way scan-strategy methodology, extended (in
tests/ci/derive_bad_alloc_membership.py) to the CORRECTED, four-condition
rule, and confirms the corrected rule's own defining claim: the population is
no longer a function of which translation unit happens to be scanned.

Requires a clang++ on PATH capable of `-Xclang -ast-dump=json` (Clang 18.1.8
locally; GitHub's `ubuntu-latest` hosted runner ships one per its published
software manifest, independent of S-HARDEN-6, design Sec3.1's "Independence
from S-HARDEN-6" note). Every test below is skipped, not failed, when no
such clang++ is available -- an environment gap is not a population defect.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import derive_bad_alloc_membership as dbam


def _clang_available() -> bool:
    try:
        dbam.run_clang_ast_dump(
            os.path.join(dbam._INCLUDE_DIR, "superslm", "sha256.h")
        )
        return True
    except dbam.ClangUnavailable:
        return False


_HAVE_CLANG = _clang_available()
requires_clang = pytest.mark.skipif(
    not _HAVE_CLANG,
    reason="no clang++ on PATH capable of -Xclang -ast-dump=json "
    "(set SUPERSLM_CLANGXX to an explicit path)",
)


def _load_pinned_oracle() -> set[tuple[str, int, str]]:
    path = os.path.join(dbam._THIS_DIR, "bad_alloc_membership_expected.txt")
    entries: set[tuple[str, int, str]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header, line_no, name = line.split(":", 2)
            entries.add((header, int(line_no), name))
    return entries


def _keyset(pop: list[dict]) -> set[tuple[str, int, str]]:
    return {(m["header"], m["line"], m["name"]) for m in pop}


@requires_clang
def test_pinned_oracle_has_nineteen_sites():
    """T-1475: proof_manifest.h's JsonEscape was promoted out of its
    anonymous namespace and into the header (T-1449), making it the
    membership rule's 19th derived member -- confirmed by executing the
    corrected rule (see derive_bad_alloc_membership.py's -std=c++20 fix,
    T-1476) against the real headers on disk, not asserted. Design Sec3.1's
    table itself (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_Design-
    2026-07-23.md) still states eighteen as of this change and is owed a
    matching amendment outside this suite's writable surface; this pin
    tracks the mechanically-derived population, which is the number this
    gate and the oracle file must agree on."""
    oracle = _load_pinned_oracle()
    assert len(oracle) == 19, (
        f"the pinned oracle should carry the nineteen sites the corrected rule "
        f"derives as of T-1475 (JsonEscape's promotion into proof_manifest.h); "
        f"has {len(oracle)} -- regenerate tests/ci/bad_alloc_membership_expected.txt "
        f"only after confirming the population change is deliberate, never silently"
    )


@requires_clang
def test_per_header_scan_matches_pinned_oracle():
    """The regression pin: if a header changes in a way that moves the
    membership population, this fails loudly and the oracle file must be
    regenerated deliberately (never silently) -- the same discipline
    check_provenance.py applies to the vendored reference's hash."""
    oracle = _load_pinned_oracle()
    derived = _keyset(dbam.derive_population_per_header())
    missing = oracle - derived
    extra = derived - oracle
    assert not missing and not extra, (
        f"per-header scan diverges from the pinned oracle -- missing {missing}, "
        f"extra {extra}. If this is an intentional header change, regenerate "
        f"tests/ci/bad_alloc_membership_expected.txt and update design Sec3.1's "
        f"table in the same change."
    )


@requires_clang
def test_scan_strategy_independence_all_three_derive_identical_population():
    """The direct re-run of the property the strike found FALSE of the prior
    rule (Claude/Loki/superslm-sharden678-bundle-strike-2026-07-23.md Sec4.3:
    per-header scan derived 14, single-TU derived 13, an odr-using TU derived
    12 -- three different answers to "run the predicate over the nine
    headers"). The corrected rule's defining claim is that this no longer
    happens; this test is that claim, executed, not asserted."""
    per_header = _keyset(dbam.derive_population_per_header())
    single_tu = _keyset(dbam.derive_population_single_tu())
    odr_tu = _keyset(dbam.derive_population_single_tu(force_odr_use_artifact_moves=True))

    assert per_header == single_tu, (
        f"per-header vs single-all-headers-TU diverge: "
        f"only-in-per-header={per_header - single_tu}, only-in-single-tu={single_tu - per_header}"
    )
    assert per_header == odr_tu, (
        f"per-header vs odr-using-TU diverge (this is exactly the artifact.h:140/141 "
        f"fracture the strike found): only-in-per-header={per_header - odr_tu}, "
        f"only-in-odr-tu={odr_tu - per_header}"
    )


@requires_clang
def test_artifact_move_special_members_are_excluded_under_every_strategy():
    """The specific fracture site (Claude/Loki strike Sec4.1/4.2):
    artifact.h:140 (move constructor) and :141 (move-assignment operator)
    must NOT appear in the population under any of the three scan
    strategies -- the prior rule admitted them under two of the three."""
    for pop, label in (
        (dbam.derive_population_per_header(), "per_header"),
        (dbam.derive_population_single_tu(), "single_tu"),
        (dbam.derive_population_single_tu(force_odr_use_artifact_moves=True), "odr_tu"),
    ):
        hit_lines = {m["line"] for m in pop if m["header"] == "artifact.h"}
        assert 140 not in hit_lines, f"{label}: artifact.h:140 (move ctor) wrongly admitted"
        assert 141 not in hit_lines, f"{label}: artifact.h:141 (move-assign) wrongly admitted"


@requires_clang
def test_vitality_injected_nineteenth_function_is_flagged(tmp_path):
    """Guard-vitality cell (design Sec3.2, "New cell -- the membership-check
    job's own discriminating power"): a throwaway function satisfying the
    rule, added to a scratch copy of the include tree, must be flagged; once
    removed, the population returns to exactly the pinned eighteen. Proves
    THIS reference tool's own discriminating power -- StandardsDocument
    Sec4's "a check that has never been shown able to fail is not shown to
    cover anything," applied to the independent derivation tool itself, ahead
    of the production CI gate that must reproduce it."""
    import shutil

    scratch_include = tmp_path / "include"
    shutil.copytree(dbam._INCLUDE_DIR, scratch_include)

    artifact_h = scratch_include / "superslm" / "artifact.h"
    original = artifact_h.read_text(encoding="utf-8")
    assert "SUPERSLM_TEST_INJECTED_NINETEENTH_FUNCTION" not in original

    injected = original.replace(
        "} // namespace superslm",
        "// Throwaway vitality probe (Curie, S-HARDEN-7 Sec3.2): a public,\n"
        "// non-noexcept, non-deleted static function taking a\n"
        "// (const uint8_t*, size_t) pair -- condition 4(a) satisfied,\n"
        "// deliberately left unwrapped. Must be flagged by the derivation.\n"
        "struct SUPERSLM_TEST_INJECTED_NINETEENTH_FUNCTION {\n"
        "\tstatic SslmStatus Probe(const uint8_t* data, size_t size);\n"
        "};\n"
        "} // namespace superslm",
        1,
    )
    assert injected != original, "the injection marker's anchor text was not found in artifact.h"
    artifact_h.write_text(injected, encoding="utf-8")

    population_with_injection = dbam.derive_population_from_headers_dir(str(tmp_path))
    injected_hits = [m for m in population_with_injection if m["name"] == "Probe"]
    assert len(injected_hits) == 1, (
        f"expected exactly one derived member named Probe after injection, got "
        f"{len(injected_hits)}: {injected_hits}"
    )
    assert injected_hits[0]["header"] == "artifact.h"

    # Every other site must be unaffected -- the injected function adds
    # exactly one member, it does not perturb the other eighteen.
    oracle = _load_pinned_oracle()
    derived_without_probe = {
        (m["header"], m["line"], m["name"]) for m in population_with_injection if m["name"] != "Probe"
    }
    assert derived_without_probe == oracle

    # Restore and confirm the population returns to exactly the pinned eighteen.
    artifact_h.write_text(original, encoding="utf-8")
    population_restored = _keyset(dbam.derive_population_from_headers_dir(str(tmp_path)))
    assert population_restored == oracle


# ---------------------------------------------------------------------------
# The production gate (design Sec3.1: "tools/ci/check_bad_alloc_contract.py")
# is built, and design Sec3.3's rename-and-wrap has landed for every one of
# the eighteen derived members (src/bad_alloc_wrap.h's WrapBadAllocContract,
# included by all five src/*.cpp files the population's members live in). The
# reference tool above (derive_bad_alloc_membership.py) proves the POPULATION
# is derivable and stable; the cells below confirm the CI GATE that reproduces
# that population reports it correctly and, now that every member is wrapped,
# reports zero unwrapped members and exits 0 -- design Sec3.3 step 3's other
# half (the RED-state half, "confirm it fails naming exactly the eighteen
# sites," was proven at the design's own authoring and is not re-proven here;
# re-creating that state would mean reverting the shipped wrap).
#
# CLI contract: `tools/ci/check_bad_alloc_contract.py --list-unwrapped` prints
# one `header:line:name` row per currently-unwrapped member of the derived
# population and exits 1 if any exist, 0 if none do.
# ---------------------------------------------------------------------------


def test_production_membership_check_tool_exists():
    """T-411's CI gate is built and wired into .github/workflows/tests.yml."""
    repo_root = dbam._REPO_ROOT
    tool_path = os.path.join(repo_root, "tools", "ci", "check_bad_alloc_contract.py")
    assert os.path.isfile(tool_path), (
        "tools/ci/check_bad_alloc_contract.py must exist -- design Sec3.1's "
        "membership-check CI gate."
    )


@requires_clang
def test_production_membership_check_reports_zero_unwrapped_members():
    """S-HARDEN-7's rename-and-wrap has landed for every one of the eighteen
    sites, so the gate's own --list-unwrapped must report none, and exit 0."""
    repo_root = dbam._REPO_ROOT
    tool_path = os.path.join(repo_root, "tools", "ci", "check_bad_alloc_contract.py")
    result = subprocess.run(
        [sys.executable, tool_path, "--list-unwrapped"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"check_bad_alloc_contract.py --list-unwrapped must exit 0 once every "
        f"derived member is wrapped; got exit {result.returncode}, stdout:\n"
        f"{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"expected the tool's own OK confirmation in stdout; got:\n{result.stdout}"
    )
