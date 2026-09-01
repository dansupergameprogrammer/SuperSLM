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
import re
import subprocess
import sys
from collections import Counter

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


def _load_pinned_oracle() -> Counter[tuple[str, str]]:
    """T-2458 (Claude/Poirot/5c82f92-t2453-ask5-tracka-confirmation.md,
    Critical 1 + Observation 2, D-SLM5528): the oracle file is keyed on
    `header:name`, one line per member, WITHOUT a line column -- the line
    component re-derived stale eight times across this population's history
    (see the oracle file's own header) and never once caught a membership
    change the name alone did not. Returned as a Counter, not a set: the
    oracle's identity is a MULTISET over (header, name), because a plain set
    would silently absorb a same-named member joining a header that already
    has one (T-2125's real `model.h:Parse` growth -- two Parse entries
    already existed and a third joined, taking that key two to three under
    the multiset, T-2467/D-SLM5560 correcting this docstring's own prior
    "fourth" miscount -- verified against git history to be exactly this
    shape -- see the oracle file's own header)."""
    path = os.path.join(dbam._THIS_DIR, "bad_alloc_membership_expected.txt")
    entries: Counter[tuple[str, str]] = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header, name = line.split(":", 1)
            entries[(header, name)] += 1
    return entries


def _multiset(pop: list[dict]) -> Counter[tuple[str, str]]:
    """Projects a derived population down to the (header, name) MULTISET the
    oracle is keyed on (see `_load_pinned_oracle`) -- line dropped, counted
    with multiplicity so a same-named member joining or leaving a header is
    still caught even where that header already carries members of that
    name."""
    return Counter((m["header"], m["name"]) for m in pop)


@requires_clang
def test_pinned_oracle_has_twenty_sites():
    """T-1475: proof_manifest.h's JsonEscape was promoted out of its
    anonymous namespace and into the header (T-1449), making it the
    membership rule's 19th derived member. T-2125: SslmAmplifyingFoldScaleView
    <Kind>::Parse (model.h) -- the T-2021/T-2029 B0b runtime-additive-LoRA
    adapter view's own parse entry point -- is the 20th; found by this suite's
    own regression pin (test_per_header_scan_matches_pinned_oracle) reddening
    against the real header with zero test-side change, confirmed a real,
    legitimate population growth (not a stale checker), and closed by a
    rename-and-wrap fix in src/model.h/model.cpp in the same round (the
    member was calling its own parse logic directly, with no
    WrapBadAllocContract around it, despite its own header comment already
    claiming the S-HARDEN-7 convention). Confirmed by executing the corrected
    rule against the real headers on disk, not asserted. Design Sec3.1's
    table itself (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_Design-
    2026-07-23.md) still states eighteen as of this change and is owed a
    matching amendment outside this suite's writable surface; this pin
    tracks the mechanically-derived population, which is the number this
    gate and the oracle file must agree on."""
    oracle = _load_pinned_oracle()
    total = sum(oracle.values())
    assert total == 20, (
        f"the pinned oracle should carry the twenty sites the corrected rule "
        f"derives as of T-2125 (SslmAmplifyingFoldScaleView<Kind>::Parse joining "
        f"proof_manifest.h's own JsonEscape-era nineteen); has {total} -- "
        f"regenerate tests/ci/bad_alloc_membership_expected.txt only after "
        f"confirming the design's own table changed, never silently"
    )


@requires_clang
def test_per_header_scan_matches_pinned_oracle():
    """The regression pin: if a header changes in a way that moves the
    membership population, this fails loudly and the oracle file must be
    regenerated deliberately (never silently) -- the same discipline
    check_provenance.py applies to the vendored reference's hash.

    T-2458 (D-SLM5528): compares MULTISETS over (header, name), not sets --
    line dropped from both sides. Counter subtraction already gives only the
    positive-count differences, so `missing`/`extra` below report exactly the
    members (and, for a duplicate name, the excess/deficit COUNT) that moved,
    the same diagnostic shape the prior (header, line, name) set comparison
    gave, minus the line noise that never once discriminated a real change."""
    oracle = _load_pinned_oracle()
    derived = _multiset(dbam.derive_population_per_header())
    missing = oracle - derived
    extra = derived - oracle
    assert not missing and not extra, (
        f"per-header scan diverges from the pinned oracle -- missing {missing}, "
        f"extra {extra}. If this is an intentional header change, regenerate "
        f"tests/ci/bad_alloc_membership_expected.txt and update design Sec3.1's "
        f"table in the same change."
    )


@requires_clang
def test_derived_population_list_length_matches_the_oracle():
    """T-2125 fix round (Poirot 242dc12-t2125-ci-drift-review.md, Significant
    4): at the time, `test_per_header_scan_matches_pinned_oracle` above
    compared KEYSETS -- (header, line, name) triples -- so it stayed green
    even while `_dedup_sort`'s old key silently stopped deduplicating a class
    template's own member: `derive_population_per_header()` returned 22
    entries for this 20-site population (one row per
    `ClassTemplateSpecializationDecl` Clang emits for
    `SslmAmplifyingFoldScaleView<Kind>::Parse`'s two real instantiations plus
    the primary template's own declaration), and nothing here compared the
    LIST's own length against the oracle it is regenerated from. (T-2458,
    D-SLM5528: `test_per_header_scan_matches_pinned_oracle` now compares a
    (header, name) MULTISET rather than a (header, line, name) SET, which
    would itself have caught this specific historical bug -- three identical
    `(header, name)` duplicates raise the count, where three identical
    `(header, line, name)` triples collapse to one set element and do not.
    This test still guards the layer beneath that projection: a correctness
    property of `_dedup_sort`'s own key, independent of which final form the
    oracle comparison takes.) `python tests/ci/derive_bad_alloc_membership.py
    --oracle` is the command
    the oracle file's own header names as its regeneration command; this pins
    that the number that command's LIST has is the population's true size,
    not the instantiation count of whichever member happens to be a
    template."""
    oracle = _load_pinned_oracle()
    total = sum(oracle.values())
    derived = dbam.derive_population_per_header()
    assert len(derived) == total, (
        f"derive_population_per_header() returned {len(derived)} entries for "
        f"a {total}-member pinned oracle -- the list is no longer the same "
        f"size as its own deduplicated keyset, which means _dedup_sort's key "
        f"is admitting more than one row per (header, line, name)"
    )


@requires_clang
def test_scan_strategy_independence_all_three_derive_identical_population():
    """The direct re-run of the property the strike found FALSE of the prior
    rule (Claude/Loki/superslm-sharden678-bundle-strike-2026-07-23.md Sec4.3:
    per-header scan derived 14, single-TU derived 13, an odr-using TU derived
    12 -- three different answers to "run the predicate over the nine
    headers"). The corrected rule's defining claim is that this no longer
    happens; this test is that claim, executed, not asserted."""
    per_header = _multiset(dbam.derive_population_per_header())
    single_tu = _multiset(dbam.derive_population_single_tu())
    odr_tu = _multiset(dbam.derive_population_single_tu(force_odr_use_artifact_moves=True))

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
def test_vitality_injected_twenty_first_function_is_flagged(tmp_path):
    """Guard-vitality cell (design Sec3.2, "New cell -- the membership-check
    job's own discriminating power"): a throwaway function satisfying the
    rule, added to a scratch copy of the include tree, must be flagged; once
    removed, the population returns to exactly the pinned twenty (T-1475
    renumbered this from eighteen to nineteen when JsonEscape was promoted
    into proof_manifest.h; T-2125 renumbered it again, nineteen to twenty,
    when SslmAmplifyingFoldScaleView<Kind>::Parse joined -- the injected
    function was, and remains, one past the pinned oracle's own count,
    whatever that count is). Proves THIS reference tool's own discriminating
    power -- StandardsDocument Sec4's "a check that has never been shown able
    to fail is not shown to cover anything," applied to the independent
    derivation tool itself, ahead of the production CI gate that must
    reproduce it."""
    import shutil

    scratch_include = tmp_path / "include"
    shutil.copytree(dbam._INCLUDE_DIR, scratch_include)

    artifact_h = scratch_include / "superslm" / "artifact.h"
    original = artifact_h.read_text(encoding="utf-8")
    assert "SUPERSLM_TEST_INJECTED_TWENTY_FIRST_FUNCTION" not in original

    injected = original.replace(
        "} // namespace superslm",
        "// Throwaway vitality probe (Curie, S-HARDEN-7 Sec3.2): a public,\n"
        "// non-noexcept, non-deleted static function taking a\n"
        "// (const uint8_t*, size_t) pair -- condition 4(a) satisfied,\n"
        "// deliberately left unwrapped. Must be flagged by the derivation.\n"
        "struct SUPERSLM_TEST_INJECTED_TWENTY_FIRST_FUNCTION {\n"
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
    # exactly one member, it does not perturb the other twenty.
    oracle = _load_pinned_oracle()
    derived_without_probe = Counter(
        (m["header"], m["name"]) for m in population_with_injection if m["name"] != "Probe"
    )
    assert derived_without_probe == oracle

    # Restore and confirm the population returns to exactly the pinned twenty.
    artifact_h.write_text(original, encoding="utf-8")
    population_restored = _multiset(dbam.derive_population_from_headers_dir(str(tmp_path)))
    assert population_restored == oracle


@requires_clang
def test_vitality_injected_same_named_duplicate_is_flagged_by_multiset_not_by_set(tmp_path):
    """Guard-vitality cell for the MULTISET identity itself (T-2467,
    Claude/Poirot/665f430-t2462-ask5-tracka-confirmation.md, Significant 2,
    D-SLM5558/D-SLM5559): `test_vitality_injected_twenty_first_function_is_flagged`
    above injects a uniquely-named `Probe`, which a plain (header, name) SET
    catches identically -- it proves *a member joining* is flagged, not that a
    MULTISET specifically was needed to flag it. This cell injects a SECOND
    member sharing an EXISTING member's own name (`artifact.h::OpenFromFile`,
    already in the pinned oracle once), the exact shape T-2125's real
    `model.h::Parse` growth took (two Parse overloads already present, a third
    joining) and the shape the round's own build log (Claude/Brunel/
    t2458-t2453-review-fixes-2026-08-31.md Sec6) and this review's own
    independent commissioning (Sec6 of the casebook above) both used to settle
    the multiset-vs-set design question by execution. A plain SET is
    unchanged by a same-named member joining a header that already has one --
    it would have silently absorbed this injection exactly as the prior rule's
    review-proposed literal-set wording would have silently absorbed T-2125's
    real third `model.h::Parse` -- so this cell asserts BOTH that the
    MULTISET comparison rejects the injection with the right excess count AND
    that a plain SET projection of the same two populations does not."""
    import shutil

    scratch_include = tmp_path / "include"
    shutil.copytree(dbam._INCLUDE_DIR, scratch_include)

    artifact_h = scratch_include / "superslm" / "artifact.h"
    original = artifact_h.read_text(encoding="utf-8")

    # A second, distinctly-signatured static overload also named `OpenFromFile`,
    # taking a `const char*` parameter named `path` (satisfies condition 4(a) the
    # same way the real member does), not noexcept, not deleted -- admitted by
    # the same four-condition rule as the real member it duplicates.
    injected = original.replace(
        "static SslmStatus OpenFromFile(const char* path, SslmArtifact& out, SslmError* err);",
        "static SslmStatus OpenFromFile(const char* path, SslmArtifact& out, SslmError* err);\n\n"
        "\t// Throwaway vitality probe (T-2467): a second, real overload sharing an\n"
        "\t// EXISTING member's own name -- proves the MULTISET identity specifically,\n"
        "\t// not merely that a joining member is flagged at all.\n"
        "\tstatic SslmStatus OpenFromFile(const char* path, uint32_t flags,\n"
        "\t                               SslmArtifact& out, SslmError* err);",
        1,
    )
    assert injected != original, "the injection anchor text was not found in artifact.h"
    artifact_h.write_text(injected, encoding="utf-8")

    oracle = _load_pinned_oracle()
    derived = _multiset(dbam.derive_population_from_headers_dir(str(tmp_path)))

    # MUST-REJECT: the multiset comparison flags the duplicate with the right
    # excess count -- exactly one extra ("artifact.h", "OpenFromFile"), nothing else.
    extra = derived - oracle
    missing = oracle - derived
    assert extra == Counter({("artifact.h", "OpenFromFile"): 1}), (
        f"expected exactly one excess (\"artifact.h\", \"OpenFromFile\") under the "
        f"multiset comparison; got extra={extra}, missing={missing}"
    )
    assert not missing

    # The property under test: a plain (header, name) SET does NOT separate this
    # injected population from the oracle -- the injection adds no NEW key, only
    # a second occurrence of one already present, which is exactly the shape a
    # set-based comparison cannot see.
    assert set(derived) == set(oracle), (
        "a plain (header, name) SET should NOT discriminate a same-named "
        "duplicate joining -- if this fails, the fixture no longer isolates the "
        "multiset-vs-set question this cell exists to answer"
    )

    # Restore and confirm the population returns to exactly the pinned twenty.
    artifact_h.write_text(original, encoding="utf-8")
    population_restored = _multiset(dbam.derive_population_from_headers_dir(str(tmp_path)))
    assert population_restored == oracle


@requires_clang
def test_oracle_regeneration_command_matches_the_committed_oracle_file():
    """Gives `derive_bad_alloc_membership.py --oracle` a caller (T-2467,
    Claude/Poirot/665f430-t2462-ask5-tracka-confirmation.md, Significant 2):
    before this cell, the committed oracle's own header names this as its
    regeneration command ("Regenerate with: python tests/ci/
    derive_bad_alloc_membership.py --oracle ... paste its output verbatim
    below this comment block"), but nothing asserted the committed body IS
    that command's current output -- a tree-wide sweep found no caller of
    `format_oracle_lines`/`--oracle` in any suite. This cell runs the real
    subprocess CLI path (not `format_oracle_lines` called in-process, which
    would leave `--argv` parsing itself unpinned) against the real headers on
    disk and diffs its stdout, byte-for-byte, against the oracle file's own
    non-comment body."""
    repo_root = dbam._REPO_ROOT
    tool_path = os.path.join(dbam._THIS_DIR, "derive_bad_alloc_membership.py")
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, tool_path, "--oracle"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"derive_bad_alloc_membership.py --oracle must exit 0; got "
        f"{result.returncode}, stderr:\n{result.stderr}"
    )
    regenerated = result.stdout.strip("\n")

    oracle_path = os.path.join(dbam._THIS_DIR, "bad_alloc_membership_expected.txt")
    with open(oracle_path, "r", encoding="utf-8") as f:
        committed_body = "\n".join(
            line.rstrip("\n") for line in f if line.strip() and not line.lstrip().startswith("#")
        )
    assert regenerated == committed_body, (
        "`python tests/ci/derive_bad_alloc_membership.py --oracle`'s own stdout no "
        "longer matches tests/ci/bad_alloc_membership_expected.txt's committed "
        "non-comment body -- regenerate the file and paste the output verbatim "
        "below its header comment, per the oracle file's own instructions"
    )

# ---------------------------------------------------------------------------
# The production gate (design Sec3.1: "tools/ci/check_bad_alloc_contract.py")
# is built, and design Sec3.3's rename-and-wrap has landed for every one of
# the twenty derived members (src/bad_alloc_wrap.h's WrapBadAllocContract,
# included by all five src/*.cpp files the population's members live in;
# T-2125 closed the twentieth, SslmAmplifyingFoldScaleView<Kind>::Parse,
# found unwrapped by this suite's own regression pin). The
# reference tool above (derive_bad_alloc_membership.py) proves the POPULATION
# is derivable and stable; the cells below confirm the CI GATE that reproduces
# that population reports it correctly and, now that every member is wrapped,
# reports zero unwrapped members and exits 0 -- design Sec3.3 step 3's other
# half (the RED-state half, "confirm it fails naming exactly the eighteen
# sites," was proven at the design's own authoring and is not re-proven here;
# re-creating that state would mean reverting the shipped wrap -- a verbatim
# quote of the design text, which described eighteen sites at that time).
#
# CLI contract: `tools/ci/check_bad_alloc_contract.py --list-unwrapped` prints
# one `header:line:name` row per currently-unwrapped member of the derived
# population and exits 1 if any exist, 0 if none do.
# ---------------------------------------------------------------------------


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def test_oracle_header_commit_count_pin_matches_the_prose_word():
    """T-2497 (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Critical 1, D-SLM5755):
    replaces `test_oracle_header_commit_count_matches_git_log_follow`, which asserted the oracle
    header's own prose word ("N total commits touch this file end to end ... through this one")
    against a FRESH `git log --follow` count taken at test-run time. That is a property of the
    CLONE, not of this file: `actions/checkout@v5` defaults to a depth-1 clone and no job in
    `.github/workflows/` sets `fetch-depth`, so `git log --follow` on a GitHub Actions runner
    counted one commit regardless of the true history and the cell failed on every Actions run --
    the first cell in `tests/ci/` to take repository history as an input, in the one job that
    collects this whole directory.

    This cell takes no git-history input at all: it asserts the prose word above against a
    companion literal, `COMMIT_COUNT_PIN: N`, that is also committed in the same file's header.
    Per StandardsDocument.md Sec5.6's preference for removing a term from a formula over adding a
    rule to remember it, the count is no longer independently re-derived from live history on
    every CI run; deriving it from `git log --follow` now happens only when a human runs
    `python tests/ci/derive_bad_alloc_membership.py --commit-count` on their own full-history
    checkout, at regeneration time, and pastes the result into both the prose word and the pin
    together -- the same shape `tests/ci/check_present_tense_defect_comments.py` (T-2387) already
    uses for the identical class: a live git call in a CI-collected cell silently degraded on a
    depth-1 checkout, closed there by vendoring the value instead of re-deriving it live. This
    cell's own job is narrower than the one it replaces -- it catches a human mistranscribing the
    word relative to the pin, not history drifting past both of them unnoticed -- which is the
    whole of what a depth-1 checkout can verify.

    T-2499 (Claude/Poirot/bc2ae29-t2498-census-fixes-confirmation.md Significant 1, D-SLM5775):
    this narrowing was real -- the class the replaced cell caught (the pin drifting from the
    file's own true history; four recorded instances: T-2467, T-2481, T-2491, T-2497) went from
    "caught on every developer checkout, broken in CI" to "caught nowhere." Restored, on the
    checkouts where it can run, by the companion cell immediately below."""
    oracle_path = os.path.join(dbam._THIS_DIR, "bad_alloc_membership_expected.txt")
    with open(oracle_path, "r", encoding="utf-8") as f:
        header = f.read()
    word_match = re.search(r"\b(\w+) total commits touch this file end to end", header)
    assert word_match, (
        "the oracle header's own commit-count sentence has moved or been reworded -- update "
        "this pin's own regex to match its new wording"
    )
    word = word_match.group(1).lower()
    assert word in _NUMBER_WORDS, (
        f"the oracle header states the commit count as {word!r}, which this pin's own "
        f"_NUMBER_WORDS does not cover -- extend it"
    )
    stated = _NUMBER_WORDS[word]

    pin_match = re.search(r"^# COMMIT_COUNT_PIN:\s*(\d+)", header, re.MULTILINE)
    assert pin_match, (
        "the oracle header's own COMMIT_COUNT_PIN line is missing or has moved -- regenerate it "
        "with `python tests/ci/derive_bad_alloc_membership.py --commit-count` on a full-history "
        "checkout and restore the line"
    )
    pinned = int(pin_match.group(1))

    assert stated == pinned, (
        f"the oracle header states {word!r} ({stated}) total commits touching this file, but "
        f"COMMIT_COUNT_PIN: {pinned} disagrees -- regenerate both together with "
        f"`python tests/ci/derive_bad_alloc_membership.py --commit-count` on a full-history "
        f"checkout before committing"
    )


def test_oracle_header_commit_count_matches_git_log_follow_on_a_full_history_checkout():
    """T-2499 (Claude/Poirot/bc2ae29-t2498-census-fixes-confirmation.md Significant 1, D-SLM5775):
    restores the detection `test_oracle_header_commit_count_pin_matches_the_prose_word` cannot
    provide by construction -- that cell only catches a human mistranscribing the prose word
    against `COMMIT_COUNT_PIN`; it cannot catch the PIN ITSELF drifting from the file's own true
    history, which is the class that has recurred four times (T-2467, T-2481, T-2491, T-2497),
    each caught only by a reviewer reading the paragraph. Executed on that exact shape (one
    comment-only commit to this file, neither number updated): a fresh `git log --follow` gives
    16 where the prose word and the pin both still read 15 -- this cell fails, the replaced cell
    (T-2497) would have failed too, and `test_oracle_header_commit_count_pin_matches_the_prose_
    word` passes throughout, because both its own numbers agree with each other while both are
    stale together.

    Guarded the same way `tools/ci/check_abi_header_inventory.py` degrades when its own
    population source is unavailable: `git rev-parse --is-shallow-repository` decides, and a
    shallow checkout SKIPS explicitly (printed by the `-v` the gating job already runs), rather
    than reporting a false pass (a shallow clone's own `git log --follow` returns 1 unconditionally,
    which would make this cell either always-red on every CI run or -- if compared to itself --
    vacuously green) or a false fail. `actions/checkout@v5` defaults to a depth-1 clone and no job
    in `.github/workflows/` sets `fetch-depth`, so this cell is not expected to run in CI; it runs
    on every developer's own full-history checkout, which is exactly where `pytest tests/ci/` was
    run before each of the four drift instances landed -- the environment the replaced cell was
    actually useful in, per T-2497's own build log (Claude/Brunel/t2497-t2496-review-fixes-
    2026-08-31.md Sec1: 'red on every developer machine ... broken in CI')."""
    is_shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=dbam._REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if is_shallow == "true":
        pytest.skip(
            "shallow checkout (git rev-parse --is-shallow-repository == true) -- this cell needs "
            "full history to run `git log --follow`; not expected in CI (actions/checkout@v5 "
            "defaults to depth-1 and no job sets fetch-depth). Run `pytest tests/ci/` on a "
            "full-history checkout to exercise it."
        )

    oracle_path = os.path.join(dbam._THIS_DIR, "bad_alloc_membership_expected.txt")
    with open(oracle_path, "r", encoding="utf-8") as f:
        header = f.read()
    word_match = re.search(r"\b(\w+) total commits touch this file end to end", header)
    assert word_match, (
        "the oracle header's own commit-count sentence has moved or been reworded -- update "
        "this pin's own regex to match its new wording"
    )
    word = word_match.group(1).lower()
    assert word in _NUMBER_WORDS, (
        f"the oracle header states the commit count as {word!r}, which this pin's own "
        f"_NUMBER_WORDS does not cover -- extend it"
    )
    stated = _NUMBER_WORDS[word]

    actual = dbam.git_log_follow_commit_count(
        os.path.relpath(oracle_path, dbam._REPO_ROOT)
    )
    assert stated == actual, (
        f"the oracle header states {word!r} ({stated}) total commits touching this file, but a "
        f"fresh `git log --follow` on this full-history checkout counts {actual} -- regenerate "
        f"the header with `python tests/ci/derive_bad_alloc_membership.py --commit-count` on a "
        f"full-history checkout and paste the result into both the prose word and "
        f"COMMIT_COUNT_PIN together"
    )


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
    """S-HARDEN-7's rename-and-wrap has landed for every one of the twenty
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
