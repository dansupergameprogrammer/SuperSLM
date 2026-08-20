"""Curie's red suite for check_ci_claims.py (T-2196 S1/S2;
`Claude/Poirot/8bf85c5-t2196-t2189-final-confirmation.md`).

Mirrors `test_check_present_tense_defect_comments.py`'s own convention:
mechanism cells (noun/verb/negation coverage, clause-boundary correctness,
exemption scoping) drive the module against constructed scratch text, never
the real production tree, per `StandardsDocument.md` Sec4's population-
validation requirement -- a check shown only to pass on unchanged input is
not shown to catch anything. The historical-population cells at the bottom
replay the seven real sites four review rounds (T-2191, T-2192, T-2195,
T-2196) found, recovered from `D:\\SuperSLM` git history, rather than a fault
this module's own author injected.
"""

import os

import check_ci_claims as cic

_FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ci_claims_historical_fixtures")


def _read(rel_fixture_name: str) -> str:
    path = os.path.join(_FIXTURES_DIR, rel_fixture_name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Rule coverage: noun x verb combinations fire; either alone does not. ---


def test_ci_noun_alone_does_not_fire():
    assert cic.find_ci_execution_claims("The CI configuration lives in a YAML file.") == []


def test_exec_verb_alone_does_not_fire():
    assert cic.find_ci_execution_claims("Every layer-budget granularity is exercised by the reference model.") == []


def test_plain_ci_plus_is_exercised_fires():
    hits = cic.find_ci_execution_claims("The Linux leg is exercised by CI on every commit.")
    assert len(hits) == 1


def test_hyphenated_continuous_integration_fires():
    hits = cic.find_ci_execution_claims("The tier is validated at continuous-integration extent.")
    assert len(hits) == 1


def test_spaced_continuous_integration_fires():
    hits = cic.find_ci_execution_claims("The tier is tested by continuous integration on every push.")
    assert len(hits) == 1


def test_ci_compound_verb_fires():
    """The `CI-exercised`/`CI-probed` adjectival shape (site3/site4's own
    historical text) -- `CI` supplies the noun and the hyphen-joined verb
    supplies the execution claim, with no separate "is" token."""
    hits = cic.find_ci_execution_claims("The tier is force-selectable and CI-exercised on this platform.")
    assert len(hits) == 1


def test_runs_on_every_push_fires():
    hits = cic.find_ci_execution_claims("The workflow runs on every push against the GitHub Actions matrix.")
    assert len(hits) == 1


def test_is_fully_green_fires():
    hits = cic.find_ci_execution_claims("The matrix is fully green on this commit, per GitHub Actions.")
    assert len(hits) == 1


def test_hosted_runner_noun_fires():
    hits = cic.find_ci_execution_claims("A hosted runner exercises the full suite on every push.")
    assert len(hits) == 1


def test_the_matrix_noun_requires_exact_phrase():
    """`\\bthe (matrix|workflows?)\\b` matches "the matrix"/"the workflow(s)"
    as consecutive tokens; a modifier between "the" and "matrix" is not
    matched by this alternative (a known, narrow scope -- the other noun
    alternatives, or a second sentence naming CI directly, are what a real
    site relies on instead; see site1-6, none of which depend on this
    alternative alone)."""
    assert cic.find_ci_execution_claims("The full matrix is exercised on every push.") == []
    assert cic.find_ci_execution_claims("The matrix is exercised on every push.") != []


# --- Negation exemption: scoped to its own clause. ---


def test_negation_in_the_same_clause_exempts_it():
    assert cic.find_ci_execution_claims("The tier is not exercised by CI yet; it is a future item.") == []


def test_negation_does_not_exempt_a_different_sentence():
    """T-2196 S1's own defect, in miniature: a hedged sentence carrying "no
    further" must not silently clear an unrelated, un-hedged sentence about
    CI naming the same noun, even when the two sit in the same paragraph."""
    text = (
        "'CI extent' means the platform is exercised by CI, and no further "
        "hardware-specific measurement has been made. The workflow runs on "
        "every push and is fully green on this commit, per GitHub Actions."
    )
    hits = cic.find_ci_execution_claims(text)
    assert len(hits) == 1, f"expected exactly the second sentence to fire, got {hits}"
    assert "runs on every push" in hits[0]


def test_no_further_exemption_does_not_fire_on_a_clean_construction():
    hits = cic.find_ci_execution_claims(
        "The platform is exercised by CI, and no further hardware-specific measurement has been made."
    )
    assert hits == []


# --- Clause-boundary correctness (T-2196 S1's own root cause). ---


def test_period_inside_a_filename_does_not_split_the_sentence():
    text = (
        "The workflows exist and define the full matrix "
        "([.github/workflows/tests.yml](.github/workflows/tests.yml)), "
        "runs on every push, and is **fully green on the 1.0 release commit**."
    )
    hits = cic.find_ci_execution_claims(text)
    assert len(hits) == 1, f"the whole sentence must be one clause carrying both the noun and the verb, got {hits}"


def test_period_inside_a_version_number_does_not_split_the_sentence():
    hits = cic.find_ci_execution_claims("Release 1.1.0's workflow runs on every push against GitHub Actions.")
    assert len(hits) == 1


def test_period_inside_an_inline_code_span_does_not_split_the_sentence():
    hits = cic.find_ci_execution_claims(
        "The check in `tests/ci/check_ci_claims.py` confirms the workflow runs on every push, per GitHub Actions."
    )
    assert len(hits) == 1


def test_a_genuine_sentence_boundary_still_splits():
    """The masking is scoped to links/code/versions only -- an ordinary
    sentence-ending period elsewhere in the same paragraph still breaks the
    clause, so a real violation is not merged into an unrelated neighbor."""
    text = "This project builds with CMake. The workflow runs on every push against GitHub Actions."
    hits = cic.find_ci_execution_claims(text)
    assert len(hits) == 1
    assert "CMake" not in hits[0]


# --- File-level scanning: the exempt section, and Windows path correctness. ---


def test_scan_text_skips_the_exempt_section_of_the_exempt_file():
    text = (
        "## CI execution status\n\n"
        "The matrix is fully green on this commit, per GitHub Actions.\n\n"
        "## Other section\n\n"
        "Some unrelated prose.\n"
    )
    assert cic.scan_text(os.path.join("docs", "platform-support.md"), text) == []


def test_scan_text_still_scans_other_sections_of_the_exempt_file():
    text = (
        "## CI execution status\n\n"
        "Honest CI text here.\n\n"
        "## Other section\n\n"
        "The matrix runs on every push against GitHub Actions.\n"
    )
    hits = cic.scan_text(os.path.join("docs", "platform-support.md"), text)
    assert len(hits) == 1


def test_scan_text_scans_every_non_exempt_file_in_full():
    text = "The matrix runs on every push against GitHub Actions."
    assert cic.scan_text("README.md", text) != []


def test_exempt_file_comparison_is_platform_correct():
    """T-2196 S2's third face: `glob.glob('docs/*.md')` yields
    `docs\\platform-support.md` on Windows, so a literal forward-slash
    string comparison never matched and the check reported a false
    violation against the clean tree on the one platform this project runs
    anything on. `os.path.join` builds the path with the current
    platform's own separator, so this cell is a real regression guard on
    Windows and a no-op (both separators equal `/`) on POSIX."""
    windows_style_path = "docs" + "\\" + "platform-support.md"
    text = "## CI execution status\n\nThe matrix is fully green, per GitHub Actions.\n"
    assert cic.scan_text(windows_style_path, text) == [] or os.sep != "\\", (
        "on Windows this must resolve to the exempt file via os.path.normpath"
    )


def test_missing_scanned_file_is_reported_distinctly():
    failures = cic.scan_files(["does-not-exist.md"], repo_root="/does/not/exist")
    assert len(failures) == 1
    assert "not found" in failures[0]


# --- CHANGELOG.md: only the latest dated release section is in scope. ---


def test_changelog_older_release_section_is_exempt():
    """T-2196's own tree-wide grep found exactly one surviving hit of this
    claim class: CHANGELOG.md's 1.0 entry, "true when written and remains
    true" -- accepted, not flagged. A dated, already-shipped release section
    is a historical record of what was true at that release, not a live
    claim about today's CI."""
    text = (
        "## [1.1.0] - 2026-08-19\n\n"
        "Nothing relevant here.\n\n"
        "## [1.0.0] - 2026-08-18\n\n"
        "- Linux and macOS are exercised at continuous-integration extent only.\n"
    )
    assert cic.scan_text("CHANGELOG.md", text) == []


def test_changelog_latest_release_section_stays_in_scope():
    text = (
        "## [1.1.0] - 2026-08-19\n\n"
        "- The workflow runs on every push against GitHub Actions.\n\n"
        "## [1.0.0] - 2026-08-18\n\n"
        "Nothing relevant here.\n"
    )
    hits = cic.scan_text("CHANGELOG.md", text)
    assert len(hits) == 1


def test_changelog_unreleased_heading_does_not_hide_the_latest_release():
    """`[Unreleased]` is a Keep-a-Changelog status heading, not a versioned
    `## [x.y.z]` release heading -- it must never be counted as one of the
    two release-heading matches that bound the scanned window. A regex that
    matched the bare `## [` would treat `[Unreleased]` as a release boundary
    and silently cut the entire latest VERSIONED section's claims out of
    scope; this fixture plants a live claim in that latest versioned
    section, with an `[Unreleased]` heading sitting above it, and requires
    the claim still be caught."""
    text = (
        "## [Unreleased]\n\n"
        "Nothing relevant here.\n\n"
        "## [1.1.0] - 2026-08-19\n\n"
        "- The workflow runs on every push against GitHub Actions.\n\n"
        "## [1.0.0] - 2026-08-18\n\n"
        "Nothing relevant here.\n"
    )
    hits = cic.scan_text("CHANGELOG.md", text)
    assert len(hits) == 1


def test_changelog_scoping_does_not_apply_to_other_files():
    text = (
        "## [1.1.0] - 2026-08-19\n\n"
        "Nothing relevant here.\n\n"
        "## [1.0.0] - 2026-08-18\n\n"
        "- Linux and macOS are exercised at continuous-integration extent only.\n"
    )
    assert cic.scan_text("docs/some-other-doc.md", text) != []


def test_changelog_with_a_single_release_heading_is_scanned_whole():
    text = "## [1.1.0] - 2026-08-19\n\n- The workflow runs on every push against GitHub Actions.\n"
    assert cic.scan_text("CHANGELOG.md", text) != []


# --- Mutation proof: a planted claim in the real target-file shape goes red. ---


def test_a_planted_present_tense_ci_claim_is_caught():
    hits = cic.find_ci_execution_claims(
        "This platform is currently tested on GitHub Actions and the CI runs on every push."
    )
    assert hits, "the check's own can-it-fire control must still fire"


# --- Historical population validation (StandardsDocument Sec4). ---
#
# tests/ci/ci_claims_historical_fixtures/site{1..7}_{pre,post}.txt hold the
# exact recovered text of every site four review rounds found this class at:
# site1/site6 (CHANGELOG.md's and docs/platform-support.md's Known-gaps
# AVX-512 bullets) and site2 (README.md's AVX-512 roadmap bullet) from
# `D:\SuperSLM`@3987cd1^ (before) and @3987cd1 (after); site3/site4/site5
# (docs/platform-support.md's SSE2/AVX2/AVX-512 kernel-tier table rows) from
# the same two commits; site7 (README.md's own CI-extent paragraph -- the
# exact sentence T-2196 S1 found this module's first version could not
# catch) from @1381076 (before) and @fb0ea6c (after). Fixing nothing here --
# these are read-only replays of text already fixed at those real commits,
# satisfying Sec4's "found independently of the check, verified, and
# reproduced before anything is fixed."

_SITE_NAMES = ("site1", "site2", "site3", "site4", "site5", "site6", "site7")


def test_fires_on_every_historical_pre_fix_site():
    misses = []
    for name in _SITE_NAMES:
        text = _read(f"{name}_pre.txt")
        hits = cic.find_ci_execution_claims(text)
        if not hits:
            misses.append(name)
    assert not misses, (
        f"rule coverage gap: the following historical PRE-fix sites were NOT flagged: {misses} "
        f"-- this check does not reproduce the full T-2191/T-2192/T-2195/T-2196 population, "
        f"which is exactly the must-reject T-2196 S1 found missing"
    )


def test_does_not_fire_on_any_historical_post_fix_site():
    false_positives = []
    for name in _SITE_NAMES:
        text = _read(f"{name}_post.txt")
        hits = cic.find_ci_execution_claims(text)
        if hits:
            false_positives.append((name, hits))
    assert not false_positives, (
        f"the check fires on the CURRENT, already-fixed text of these sites: {false_positives} "
        f"-- it would never stop reporting the correct, current prose"
    )


def test_historical_population_is_the_full_named_set_not_a_subset():
    """Guards the fixture set itself: seven pre and seven post files, one
    pair per site this class was found at across four review rounds, so a
    fixture accidentally left out does not silently shrink the population
    this suite claims to validate against."""
    for name in _SITE_NAMES:
        for suffix in ("pre", "post"):
            path = os.path.join(_FIXTURES_DIR, f"{name}_{suffix}.txt")
            assert os.path.isfile(path), f"missing historical fixture: {path}"


def test_the_must_reject_site_scans_clean_via_the_full_file_level_path():
    """site7 end-to-end through scan_text (not just find_ci_execution_claims
    directly) -- the exact shape the real README.md scan takes, confirming
    the fix holds through _strip_exempt_section too (README.md is never the
    exempt file, so this also confirms non-exempt files are unaffected by
    that code path)."""
    pre_text = _read("site7_pre.txt")
    assert cic.scan_text("README.md", pre_text) != []
    post_text = _read("site7_post.txt")
    assert cic.scan_text("README.md", post_text) == []
