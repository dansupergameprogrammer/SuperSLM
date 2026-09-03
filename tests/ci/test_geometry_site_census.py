"""Red suite for tools/geometry_site_census.py, T-2475: closes the census-exclusion class
Claude/Mendeleev/t2468-census-recommissioning-2026-08-31.md found NOT COMMISSIONED (F1,
D-SLM5603) -- `_part2_excluded_text_hit`'s substring match (`text in line`) excused the WHOLE
physical line the instant an excluded fragment's text appeared anywhere in it, so a genuinely
new, unregistered geometry-defect statement appended onto the SAME line as an already-excused
fragment was silently absorbed: PASS, exit 0, unaudited site in the tree. This is the second
instance of one class -- the first (line-keyed, T-2467) excused a whole line on a stale line
number; the second (text-keyed, this file) excused a whole line on a partial text match. Both
share the same shape: the exclusion excuses more than the thing it names.

T-2475's own property: an exclusion excuses exactly the statement it names and nothing else on
that line. `_part2_excise_excluded_text` removes the excused fragment's own text from a line
before the pattern regexes see it, rather than skipping the whole line -- new content sharing
the line, whichever side of the fragment it lands on, is left in the remainder and scanned like
ordinary code. `model.h`'s and `proof_manifest.h`'s own `_PART2_EXCLUDED_TEXT` fragments were
widened from a uniqueness-only prefix (the enumerator name alone) to their own full stripped
line, because their trailing explanatory comment -- not the enumerator declaration -- is what
actually trips `_r1_multiply_hit`; excising only the prefix would have left that comment's own
hidden_size/head_dim/num_attention_heads/`*` text in the remainder and turned the untouched
baseline red.

Structure, in the order `StandardsDocument.md` Sec4/Sec5.4 requires (population derived and
verified at source before the mechanism was built, mechanism built, population reproduced
against the built mechanism):

  - Mechanism cells (`test_excise_*`): `_part2_excise_excluded_text` in isolation, synthetic
    lines, no file I/O -- proves the excision itself, independent of the real tree.
  - Real-tree population cells, against the REAL checkout via `_mutated` (mutate one real
    production file for the duration of one test, restore its exact original bytes after --
    this module imports `census` ONCE, at collection time, so `census._REGISTRY_PATH` is bound
    to this repo's own `tools/` directory for every call in this file; calling
    `census.run_census(tmp_root)` against a synthetic tmp tree would still load THIS repo's real
    registry, mismatched against the tmp tree's own paths, not a copy of it -- a fresh subprocess
    invoking a copy of `tools/` alongside a copy of `src/`/`include/` genuinely does carry its own
    registry (Claude/Poirot/dcefab3-t2486-census-content-keying-confirmation.md's own lab does
    exactly this), which this suite does not use because every cell here calls the already-
    imported module in-process; mirrors T-2468's own mutate/run/revert discipline):
      * `test_real_tree_baseline_is_clean_today` -- cell zero, unmutated.
      * `test_must_accept_*` (T-2468 Sec1, D-SLM5601) -- comment inserted above each of the
        three excluded lines; must stay PASS (position-independence, unaffected by this fix).
      * `test_must_reject_*` (T-2468 Sec2, D-SLM5602) -- a genuinely new, unregistered site in
        a zero-marker production header; must FAIL and cite the injected line.
      * `test_defeats_the_prior_mechanism_*` (T-2468 Sec3, D-SLM5603 -- THE finding this ticket
        closes) -- a genuinely new statement appended onto the SAME physical line as an
        already-excused fragment, on each of the three exclusions; must now FAIL (was PASS
        under `_part2_excluded_text_hit`).
      * `test_prepended_content_on_an_excused_line_is_also_caught` -- this ticket's own added
        construction: new content BEFORE the fragment, not just after -- the excise-and-scan
        property is symmetric, unlike a hypothetical "only widen at the end" remedy.
      * `test_safe_direction_rename_and_reformat_*` (T-2468 Sec4a cases 1-2, D-SLM5604) -- the
        brief's own named safe-direction Note: renaming or reformatting the excused line itself
        still re-triggers the census (a real cost, not a hole a defect can hide in). Confirms
        this fix does not make the exclusion stickier.
      * `test_wrap_comment_split_is_now_reported_as_a_dead_exclusion_not_a_silent_pass` (T-2468
        Sec4a case 3, reclassified by T-2475's own fragment-widening + dead-entry check) --
        splitting the excused declaration from its own comment removes the pattern-triggering
        co-occurrence entirely AND breaks the widened fragment's exact-text match; the case is
        no longer a silent, uninformative PASS -- it is now a named DEAD EXCLUSION finding.
      * `test_two_line_split_is_the_documented_f2_limitation_still_open` (T-2468 Sec4a case 4,
        F2) -- a genuinely new QOW site split across two physical lines still evades Part 2;
        pre-existing, not this ticket's own diff (Part 2's per-line hit detection is unchanged),
        reproduced here so the PASS this case still returns is a proven, not assumed, fact, and
        cross-checked against the KNOWN LIMITATION text `main()` now prints.

T-2481 fold-in (Claude/Bach/briefs/t2481.md; Claude/Mendeleev/t2480-census-recommissioning-
2026-08-31.md F1, D-SLM5647/D-SLM5659; Claude/Poirot/f363c2a-t2479-census-class-confirmation.md
Significant 1/2, D-SLM5651/D-SLM5652): four sources bundled into one round --

  - Part 3's occurrence-ordinal keying is defeated by REORDERING two existing occurrences of a
    same-file, multi-occurrence site -- nothing added or removed, marker count unchanged, Part 1
    blind to it -- which desyncs the ordinal-to-offset lookup and silently absorbs a genuine
    revert while citing an untouched line (`test_part3_reorder_*`, below). Replaced with
    CONTENT-ADDRESSED occurrence identification (`required_token_scopes`, registry) -- see
    `tools/geometry_site_census.py`'s own Part 3 header comment and the registry's own header
    comment for the full account.
  - Part 2's exclusion mechanism excuses EVERY literal copy of an excused fragment in its own
    file, not only the one statement it names -- a genuinely new, unregistered site written as a
    literal copy of an excused statement is excised for free (`test_part2_ambiguous_exclusion_*`,
    below). Closed by counting non-comment-line matches instead of recording mere presence.
  - Two of the four cells pinning T-2475's own headline remedy exercised text a compiler never
    sees (append AFTER a fragment's own trailing `//` comment); the two prior cells renamed to
    state precisely what they do prove (a dead exclusion, not a caught defect statement). T-2481
    also added two cells over what it believed was the producible shape for these two files
    (insert BEFORE the trailing comment) -- corrected below, T-2491: it is not producible.
  - The S2 red-check comment block's own cell count ("exactly four") is corrected to what a
    full, unfiltered run of this file actually reddens under the same reversion.

T-2491 fold-in (Claude/Bach/briefs/t2491.md; Claude/Poirot/dcefab3-t2486-census-content-keying-
confirmation.md Significant 1/2, D-SLM5716/D-SLM5717/D-SLM5718; D-SLM5732): fix round on T-2486's
confirmation of T-2481's diff --

  - T-2481's own `test_defeats_the_prior_mechanism_same_line_insert_before_comment_model_h`/
    `_proof_manifest_h` were believed to pin the producible same-line shape for `model.h` and
    `proof_manifest.h`. Both excused lines are enumerators inside an `enum class` body
    (`SslmModelStatus`, `ConfigGeometryStatus`); an enum body admits enumerators, not statements,
    so the construction does not compile -- executed with the CI's own pinned compiler, clang
    18.1.8, `-fsyntax-only -std=c++20`: `error: missing ',' between enumerators` on both headers.
    The cells also have zero discrimination, executed both with the QOW family forced false and
    with the injected identifier replaced by an inert statement: the assertion still holds either
    way, because what actually fires is the trailing prose comment's own R1 co-occurrence once
    the exclusion's exact-text match goes dead, not recognition of the injected statement. Deleted
    rather than replaced with a third attempt at the SAME shape.

    T-2491's own replacement text (D-SLM5732, D-SLM5740) went further than the compile finding
    supports: "there is no producible same-line geometry defect an enum body can host." Refuted
    by construction (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Significant 1,
    D-SLM5756/D-SLM5757): an enum body admits enumerators, not statements, but a same-line
    ENUMERATOR prepend is both legal C++ and enough. Because the excused fragment is the whole
    stripped line, a prepend leaves it intact -- the exclusion still fires and the injected text
    is scanned as ordinary remainder. Executed at the CI's pinned compiler, clang 18.1.8,
    `-fsyntax-only -std=c++20`: `q_proj_out_channels_hidden_size, ` prepended to `model.h`'s
    fragment and `o_proj_out_channels_hidden_size, ` prepended to `proof_manifest.h`'s both
    compile clean and fire `UNMARKED QOW PATTERN HIT`, discriminating both ways (silent with the
    QOW family disabled, silent with an inert enumerator in place of the defect). T-2497 adds one
    cell per header (`test_defeats_the_prior_mechanism_same_line_enumerator_prepend_model_h`/
    `_proof_manifest_h`, below) pinning exactly this. The corrected, narrower fact: a same-line
    *statement* is not producible inside an `enum class` body; a same-line *enumerator* is, and
    the census catches it. `adapter_marshal.h`'s excused line sits inside a function body, where
    both statement forms (`..._append_adapter_marshal`/
    `test_prepended_content_on_an_excused_line_is_also_caught`, below) are real.
  - `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_all`
    restored `tools/geometry_site_registry.json` via `open(path, "w", encoding="utf-8")`, which
    translates `\\n` to `\\r\\n` on this platform; the file is `attr/text eol=lf`, so a clean
    checkout came back `w/crlf` and ` M` after one run. Fixed with `newline=""` on both the write
    and the restore. `_mutated`, below, is hardened the same round -- not with a blanket
    `newline=""` (every `.h`/`.cpp`/`.hlsl` file it mutates is `attr/text`, checked out CRLF, and
    an unconditional `newline=""` would restore those as LF instead, the same defect in the other
    direction), but by detecting each target file's own newline convention from its raw bytes
    before writing, so the next `eol=lf` file it is pointed at inherits the correct behavior.
  - `AMBIGUOUS SCOPE ANCHOR` (Part 3's third fail-closed diagnostic) had no cell -- pinned by
    `test_part3_ambiguous_scope_anchor_fires_when_two_records_own_anchors_collide_in_one_window`,
    below, reproducing the reviewer's own construction.
  - GS-12's own nine occurrences are anchored on their own explanatory COMMENT text (this file's
    own Part 3 header comment explains why: a bare code prefix collided with unrelated, unmarked
    code once a reorder moved it), so the match key includes comment prose for all nine of GS-12's
    occurrences, through eight registered records (two occurrences, `occ2`/`occ7`, share one
    record -- see the registry's own header comment) -- a developer who rewords one of those
    comments, no code touched, reddens the
    census with `MISSING SCOPE ANCHOR` (Claude/Poirot/dcefab3-t2486-census-content-keying-
    confirmation.md Sec7 O2, D-SLM5726, executed: "the normed" to "the normalised" in one GS-12
    comment). This was disclosed nowhere a reader of the tool would find it; disclosed now in
    `tools/geometry_site_census.py`'s own Part 3 header comment and the registry's own header
    comment, both of which also say the correct response to that red: re-derive the occurrence's
    own anchor from the new comment text, not treat the finding as a caught regression.
  - This module's own docstring (below) and the S2 red-check comment block (further down this
    file) each carried a stale claim untouched by T-2481's diff -- corrected in place rather than
    superseded a third time; see each site's own comment for what was wrong and how it was
    checked.

T-2497 fold-in (Claude/Bach/briefs/t2497.md; Claude/Poirot/ba29de4-t2496-census-fixes-
confirmation.md Significant 1/2, D-SLM5756/D-SLM5757/D-SLM5758): fix round on T-2496's
confirmation of T-2491's diff --

  - T-2491's own replacement fact for the two deleted pins -- "there is no producible same-line
    geometry defect an enum body can host" -- overclaimed. Refuted by construction: an enum body
    admits enumerators, not statements, but a same-line ENUMERATOR prepend is legal C++, compiles
    clean at the CI's pinned compiler, and fires the census from the excused fragment's own
    remainder. Two cells restore what the deletion left uncovered
    (`test_defeats_the_prior_mechanism_same_line_enumerator_prepend_model_h`/
    `_proof_manifest_h`, below); the corrected, narrower fact -- a same-line statement is not
    producible inside an enum class body, a same-line enumerator is, and is caught -- replaces
    the overclaim at every site it reached (this docstring, the S2 red-check comment block, both
    T-2481/T-2491 build logs, and D-SLM5732/D-SLM5740).
  - Neither production change T-2491 made is detectable by any cell in this file: reverting
    `newline=""` on `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_
    registry_record_at_all`'s own registry write still passes while the registry's own bytes
    change underneath it, and reverting `_mutated`'s convention-detection hardening leaves this
    whole file green. `_mutated_targets_and_registry_are_byte_identical_after_the_module_runs`,
    below, is a module-scoped, autouse fixture that snapshots the `_mutated` targets plus
    the registry before this module's own suite runs and asserts them byte-identical after --
    closing both halves of the dirty-checkout class at once, for the paths
    `_MUTATED_TARGETS_AND_REGISTRY_PATHS` names (T-2526: no count restated here -- see that
    tuple's own comment for why).

T-2499 fold-in (Claude/Bach/briefs/t2499.md; Claude/Poirot/bc2ae29-t2498-census-fixes-
confirmation.md Significant 1/2, D-SLM5775/D-SLM5776): fix round on T-2498's confirmation of
T-2497's diff --

  - `test_oracle_header_commit_count_pin_matches_the_prose_word`'s own remedy (T-2497) removed
    the only detection of the class it replaced: the oracle header's `COMMIT_COUNT_PIN` and prose
    word can drift together from the file's own true history with that cell green throughout,
    executed (a comment-only commit to the oracle, neither number updated: a full-history `git
    log --follow` gives 16 where both the prose word and the pin still read 15). Restored by
    `test_oracle_header_commit_count_matches_git_log_follow_on_a_full_history_checkout` in
    `test_membership_check_population.py`, guarded on `git rev-parse --is-shallow-repository` the
    same way `tools/ci/check_abi_header_inventory.py` degrades when its own population source is
    unavailable -- skips explicitly on a shallow checkout (never expected to run in CI), runs and
    catches the drift on every developer's own full-history checkout.
  - The T-2497 fold-in bullet immediately above overclaimed the byte-identity fixture's own
    scope: "every future cell that touches the real tree" is false as implemented -- executed, a
    cell writing `include/superslm/forward_sites.h` without restoring gives `41 passed`, fixture
    silent, file left modified, because that path was never in
    `_MUTATED_TARGETS_AND_REGISTRY_PATHS`. The claim is narrowed to what the fixture actually
    guarantees (its own docstring, below, and the comment above the tuple) -- exactly the paths
    `_MUTATED_TARGETS_AND_REGISTRY_PATHS` names, no more. The tuple's own `src/` entries, formerly
    duplicated as separate `os.path.join(...)` literals 430-odd lines below the tuple, are replaced
    with references to `_PROOF_MANIFEST_CPP`/`_SUPERSLM_GPU_CPP`, moved up and defined once, so the
    two spellings cannot desync.
  - `derive_bad_alloc_membership.py`'s `git_log_follow_commit_count` and its `--commit-count` CLI
    branch (T-2497) had no cell; `test_membership_check_population.py`'s `--oracle` mode is
    deliberately pinned through the real subprocess CLI path rather than called in-process, with
    the reason written down, and this round follows that same precedent for `--commit-count`.
  - The item-3 sweep (Claude/Bach/briefs/t2499.md; every production change landed by T-2491,
    T-2497, and this round, checked against a detecting cell) found `_mutated`'s own newline-
    convention branch (T-2491) genuinely undetectable by anything in this suite -- every real
    `_mutated` target is pure CRLF today, so the branch has never been exercised on an `eol=lf`
    input. Cheap to close: extracted into its own pure function, `_write_newline_for`, pinned
    directly on synthetic bytes by `test_write_newline_for_picks_the_convention_from_raw_bytes`,
    below, with no real file and no new `eol=lf` fixture added to the repo.

T-2518 fold-in (Claude/Bach/briefs/t2518-census-ci-regression-fix.md; Claude/Poirot/a3a20bc-
t2509-adapter-geometry-review.md Critical 1, D-SLM5857): fix round on T-2509's own diff, which
rewrote `tools/geometry_site_census.py` (the T-2509 widening -- see the module docstring's own
"WIDENED T-2509" note) and fixed BOTH of adapter_marshal.h's own geometry defects (GS-28, GS-29)
without running this file, reddening 13 of its then-41 cells --

  - `_ADAPTER_FRAGMENT`/`_MODEL_FRAGMENT` (module-level constants, above) named text that no
    longer exists: T-2509 fixed adapter_marshal.h's own q_proj/o_proj residual (the exact excused
    line these constants pinned is deleted, replaced by real per-projection branches, registered
    and marked instead of excused) and restructured model.h's ConfigGeometryHiddenSizeMismatch
    comment so `head_dim` no longer shares a physical line with `hidden_size` there either --
    `_PART2_EXCLUDED_TEXT` (census.py) no longer carries an entry for either file. Every cell
    built on `_ADAPTER_FRAGMENT`/`_MODEL_FRAGMENT` either raised constructing its own fixture (an
    internal `assert ... == 1` uniqueness check against text no longer present) or silently
    no-opped (mutated nothing, asserted against an unmutated tree) -- ten cells across the
    excise-mechanism, must-accept, same-line-append, enumerator-prepend, prepended-content,
    safe-direction, dead-exclusion, and ambiguous-exclusion populations. Each population's own
    adapter_marshal.h and/or model.h member is DELETED (ten cells; see each section's own header
    comment, at the site of its own deletion, for the specific account) rather than retargeted at
    a synthetic (path, text) pair invented for the sole purpose of keeping the cell count
    unchanged -- this module's own standard (above, "population derived and verified at source
    before the mechanism was built") is that a construction is grounded in the real tree, not
    fabricated to exercise a mechanism in the abstract, and neither file has a real, compilable
    subject left for any of these shapes. All ten are genuinely absent from the tip file -- none
    is retargeted. Five SURVIVING cells, never among the ten, are separately retargeted onto
    `proof_manifest.h` (that file's own exclusion is untouched by T-2509 and still live, so the
    underlying property each proves -- one member of the excise-mechanism population
    (`test_excise_does_not_touch_an_unrelated_line_in_a_registered_file`), both members of the
    safe-direction population (`test_safe_direction_rename_of_the_excused_enumerator_reddens`,
    `test_safe_direction_reformat_of_the_excused_line_reddens`), one member of the dead-exclusion
    population (`test_dead_exclusion_is_reported_when_its_subject_is_deleted`), and one member of
    the ambiguous-exclusion population (`test_part2_ambiguous_exclusion_fires_on_a_literal_
    duplicate_copy`) -- still has a real subject and is still proven, just against the one file
    where it remains true). One cell (`test_two_line_split_is_the_documented_f2_limitation_still_
    open`) is SPLIT in two rather than deleted or retargeted: T-2509's own window widening is
    QOW-only, deliberately (census.py's own comment above the window loop), so half of what this
    cell's own construction proved -- the QOW-shaped two-line split -- now FAILS the census where
    it used to PASS, while the other half (an R1-shaped two-line split, a new construction this
    round adds) is still NOT caught, exactly as documented. Net (name-diffed against the tip file:
    11 removed, 2 added, 30 kept): ten deletions, one two-way split, thirty-two cells passing (was
    forty-one; thirty kept plus the split's own two halves).

    T-2524 correction (Poirot e4bcaeb-t2518-census-fix-confirmation.md Minor 1, D-SLM5883): the
    paragraph above originally said "29 pre-existing plus the new R1 cell" (= 30, not 32) and
    named "four of the ten ... RETARGETED onto proof_manifest.h" as must-accept/dead-exclusion/
    ambiguous-exclusion/safe-direction -- both wrong, checked at source: no deleted cell is
    retargeted, the five retargets above are all SURVIVING cells, must-accept's own two
    non-`proof_manifest.h` members are among the ten DELETED (not retargeted) with its
    `proof_manifest.h` member simply untouched, and the excise-mechanism population (omitted from
    the original four) has one retarget of its own. Two of the ten -- `test_excise_leaves_
    appended_new_content_in_the_remainder` and `test_excise_leaves_prepended_new_content_in_the_
    remainder` -- are additionally RESTORED (not merely corrected in the account): see the
    mechanism-cells section's own header comment, above, for why they were deleted in error.
  - Poirot's own Significant 1 (`include/superslm/model.h`'s ConfigGeometryHiddenSizeMismatch
    comment falsely claims "UNREACHABLE, no live producer returns this" when four `src/model.cpp`
    returns are live) is fixed in place: the two clauses T-2509 dropped while copying
    `proof_manifest.h`'s own already-correct form are restored, verified by execution to leave
    the census unaffected (head_dim/the multiply still sit on the comment-only continuation line,
    exactly as before this fix).
  - Poirot's own Significant 2 (the fix's own two new construction sites, GS-32/GS-33, are
    registered `confirmed-correct` -- carrying no `required_tokens` -- so Part 3 never checks
    them, and reverting either to the pre-fix `q_width = hidden_size` defect leaves the census
    PASSing) is fixed per Poirot's own routed, execution-verified remedy: both flipped to
    `status: "fixed"` with `required_tokens: ["num_attention_heads"]` and a null-anchor,
    offset-6 `required_token_scopes` record for their own file. Verified by construction: reverting
    either site's own `q_width` computation to `bc.hidden_size`/`model->hidden_size` now fails the
    census with `REGRESSED SITE`, restored clean.
  - Poirot's own Significant 3 (the window's own doc comment claims a comment-only line is "not
    counted toward window width" -- false; it is skipped from the JOINED text but still consumes
    one of the window's own `_WINDOW_SIZE` physical-line slots, so `_WINDOW_SIZE` or more
    intervening comment lines between a real co-occurrence's two halves are not reached) is
    corrected in place (census.py's own comment above the window loop) -- a documentation fix,
    the second of Poirot's own two offered remedies, folded into the same PASS-line KNOWN
    LIMITATION disclosure S4 below already needed.
  - Poirot's own Significant 4 (the PASS line claims the T-2468 F2 limitation is closed whole; it
    is closed only for QOW, and the R1 half is exactly as open as before) is corrected: the PASS
    line (census.py's own `main()`) now states what closed (QOW, across physical lines) and what
    did not (R1, still single-physical-line only; and Significant 3's own comment-width bound),
    on the PASS line itself, not only in the docstring -- matching the two-way split above.
  - Poirot's own Minor 1 (the window check tests only whether its own START line already produced
    a single-line hit, so a single-line hit at `i+1` is re-reported a second time by a window
    starting at `i`) is fixed: the window loop (census.py) now skips any window whose own SPAN
    contains a single-line hit anywhere in it, verified by construction (a minimal probe: one
    QOW-shaped statement now produces exactly one finding, not two).
  - Poirot's own Minor 2 (the `_OUT_CHANNELS_RE` comment claims every genuine `out_channels` use
    tree-wide is a bare identifier -- false; prefixed identifiers exist tree-wide) is corrected:
    the comment (census.py, above `_OUT_CHANNELS_RE`) now states the narrower, verified-true claim
    the tightening actually rests on (no CODE line in Part 2's own swept scope pairs `hidden_size`
    with a prefixed `*_out_channels`), independently re-verified by grep and by a direct sweep of
    the swept scope. The exact per-identifier count is not restated here -- see census.py's own
    comment for the current figure and its disclosed classifier; T-2524 (Poirot e4bcaeb-t2518-
    census-fix-confirmation.md Minor 2, D-SLM5883) found this bullet's own prior restatement of
    that count already stale at the commit that carried it, which is the drift this indirection
    is meant to stop recurring.
  - Poirot's own Minor 3 (`tools/t2113_b6b_adapter_delta_smoke.cpp`'s own `StepCpu` loads an
    adapter at `q_width` but calls `RunLayerLoop` without threading `q_width` through, taking the
    `= 0` default that resolves to `effective_q_width = hidden_size` -- an under-read on a
    non-square artifact) is fixed: `StepCpu` now takes and threads `q_width` (both call sites pass
    `base_geom.q_width`, the same construction the loader already uses), verified to compile
    clean at the CI's own pinned compiler (clang 18.1.8, `-fsyntax-only -std=c++20`).
  - Poirot's own Observation 1 (a merged window group's coverage is checked only at its own start
    line, so a group longer than `_covered`'s own +-25-line marker window with a covered start but
    an uncovered end would be suppressed -- no live instance today, all 20 real groups measured
    span <=12 lines and are covered at both ends) is closed on sight, per this file's own
    established convention of closing a cheap gap rather than filing a note for one: the check
    (census.py) now reads `_covered(group_start) or _covered(group_end)`.

    T-2524 correction (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 1, D-SLM5880):
    the expression above is INVERTED -- it emits only when NEITHER end is covered, which is a
    strict narrowing of the old single-end check, not the superset this bullet's own stated intent
    describes. Executed on the real tree: four measured gaps where `or` suppresses a genuine
    finding the pre-T-2518 code used to report, and the (covered start, uncovered end) case this
    Observation names is still missed. Corrected to `_covered(group_start) and _covered(group_end)`
    (census.py) -- emit unless BOTH ends are covered, the superset the bullet above always intended.
  - D-SLM5859 (the census sweeps `.worktrees`, so its result depends on which checkout it is
    invoked from -- `D:/SuperSLM` itself holds roughly sixty nested full-source checkouts under
    its own `.worktrees` directory) is fixed rather than deferred: `.worktrees` is added to
    `_SKIP_DIR_NAMES` (census.py), the same mechanism that already excludes `.git`/`out`/`build`.
    Verified by construction: invoking the fixed census with `--repo-root D:/SuperSLM` sweeps 446
    files (none under `.worktrees`) in 0.15s, where the unfixed script measured 212.17s and a
    result that depended on caller cwd.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import geometry_site_census as census  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_ADAPTER_H = os.path.join("include", "superslm", "adapter_marshal.h")
_MODEL_H = os.path.join("include", "superslm", "model.h")
_PROOF_H = os.path.join("include", "superslm", "proof_manifest.h")
_MATMUL_H = os.path.join("include", "superslm", "matmul.h")
_FORWARD_SITES_CPP_T2481 = os.path.join("src", "forward", "forward_sites.cpp")
# T-2499 (Claude/Poirot/bc2ae29-t2498-census-fixes-confirmation.md Significant 2, D-SLM5776):
# moved up from this file's own T-2475 fold-in section (formerly defined at what were then lines
# 674-675, well below `_MUTATED_TARGETS_AND_REGISTRY_PATHS`, below) so that tuple can reference
# these two constants directly instead of duplicating their spelling as separate `os.path.join(...)`
# literals -- a duplicate spelling that could (and did) go unnoticed if the constant below it in
# the file ever changed. One definition, referenced from both places.
_PROOF_MANIFEST_CPP = os.path.join("src", "proof_manifest.cpp")
_SUPERSLM_GPU_CPP = os.path.join("src", "gpu", "superslm_gpu.cpp")
# T-2524 (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 3, D-SLM5882): GS-32's and
# GS-33's own new construction sites, neither previously a `_mutated` target.
_SSLM_ABI_CPP = os.path.join("src", "sslm_abi.cpp")
_GPU_1P0_CPP = os.path.join("src", "gpu", "gpu_1p0.cpp")

_ADAPTER_FRAGMENT = 'if (proj == "q_proj" || proj == "o_proj" || proj == "down_proj") return hidden_size;'
_PROOF_FRAGMENT = "HiddenSizeGeometryMismatch, // hidden_size != num_attention_heads * head_dim -- R1, REMOVED"
# T-2524 (Poirot e4bcaeb-t2518-census-fix-confirmation.md Minor 4, D-SLM5883): `_MODEL_FRAGMENT`
# (formerly defined here) was referenced only by its own definition and by prose comments after
# T-2518 deleted every cell built on it -- dead in HEAD (StandardsDocument.md §6.6). Removed; the
# historical fold-in bullets above still name it in prose, which needs no live constant to be true.


def _write_newline_for(raw_bytes: bytes) -> str:
    """T-2491 (Poirot dcefab3-t2486-census-content-keying-confirmation.md Significant 2,
    D-SLM5718), extracted as its own pure function by T-2499 (Claude/Poirot/bc2ae29-t2498-census-
    fixes-confirmation.md item 3 sweep, D-SLM5778): picks the `open(..., newline=...)` argument
    that reproduces a file's own newline convention on write-back, from that file's own raw bytes
    rather than from its extension or the platform default. `.h`/`.cpp`/`.hlsl` files are
    `attr/text` (tool-native, `w/crlf` on this checkout) and the platform-default write already
    reproduced that -- the defect this closes is an `attr/text eol=lf` file (e.g. the JSON
    registry): the platform-default write translates every `\\n` to `\\r\\n` regardless of the
    file's own pinned convention, so a byte-for-byte restore of an eol=lf file silently comes back
    CRLF.

    Pulled out of `_mutated`, below, so this branch can be pinned directly on synthetic bytes
    (`test_write_newline_for_picks_the_convention_from_raw_bytes`) rather than only through a real
    repo file. T-2526 (Poirot 96abd9e-t2524-census-confirmation2.md Minor 1, D-SLM5899) widened
    `tools/geometry_site_registry.json` -- `attr/text eol=lf`, genuinely LF on disk (confirmed by
    direct execution, T-2531) -- from a directly-written path into a real `_mutated` target too,
    so this function's own eol=lf branch IS now exercised through the real tree by two cells
    (`test_part2_window_group_covered_start_uncovered_end_is_still_reported` and its sibling,
    below).

    What `Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md` Sec7 (D-SLM5758) measured, on
    this Windows/CRLF checkout, is BOTH of `_mutated`'s writes reverted to the platform default --
    `open(full, "w", encoding="utf-8")`, no `newline=` argument, not a blanket `newline=""` --
    leaving the whole file green and the regression undetected (Claude/Poirot/
    ed0c67d-t2502-census-fixes-confirmation.md Significant 2, D-SLM5785). T-2499's extraction of
    this function did NOT close that finding, whatever the module's own prose once claimed
    (Claude/Poirot/5e128ee-t2530-superslm-ci-green-review.md C-1, D-SLM5936): a second, separate
    writer --
    `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_all`
    (below) hand-rolled its own hardcoded `newline=""` against this same registry path instead of
    calling this function, and that cell's own still-correct write, running after the two
    window-group cells corrupt the registry through a REVERTED `_mutated`, silently restored the
    checkout to its correct bytes as a side effect of its own teardown -- so the module-scoped
    `_mutated_targets_and_registry_are_byte_identical_after_the_module_runs` fixture never saw a
    mismatch, and the regression stayed undetectable end to end (reproduced by direct execution,
    T-2531: reverting `_mutated`'s `newline=_write_newline` argument with that cell unchanged left
    the FULL module at 39 passed, 0 failed). T-2531 routed that cell through `_mutated` too, so
    every real-tree write in this module decides its newline convention through this ONE
    function.

    T-2533 correction (Poirot 4187739-t2532-superslm-ci-green-confirmation.md C-1n): the closure
    claim T-2531 wrote here -- "the byte-identity fixture fires regardless of which cell happens
    to run last" -- is FALSE as an unqualified statement, and D-SLM5944 recorded the same false
    breadth. Executed both ways, on byte-preserved trees: on a CRLF checkout the identical revert
    still fails the module (`39 passed, 1 error`, byte mismatch on
    `tools/geometry_site_registry.json`); on a genuine LF checkout -- built by cloning this repo
    fresh into WSL/Ubuntu, matching every `ubuntu-latest` CI job that runs this module -- the
    IDENTICAL revert leaves `39 passed, 0 failed`: `tools/geometry_site_registry.json` is already
    LF there, Linux's own platform default is also LF, and a platform-default write to it is a
    no-op, so the real tree has nothing left to discriminate. The call-site regression D-SLM5785/
    D-SLM5758 name is real and stays undetectable by any REAL-TREE cell in this module on the
    platform every CI job that runs it actually uses.

    Closed for that platform by `test_mutated_call_site_preserves_a_forced_crlf_file_regardless_
    of_checkout_platform` (below): a synthetic scratch file carrying FORCED CRLF bytes, outside
    `_MUTATED_TARGETS_AND_REGISTRY_PATHS` and independent of the real tree's own OS-dependent
    state, round-tripped through `_mutated`'s own call site -- discriminating the regression on
    every platform, including the LF one where nothing else in this module can."""
    return "\r\n" if b"\r\n" in raw_bytes else ""


@contextlib.contextmanager
def _mutated(rel_path: str, transform):
    """Mutates a real production file, at `rel_path` under the repo root, for the duration of
    the `with` block via `transform(original_text) -> new_text`, then restores the original
    content byte-for-byte. See this module's own docstring for why the real tree is mutated
    in place rather than exercised against a synthetic `tmp` repo root. See `_write_newline_for`,
    above, for how the write-back newline convention is chosen.

    T-2535 (Poirot 2945361-t2534-superslm-ci-green-confirmation2.md O-1): `rel_path` is also
    called with an ABSOLUTE path by both forced-bytes `_mutated`-call-site cells (below
    `_write_newline_for`) -- relied on, not stated, until this note. `os.path.join(_REPO_ROOT,
    rel_path)` discards `_REPO_ROOT` entirely when `rel_path` is already absolute (`os.path`'s
    own documented behaviour, identical on POSIX and Windows: the last absolute component wins),
    so passing a scratch file's own absolute path here operates on that file directly rather
    than joining it under the repo root, with no code change needed -- this is what lets those
    two cells build a synthetic scratch file OUTSIDE the checked-out tree and still route it
    through this exact function.
    """
    full = os.path.join(_REPO_ROOT, rel_path)
    with open(full, "rb") as f:
        _write_newline = _write_newline_for(f.read())
    with open(full, "r", encoding="utf-8") as f:
        original = f.read()
    try:
        with open(full, "w", encoding="utf-8", newline=_write_newline) as f:
            f.write(transform(original))
        yield
    finally:
        with open(full, "w", encoding="utf-8", newline=_write_newline) as f:
            f.write(original)


def test_write_newline_for_picks_the_convention_from_raw_bytes():
    """T-2499 (item 3 sweep, D-SLM5778): pins `_write_newline_for`'s own branch directly on
    synthetic bytes rather than only through a real repo file.

    T-2533 (Poirot 4187739-t2532-superslm-ci-green-confirmation.md M-2n) correction: this
    docstring used to claim "every one of `_mutated`'s real targets is pure CRLF today" --
    stale since T-2526 (`tools/geometry_site_registry.json` has been a real, LF-carrying
    `_mutated` target since that round) and doubly so since T-2531's own C-1 fix (below,
    `test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_
    all` now calls `_mutated` too) -- the exact self-contradiction M-2n names, a false claim
    sitting 45 lines from the true one this same diff introduced. What remains true, and is
    this cell's own reason to exist: reverting this FUNCTION's own logic (its `return`
    statement) is caught here directly, on synthetic bytes, regardless of platform or of any
    real file's own checked-out line-ending convention. What this cell does NOT catch --
    proven by direct execution, T-2533 (see `test_mutated_call_site_preserves_a_forced_crlf_
    file_regardless_of_checkout_platform`, below) -- is a revert of the CALL SITE's own
    `newline=_write_newline` argument (`_mutated`, immediately above): on a real LF checkout
    (every `ubuntu-latest` CI job that runs this module), `tools/geometry_site_registry.json`
    is ALREADY LF, and Linux's own platform default is ALSO LF, so a platform-default write to
    it is a no-op -- there is nothing for the real tree's own state to discriminate. That gap
    is what the two forced-bytes `_mutated`-call-site cells below close, on any platform.
    """
    assert _write_newline_for(b"a line\r\nanother line\r\n") == "\r\n"
    assert _write_newline_for(b"a line\nanother line\n") == ""
    assert _write_newline_for(b"") == ""

def test_mutated_call_site_preserves_a_forced_crlf_file_regardless_of_checkout_platform():
    """T-2533 (Poirot 4187739-t2532-superslm-ci-green-confirmation.md C-1n): D-SLM5785's own
    regression -- `_mutated`'s own `newline=_write_newline` argument reverted to the platform
    default -- is undetectable by every OTHER cell in this module on a real LF checkout (every
    `ubuntu-latest` CI job that runs it): `tools/geometry_site_registry.json`, the one real
    `_mutated` target whose own convention (`eol=lf`) differs from the platform default on
    Windows, is ALREADY LF on a Linux checkout, where the platform default is ALSO LF -- a
    platform-default write to it is a no-op there, so the real tree's own checked-out state has
    nothing left to discriminate (confirmed by direct execution, T-2533: on a fresh `git clone`
    into WSL/Ubuntu -- matching `ubuntu-latest`'s own checkout convention -- the identical
    revert this cell reproduces leaves the WHOLE module at `39 passed, 0 failed`).

    This cell does not depend on the checked-out tree's own OS-dependent line-ending state at
    all: it constructs a scratch file (never a tracked path -- outside
    `_MUTATED_TARGETS_AND_REGISTRY_PATHS`, so the module-scoped byte-identity fixture does not
    also need to know about it) with FORCED CRLF bytes, a convention that diverges from the
    platform default on every OS this suite runs on except Windows itself, and proves
    `_mutated`'s own call-site argument round-trips it byte-for-byte through a real mutate+
    restore cycle -- discriminating the D-SLM5785/D-SLM5758 regression on Linux, where every
    other real-tree cell in this module cannot, because it does not depend on the real tree
    agreeing with the platform default to begin with.
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix="t2533_crlf_scratch_")
    os.close(fd)
    original = b'{\r\n  "a": 1\r\n}\r\n'
    try:
        with open(path, "wb") as f:
            f.write(original)
        with _mutated(path, lambda text: text.replace('"a": 1', '"a": 2')):
            with open(path, "rb") as f:
                mutated_bytes = f.read()
            assert mutated_bytes == b'{\r\n  "a": 2\r\n}\r\n', (
                "the mutation itself must preserve CRLF while changing content -- got %r" % mutated_bytes
            )
        with open(path, "rb") as f:
            restored = f.read()
        assert restored == original, (
            "_mutated must restore a forced-CRLF scratch file byte-for-byte, on every platform -- "
            "got %r, want %r (this is D-SLM5785/D-SLM5758's own regression, reproduced without "
            "depending on the checked-out tree's own OS-dependent line endings)" % (restored, original)
        )
    finally:
        os.remove(path)


def test_mutated_call_site_preserves_a_forced_lf_file_regardless_of_checkout_platform():
    """T-2533 (Poirot 4187739-t2532-superslm-ci-green-confirmation.md C-1n), sibling of the
    CRLF cell above: the same construction with FORCED LF bytes, proving `_mutated`'s own
    call site also round-trips the LF branch correctly on every platform. This direction
    already discriminates a blanket-`newline=""` call-site reversion on Windows (LF diverges
    from that platform's own CRLF default there); included for symmetry with the CRLF cell
    above and so this module tests both `_write_newline_for` branches at the call site, not
    only in `_write_newline_for` itself (`test_write_newline_for_picks_the_convention_from_
    raw_bytes`, above).
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix="t2533_lf_scratch_")
    os.close(fd)
    original = b'{\n  "a": 1\n}\n'
    try:
        with open(path, "wb") as f:
            f.write(original)
        with _mutated(path, lambda text: text.replace('"a": 1', '"a": 2')):
            with open(path, "rb") as f:
                mutated_bytes = f.read()
            assert mutated_bytes == b'{\n  "a": 2\n}\n', (
                "the mutation itself must preserve LF while changing content -- got %r" % mutated_bytes
            )
        with open(path, "rb") as f:
            restored = f.read()
        assert restored == original, (
            "_mutated must restore a forced-LF scratch file byte-for-byte, on every platform -- "
            "got %r, want %r" % (restored, original)
        )
    finally:
        os.remove(path)




# T-2497 (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Significant 2, D-SLM5758)
# established this tuple: every real path this module's own cells touch, closing per-cell
# dirty-checkout drift at the root (the module-scoped byte-identity fixture below) rather than
# trusting each cell to restore what it mutates. The two `src/` entries are the same
# `_PROOF_MANIFEST_CPP`/`_SUPERSLM_GPU_CPP` constants this file's own T-2475-fold-in cells use,
# defined once above (not duplicated here), so the two spellings cannot desync.
#
# CURRENT STATE (T-2535, Poirot 2945361-t2534-superslm-ci-green-confirmation2.md M-4): ten
# entries. Every one except `_MODEL_H` is a real `_mutated` target, reached EXCLUSIVELY through
# that one function -- there is no direct-write path to any entry of THIS TUPLE anywhere in
# this module. (T-2537 correction, Poirot 67bfcbf-t2536-superslm-ci-green-confirmation3.md
# M-2: the unqualified form of that sentence -- "there is no direct-write path anywhere in
# this module", with no "to any entry of this tuple" -- was false the moment it was written:
# the two forced-bytes cells this same round's own C-1n fix added
# (`test_mutated_call_site_preserves_a_forced_crlf_file_regardless_of_checkout_platform` and
# its LF sibling, below `_write_newline_for`) write their own `tempfile.mkstemp()` scratch
# paths directly with `open(path, "wb")` -- by design, since those paths are deliberately
# OUTSIDE this tuple and O-1's own note on `_mutated` documents exactly why. The true
# statement is scoped to this tuple's own entries, never to every write statement in the
# module.) `_MODEL_H` alone
# is a harmless superset guard: mutated by nothing (`_MODEL_FRAGMENT`, its own would-be writer,
# is dead and removed; see the mechanism-cells section's own header comment), a byte-identity
# check over an untouched path costs nothing to keep. It does NOT close every future cell that
# touches the real tree: a cell added later that mutates a real-tree path outside this tuple is
# NOT caught here -- extend `_MUTATED_TARGETS_AND_REGISTRY_PATHS` in the same change that adds
# such a cell.
#
# This statement has been wrong three times before as this module's own construction changed
# under it without the comment being updated in the same change -- T-2524 (Minor 4, D-SLM5883:
# an already-stale entry count), T-2526 (Minor 1, D-SLM5899: an already-stale direct/`_mutated`
# split), T-2533 (its own M-2n: the same split, stale again). Each time it was restated in place
# rather than left standing -- and restating it is exactly the failure mode M-4 named: a reader
# who stops at an early generation reads a false statement, and reproducing the whole chain
# every round guarantees a further stale generation on the next touch. `git log -p` on this
# hunk carries that chain for anyone tracing what changed and why; it is not reproduced here
# again.
_MUTATED_TARGETS_AND_REGISTRY_PATHS = (
    _ADAPTER_H,
    _MATMUL_H,
    _MODEL_H,
    _PROOF_H,
    _FORWARD_SITES_CPP_T2481,
    _PROOF_MANIFEST_CPP,
    _SUPERSLM_GPU_CPP,
    _SSLM_ABI_CPP,
    _GPU_1P0_CPP,
    os.path.join("tools", "geometry_site_registry.json"),
)


@pytest.fixture(scope="module", autouse=True)
def _mutated_targets_and_registry_are_byte_identical_after_the_module_runs():
    """Session-scoped in spirit, module-scoped in fact (this module's own `census` import is
    already bound to this repo's `tools/`, per this file's own docstring above) -- snapshots the
    raw bytes of every path in `_MUTATED_TARGETS_AND_REGISTRY_PATHS` BEFORE the first cell in
    this module runs, and asserts them byte-identical AFTER the last one has, whatever mix of
    `_mutated()` blocks ran in between -- every entry except `_MODEL_H` is reached exclusively
    through `_mutated`, there is no direct-write path to any entry of THIS TUPLE anywhere in
    this module (T-2537 correction, M-2: not "no direct-write path in this module" --
    scratch files outside this tuple ARE written directly, by design; see the tuple's own
    "CURRENT STATE" comment, above). Closes T-2496's Significant 2
    (D-SLM5758) at the root rather than per-cell: a fix that touches one of the paths named
    in `_MUTATED_TARGETS_AND_REGISTRY_PATHS` and leaves it modified fails HERE regardless of what
    that fix's own cell asserts.

    Its guarantee is exactly that list, not the whole real tree (T-2499, Claude/Poirot/bc2ae29-
    t2498-census-fixes-confirmation.md Significant 2, D-SLM5776): a future cell that mutates a
    real-tree path outside `_MUTATED_TARGETS_AND_REGISTRY_PATHS` and fails to restore it is NOT
    caught here -- extend that tuple in the same change that adds such a cell, the same discipline
    every prior real-tree cell in this module already follows via `_mutated`."""
    paths = [os.path.join(_REPO_ROOT, rel) for rel in _MUTATED_TARGETS_AND_REGISTRY_PATHS]
    before = {}
    for p in paths:
        with open(p, "rb") as f:
            before[p] = f.read()
    yield
    changed = []
    for p in paths:
        with open(p, "rb") as f:
            after = f.read()
        if after != before[p]:
            changed.append(os.path.relpath(p, _REPO_ROOT))
    assert not changed, (
        "this module's own suite left the checkout modified -- byte mismatch after the run on: "
        + ", ".join(changed)
    )


# --- Mechanism cells: _part2_excise_excluded_text in isolation. ---
#
# T-2518 (Claude/Poirot/a3a20bc-t2509-adapter-geometry-review.md Critical 1): this section, and
# the population below it, used to carry cells keyed on `_ADAPTER_H`/`_ADAPTER_FRAGMENT` and
# `_MODEL_H`/`_MODEL_FRAGMENT` -- `test_excise_removes_the_adapter_marshal_fragment_entirely`,
# `test_excise_leaves_appended_new_content_in_the_remainder`,
# `test_excise_leaves_prepended_new_content_in_the_remainder`, and
# `test_excise_removes_the_full_widened_model_h_fragment_including_its_own_comment`. T-2509 fixed
# BOTH residuals these cells' own subjects named (adapter_marshal.h's AdapterOutChannelsFor q_proj
# branch, GS-29; and restructured model.h's ConfigGeometryHiddenSizeMismatch comment so head_dim
# no longer shares a physical line with hidden_size there) -- `_PART2_EXCLUDED_TEXT` no longer
# carries an entry for either file (module docstring's own "WIDENED T-2509" note; census.py's own
# header comment above `_PART2_EXCLUDED_TEXT`), so `_ADAPTER_FRAGMENT`/`_MODEL_FRAGMENT` no longer
# match anything in the real tree. T-2518 deleted all four rather than retargeting any of them.
#
# T-2524 correction (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 2, D-SLM5881):
# that reasoning does not hold for two of the four. `test_excise_removes_the_adapter_marshal_
# fragment_entirely` and `test_excise_removes_the_full_widened_model_h_fragment_including_its_
# own_comment` call `_part2_excise_excluded_text` against each file's OWN fragment constant -- once
# neither fragment exists in the tree, both silently no-op (mutate nothing, assert against an
# unmutated tree) and are correctly deleted; `test_excise_removes_the_adapter_marshal_fragment_
# entirely` additionally adds nothing even retargeted (a whole-line skip also removes the fragment
# outright, so it does not discriminate from the surviving `..._proof_manifest_h_fragment_
# including_its_own_comment` cell below). But `test_excise_leaves_appended_new_content_in_the_
# remainder` and `test_excise_leaves_prepended_new_content_in_the_remainder` call
# `_part2_excise_excluded_text(path, line)` on a SYNTHETIC STRING LITERAL -- they never touch the
# tree and never needed a compilable subject, only a REGISTERED path, which `proof_manifest.h`
# still is. Restored below, retargeted onto `_PROOF_H`/`_PROOF_FRAGMENT` -- executed: pass on the
# clean tree, and BOTH fail under `_part2_excise_excluded_text` reverted to skip-whole-line-on-
# match, the mutation this file's own S2 red-check comment block (further down this file) exists
# to measure. These two are the excise mechanism's only UNIT-LEVEL pins for "new content sharing
# an excused line is left in the remainder" -- the surviving end-to-end population further down
# this file (`test_appending_after_proof_manifest_h_s_trailing_comment_fires_on_apparatus_unsound_
# grounds`, `test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h`) both
# depend on `proof_manifest.h` still carrying a live exclusion, one edit away from having no
# subject left at all -- these two take their input as an argument instead, and do not.
#
# `_part2_excise_excluded_text`'s generic contract -- removes a registered fragment's own text,
# leaves everything else -- is proven by all four surviving cells in this section together.

def test_excise_does_not_touch_a_line_in_an_unregistered_file():
    line = _ADAPTER_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(os.path.join("include", "superslm", "other.h"), line)
    assert out == line


def test_excise_does_not_touch_an_unrelated_line_in_a_registered_file():
    # T-2518: retargeted from _ADAPTER_H to _PROOF_H -- adapter_marshal.h is no longer a
    # registered exclusion (this section's own header comment, above), so this cell's own premise
    # ("a registered file") now needs the one file that actually is registered.
    line = "uint64_t unrelated_thing = 1;\n"
    out = census._part2_excise_excluded_text(_PROOF_H, line)
    assert out == line


def test_excise_removes_the_full_widened_proof_manifest_h_fragment_including_its_own_comment():
    line = "\t" + _PROOF_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(_PROOF_H, line)
    assert "head_dim" not in out
    assert "hidden_size" not in out


def test_excise_leaves_appended_new_content_in_the_remainder():
    # T-2524 (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 2, D-SLM5881):
    # restored, retargeted from _ADAPTER_H/_ADAPTER_FRAGMENT to _PROOF_H/_PROOF_FRAGMENT -- this
    # cell never mutates the tree; it calls `_part2_excise_excluded_text` directly on a synthetic
    # string and needs only a REGISTERED path, which `proof_manifest.h` still is (unlike
    # adapter_marshal.h, whose own exclusion T-2509 fixed away). See this section's own header
    # comment for why T-2518 deleted this cell in error.
    line = "\t" + _PROOF_FRAGMENT + ' if (proj == "k_proj") { uint64_t out_channels = hidden_size; return out_channels; }\n'
    out = census._part2_excise_excluded_text(_PROOF_H, line)
    assert "k_proj" in out
    assert "num_attention_heads" not in out, (
        "the excused fragment's own text must be gone from the remainder"
    )


def test_excise_leaves_prepended_new_content_in_the_remainder():
    # T-2524: restored, retargeted from _ADAPTER_H/_ADAPTER_FRAGMENT to _PROOF_H/_PROOF_FRAGMENT --
    # see the cell above and this section's own header comment.
    line = 'uint64_t sneaky_new_site = hidden_size; /* alias */ \t' + _PROOF_FRAGMENT + "\n"
    out = census._part2_excise_excluded_text(_PROOF_H, line)
    assert "sneaky_new_site" in out


# --- Cell zero: the real, unmutated tree is clean today. ---

def test_real_tree_baseline_is_clean_today():
    assert census.run_census(_REPO_ROOT) == []


# --- Must-accept (T-2468 Sec1, D-SLM5601): comment inserted above each excluded line --
# position-independence, the property text-keying was built for, unaffected by this fix.
#
# T-2518 (Poirot Critical 1): this population used to have three members, one per excluded file.
# The adapter_marshal.h and model.h members are deleted, not retargeted -- both files' own
# exclusions are gone (T-2509 fixed the residuals they named; see the mechanism-cells section's
# own header comment, above, for the full account), so `_ADAPTER_FRAGMENT`/`_MODEL_FRAGMENT` no
# longer match either file and both cells silently no-op (mutate nothing, assert against an
# unmutated tree) rather than exercising position-independence at all. Only proof_manifest.h's
# exclusion is still live; that member is unaffected and remains the population's sole survivor. ---

def test_must_accept_comment_above_proof_manifest_h_exclusion():
    def _t(text):
        return text.replace("\t" + _PROOF_FRAGMENT, "\t// a harmless comment, no defect\n\t" + _PROOF_FRAGMENT, 1)
    with _mutated(_PROOF_H, _t):
        assert census.run_census(_REPO_ROOT) == []


# --- Must-reject (T-2468 Sec2, D-SLM5602): a genuinely new, unregistered site in a
# zero-marker production header -- the obvious candidate a defeated exclusion would let hide. ---

def test_must_reject_new_unmarked_site_in_zero_marker_header():
    assert census._MARKER_RE.search(open(os.path.join(_REPO_ROOT, _MATMUL_H), encoding="utf-8").read()) is None, (
        "this cell's premise is a zero-marker file -- no marker-proximity window to absorb the new site"
    )
    injected = (
        "\ninline uint64_t T2475ProbeOutWidthFor(const std::string& proj, uint64_t hidden_size) {\n"
        '\tif (proj == "q_proj" || proj == "o_proj") return hidden_size;\n'
        "\treturn 0;\n"
        "}\n"
    )

    def _t(text):
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    with _mutated(_MATMUL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a new, unmarked QOW site in a zero-marker header must fail the census"
    assert any("matmul.h" in f and "q_proj" in f for f in failures)


# --- THE finding this ticket closes (T-2468 Sec3, D-SLM5603): a genuinely new statement
# appended onto the SAME physical line as an already-excused fragment.
#
# T-2518 (Poirot Critical 1): this population used to be reproduced on all three exclusions.
# `test_defeats_the_prior_mechanism_same_line_append_adapter_marshal` (adapter_marshal.h) and the
# model.h half of the pair below are deleted, not retargeted -- both files' own exclusions are
# gone (T-2509 fixed the residuals they named; see the mechanism-cells section's own header
# comment, near the top of this file, for the full account), so `_ADAPTER_FRAGMENT`/
# `_MODEL_FRAGMENT` no longer match either file. Only proof_manifest.h's exclusion is still live;
# that member of the population is unaffected and remains below. ---

def test_appending_after_proof_manifest_h_s_trailing_comment_fires_on_apparatus_unsound_grounds():
    # T-2479 Significant 2 (Poirot f363c2a-t2479-census-class-confirmation.md Sec5, D-SLM5652):
    # this construction's own injected text lands AFTER the fragment's trailing `//` comment, so
    # from a real compiler's own point of view the ENTIRE remainder of the physical line --
    # including the appended `uint64_t o_proj_out_channels = hidden_size;` -- is commented-out
    # prose, never compiled, and cannot be a geometry site. Census.py does not know that: once
    # the fragment (enumerator PLUS its own trailing `// R1: ...` comment) is excised as one
    # exact-text match, only the appended text remains in the scanned line, and that text
    # independently looks QOW-shaped (`hidden_size` plus the `out_channels` substring inside its
    # own identifier) regardless of ever having sat inside a comment. Executed: this construction
    # DOES fire, on `UNMARKED QOW PATTERN HIT`, not merely a `DEAD EXCLUSION` -- but the
    # right-verdict-for-the-wrong-reason gap D-SLM5652 names still holds: nothing here proves the
    # census recognizes commented-out text as inert, only that this SPECIFIC probe's own
    # appended identifier happens to independently trip a pattern. `ConfigGeometryHiddenSizeMismatch`
    # and `HiddenSizeGeometryMismatch` are enumerators inside an `enum class` body, which admits
    # enumerators, not statements, so there is no producible same-line STATEMENT for
    # model.h/proof_manifest.h (T-2491, Poirot dcefab3-t2486-census-content-keying-confirmation.md
    # Sec4, D-SLM5716/D-SLM5717, D-SLM5732, executed with the CI's own pinned compiler, clang
    # 18.1.8, `-fsyntax-only -std=c++20`: `error: missing ',' between enumerators`). A same-line
    # ENUMERATOR, by contrast, IS producible and caught -- see
    # `test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h`, below
    # (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Significant 1, D-SLM5756/D-SLM5757).
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\t" + _PROOF_FRAGMENT + "  uint64_t o_proj_out_channels = hidden_size; // T-2475 probe",
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "appending after proof_manifest.h's own trailing comment must still redden the census"
    assert any("UNMARKED QOW PATTERN HIT" in f and "proof_manifest.h" in f and "out_channels" in f
               for f in failures)


# --- No same-line STATEMENT is producible for model.h/proof_manifest.h (T-2491, Poirot dcefab3-
# t2486-census-content-keying-confirmation.md Sec4, D-SLM5716/D-SLM5717, D-SLM5732): both excused
# lines are enumerators inside an `enum class` body (`SslmModelStatus`, `ConfigGeometryStatus`),
# which admits enumerators, not statements. T-2481's two cells here inserted a statement between
# the enumerator and its own trailing comment, believing that construction was real, compiled code
# the production path could emit; executed with the CI's own pinned compiler (clang 18.1.8,
# `-fsyntax-only -std=c++20`), both headers fail with `error: missing ',' between enumerators`.
# The cells also had zero discrimination, executed both with the QOW family forced false and with
# the injected identifier replaced by an inert statement: the assertion held either way, because
# what actually fired was the trailing prose comment's own R1 co-occurrence once the exclusion's
# exact-text match went dead -- not recognition of the injected statement. Deleted rather than
# replaced with a third attempt at the SAME (statement) shape.
#
# T-2491's own replacement text overclaimed: "there is no producible same-line geometry defect an
# enum body can host" is false -- refuted by construction (Claude/Poirot/ba29de4-t2496-census-
# fixes-confirmation.md Significant 1, D-SLM5756/D-SLM5757). An enum body admits enumerators, not
# statements, but a same-line ENUMERATOR prepend is legal C++ and enough: the excused fragment is
# the whole stripped line, so a prepend leaves it intact -- the exclusion still fires and the
# injected text is scanned as ordinary remainder. Executed at the CI's pinned compiler, clang
# 18.1.8, `-fsyntax-only -std=c++20`: `q_proj_out_channels_hidden_size, ` prepended to `model.h`'s
# fragment and `o_proj_out_channels_hidden_size, ` prepended to `proof_manifest.h`'s both compile
# clean and fire `UNMARKED QOW PATTERN HIT`; both discriminated (silent with the QOW family
# disabled, silent with an inert enumerator instead of the defect) -- the corrected fact is: a
# same-line statement is not producible inside an enum body; a same-line enumerator is, and the
# census catches it.
#
# T-2518 (Poirot Critical 1): this population, and the same-line-append/prepend population above
# it, used to have BOTH an adapter_marshal.h member (`test_defeats_the_prior_mechanism_same_
# line_append_adapter_marshal`, `test_prepended_content_on_an_excused_line_is_also_caught` -- the
# same-line STATEMENT class's whole population, since adapter_marshal.h's excused line sat inside
# a function body, not an enum) and a model.h member (`test_defeats_the_prior_mechanism_same_
# line_enumerator_prepend_model_h`). T-2509 fixed the residual adapter_marshal.h's own exclusion
# named (GS-29) and restructured model.h's comment so its exclusion is no longer needed either
# (see the mechanism-cells section's own header comment, near the top of this file, for the full
# account) -- neither file carries a live exclusion any more, so there is no longer a real,
# compilable subject for either shape on either file. Both members deleted rather than retargeted
# at a synthetic exclusion invented for the sole purpose of keeping the cell count unchanged; the
# statement-class property (same-line append/prepend defeats a function-body exclusion) has no
# live subject left anywhere in this tree. The enumerator-prepend property is still fully proven,
# against the one exclusion still live, by the surviving proof_manifest.h cell below. ---

def test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h():
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\to_proj_out_channels_hidden_size, " + _PROOF_FRAGMENT,
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, (
        "a new enumerator prepended before proof_manifest.h's own excused enumerator must FAIL"
    )
    assert any("UNMARKED QOW PATTERN HIT" in f and "proof_manifest.h" in f and "out_channels" in f
               for f in failures)
    assert not any("DEAD EXCLUSION" in f for f in failures)


# --- Safe-direction Note (T-2468 Sec4a cases 1-2, D-SLM5604): rename/reformat of the excused
# line itself must still re-trigger the census. This fix must not make exclusions stickier.
#
# T-2518 (Poirot Critical 1): both cells below used to target model.h (rename) and adapter_
# marshal.h (reformat) -- one representative file per case, rather than all three, to avoid a full
# 3x3 duplication. Neither file carries a live exclusion any more (see the mechanism-cells
# section's own header comment, near the top of this file), so both are retargeted onto
# proof_manifest.h, the one file whose exclusion is still live and whose fragment sits on a single
# physical line the way the original model.h fragment did (unlike the corrected, multi-line
# model.h comment this round leaves behind). ---

def test_safe_direction_rename_of_the_excused_enumerator_reddens():
    def _t(text):
        return text.replace("HiddenSizeGeometryMismatch,", "HiddenSizeGeometryMismatchRenamed,", 1)
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "renaming the excused enumerator must re-trigger the census (safe direction)"


def test_safe_direction_reformat_of_the_excused_line_reddens():
    def _t(text):
        return text.replace(_PROOF_FRAGMENT, _PROOF_FRAGMENT.replace(" != ", "  !=  "), 1)
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "whitespace-only reformat of the excused line must re-trigger the census (safe direction)"


# --- Wrap/comment-split (T-2468 Sec4a case 3): splitting the excused declaration from its own
# comment removes the pattern co-occurrence entirely. Under T-2468's own narrow fragment (the
# enumerator name alone) this PASSed uninformatively -- Part 2 had nothing to exclude, so the
# PASS was never evidence the exclusion survived. T-2475 widened this exclusion's own fragment
# to its full original line (declaration + trailing comment together, see the header comment
# above `_PART2_EXCLUDED_TEXT`), so the split ALSO breaks the widened fragment's own exact-text
# match -- and the dead-exclusion check (this file's own `test_dead_exclusion_*` cells) now
# reports it by name instead of passing silently. Strictly more informative than the T-2468-era
# behavior: a human re-reading a DEAD EXCLUSION for this entry after a wrap/split would correctly
# read it as "re-derive or drop this exclusion," where a bare PASS gave no such signal. ---

def test_wrap_comment_split_is_now_reported_as_a_dead_exclusion_not_a_silent_pass():
    def _t(text):
        return text.replace(
            "\t" + _PROOF_FRAGMENT,
            "\tHiddenSizeGeometryMismatch,\n\t// hidden_size != num_attention_heads * head_dim -- R1, REMOVED",
            1,
        )
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, (
        "splitting the widened proof_manifest.h fragment across two lines must now be reported "
        "as a DEAD EXCLUSION (T-2475's own dead-entry check), not silently PASS as it did under "
        "T-2468's narrower, enumerator-only fragment"
    )
    assert any("DEAD EXCLUSION" in f and "proof_manifest.h" in f for f in failures)
    # And it must not ALSO be reported as an UNMARKED pattern hit -- the split still removes the
    # co-occurrence from every resulting line, exactly as T-2468 found; only the exclusion's own
    # aliveness changed, not whether the split content itself trips a pattern.
    assert not any("UNMARKED" in f for f in failures)


# --- F2 (T-2468 Sec4a case 4): a genuinely new site whose co-occurring tokens land on different
# physical lines. T-2509 widened Part 2 with a sliding multi-line window (module docstring's own
# "WIDENED T-2509" note), but QOW-ONLY, deliberately -- R1's own bare `"*" in line` test is not
# safely windowable (census.py's own comment above the window loop). So F2 is now HALF closed:
# the QOW two-line split below IS caught; the R1 two-line split is not, and is still the
# documented KNOWN LIMITATION main() prints (T-2518 correction, Poirot Significant 4 -- this
# split reproduces both halves so the PASS/FAIL for each is proven, not assumed, and cross-checked
# against the KNOWN LIMITATION text main() now prints, so the two cannot drift apart silently). ---

def test_two_line_split_qow_case_is_now_caught_by_the_window():
    # Neither physical line carries BOTH a QOW token and hidden_size on its own -- the
    # condition line names q_proj/o_proj without hidden_size, and the return line names
    # hidden_size without any QOW token, exactly the shape an ordinary wrapped conditional
    # produces (T-2468 Sec4a case 4's own construction, reproduced here). T-2509's own widened
    # window closes this half of F2 -- unlike at T-2468/T-2475/T-2481/T-2491/T-2497/T-2499, this
    # construction now FAILS the census (T-2518 correction, Poirot Significant 4).
    injected = (
        "\ninline uint64_t T2475TwoLineProbe(const std::string& proj, uint64_t hidden_size) {\n"
        '\tif (proj == "q_proj" || proj == "o_proj")\n'
        "\t\treturn hidden_size;\n"
        "\treturn 0;\n"
        "}\n"
    )

    def _t(text):
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    with _mutated(_MATMUL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, (
        "a new QOW site split across two physical lines, within _WINDOW_SIZE, must now be caught "
        "by the T-2509 window -- if this reverts to passing, the window has regressed"
    )
    assert any("UNMARKED QOW PATTERN HIT (window)" in f and "matmul.h" in f for f in failures)


def test_two_line_split_r1_case_is_still_the_documented_f2_limitation():
    # R1's own co-occurrence (head_dim, a heads-token, and "*") split across two physical lines --
    # the heads-token on the condition line, `head_dim` and the multiply on the next -- so neither
    # line alone satisfies `_r1_multiply_hit`, and the window (QOW-only, deliberately -- see this
    # section's own header comment) never tests R1 at all, windowed or not. This is the half of F2
    # T-2509 does NOT close; a change to this assertion means R1 was widened into the window and
    # the KNOWN LIMITATION text below (and the module docstring) must be updated in the same
    # change.
    injected = (
        "\ninline uint64_t T2518TwoLineR1Probe(uint64_t num_attention_heads, uint64_t head_dim) {\n"
        "\tuint64_t reconstructed_hidden_size = num_attention_heads;\n"
        "\treconstructed_hidden_size *= head_dim;\n"
        "\treturn reconstructed_hidden_size;\n"
        "}\n"
    )

    def _t(text):
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    with _mutated(_MATMUL_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], (
        "a new R1 site split across two physical lines is NOT caught today -- this is F2's own "
        "R1 half, deliberately left open (the window is QOW-only) rather than closed by this "
        "ticket; a change to this assertion means R1 was widened into the window and the KNOWN "
        "LIMITATION text below (and the module docstring) must be updated in the same change"
    )


# =====================================================================================
# T-2524 (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 1, D-SLM5880): a merged
# window group's own coverage check was `_covered(group_start) or _covered(group_end)` -- this
# EMITS ONLY when NEITHER end is covered, a strict narrowing of the pre-T-2518 single-end check,
# not the superset the comment above it always described. Fixed to `and` -- emit unless BOTH
# ends are covered. Nothing in the suite asserted that a finding IS reported for a genuinely
# unmarked group with one end near an unrelated marker, in either direction; these two cells
# close that gap directly, each planting its own disposable marker (`_marker_line`, registered
# against a `confirmed-correct` registry entry with no `required_tokens`, so Part 3 never checks
# it) at a swept distance from a genuinely unmarked two-line QOW split -- reproducing Poirot's own
# executed construction. A 4-line comment-only buffer on every side keeps the real surrounding
# tree's own markers and window scans from reaching in or being reached.
# =====================================================================================

# T-2526 (Poirot 96abd9e-t2524-census-confirmation2.md Observation 3): the filler count below is
# sized against `_covered`'s own +-line window rather than a hard-coded copy of it --
# `census._COVERED_WINDOW`, imported from `tools/geometry_site_census.py`, so a future widening of
# that window cannot silently desync these two probes from the boundary they exist to sit on.
_COVERED_WINDOW_FILLER_LINES = census._COVERED_WINDOW - 1


def _covered_start_uncovered_end_probe_lines() -> list[str]:
    # Marker ABOVE the group, `census._COVERED_WINDOW` lines from the group's own START (covered)
    # and therefore one more than that from its END (uncovered) -- this is the case Poirot's own
    # Observation 1 named: a covered start suppressing a genuine finding for an uncovered end.
    # `or` left this open; `and` closes it (not(True and False) == True -> emit).
    return (
        ["\t// T-2524 buffer\n"] * 4
        + [_marker_line("\t", "GS-9001")]
        + ["\t// T-2524 filler\n"] * _COVERED_WINDOW_FILLER_LINES
        + ['\tif (proj == "q_proj") {\n', "\t\treturn hidden_size;\n"]
        + ["\t// T-2524 buffer\n"] * 4
    )


def _uncovered_start_covered_end_probe_lines() -> list[str]:
    # Marker BELOW the group, `census._COVERED_WINDOW` lines from the group's own END (covered) and
    # therefore one more than that from its START (uncovered) -- the case the pre-T-2518
    # single-end check already caught (checking `group_start` alone), and the case T-2518's own
    # `or` silently stopped reporting. Guards against a re-regression to `or` as surely as the cell
    # above guards the Observation's own named case.
    return (
        ["\t// T-2524 buffer\n"] * 4
        + ['\tif (proj == "q_proj") {\n', "\t\treturn hidden_size;\n"]
        + ["\t// T-2524 filler\n"] * _COVERED_WINDOW_FILLER_LINES
        + [_marker_line("\t", "GS-9002")]
        + ["\t// T-2524 buffer\n"] * 4
    )


def _with_probe_registry_entry(gs_id: str):
    def _t(registry_json):
        import json
        data = json.loads(registry_json)
        assert not any(s["id"] == gs_id for s in data["sites"]), (
            f"{gs_id} is no longer free -- pick a different probe id"
        )
        data["sites"].append({
            "id": gs_id,
            "family": "QOW",
            "file": "include/superslm/matmul.h",
            "function": "T-2524 disposable probe (window group coverage, not a real geometry site)",
            "status": "confirmed-correct",
            "expected_marker_count": 1,
        })
        return json.dumps(data, indent=2)
    return _t


def test_part2_window_group_covered_start_uncovered_end_is_still_reported():
    injected = "".join(_covered_start_uncovered_end_probe_lines())

    def _t(text):
        assert "}  // namespace superslm" in text
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    registry_rel = os.path.join("tools", "geometry_site_registry.json")
    with _mutated(_MATMUL_H, _t), _mutated(registry_rel, _with_probe_registry_entry("GS-9001")):
        failures = census.run_census(_REPO_ROOT)
    assert any("UNMARKED QOW PATTERN HIT (window)" in f and "matmul.h" in f for f in failures), (
        "a genuinely unmarked two-line QOW split with a COVERED start and an UNCOVERED end must "
        "still be reported -- if this reverts to passing, S1's own `and` fix has regressed to `or`"
    )


def test_part2_window_group_uncovered_start_covered_end_is_still_reported():
    injected = "".join(_uncovered_start_covered_end_probe_lines())

    def _t(text):
        assert "}  // namespace superslm" in text
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)

    registry_rel = os.path.join("tools", "geometry_site_registry.json")
    with _mutated(_MATMUL_H, _t), _mutated(registry_rel, _with_probe_registry_entry("GS-9002")):
        failures = census.run_census(_REPO_ROOT)
    assert any("UNMARKED QOW PATTERN HIT (window)" in f and "matmul.h" in f for f in failures), (
        "a genuinely unmarked two-line QOW split with an UNCOVERED start and a COVERED end must "
        "still be reported -- this is the shape the pre-T-2518 single-end check already caught, "
        "and T-2518's own `or` silently stopped reporting"
    )


def test_known_limitation_for_f2_is_printed_on_a_passing_run():
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "tools", "geometry_site_census.py")],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "KNOWN LIMITATION" in proc.stdout
    assert "F2" in proc.stdout


# --- End-to-end against the real tree via the real CLI entry point, not just run_census(). ---

def test_main_end_to_end_via_subprocess_is_green_today():
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "tools", "geometry_site_census.py")],
        cwd=_REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "PASS" in proc.stdout


# =====================================================================================
# T-2475 fold-in (Claude/Poirot/6597903-t2472-ask5-tracka-confirmation.md Sec4-5,
# Significant 1 and 2 -- both in this same file, folded into this same round rather than
# left for a second builder):
#
# S1: Part 3's `required_token_scope_ends` was ruled self-detecting on the reasoning that
# the LOOKUP key's `marker_line` is recomputed fresh every run. True, and irrelevant -- the
# STORED key and its end line were both absolute line numbers frozen at authoring time, so
# an edit anywhere ELSE in the same file (the reviewer's own construction: one comment line
# at the very top) shifted every marker below it without shifting the registry to match,
# producing a false `MISSING ... ENTRY` on untouched, correctly-fixed code. Closed by
# re-keying `required_token_scope_end_offsets` on occurrence ORDINAL (this site's Nth
# marker in this file) and OFFSET (marker-to-scope-end distance) instead of absolute line
# numbers -- see `tools/geometry_site_census.py`'s own Part 3 section and the registry's
# header comment for the full account.
#
# S2: this file's own suite IS the pin Significant 2 asked for -- the census had no cell at
# all before it, and the reviewer's own mutation-proof (reverting the text-keyed exclusion
# remedy to skip-whole-line-on-match leaves every gate green) is reproduced by this file's own
# same-line population cells, below -- see the S2 red-check comment block further down this
# file for the exact, executed cell list and count.
# =====================================================================================

_GS14_FIXED_LINE = "\t\t\tplan.out_channels = (q_width != UINT32_MAX) ? q_width : hidden_size;\n"
_GS14_REVERTED_LINE = "\t\t\tplan.out_channels = hidden_size;\n"


# --- S1: the founding demonstration (GS-14 revert, marker untouched) must stay caught. ---

def test_part3_gs14_revert_with_marker_untouched_is_still_caught():
    def _t(text):
        assert _GS14_FIXED_LINE in text, "GS-14's own fixed line has moved -- update this fixture"
        return text.replace(_GS14_FIXED_LINE, _GS14_REVERTED_LINE, 1)
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "GS-14's fix reverted with its marker left in place must still fail Part 3"
    assert any("GS-14" in f and "REGRESSED SITE" in f for f in failures)


# --- S1: the finding this fold-in closes -- an edit far from any marker must NOT false-FAIL. ---

def test_part3_unrelated_edit_elsewhere_in_the_file_does_not_false_fail():
    # The reviewer's own construction: one comment line inserted at the very TOP of a file
    # carrying a Part-3-checked marker (GS-01, src/proof_manifest.cpp), nowhere near any
    # site's own code -- under the prior absolute-line keying this shifted every marker
    # below it and produced `MISSING required_token_scope_ends ENTRY` on untouched, correct
    # code (T-2472 Significant 1, reproduced live against `6597903` before this fix landed).
    def _t(text):
        return "// T-2475 S1 reproduction: one comment line at the top of the file\n" + text
    with _mutated(_PROOF_MANIFEST_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], (
        "an edit far from every marker must not desync the ordinal/offset keying -- if this "
        "fails, S1's own remedy has regressed"
    )


# =====================================================================================
# T-2524 (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 3, D-SLM5882): T-2518's own
# two new production changes -- GS-32/GS-33 flipped to `status: "fixed"` with `required_tokens`,
# and `.worktrees` added to `_SKIP_DIR_NAMES` -- were guarded by nothing. Reverting either left the
# census PASSing and the suite green, the same shape as T-2514's own Significant 2 one level out.
# Three cells close it: two drive the GS-32/GS-33 revert mutation the way `test_part3_gs14_
# revert_with_marker_untouched_is_still_caught` (above) already does for GS-14, and one plants a
# `.worktrees` subtree under a scratch root rather than depending on how many worktrees this
# checkout happens to carry.
# =====================================================================================

_GS32_FIXED_LINE = "\tbase_geom.q_width = static_cast<uint64_t>(bc.num_attention_heads) * bc.head_dim;\n"
_GS32_REVERTED_LINE = "\tbase_geom.q_width = bc.hidden_size;\n"


def test_part3_gs32_revert_to_hidden_size_is_caught():
    def _t(text):
        assert _GS32_FIXED_LINE in text, "GS-32's own fixed line has moved -- update this fixture"
        return text.replace(_GS32_FIXED_LINE, _GS32_REVERTED_LINE, 1)
    with _mutated(_SSLM_ABI_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "GS-32's fix reverted to bc.hidden_size with its marker left in place must still fail Part 3"
    assert any("GS-32" in f and "REGRESSED SITE" in f for f in failures)


_GS33_FIXED_LINE = "\tbase_geom.q_width = static_cast<uint64_t>(model->num_attention_heads) * model->head_dim;\n"
_GS33_REVERTED_LINE = "\tbase_geom.q_width = model->hidden_size;\n"


def test_part3_gs33_revert_to_hidden_size_is_caught():
    def _t(text):
        assert _GS33_FIXED_LINE in text, "GS-33's own fixed line has moved -- update this fixture"
        return text.replace(_GS33_FIXED_LINE, _GS33_REVERTED_LINE, 1)
    with _mutated(_GPU_1P0_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "GS-33's fix reverted to model->hidden_size with its marker left in place must still fail Part 3"
    assert any("GS-33" in f and "REGRESSED SITE" in f for f in failures)


def test_worktrees_subtree_under_a_scratch_repo_root_is_not_swept(tmp_path):
    # Planted rather than asserted against the real tree -- a scratch root's own count is stable
    # regardless of how many worktrees this checkout happens to carry (D-SLM5859 measured roughly
    # sixty under `D:/SuperSLM` itself; that number is not this cell's own subject).
    assert ".worktrees" in census._SKIP_DIR_NAMES
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.h").write_text("// kept\n", encoding="utf-8")
    nested = tmp_path / ".worktrees" / "some-branch" / "src"
    nested.mkdir(parents=True)
    (nested / "excluded.h").write_text("// must not be swept\n", encoding="utf-8")
    found = list(census._iter_source_files(str(tmp_path)))
    names = {os.path.basename(p) for p in found}
    assert "kept.h" in names
    assert "excluded.h" not in names, ".worktrees must be excluded from the sweep"


# --- S2 red-check: reverting `_part2_excise_excluded_text` to skip-whole-line-on-match (the
# TEXT-keyed exclusion mechanism the reviewer's own mutation-proof reverted to -- `Claude/Brunel/
# t2475-census-exclusion-class-2026-08-31.md` Sec10 records what was actually run; NOT a return
# to (path, line) keying) reddened EIGHT cells at T-2497 (T-2480 F2, Poirot f363c2a-t2479-census-
# class-confirmation.md Sec8 M2, D-SLM5654; re-verified against this file's own current, grown
# population by Poirot dcefab3-t2486-census-content-keying-confirmation.md Sec8 M2, D-SLM5721;
# T-2497 adds the two enumerator-prepend cells to the population, re-executed against the same
# reversion -- 8 failed, 32 passed, up from the prior 6 failed/34 passed).
#
# T-2518 re-execution (Poirot Critical 1's own fold-in, this docstring's own T-2518 fold-in
# bullet above): reddened exactly TWO cells -- 2 failed, 30 passed. Four of the prior eight are
# gone from the population, correctly: `test_defeats_the_prior_mechanism_same_line_append_
# adapter_marshal`, `test_appending_after_model_h_s_trailing_comment_fires_on_apparatus_unsound_
# grounds`, `test_prepended_content_on_an_excused_line_is_also_caught`, and `test_defeats_the_
# prior_mechanism_same_line_enumerator_prepend_model_h` are all deleted (this file's own T-2518
# fold-in bullet, above, names why; all four are keyed on `_ADAPTER_H`/`_ADAPTER_FRAGMENT` or
# `_MODEL_H`/`_MODEL_FRAGMENT`, both fixed away). The two that remained --
# `test_appending_after_proof_manifest_h_s_trailing_comment_fires_on_apparatus_unsound_grounds`
# and `test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h` -- were
# both still keyed on `proof_manifest.h`'s own exclusion, untouched by T-2509.
#
# T-2524 correction (Poirot e4bcaeb-t2518-census-fix-confirmation.md Significant 2, D-SLM5881):
# `test_excise_leaves_appended_new_content_in_the_remainder` and `test_excise_leaves_prepended_
# new_content_in_the_remainder` were deleted alongside the four above on the SAME stated reason --
# incorrectly: both call `_part2_excise_excluded_text(path, line)` on a synthetic string literal,
# never touch the tree, and need only a REGISTERED path, which `proof_manifest.h` still is.
# Restored, retargeted onto `_PROOF_H`/`_PROOF_FRAGMENT` (mechanism-cells section, above).
# Re-executed against the same reversion: the two cells restored just above and the two end-to-end
# survivors the T-2518 bullet above names
# (`test_appending_after_proof_manifest_h_s_trailing_comment_fires_on_apparatus_unsound_grounds`,
# `test_defeats_the_prior_mechanism_same_line_enumerator_prepend_proof_manifest_h`) are the FOUR
# cells that now redden -- named here rather than counted against the suite's own total, which
# changes every time this file gains a cell (T-2526, Poirot 96abd9e-t2524-census-confirmation2.md
# Minor 2, D-SLM5899: the "4 failed, 32 passed" this correction stated was already stale when
# written -- two more cells landed in this same round before the commit that carries it, giving
# 4 failed, 35 passed at the tip). The two that remained after T-2518 are the whole surviving
# END-TO-END population; these two restored cells are the excise mechanism's only UNIT-LEVEL pins
# -- they take their subject as an argument rather than reading it off the tree, so they do not
# depend on `proof_manifest.h`
# still carrying a live exclusion the way the other two do. The two cells T-2491 deletes (formerly
# the `..._insert_before_comment_*` pair) were never among either population -- executed and
# confirmed at the time: neither depended on the excision mechanism at all, so reverting it left
# both unaffected regardless of whether they existed.


# --- Observation carried into this round (Poirot 6597903-t2472-ask5-tracka-confirmation.md
# Sec10 O1, D-SLM5617): a `_PART2_EXCLUDED_TEXT` entry whose subject is deleted used to go
# silent (exit 0, no diagnostic) -- unsafe for nobody (a genuinely new site elsewhere in the
# same file is still caught), but the disclosure of a deliberate, documented residual quietly
# disappearing is itself a defect in what a PASS claims. Closed the same way as the class this
# round exists to close: excision now tracks which fragments actually matched something during
# the walk, and an entry that matched nothing anywhere is reported by name. ---

def test_dead_exclusion_is_reported_when_its_subject_is_deleted():
    # T-2518 (Poirot Critical 1): retargeted from _ADAPTER_H to _PROOF_H -- adapter_marshal.h no
    # longer carries a live exclusion (T-2509 fixed the residual it named; see the mechanism-cells
    # section's own header comment, near the top of this file), so proof_manifest.h is the only
    # file left where "delete the excused statement outright" is a real, non-vacuous mutation.
    def _t(text):
        assert (_PROOF_FRAGMENT + "\n") in text
        return text.replace(_PROOF_FRAGMENT + "\n", "", 1)
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "deleting an excused statement outright must be reported, not silent"
    assert any("DEAD EXCLUSION" in f and "proof_manifest.h" in f for f in failures)


def test_dead_exclusion_check_does_not_false_fire_on_the_untouched_tree():
    # Sanity companion to the cell above -- the baseline test already asserts this globally,
    # restated here so a DEAD EXCLUSION regression is locatable by name if it ever recurs.
    failures = census.run_census(_REPO_ROOT)
    assert not any("DEAD EXCLUSION" in f for f in failures)


# =====================================================================================
# T-2481 fold-in, item 2 (Poirot f363c2a-t2479-census-class-confirmation.md Significant 1,
# D-SLM5651/D-SLM5670): `_part2_excise_excluded_text` matches an excused fragment by exact-text
# PRESENCE, so a genuinely new, unregistered site written as a LITERAL COPY of an excused
# fragment, anywhere else in that same file, is excised for free -- exit 0, unaudited site in
# the tree. Closed by counting non-comment-line matches instead of recording mere presence and
# failing above one. This is the same class as item 1 below one level apart (a key that names
# LESS than the thing it stands for -- here, "this text exists somewhere in the file" instead of
# "this text is THE one statement it was written to excuse").
# =====================================================================================

def test_part2_ambiguous_exclusion_fires_on_a_literal_duplicate_copy():
    # T-2518 (Poirot Critical 1): retargeted from _ADAPTER_H to _PROOF_H -- adapter_marshal.h no
    # longer carries a live exclusion to duplicate (see the mechanism-cells section's own header
    # comment, near the top of this file), so proof_manifest.h is the only file left where a
    # literal duplicate copy of an excused fragment is a real, non-vacuous mutation.
    def _t(text):
        assert text.count(_PROOF_FRAGMENT) == 1
        # A new, unrelated `enum class` with an enumerator carrying the SAME literal text as the
        # excused fragment -- legal, compiling C++ (a scoped enum's own enumerators are scoped to
        # it, so the name collides with nothing), and a literal duplicate copy of the fragment on
        # a new non-comment line, exactly the shape T-2481's own construction intends.
        injected = (
            "\nenum class T2518DuplicateProofEnum : uint8_t {\n"
            "\t" + _PROOF_FRAGMENT + "\n"
            "};\n"
        )
        assert "}  // namespace superslm" in text
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)
    with _mutated(_PROOF_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a literal duplicate copy of an excused fragment must FAIL, not be excised for free"
    assert any("AMBIGUOUS EXCLUSION" in f and "proof_manifest.h" in f and "2 non-comment lines" in f
               for f in failures)


def test_part2_ambiguous_exclusion_does_not_false_fire_on_a_near_copy():
    # A near-copy (the excused fragment's own text disturbed, not reproduced literally) is NOT
    # a duplicate -- it is a genuinely different statement, and must be caught the ordinary way
    # (UNMARKED ... PATTERN HIT), not misreported as an ambiguous exclusion.
    def _t(text):
        near_copy = _ADAPTER_FRAGMENT.replace("proj ==", "proj  ==", 1)
        assert near_copy != _ADAPTER_FRAGMENT
        injected = (
            "\ninline uint64_t T2481NearCopyProbe(const std::string& proj, uint64_t hidden_size) {\n"
            "\t" + near_copy + "\n"
            "\treturn 0;\n"
            "}\n"
        )
        return text.replace("}  // namespace superslm", injected + "}  // namespace superslm", 1)
    with _mutated(_ADAPTER_H, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a near-copy (not a literal duplicate) must still FAIL as an unmarked site"
    assert any("UNMARKED QOW PATTERN HIT" in f and "adapter_marshal.h" in f for f in failures)
    assert not any("AMBIGUOUS EXCLUSION" in f for f in failures)


def test_part2_ambiguous_exclusion_does_not_false_fire_on_the_untouched_tree():
    failures = census.run_census(_REPO_ROOT)
    assert not any("AMBIGUOUS EXCLUSION" in f for f in failures)


# =====================================================================================
# T-2481 fold-in, item 1 (Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md F1,
# D-SLM5647; D-SLM5659): Part 3's occurrence-ORDINAL keying is defeated by REORDERING two of a
# multi-occurrence site's own existing occurrences within their shared file -- nothing added or
# removed, marker count unchanged, Part 1 blind to it -- which desyncs the ordinal-to-offset
# lookup and silently absorbs a genuine revert while citing an untouched line. Demonstrated on
# GS-10 (D-SLM5647); T-2481 found the SAME exposure, independently, on GS-11 (same two-occurrence
# shape, omitted from the prior commissioning's own exposure list) before building the remedy.
# `required_token_scopes` (registry) replaces ordinal with CONTENT-ADDRESSED anchors -- these
# cells prove the reorder-plus-revert construction that produced byte-identical output under the
# old scheme now correctly discriminates: PASS on the healthy reorder, FAIL citing the RELOCATED
# occurrence's own line on the reverted one.
# =====================================================================================

def _marker_line(indent: str, gs_id: str) -> str:
    """Builds a real marker's own literal line at RUNTIME rather than embedding it as a
    contiguous string constant in this module -- this test file is itself swept by Part 1's
    tree-wide marker scan (`_iter_source_files` with no `production_only` filter includes
    tests/**/*.py), so a literal "SSLM-GEOMETRY-SITE: GS-NN" substring written directly into
    this file's own source would be picked up as a genuine (orphan/duplicate) marker occurrence
    of THIS file, not just of the mutated production file a fixture targets."""
    return indent + "// SSLM-GEOMETRY" + "-SITE: " + gs_id + "\n"


def _swap_and_optionally_revert(text, chunk_a, chunk_b, revert_from=None, revert_to=None):
    """Swaps `chunk_a` and `chunk_b` bodily (each occupies the other's former physical position),
    optionally reverting one occurrence of a literal substring (`revert_from` -> `revert_to`,
    the FIRST one found) in the RESULT -- i.e. after the swap, wherever that text now physically
    sits. Mirrors Claude/Mendeleev/t2480-probes/t2480_q2_part3_edit_survivability.py's own
    reorder-plus-revert construction. `revert_from` need not be unique in the swapped text: two
    occurrences whose own local code is genuinely identical (GS-12's occ2/occ7) legitimately
    share the same revertible text, and reverting either one is an equally valid probe of that
    pair's own harmless-collision property."""
    assert text.count(chunk_a) == 1, "chunk_a not found uniquely -- update this fixture"
    assert text.count(chunk_b) == 1, "chunk_b not found uniquely -- update this fixture"
    swapped = text.replace(chunk_a, "\0T2481CHUNKB\0").replace(chunk_b, chunk_a)
    swapped = swapped.replace("\0T2481CHUNKB\0", chunk_b)
    if revert_from is None:
        return swapped
    assert swapped.count(revert_from) >= 1, "revert_from not found at all after the swap"
    return swapped.replace(revert_from, revert_to, 1)


_GS10_CHUNK_A = (
    _marker_line("\t", "GS-10") +
    "\t// T-2432 (Track A step 5, design §2.1 item 5/§6 Track A step 5, GS-10): "
    "q_proj.weight's\n"
    "\t// real shape is [q_width, hidden_size] -- byte extent q_width * hidden_size, not\n"
    "\t// hidden_size * hidden_size.\n"
    "\tL.off[2] = cur; cur += Align8U32(effective_q_width * hidden_size);  // q_weight (int8)\n"
)
_GS10_CHUNK_B = (
    _marker_line("\t\t", "GS-10") +
    "\t\tfor (uint32_t i = 0; i < QW * H; ++i) lw_bytes[base + layout.off[2] + i] = "
    "static_cast<uint8_t>(lw.q_weight[i]);\n"
)


def test_part3_reorder_gs10_healthy_swap_stays_clean():
    # The bare reorder, with neither occurrence's own fix touched -- content-addressing must not
    # false-fire just because the two occurrences trade physical positions.
    with _mutated(_SUPERSLM_GPU_CPP, lambda t: _swap_and_optionally_revert(t, _GS10_CHUNK_A, _GS10_CHUNK_B)):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], "reordering two healthy occurrences must not, by itself, redden the census"


def test_part3_reorder_gs10_plus_revert_is_caught_at_the_relocated_line():
    # Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md's own founding construction
    # (D-SLM5647): under the pre-T2481 ordinal keying this produced BYTE-IDENTICAL output to the
    # healthy-swap cell above, citing the UNTOUCHED occurrence's own line while the RELOCATED
    # occurrence's own genuine revert went silently absorbed.
    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS10_CHUNK_A, _GS10_CHUNK_B,
            revert_from="for (uint32_t i = 0; i < QW * H; ++i)",
            revert_to="for (uint32_t i = 0; i < H * H; ++i)",
        )
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a genuine revert of the relocated occurrence must FAIL, not be silently absorbed"
    # (T-2575): superslm_gpu.cpp's ShaderPath gained the shader-binary freshness guard and its
    # three helpers, shifting this occurrence from :723 to :838 in the mutated (chunk A/B
    # swapped) copy this test builds -- re-derived, as the T-2560 correction before it was, by
    # running the census against the real, current file through the same mutation this test
    # applies, never by adding a line-count offset to the previous number.
    # CORRECTED 2026-09-03 (T-2577): superslm_gpu.cpp gained the model_generation cache-key
    # parameter, its own top-level gate function and header comments (S1), the
    # GpuShaderBinaryStaleError type and catch clauses (S2), and the per-site saturation-count
    # offsets and packing (S3) -- all landing above this occurrence, shifting it from :838 to
    # :923 in the same mutated copy. Re-derived the same way: `python -c` driving
    # `_swap_and_optionally_revert`/`census.run_census` directly against the real, current file
    # through the identical mutation, never by offset.
    assert any("REGRESSED SITE" in f and "GS-10" in f and ":923" in f for f in failures), (
        "the finding must cite the RELOCATED occurrence's own (now-first) line, not the "
        "untouched occurrence -- a wrong-line citation is exactly what the prior ordinal "
        "keying produced"
    )


_GS11_CHUNK_A = (
    _marker_line("\t", "GS-11") +
    "\t// T-2432 (Track A step 5, GS-11): o_proj.weight's real shape is [hidden_size, q_width] --\n"
    "\t// byte extent hidden_size * q_width, not hidden_size * hidden_size. Unlike GS-07's own\n"
    "\t// per-output-channel fold count (o_fold_identity/mult/shift below, GS-09, confirmed\n"
    "\t// correct as hidden_size-sized and NOT touched by this step), the weight MATRIX itself\n"
    "\t// genuinely decouples on its input axis.\n"
    "\tL.off[25] = cur; cur += Align8U32(hidden_size * effective_q_width);  // o_weight (int8)\n"
)
_GS11_CHUNK_B = (
    _marker_line("\t\t", "GS-11") +
    "\t\tfor (uint32_t i = 0; i < H * QW; ++i) lw_bytes[base + layout.off[25] + i] = "
    "static_cast<uint8_t>(lw.o_weight[i]);\n"
)


def test_part3_reorder_gs11_healthy_swap_stays_clean():
    # GS-11 shares GS-10's own two-occurrence, one-file shape (offsets 6 and 1) and is
    # independently exposed to the same ordinal-desync defect (T-2481, D-SLM5659) -- the prior
    # commissioning's own exposure list ("GS-10, GS-12, GS-19") omitted it. Verified by
    # execution: this exact swap produced a false REGRESSED SITE on healthy, unreverted code
    # under the pre-T2481 ordinal keying (a safe-direction false alarm, not a silent pass, for
    # this specific construction -- contingent on what unrelated text fell inside the misapplied
    # window, per Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md F3).
    with _mutated(_SUPERSLM_GPU_CPP, lambda t: _swap_and_optionally_revert(t, _GS11_CHUNK_A, _GS11_CHUNK_B)):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], "reordering two healthy GS-11 occurrences must not redden the census"


def test_part3_reorder_gs11_plus_revert_is_caught_at_the_relocated_line():
    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS11_CHUNK_A, _GS11_CHUNK_B,
            revert_from="for (uint32_t i = 0; i < H * QW; ++i)",
            revert_to="for (uint32_t i = 0; i < H * H; ++i)",
        )
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a genuine revert of the relocated GS-11 occurrence must FAIL"
    # (T-2575): shifted from :750 to :865 in the mutated copy, by the same ShaderPath freshness
    # guard named at GS-10's own identical correction above, and re-derived the same way.
    # CORRECTED 2026-09-03 (T-2577): shifted from :865 to :950, by the same S1/S2/S3 additions
    # GS-10's own identical correction above names, re-derived the same way.
    assert any("REGRESSED SITE" in f and "GS-11" in f and ":950" in f for f in failures)


_GS12_OCC0 = (
    _marker_line("\t", "GS-12") +
    "\t// T-2432 (Track A step 3, design §2.1 item 5/§6 Track A step 3): "
    "q_codes/q_rot/ctx_codes\n"
    "\t// are Q's own output-width buffers -- sized `effective_q_width`, not `hidden_size`.\n"
    "\t// k_rot is written at `h * head_dim` for `h` up to `num_heads` (this loop's own "
    "query-head\n"
    "\t// index, not the KV-head index LandTokenKVRow uses to size its own K store) -- the same\n"
    "\t// query-head-count bound q_rot uses, so it needs the identical widening or an\n"
    "\t// out-of-bounds write follows the moment `num_heads` exceeds `hidden_size / head_dim`\n"
    "\t// (a mechanical consequence of widening `num_heads`, not a separate design decision --\n"
    "\t// the design's own §6 Track A step 3 text names q_codes/q_rot/ctx_wide/ctx_codes "
    "and does\n"
    "\t// not separately name k_rot because k_rot did not yet exist as a distinct local at the\n"
    "\t// text's own citation range; its indexing is identical to q_rot's).\n"
    "\tstd::vector<int8_t> normed(hidden_size), q_codes(effective_q_width), o_codes(hidden_size);\n"
)
_GS12_OCC1 = (
    _marker_line("\t\t", "GS-12") +
    "\t\t// T-2432 (Track A step 3): q_proj's INPUT width stays hidden_size (the normed\n"
    "\t\t// residual stream is unchanged by this ask); its OUTPUT width is effective_q_width.\n"
    "\t\tst = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, hidden_size, "
    "effective_q_width,\n"
)


def test_part3_reorder_gs12_two_differently_offset_occurrences_plus_revert_is_caught():
    # GS-12 has nine occurrences in one file with a mix of offsets (11, 3, 3, 4, 3, 4, 4, 3, 4)
    # -- occ0 (offset 11) and occ1 (offset 3) genuinely differ, unlike the occ2/occ7 pair below.
    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS12_OCC0, _GS12_OCC1,
            revert_from=("st = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, "
                          "hidden_size, effective_q_width,"),
            revert_to=("st = ProjectAndFunnel(normed.data(), normed_scale, lw.q_weight, "
                        "hidden_size, hidden_size,"),
        )
    with _mutated(_FORWARD_SITES_CPP_T2481, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a genuine revert of the relocated GS-12 occurrence must FAIL"
    # T-2551 (design §6 Track B steps 1/2): ApplyQkNormSite landed between LandTokenKVRow and
    # RunLayerLoopImpl, shifting this occurrence from :1681 to :1721 in the mutated (OCC0/OCC1
    # swapped) copy this test builds -- re-derived by running the census against the real,
    # current file (through the same mutation this test applies), not by applying a line-count
    # offset to the unmutated file's own line number.
    # (carried-scale delta, T-2560): ApplyQkNormSite's own header comment (forward_sites.h,
    # not this file) grew, but this file's own Q-per-head/K-relanding restructuring inside
    # RunLayerLoopImpl (above ApplyWeightScaleFold's own call site, above this occurrence)
    # added lines too -- shifting this occurrence from :1721 to :1732 in the same mutated
    # copy. Re-derived the same way, against the real, current, mutated file.
    # CORRECTED 2026-09-02 (T-2564, S3): ApplyQkNormSite's own signature gained an
    # `out_saturation_count` parameter, above this occurrence -- shifting it from :1732 to
    # :1734 in the same mutated copy. Re-derived by running the census against the real,
    # current, mutated file, not by applying +2 as an offset to the prior citation.
    # CORRECTED 2026-09-03 (T-2572, D-SLM6263, external review Significant 1): RopeApplySite's
    # own predicated-increment saturation counter (forward_sites.cpp, this ticket's own fix)
    # added lines above this occurrence -- shifting it from :1734 to :1745 in the same mutated
    # copy. Re-derived by running the census against the real, current, mutated file.
    # CORRECTED AGAIN 2026-09-03 (T-2577, D-SLM6274 S3/O1): the per-site saturation-count
    # parameters and their call-site wiring (LandTokenKVRow/ApplyQkNormSite/RopeApplySite, and
    # RunLayerLoopChunkBatched's own four new trailing parameters) added lines above this
    # occurrence -- shifting it from :1745 to :1775 in the same mutated copy. Re-derived the
    # same way, against the real, current, mutated file.
    assert any("REGRESSED SITE" in f and "GS-12" in f and ":1775" in f for f in failures)


_GS12_OCC2 = (
    _marker_line("\t\t\t", "GS-12") +
    "\t\t\t// T-2432 (Track A step 3): ctx_wide is the pre-fold attention-context accumulator,\n"
    "\t\t\t// one head_dim-wide slice per query head -- sized effective_q_width, not "
    "hidden_size.\n"
    "\t\t\tstd::vector<int64_t> ctx_wide(effective_q_width);\n"
)
_GS12_OCC7 = (
    _marker_line("\t\t\t\t", "GS-12") +
    "\t\t\t\t// T-2432 (Track A step 3): ctx_wide is the pre-fold attention-context accumulator,\n"
    "\t\t\t\t// one head_dim-wide slice per query head -- sized effective_q_width, not "
    "hidden_size.\n"
    "\t\t\t\tstd::vector<int64_t> ctx_wide(effective_q_width);\n"
)


def test_part3_reorder_gs12_identical_shared_anchor_pair_is_harmless():
    # occ2 and occ7 share literally identical local comment text (a copy-pasted explanation for
    # equivalent code in two sibling functions) and the SAME offset (3) -- the registry
    # registers one shared anchor record for both. Reordering these two specifically must stay
    # harmless in both directions: healthy swap clean, and a revert of EITHER relocated
    # occurrence still correctly caught (misassigning between two occurrences whose own
    # registered offset agrees changes nothing about what gets checked).
    with _mutated(_FORWARD_SITES_CPP_T2481, lambda t: _swap_and_optionally_revert(t, _GS12_OCC2, _GS12_OCC7)):
        failures = census.run_census(_REPO_ROOT)
    assert failures == [], "reordering the two identical-anchor GS-12 occurrences must not redden the census"

    def _t(text):
        return _swap_and_optionally_revert(
            text, _GS12_OCC2, _GS12_OCC7,
            revert_from="std::vector<int64_t> ctx_wide(effective_q_width);",
            revert_to="std::vector<int64_t> ctx_wide(hidden_size);",
        )
    with _mutated(_FORWARD_SITES_CPP_T2481, _t):
        failures = census.run_census(_REPO_ROOT)
    assert failures, "a revert of either identical-anchor occurrence must still be caught"
    assert any("REGRESSED SITE" in f and "GS-12" in f for f in failures)


def test_part3_insert_unregistered_occurrence_between_two_existing_is_caught_without_corrupting_others():
    # Claude/Mendeleev/t2480-census-recommissioning-2026-08-31.md Sec3.2's own "add an
    # occurrence between two existing ones" construction, F3: under the pre-T2481 ordinal
    # keying this cascaded an ordinal shift through every downstream occurrence, contingently
    # producing a mix of false REGRESSED SITEs on untouched code. Content-addressing has no
    # ordinal to cascade: only the new, genuinely unregistered marker is flagged, and every
    # pre-existing occurrence -- unaffected by an edit elsewhere in the file -- stays clean.
    def _t(text):
        anchor_line = ("\tstd::vector<int8_t> normed(hidden_size), q_codes(effective_q_width), "
                        "o_codes(hidden_size);\n")
        assert text.count(anchor_line) == 1
        injected = (_marker_line("\t", "GS-12") +
                    "\tstd::vector<int8_t> ctx_extra(effective_q_width);\n")
        return text.replace(anchor_line, anchor_line + injected, 1)
    with _mutated(_FORWARD_SITES_CPP_T2481, _t):
        failures = census.run_census(_REPO_ROOT)
    assert any("MARKER COUNT MISMATCH" in f and "GS-12" in f for f in failures)
    assert any("MISSING SCOPE ANCHOR" in f and "GS-12" in f for f in failures), (
        "the new, unregistered occurrence must be named by its own marker line"
    )
    assert not any("REGRESSED SITE" in f for f in failures), (
        "no PRE-EXISTING occurrence may be falsely flagged just because an unrelated new "
        "marker was inserted elsewhere in the file"
    )


def test_part3_missing_scopes_entry_is_reported_when_a_fixed_site_has_no_registry_record_at_all():
    # A `fixed` site with `required_tokens` but NO `required_token_scopes` entry for a file it
    # has a marker in -- the registry-side twin of `MISSING SCOPE ANCHOR` (a marker with no
    # matching record at all, rather than one that fails to match any of its site's records).
    #
    # T-2531 (Poirot 5e128ee-t2530-superslm-ci-green-review.md C-1, D-SLM5785/D-SLM5936): this
    # cell used to read/write tools/geometry_site_registry.json directly, hand-rolling its own
    # hardcoded `newline=""` rather than calling `_write_newline_for` (above) -- a second,
    # independent decision of the identical newline convention that function exists to be the
    # ONE place deciding. Two writers agreeing is indistinguishable from one, until one of them
    # reverts and the other does not: executed at this tip, reverting `_mutated`'s own
    # `newline=_write_newline` argument (D-SLM5785's exact regression -- the platform-default
    # write, not a blanket `newline=""`) while THIS cell still hand-rolled its own correct write
    # left the FULL module at 39 passed, 0 failed -- this cell's own unrelated, still-correct
    # write, running after the two window-group cells that corrupt the registry through
    # `_mutated` (test_part2_window_group_covered_start_uncovered_end_is_still_reported and its
    # sibling, above), silently restored the correct bytes as a side effect of its own teardown,
    # and the module-scoped byte-identity fixture
    # (`_mutated_targets_and_registry_are_byte_identical_after_the_module_runs`) never saw a
    # mismatch -- confirming D-SLM5785 is still live at this tip exactly as the review found.
    # Routed through `_mutated` here instead of hand-rolling the write: every real-tree write in
    # this module now goes through the one function that decides the convention, so the
    # instrument-level flaw (a second writer disagreeing with the first) is closed everywhere.
    #
    # T-2535 correction (Poirot 2945361-t2534-superslm-ci-green-confirmation2.md S-1): the
    # sentence this paragraph used to close on -- "reverting it corrupts every write alike and
    # the byte-identity fixture fires regardless of which cell happens to run last" -- and the
    # one after it -- "reverting `_mutated`'s own `newline=_write_newline` argument now fails
    # the module (byte mismatch reported on `tools\geometry_site_registry.json`) rather than
    # passing silently" -- are both false as unqualified statements, on the SAME cell C-1n's own
    # closure was already qualified against and the docstring above (`_write_newline_for`) now
    # states correctly: on a CRLF checkout (this platform) the revert fails the module with a
    # byte mismatch on the registry, exactly as both sentences here claim; on a fresh LF checkout
    # (`git clone` into WSL/Ubuntu, what every `ubuntu-latest` job that runs this module actually
    # checks out) the IDENTICAL revert leaves `40 passed, 1 failed` with NO byte mismatch and the
    # registry untouched -- the registry is already LF there, Linux's own platform-default write
    # is also LF, so the call-site regression is a no-op on that cell and nothing about it
    # corrupts anything, let alone every write alike. The one failure on LF is
    # `test_mutated_call_site_preserves_a_forced_crlf_file_regardless_of_checkout_platform`'s own
    # synthetic-bytes cell (below `_write_newline_for`), not this module's real-tree state --
    # confirmed by direct execution, both ways, on byte-preserved trees.
    def _t(registry_json):
        import json
        data = json.loads(registry_json)
        for site in data["sites"]:
            if site["id"] == "GS-01":
                del site["required_token_scopes"]
                break
        else:
            raise AssertionError("GS-01 not found in registry")
        return json.dumps(data, indent=2)
    registry_rel = os.path.join("tools", "geometry_site_registry.json")
    with _mutated(registry_rel, _t):
        failures = census.run_census(_REPO_ROOT)
    assert any("MISSING required_token_scopes ENTRY" in f and "GS-01" in f for f in failures)


# =====================================================================================
# T-2491 fold-in (Claude/Poirot/dcefab3-t2486-census-content-keying-confirmation.md O1,
# D-SLM5725): `AMBIGUOUS SCOPE ANCHOR` is the third of Part 3's three fail-closed diagnostics and
# the only one this file never exercised. It is a live guard, not a dead one -- the reviewer
# fired it by placing a second record's own anchor text inside a first record's own claimed
# window, with no reorder needed. Reproduced here on GS-10: occurrence 0's own registered record
# (`anchor="L.off[2] = cur; cur += Align8U32(", offset=4`) claims a 5-line window starting at its
# marker (its own 5-line chunk, marker through its own governed code); occurrence 1's own
# registered anchor (`"lw_bytes[base + layout.off[2] + i] = static_cast<uint8_t>(lw.q_weight[i]);
# "`) is appended onto occurrence 0's OWN marker line -- inside occurrence 0's own window and
# without adding or removing a line, so occurrence 0's own real anchor stays exactly where it
# was. Both records now match occurrence 0's marker, and they disagree on offset (4 vs 1).
# =====================================================================================

def test_part3_ambiguous_scope_anchor_fires_when_two_records_own_anchors_collide_in_one_window():
    def _t(text):
        assert _GS10_CHUNK_A in text, "GS-10 occurrence 0's own chunk has moved -- update this fixture"
        marker = _marker_line("\t", "GS-10")
        assert marker in _GS10_CHUNK_A
        collided_marker = marker.rstrip("\n") + (
            "  // collision probe: lw_bytes[base + layout.off[2] + i] = "
            "static_cast<uint8_t>(lw.q_weight[i]);\n"
        )
        collided = _GS10_CHUNK_A.replace(marker, collided_marker, 1)
        assert collided != _GS10_CHUNK_A
        assert collided.count("\n") == _GS10_CHUNK_A.count("\n"), (
            "must not change the chunk's own line count -- occurrence 0's real anchor line "
            "would shift out of its own registered window"
        )
        return text.replace(_GS10_CHUNK_A, collided, 1)
    with _mutated(_SUPERSLM_GPU_CPP, _t):
        failures = census.run_census(_REPO_ROOT)
    assert any("AMBIGUOUS SCOPE ANCHOR" in f and "GS-10" in f for f in failures), (
        "two registered records whose own anchors both match within one physical marker's own "
        "window, with disagreeing offsets, must be reported by name rather than guessed at"
    )
    assert not any("REGRESSED SITE" in f and "GS-10" in f for f in failures), (
        "an ambiguous match must not fall through and silently pass -- this second assertion is "
        "itself inert (Claude/Poirot/ba29de4-t2496-census-fixes-confirmation.md Sec8 M2, "
        "D-SLM5760: with the guard disabled the census reports nothing at exit 0, never a false "
        "REGRESSED SITE); the first assertion above carries the whole pin, and what the guard "
        "actually prevents is that silent absorption, not a required-token check against either "
        "record's own window"
    )
