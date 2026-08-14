"""Red suite for tests/ci/check_gpu_guard_status_parity.py.

Mechanism cells (constructed fixture text, mirroring test_check_no_forward_
leaf_calls.py's own convention of exercising the module against synthetic
source rather than mutating the real tree for every case) prove the module's
functions behave correctly in isolation. The end-to-end cells at the bottom
drive `run_all_checks`/`main` against the REAL repo tree, proving the real
population passes today and that the wiring (real paths, real anchors) is
not vacuous -- the same two-tier discipline check_no_forward_leaf_calls.py's
own test file already uses (a fixture-driven mechanism proof plus one
"against the real default glob" cell).

T-2055 (Claude/Poirot/db73b22-gpu-serial-port-final-confirmation-review.md,
P1): the three "mutation" cells below (`test_mutation_a_*`,
`test_mutation_b_*`, `test_mutation_c_*`) are the FIXTURE-LEVEL
reproductions of the review's own three real, on-hardware mutations --
proving this module's OWN mechanism catches (or, honestly, does not catch)
the same shapes the review demonstrated against the real 1500-line files.
The real-hardware mutation runs themselves (against actual `forward_sites.
cpp`/`superslm_gpu.cpp`/`gpu_layer_loop_guards.def` copies in disposable
scratch, each built and run once) are recorded in the build log, not
reproduced here as unit cells -- this file's job is the mechanism, not a
second on-hardware proof run.

T-2062 (Claude/Poirot/a3d44e7-gpu-serial-port-ship-confirmation-review.md,
S2): `test_mutation_d_*` are the fixture-level reproduction of the review's
OWN second falsifying mutation -- a new-status guard placed past the old
`CPU_GUARD_REGION_END_MARKER`, invisible to the marker-truncated extraction
this round replaced. `test_find_matching_close_brace_*` and `test_extract_
function_body_*` are direct mechanism cells for the new brace-based
extractor those mutations exercise indirectly.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_gpu_guard_status_parity as chk  # noqa: E402


# --- Small, synthetic stand-ins for the three real files' relevant shape. ---

_CPU_FIXTURE = """\
namespace superslm {
static SslmForwardStatus RunLayerLoopImpl(SequenceLayerState& seq, int x) {
	// a leading comment naming InvalidContextCap should NOT count as a return
	if (x == 0) return SslmForwardStatus::InvalidLayerBudget;
	if (x < 1) return SslmForwardStatus::InvalidContextCap;
	/* a block comment naming HeadDimGeometryMismatch should NOT count either */
	if (x == 2) {
		return SslmForwardStatus::HeadDimGeometryMismatch;
	}
	const int64_t position = seq.context_length;
	(void)position;
	return SslmForwardStatus::Ok;
}
}  // namespace superslm
"""

_GPU_FIXTURE = """\
namespace superslm_gpu {
superslm::SslmForwardStatus RunLayerLoopGpu(superslm::SequenceLayerState& seq, int x) {
	if (x == 0) return superslm::SslmForwardStatus::InvalidLayerBudget;  // LayerBudgetZero
	if (x < 1) return superslm::SslmForwardStatus::InvalidContextCap;  // ContextCapNonPositive
	if (x == 2) {
		return superslm::SslmForwardStatus::HeadDimGeometryMismatch;  // HeadDimGeometryMismatch
	}
	static_assert(static_cast<int>(superslm_gpu::GpuLayerLoopGuard::kCount) == 3, "x");
	if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;
	// T-2098 (S3): the fixture carries the real residency assignment, because the
	// before/after path-count derivation CUTS on it. Round 15's fixture had no such
	// statement at all, so `derive_lwuws_before_decision_count` silently fell back to
	// the whole body and every cell below measured a region the production path does
	// not have -- a fixture that lacks the boundary the check cuts on cannot exercise
	// the check, which is why the repaired derivation raises instead of falling back.
	const bool weights_resident = g_resident_weights.valid;
	g_last_weight_upload_was_skipped = weights_resident;
	// the real function ends via `return DecodeStickyTag(sticky_tag);` --
	// not a literal `return SslmForwardStatus::X;` -- so this fixture ends
	// the same unmatched way, never contributing a spurious status.
	return DecodeStickyTag(sticky_tag);
}
}  // namespace superslm_gpu
"""

_DEF_FIXTURE = """\
// fixture .def
#ifndef SSLM_GPU_LAYER_LOOP_GUARD
#error "define it"
#endif
SSLM_GPU_LAYER_LOOP_GUARD(LayerBudgetZero, InvalidLayerBudget, "forward_sites.cpp:4")
SSLM_GPU_LAYER_LOOP_GUARD(ContextCapNonPositive, InvalidContextCap, "forward_sites.cpp:5")
SSLM_GPU_LAYER_LOOP_GUARD(HeadDimGeometryMismatch, HeadDimGeometryMismatch, "forward_sites.cpp:7-9")
#undef SSLM_GPU_LAYER_LOOP_GUARD
"""


def _write_fixture_tree(tmpdir: str, *, cpu: str = _CPU_FIXTURE, gpu: str = _GPU_FIXTURE,
                         def_text: str = _DEF_FIXTURE) -> tuple[str, str, str]:
    forward_dir = os.path.join(tmpdir, "src", "forward")
    gpu_dir = os.path.join(tmpdir, "src", "gpu")
    inc_dir = os.path.join(tmpdir, "include", "superslm")
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(gpu_dir, exist_ok=True)
    os.makedirs(inc_dir, exist_ok=True)
    cpu_path = os.path.join(forward_dir, "forward_sites.cpp")
    gpu_path = os.path.join(gpu_dir, "superslm_gpu.cpp")
    def_path = os.path.join(inc_dir, "gpu_layer_loop_guards.def")
    with open(cpu_path, "w", encoding="utf-8") as f:
        f.write(cpu)
    with open(gpu_path, "w", encoding="utf-8") as f:
        f.write(gpu)
    with open(def_path, "w", encoding="utf-8") as f:
        f.write(def_text)
    return cpu_path, gpu_path, def_path


# --- strip_comments ---

def test_strip_comments_removes_line_and_block_comments():
    text = 'a // line comment with InvalidContextCap\nb /* block\ncomment */ c'
    out = chk.strip_comments(text)
    assert "InvalidContextCap" not in out
    assert "a " in out and "b " in out and " c" in out


# --- extract_region ---

def test_extract_region_finds_text_between_real_markers():
    region = chk.extract_region(_CPU_FIXTURE, chk.CPU_FUNC_SIGNATURE, chk.CPU_GUARD_REGION_END_MARKER,
                                 label="fixture")
    assert "InvalidLayerBudget" in region
    assert "InvalidContextCap" in region
    assert "HeadDimGeometryMismatch" in region
    # Ok is returned AFTER the end marker -- must not be swept into the guard region.
    assert "Ok" not in region


def test_extract_region_raises_when_start_marker_absent():
    try:
        chk.extract_region("no signature here", chk.CPU_FUNC_SIGNATURE, chk.CPU_GUARD_REGION_END_MARKER,
                            label="fixture")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "start marker" in str(e)


def test_extract_region_raises_when_end_marker_absent():
    try:
        chk.extract_region(chk.CPU_FUNC_SIGNATURE + "\nno end marker here", chk.CPU_FUNC_SIGNATURE,
                            chk.CPU_GUARD_REGION_END_MARKER, label="fixture")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "end marker" in str(e)


# --- extract_status_set ---

def test_extract_status_set_ignores_comment_only_mentions():
    region = chk.extract_region(_CPU_FIXTURE, chk.CPU_FUNC_SIGNATURE, chk.CPU_GUARD_REGION_END_MARKER,
                                 label="fixture")
    s = chk.extract_status_set(region)
    # The fixture's leading `//` comment names InvalidContextCap and its block
    # comment names HeadDimGeometryMismatch -- both are ALSO real returns, so
    # this alone would not distinguish comment-inflation from a genuine
    # return; the point is proven by test_extract_status_set_comment_only_
    # mention_does_not_inflate_set below, against a name that is NEVER
    # returned.
    assert s == {"InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"}


# --- find_matching_close_brace / extract_function_body (T-2062, S2) ---

def test_find_matching_close_brace_skips_braces_in_comments_and_strings():
    text = '{ // a { comment\n /* another { */ "a } string" \'{\' x; }'
    close = chk.find_matching_close_brace(text, 0)
    assert close == len(text) - 1


def test_find_matching_close_brace_handles_nested_braces():
    text = "{ a { b { c } d } e }"
    close = chk.find_matching_close_brace(text, 0)
    assert close == len(text) - 1


def test_find_matching_close_brace_raises_when_unterminated():
    try:
        chk.find_matching_close_brace("{ a { b }", 0)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no matching" in str(e)


def test_find_matching_close_brace_raises_when_not_a_brace():
    try:
        chk.find_matching_close_brace("abc", 0)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "is not '{'" in str(e)


def test_extract_function_body_finds_the_real_closing_brace_past_the_old_marker():
    body = chk.extract_function_body(_CPU_FIXTURE, chk.CPU_FUNC_SIGNATURE, label="fixture")
    # Unlike extract_region (marker-truncated), the whole body -- including
    # the trailing Ok past the old marker -- is present.
    assert "InvalidLayerBudget" in body
    assert "Ok" in body
    # The enclosing `}  // namespace superslm` line is NOT included -- the
    # function's own closing brace, not a later one, ends the extraction.
    assert "namespace superslm" not in body
    assert body.rstrip().endswith("SslmForwardStatus::Ok;")


def test_extract_function_body_raises_when_signature_absent():
    try:
        chk.extract_function_body("nothing here", chk.CPU_FUNC_SIGNATURE, label="fixture")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "signature not found" in str(e)


# --- cpu_guard_status_set ---

def test_cpu_guard_status_set_subtracts_the_named_below_guard_statuses():
    # Using the FIXTURE's own reduced subtraction set (only "Ok" applies to
    # this fixture; the other five don't appear in it) rather than the real
    # production six, to keep this cell independent of the production list's
    # own membership.
    s = chk.cpu_guard_status_set(_CPU_FIXTURE, below_guard_arithmetic_statuses=frozenset({"Ok"}))
    assert s == {"InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"}


def test_cpu_guard_status_set_with_default_subtraction_set_matches_extract_region_on_the_fixture():
    # The fixture's only below-guard status is "Ok", which IS in the real
    # production CPU_BELOW_GUARD_ARITHMETIC_STATUSES default -- so the new,
    # whole-function extraction with the default subtraction set agrees with
    # the OLD marker-truncated extraction on this fixture, which has nothing
    # past the marker except Ok.
    old_way = chk.extract_status_set(
        chk.extract_region(_CPU_FIXTURE, chk.CPU_FUNC_SIGNATURE, chk.CPU_GUARD_REGION_END_MARKER, label="fixture")
    )
    new_way = chk.cpu_guard_status_set(_CPU_FIXTURE)
    assert new_way == old_way == {"InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"}


def test_extract_status_set_comment_only_mention_does_not_inflate_set():
    region = "// KvCapacityExhausted is mentioned here only\nreturn SslmForwardStatus::InvalidLayerBudget;"
    s = chk.extract_status_set(region)
    assert s == {"InvalidLayerBudget"}
    assert "KvCapacityExhausted" not in s


# --- parse_def_rows / def_status_set ---

def test_parse_def_rows_extracts_all_three_fixture_rows_in_order():
    rows = chk.parse_def_rows(_DEF_FIXTURE)
    assert [r.enum_name for r in rows] == ["LayerBudgetZero", "ContextCapNonPositive", "HeadDimGeometryMismatch"]
    assert [r.status_name for r in rows] == ["InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"]
    assert rows[2].citation == "forward_sites.cpp:7-9"


def test_def_status_set_matches_fixture():
    rows = chk.parse_def_rows(_DEF_FIXTURE)
    assert chk.def_status_set(rows) == {"InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"}


# --- check_citation ---

def test_check_citation_passes_for_a_real_matching_single_line():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, _, _ = _write_fixture_tree(tmp)
        row = chk.DefRow("LayerBudgetZero", "InvalidLayerBudget", "forward_sites.cpp:4")
        assert chk.check_citation(row, repo_root=tmp) is None


def test_check_citation_passes_for_a_real_matching_multi_line_range():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        row = chk.DefRow("HeadDimGeometryMismatch", "HeadDimGeometryMismatch", "forward_sites.cpp:7-9")
        assert chk.check_citation(row, repo_root=tmp) is None


def test_check_citation_fails_when_cited_line_holds_a_different_status():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        row = chk.DefRow("LayerBudgetZero", "InvalidContextCap", "forward_sites.cpp:4")
        failure = chk.check_citation(row, repo_root=tmp)
        assert failure is not None
        assert "does not hold a rejecting" in failure


def test_check_citation_fails_when_cited_line_is_past_end_of_file():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        row = chk.DefRow("Ghost", "InvalidLayerBudget", "forward_sites.cpp:9999")
        failure = chk.check_citation(row, repo_root=tmp)
        assert failure is not None
        assert "exceeds" in failure


def test_check_citation_fails_when_cited_file_is_missing():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        row = chk.DefRow("Ghost", "InvalidLayerBudget", "nonexistent_file.cpp:1")
        failure = chk.check_citation(row, repo_root=tmp)
        assert failure is not None
        assert "not found" in failure


def test_check_citation_fails_for_a_malformed_citation():
    row = chk.DefRow("Ghost", "InvalidLayerBudget", "not a citation at all")
    failure = chk.check_citation(row, repo_root=os.getcwd())
    assert failure is not None
    assert "does not start with a file:line prefix" in failure


# --- run_all_checks: full mechanism, fixture population equal at 3 ---

def test_run_all_checks_clean_on_the_matching_fixture_tree():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures == []


# --- Mutation A: a tenth (here, fourth) guard added to BOTH ladders with no
# .def row -- must redden IFF the added status is new to the set; must NOT
# redden (the review's own "honest residual") if it reuses an existing one.

def test_mutation_a_new_status_with_no_def_row_reddens():
    cpu = _CPU_FIXTURE.replace(
        "if (x == 2) {",
        'if (x == 3) return SslmForwardStatus::KvCapacityExhausted;\n\tif (x == 2) {',
    )
    gpu = _GPU_FIXTURE.replace(
        "if (x == 2) {",
        'if (x == 3) return superslm::SslmForwardStatus::KvCapacityExhausted;  // Mutated\n\tif (x == 2) {',
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, cpu=cpu, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures, "a tenth guard with a status absent from the .def must fail the check"
        assert any("disagree" in f for f in failures)


def test_mutation_a_reused_status_with_no_def_row_does_not_redden_the_honest_residual():
    # The added guard reuses InvalidLayerBudget, already a member of all
    # three sets -- the set-equality check cannot see a fourth guard was
    # added at all. This is the residual this module's own docstring names
    # explicitly, proven here rather than merely asserted.
    cpu = _CPU_FIXTURE.replace(
        "if (x == 2) {",
        'if (x == 3) return SslmForwardStatus::InvalidLayerBudget;\n\tif (x == 2) {',
    )
    gpu = _GPU_FIXTURE.replace(
        "if (x == 2) {",
        'if (x == 3) return superslm::SslmForwardStatus::InvalidLayerBudget;  // Mutated\n\tif (x == 2) {',
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, cpu=cpu, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures == [], "reusing an existing status is the documented residual -- must stay green"


# --- Mutation B: one row added to the .def with no matching ladder guard on
# either side -- must redden (both CPU and GPU sets now disagree with .def).

def test_mutation_b_def_row_with_no_matching_ladder_guard_reddens():
    def_text = _DEF_FIXTURE.replace(
        '#undef SSLM_GPU_LAYER_LOOP_GUARD',
        'SSLM_GPU_LAYER_LOOP_GUARD(Phantom, KvCapacityExhausted, "forward_sites.cpp:1")\n'
        '#undef SSLM_GPU_LAYER_LOOP_GUARD',
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, def_text=def_text)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures, "a .def row naming a status neither ladder returns must fail the check"


# --- Mutation C: a guard removed from the GPU ladder only -- must redden
# (GPU set now disagrees with both CPU and .def).

def test_mutation_c_guard_removed_from_gpu_ladder_only_reddens():
    gpu = _GPU_FIXTURE.replace(
        'if (x < 1) return superslm::SslmForwardStatus::InvalidContextCap;  // ContextCapNonPositive\n', ''
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures, "a guard present in CPU/.def but removed from the GPU ladder must fail the check"
        assert any("disagree" in f for f in failures)


# --- Mutation D (T-2062, S2): a new-status guard placed PAST the old marker
# -- the review's own second falsifying mutation, invisible to the
# marker-truncated extraction, now caught by the whole-function extraction.

def test_mutation_d_new_status_guard_placed_after_the_old_end_marker_reddens():
    # CPU-only edit, exactly mirroring the real-file mutation: the GPU
    # ladder and the .def are untouched, so a genuinely new CPU-side status
    # appearing anywhere in RunLayerLoopImpl's own body -- guard region or
    # not -- must now surface as a real disagreement.
    cpu = _CPU_FIXTURE.replace(
        "const int64_t position = seq.context_length;",
        "const int64_t position = seq.context_length;\n"
        "\tif (x == 99) return SslmForwardStatus::KvCapacityExhausted;  // new, past the old marker",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, cpu=cpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures, "a new-status guard placed after the old end marker must now be caught (S2)"
        assert any("disagree" in f for f in failures)
        assert any("KvCapacityExhausted" in f for f in failures)


def test_mutation_d_variant_reused_status_after_marker_stays_green_the_widened_residual():
    # The widened honest residual: a status added past the marker that
    # reuses one already in the derived set is indistinguishable from "no
    # guard was added at all" -- exactly the same shape as mutation A's own
    # documented residual, now proven true past the old marker too.
    cpu = _CPU_FIXTURE.replace(
        "const int64_t position = seq.context_length;",
        "const int64_t position = seq.context_length;\n"
        "\tif (x == 99) return SslmForwardStatus::InvalidLayerBudget;  // reused status, past the old marker",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, cpu=cpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures == [], "reusing an existing status past the marker is the (widened) documented residual"


# --- Mutation E (T-2069, S3): the symmetric twin of mutation D, aimed at the
# GPU side -- a new-status guard placed PAST GPU_GUARD_REGION_END_MARKER
# (the old static_assert cut), the review's own falsifying mutation, run
# both directions per the coordinator's own instruction: after the marker
# must redden; before the marker (already caught pre-fix) must stay
# reddening too, proving the fix changes nothing about the case that
# already worked.

def test_mutation_e_new_status_guard_placed_after_the_gpu_marker_reddens():
    # GPU-only edit, exactly mirroring the real-file mutation: CPU and the
    # .def are untouched, so a genuinely new GPU-side status appearing
    # anywhere in RunLayerLoopGpu's own body -- ladder or not -- must now
    # surface as a real disagreement. Placed after the fixture's own
    # static_assert line (the GPU marker), inside the device-capability
    # region the real file already carries a rejection in.
    gpu = _GPU_FIXTURE.replace(
        "if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;",
        "if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;\n"
        "\tif (x == 99) return superslm::SslmForwardStatus::KvCapacityExhausted;  // new, past the GPU marker",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures, "a new-status GPU guard placed after the old marker must now be caught (S3)"
        assert any("disagree" in f for f in failures)
        assert any("KvCapacityExhausted" in f for f in failures)


def test_mutation_e_control_new_status_guard_placed_before_the_gpu_marker_reddens_both_before_and_after_the_fix():
    # The review's own control run: the IDENTICAL guard placed BEFORE the
    # marker (inside the ladder proper) was already caught by the
    # pre-T-2069 marker-truncated extraction, and must stay caught after
    # the fix too -- proving the fix is additive (closes the gap below the
    # marker) rather than a behavior change to the case that already
    # worked. This is the one-line difference (before vs. after the
    # marker) the review's own report names as "the only difference
    # between the two runs."
    gpu = _GPU_FIXTURE.replace(
        "if (x == 2) {",
        'if (x == 3) return superslm::SslmForwardStatus::KvCapacityExhausted;  // new, before the GPU marker\n'
        '\tif (x == 2) {',
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures, "the identical guard placed before the marker must redden, same as before this fix"
        assert any("disagree" in f for f in failures)
        assert any("KvCapacityExhausted" in f for f in failures)


def test_mutation_e_variant_reused_status_after_gpu_marker_stays_green_the_widened_residual():
    # The widened honest residual, GPU side: a status added past the GPU
    # marker that reuses one already in the derived set (including the one
    # subtracted GPU_BELOW_LADDER_STATUSES name) is indistinguishable from
    # "no rejection was added at all."
    gpu = _GPU_FIXTURE.replace(
        "if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;",
        "if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;\n"
        "\tif (x == 99) return superslm::SslmForwardStatus::InvalidLayerBudget;  // reused, past the GPU marker",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures == [], "reusing an existing status past the GPU marker is the (widened) documented residual"


def test_mutation_e_variant_reused_below_ladder_status_after_gpu_marker_stays_green():
    # Reusing the ONE subtracted GPU_BELOW_LADDER_STATUSES name itself
    # (KvPrecisionUnsupported, already returned once by the fixture's own
    # `!dev.available` line) must also stay green -- the widened residual
    # explicitly includes the subtracted names, not only the guard names.
    gpu = _GPU_FIXTURE.replace(
        "if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;",
        "if (!dev.available) return superslm::SslmForwardStatus::KvPrecisionUnsupported;\n"
        "\tif (x == 99) return superslm::SslmForwardStatus::KvPrecisionUnsupported;  // reused below-ladder status",
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp,
                                       gpu_port_h_path=None, check_decode_sticky_tag=False,
                                       build_bat_path=None,
                                       check_marked_citation_scan=False,
                                       run_symbol_integrity_scan=False,
                                       check_lwuws_path_count=False)
        assert failures == [], "reusing the subtracted below-ladder status itself is also the documented residual"


# --- gpu_ladder_status_set: direct mechanism cells (T-2069, S3) ---

def test_gpu_ladder_status_set_subtracts_the_named_below_ladder_status():
    s = chk.gpu_ladder_status_set(_GPU_FIXTURE)
    assert s == {"InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"}


def test_gpu_ladder_status_set_default_matches_marker_truncated_extraction_on_the_unmutated_fixture():
    old_way = chk.extract_status_set(
        chk.extract_region(_GPU_FIXTURE, chk.GPU_FUNC_SIGNATURE, chk.GPU_GUARD_REGION_END_MARKER, label="fixture")
    )
    new_way = chk.gpu_ladder_status_set(_GPU_FIXTURE)
    assert new_way == old_way == {"InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch"}


def test_real_tree_gpu_full_body_raw_set_before_subtraction_matches_the_review_own_enumeration():
    # The review's own S3 finding enumerated, at source, the one status
    # legitimately returned below the GPU marker (KvPrecisionUnsupported,
    # twice, both device-capability rejections). Confirmed against the real
    # file: the RAW (pre-subtraction) set over the whole function body is
    # exactly the named nine guards plus that one.
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        gpu_text = f.read()
    body = chk.extract_function_body(gpu_text, chk.GPU_FUNC_SIGNATURE, label="superslm_gpu.cpp")
    raw = chk.extract_status_set(body)
    assert raw - chk.GPU_BELOW_LADDER_STATUSES == {
        "InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch",
        "KvHeadGeometryMismatch", "WorkspaceTooSmall", "InvalidHiddenCodes",
        "SequenceAlreadyComplete", "PositionOverCap", "KvCapacityExhausted",
    }
    assert raw & chk.GPU_BELOW_LADDER_STATUSES == chk.GPU_BELOW_LADDER_STATUSES, (
        "the one named below-ladder status must actually appear at source -- an unused name in the "
        "subtraction set would silently widen the residual for nothing"
    )
    # O19 (same casebook): GpuAllocationFailed/GpuDeviceRemoved are returned
    # through a ternary, not a literal `return SslmForwardStatus::X;` -- they
    # must NOT appear in the raw regex-derived set at all, on the real file.
    assert "GpuAllocationFailed" not in raw
    assert "GpuDeviceRemoved" not in raw


# --- decode_sticky_tag_status_set / check_decode_sticky_tag_range
# (T-2083, O35: gpu_port.h's own DecodeStickyTag range citation covers two
# ENDPOINTS, not the function's own content -- these pin the content directly) ---

_DECODE_STICKY_TAG_FIXTURE = """\
namespace superslm_gpu {
superslm::SslmForwardStatus DecodeStickyTag(int64_t tag) {
	using S = superslm::SslmForwardStatus;
	switch (tag) {
		case 0: return S::Ok;
		case 1: return S::ChainInputOutOfDomain;
		case 2: return S::RopeTableTensorMissing;
		default: return S::KvPrecisionUnsupported;
	}
}
}  // namespace superslm_gpu
"""


def test_decode_sticky_tag_status_set_extracts_the_real_body():
    names = chk.decode_sticky_tag_status_set(_DECODE_STICKY_TAG_FIXTURE)
    assert names == {"Ok", "ChainInputOutOfDomain", "RopeTableTensorMissing", "KvPrecisionUnsupported"}


def test_decode_sticky_tag_status_set_ignores_comment_only_mentions():
    fixture = _DECODE_STICKY_TAG_FIXTURE.replace(
        "using S = superslm::SslmForwardStatus;",
        "using S = superslm::SslmForwardStatus;  // S::SiluCompositionScaleOutOfDomain is NOT returned here",
    )
    names = chk.decode_sticky_tag_status_set(fixture)
    assert "SiluCompositionScaleOutOfDomain" not in names
    assert names == {"Ok", "ChainInputOutOfDomain", "RopeTableTensorMissing", "KvPrecisionUnsupported"}


def test_decode_sticky_tag_status_set_reddens_when_an_interior_case_is_deleted():
    # T-2083 (O35's own falsifying case, reproduced at the fixture level):
    # deleting one interior case changes what THIS extraction counts,
    # regardless of the file's own total line count -- unlike the two-
    # endpoint range citation, which the review's own real-file mutation
    # (case 13 deleted, line count preserved) left green.
    mutated = _DECODE_STICKY_TAG_FIXTURE.replace("\t\tcase 2: return S::RopeTableTensorMissing;\n", "")
    before = chk.decode_sticky_tag_status_set(_DECODE_STICKY_TAG_FIXTURE)
    after = chk.decode_sticky_tag_status_set(mutated)
    assert after == before - {"RopeTableTensorMissing"}
    assert len(after) == len(before) - 1


def test_real_tree_decode_sticky_tag_range_matches_the_citation_today():
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        gpu_text = f.read()
    failures = chk.check_decode_sticky_tag_range(gpu_text)
    assert failures == [], f"DecodeStickyTag's own real body should match gpu_port.h's citation today: {failures}"


def test_real_tree_decode_sticky_tag_status_set_is_the_named_fourteen():
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        gpu_text = f.read()
    names = chk.decode_sticky_tag_status_set(gpu_text)
    assert names == {
        "Ok", "CarriedScaleMantissaOutOfDomain", "ChainInputOutOfDomain",
        "RoundingDivideByPotExponentOutOfDomain", "BiasReconcileProductOutOfDomain",
        "RopeTableTensorMissing", "RopeTableExtentExceeded", "PositionOverCap",
        "SoftmaxRowWidthOutOfDomain", "IExpScaleDerivationOutOfDomain",
        "SoftmaxKernelRefusedAfterGateAccepted", "ResidualReconciliationMagnitudeOutOfDomain",
        "SiluCompositionScaleOutOfDomain", "KvPrecisionUnsupported",
    }


# --- find_marked_citations / check_marked_citation / check_marked_citations
# (T-2094, S1's own structural remedy: line-number citations removed entirely, replaced by
# `` `Name` (file.ext) `` existence-checked marked citations, whole-tree, no chunk boundary) ---

def test_find_marked_citations_extracts_every_occurrence():
    text = "see `InvalidLayerBudget` (superslm_gpu.cpp) and `DecodeStickyTag` (superslm_gpu.cpp) too"
    found = chk.find_marked_citations(text, "//")
    assert found == [("InvalidLayerBudget", "superslm_gpu.cpp"), ("DecodeStickyTag", "superslm_gpu.cpp")]


def test_find_marked_citations_joins_a_line_wrap_between_name_and_file():
    text = "see `InvalidLayerBudget`\n// (superslm_gpu.cpp) for real"
    found = chk.find_marked_citations(text, "//")
    assert found == [("InvalidLayerBudget", "superslm_gpu.cpp")]


def test_find_marked_citations_ignores_ordinary_english_parentheticals():
    # "(below)"/"(inclusive)" and a hex literal are not filenames -- the required real extension
    # (.cpp/.h/.py/.def/.bat) is what keeps this pattern from firing on prose that merely happens
    # to sit after a backtick-quoted word.
    text = "`gpu_ladder_status_set` (below) and `STATUS_ACCESS_VIOLATION` (0xC0000005)"
    assert chk.find_marked_citations(text, "//") == []


def test_check_marked_citation_passes_for_a_real_identifier():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp)
        assert chk.check_marked_citation("InvalidLayerBudget", "superslm_gpu.cpp", repo_root=tmp) is None


def test_check_marked_citation_fails_when_name_does_not_resolve():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        failure = chk.check_marked_citation("NeverExisted", "superslm_gpu.cpp", repo_root=tmp)
        assert failure is not None
        assert "does not resolve" in failure


def test_check_marked_citation_fails_for_an_unrecognized_target_file():
    failure = chk.check_marked_citation("Anything", "not_a_real_file.cpp", repo_root=os.getcwd())
    assert failure is not None
    assert "not a recognized citation target" in failure


def test_check_marked_citation_passes_for_a_member_access_anchor():
    # `dev.available` is not identifier-shaped (contains a dot) -- resolved as a literal substring
    # of the comment-stripped code instead of via the identifier-token universe.
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp)
        assert chk.check_marked_citation("dev.available", "superslm_gpu.cpp", repo_root=tmp) is None


def test_check_marked_citation_passes_for_an_anchor_comment_marker():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(
            tmp, gpu=_GPU_FIXTURE.replace(
                "if (!dev.available)",
                "// ANCHOR:my_test_anchor\n\tif (!dev.available)",
            ),
        )
        assert chk.check_marked_citation("my_test_anchor", "superslm_gpu.cpp", repo_root=tmp) is None


def test_check_marked_citations_reddens_on_one_stale_citation_among_several():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp)
        gph_dir = os.path.join(tmp, "include", "superslm")
        os.makedirs(gph_dir, exist_ok=True)
        gph_path = os.path.join(gph_dir, "gpu_port.h")
        with open(gph_path, "w", encoding="utf-8") as f:
            f.write(
                "// see `InvalidLayerBudget` (superslm_gpu.cpp), real\n"
                "// and `TotallyMadeUp` (superslm_gpu.cpp), not real\n"
            )
        failures = chk.check_marked_citations(
            scanned_files=(("include/superslm/gpu_port.h", "//"),), repo_root=tmp,
        )
        assert len(failures) == 1
        assert "TotallyMadeUp" in failures[0]


def test_check_marked_citations_reddens_regardless_of_WHERE_in_the_file_the_citation_sits():
    # T-2094's own closure of S1's class: no chunk boundary exists any more -- a marked citation
    # placed ANYWHERE in a scanned file's own prose is checked, unlike SELF_CITATIONS' own four
    # hand-picked chunks (T-2091), which the exact M2 defect could hide outside of.
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp)
        gph_dir = os.path.join(tmp, "include", "superslm")
        os.makedirs(gph_dir, exist_ok=True)
        gph_path = os.path.join(gph_dir, "gpu_port.h")
        # A long file with the stale citation at the very END, nowhere near any conventional
        # "paragraph" a chunk-based scan might have bounded.
        padding = "\n".join(f"// padding line {i}, nothing relevant here" for i in range(200))
        with open(gph_path, "w", encoding="utf-8") as f:
            f.write(padding + "\n// finally, `NeverExisted` (superslm_gpu.cpp) cited at the very end\n")
        failures = chk.check_marked_citations(
            scanned_files=(("include/superslm/gpu_port.h", "//"),), repo_root=tmp,
        )
        assert failures, "a stale citation at the end of a long file, no chunk involved, must redden"
        assert "NeverExisted" in failures[0]


def test_real_tree_check_marked_citations_passes_today():
    assert chk.check_marked_citations() == []


# --- End-to-end against the real tree: not vacuous, and green today. ---

def test_main_end_to_end_against_the_real_tree_is_green_today():
    failures = chk.run_all_checks()
    assert failures == [], f"real tree should be clean today: {failures}"


def test_real_tree_status_set_is_the_named_nine():
    with open(chk.FORWARD_SITES_CPP, "r", encoding="utf-8") as f:
        cpu_text = f.read()
    cpu_set = chk.cpu_guard_status_set(cpu_text)
    assert cpu_set == {
        "InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch",
        "KvHeadGeometryMismatch", "WorkspaceTooSmall", "InvalidHiddenCodes",
        "SequenceAlreadyComplete", "PositionOverCap", "KvCapacityExhausted",
    }


def test_real_tree_cpu_full_body_raw_set_before_subtraction_matches_the_review_own_enumeration():
    # The review's own S2 finding enumerated, at source, the six statuses
    # legitimately returned below the old marker (five per-site arithmetic
    # rejections plus Ok). Confirmed here against the real file: the RAW
    # (pre-subtraction) set over the whole function body is exactly the
    # named nine guards plus those six.
    with open(chk.FORWARD_SITES_CPP, "r", encoding="utf-8") as f:
        cpu_text = f.read()
    body = chk.extract_function_body(cpu_text, chk.CPU_FUNC_SIGNATURE, label="forward_sites.cpp")
    raw = chk.extract_status_set(body)
    assert raw - chk.CPU_BELOW_GUARD_ARITHMETIC_STATUSES == {
        "InvalidLayerBudget", "InvalidContextCap", "HeadDimGeometryMismatch",
        "KvHeadGeometryMismatch", "WorkspaceTooSmall", "InvalidHiddenCodes",
        "SequenceAlreadyComplete", "PositionOverCap", "KvCapacityExhausted",
    }
    assert raw & chk.CPU_BELOW_GUARD_ARITHMETIC_STATUSES == chk.CPU_BELOW_GUARD_ARITHMETIC_STATUSES, (
        "every one of the six named below-guard statuses must actually appear at source -- "
        "an unused name in the subtraction list would silently widen the residual for nothing"
    )


def test_main_returns_0_against_the_real_tree(capsys):
    rc = chk.main()
    out = capsys.readouterr()
    assert rc == 0
    assert "OK" in out.out


# --- check_build_bat_defines_o11_gate (T-2088, S1's own structural remedy; T-2091 Minor 2:
# landed round 11 with zero cells of either kind -- fixed here, the fixture is four lines. ---

_BUILD_BAT_FIXTURE_CLEAN = """\
cl /nologo /std:c++20 /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
    test_main.cpp /Fe:out\\superslm_tests.exe
cl /nologo /std:c++20 /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
    t2039_c5_harness.cpp /Fe:out\\t2039_c5_harness.exe
"""


def test_check_build_bat_defines_o11_gate_passes_when_flag_is_on_the_first_cl_line():
    assert chk.check_build_bat_defines_o11_gate(_BUILD_BAT_FIXTURE_CLEAN) == []


def test_check_build_bat_defines_o11_gate_reddens_when_flag_is_absent():
    mutated = _BUILD_BAT_FIXTURE_CLEAN.replace(" /DSUPERSLM_O11_ALLOC_INJECTION", "")
    failures = chk.check_build_bat_defines_o11_gate(mutated)
    assert failures, "removing the flag from the test-binary line must redden"
    assert "no longer defines" in failures[0]


def test_check_build_bat_defines_o11_gate_reddens_when_flag_is_only_on_the_second_cl_line():
    # T-2089's own MUT-1b: the docstring's own anchoring argument turns on this exact case --
    # the flag existing SOMEWHERE in the file is not what this check verifies.
    moved = (
        "cl /nologo /std:c++20 /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^\n"
        "    test_main.cpp /Fe:out\\superslm_tests.exe\n"
        "cl /nologo /std:c++20 /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^\n"
        "    t2039_c5_harness.cpp /Fe:out\\t2039_c5_harness.exe\n"
    )
    failures = chk.check_build_bat_defines_o11_gate(moved)
    assert failures, "the flag on the C5-harness line alone must not satisfy the test-binary check"


def test_check_build_bat_defines_o11_gate_reddens_when_no_cl_invocation_exists():
    failures = chk.check_build_bat_defines_o11_gate("echo nothing to build here\n")
    assert failures, "no 'cl /nologo' invocation at all must redden with a named message"
    assert "no 'cl /nologo' invocation found" in failures[0]


def test_real_tree_check_build_bat_defines_o11_gate_passes_today():
    with open(chk.BUILD_BAT, "r", encoding="utf-8") as f:
        build_bat_text = f.read()
    assert chk.check_build_bat_defines_o11_gate(build_bat_text) == []


# --- check_symbol_integrity (T-2091, S2's own structural remedy, build log §27): round 11's own
# M1 routed this and round 12 built a manual fragment-grep instead, without disclosing the
# deferral. T-2094's own M1 correction: the ORIGINAL claimed red-proof (restoring gpu_port.h's
# pre-fix split alone) is false on the real tree -- a split of a symbol that is STILL LIVE
# resolves by design; the real defect this check catches is a RENAME that misses the split
# occurrence, reproduced below. ---

def test_check_symbol_integrity_passes_on_the_real_tree_today():
    assert chk.check_symbol_integrity() == []


def test_check_symbol_integrity_reddens_on_a_split_identifier_with_no_real_symbol():
    with tempfile.TemporaryDirectory() as tmp:
        fake_dir = os.path.join(tmp, "fake")
        os.makedirs(fake_dir, exist_ok=True)
        fake_path = os.path.join(fake_dir, "fake.cpp")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write(
                "// a comment naming `SomeRenamedSymbol\n"
                "// ThatNoLongerExists` anywhere in this fixture\n"
                "void RealFunction() {}\n"
            )
        failures = chk.check_symbol_integrity(
            scanned_files=(("fake/fake.cpp", "//"),), allowlist=frozenset(), repo_root=tmp,
        )
        assert failures, "a split identifier with no real symbol and no allowlist entry must redden"
        assert "SomeRenamedSymbolThatNoLongerExists" in failures[0]


def test_check_symbol_integrity_stays_clean_when_the_split_identifier_resolves():
    with tempfile.TemporaryDirectory() as tmp:
        fake_dir = os.path.join(tmp, "fake")
        os.makedirs(fake_dir, exist_ok=True)
        fake_path = os.path.join(fake_dir, "fake.cpp")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write(
                "// a comment naming `RealFunc\n"
                "// tion` split across a line-wrap\n"
                "void RealFunction() {}\n"
            )
        failures = chk.check_symbol_integrity(
            scanned_files=(("fake/fake.cpp", "//"),), allowlist=frozenset(), repo_root=tmp,
        )
        assert failures == [], f"a split identifier that resolves to a real symbol must not redden: {failures}"


def test_check_symbol_integrity_stays_clean_when_the_split_identifier_is_allowlisted():
    with tempfile.TemporaryDirectory() as tmp:
        fake_dir = os.path.join(tmp, "fake")
        os.makedirs(fake_dir, exist_ok=True)
        fake_path = os.path.join(fake_dir, "fake.cpp")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write("// a historical name `RetiredSym\n// bol` kept on purpose\n")
        failures = chk.check_symbol_integrity(
            scanned_files=(("fake/fake.cpp", "//"),),
            allowlist=frozenset({"RetiredSymbol"}),
            repo_root=tmp,
        )
        assert failures == [], f"an allowlisted split identifier must not redden: {failures}"


def test_check_symbol_integrity_reddens_when_a_rename_misses_the_split_occurrence():
    # T-2094's own M1 correction (build log §28): the ORIGINAL claim here ("restoring gpu_port.h's
    # pre-fix split alone must redden") was FALSE on the real tree -- executed, restoring that
    # exact split leaves the check green, because the reconstructed name still resolves to the
    # live function. The real defect class is a RENAME that sweeps every UNSPLIT occurrence and
    # misses the split one -- reproduced for real here: a scanned "library" file where the real
    # function is named `RenamedFunction`, and a scanned "header" file whose own split citation
    # still names the OLD, pre-rename form across a line-wrap.
    with tempfile.TemporaryDirectory() as tmp:
        lib_dir = os.path.join(tmp, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        lib_path = os.path.join(lib_dir, "lib.cpp")
        with open(lib_path, "w", encoding="utf-8") as f:
            f.write("void RenamedFunction() {}\n")  # the rename landed here...
        header_dir = os.path.join(tmp, "include")
        os.makedirs(header_dir, exist_ok=True)
        header_path = os.path.join(header_dir, "header.h")
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(
                "// pinned assertion in `lib.cpp` (`OldFunction\n"
                "// Name`), not this sentence alone.\n"  # ...but not swept here, hidden by the wrap
            )
        failures = chk.check_symbol_integrity(
            scanned_files=(("lib/lib.cpp", "//"), ("include/header.h", "//")),
            allowlist=frozenset(), repo_root=tmp,
        )
        assert failures, "a rename that misses a split occurrence must redden the reconstructed old name"
        assert "OldFunctionName" in failures[0]


def test_check_symbol_integrity_stays_clean_when_a_split_symbol_is_still_live():
    # The corrected claim's OTHER half, stated as its own cell rather than left implicit: a split
    # of a symbol that IS still live (no rename happened) resolves by design and must NOT redden --
    # this is `test_check_symbol_integrity_stays_clean_when_the_split_identifier_resolves` above,
    # confirmed again here against the exact real-world shape (a file citation split mid-name) M1
    # found the original test's own claim wrong about.
    with tempfile.TemporaryDirectory() as tmp:
        lib_dir = os.path.join(tmp, "lib")
        os.makedirs(lib_dir, exist_ok=True)
        lib_path = os.path.join(lib_dir, "lib.cpp")
        with open(lib_path, "w", encoding="utf-8") as f:
            f.write("void StillLiveFunction() {}\n")
        header_dir = os.path.join(tmp, "include")
        os.makedirs(header_dir, exist_ok=True)
        header_path = os.path.join(header_dir, "header.h")
        with open(header_path, "w", encoding="utf-8") as f:
            f.write("// pinned assertion in `lib.cpp` (`StillLiveFunction\n// `), not this sentence alone.\n")
        failures = chk.check_symbol_integrity(
            scanned_files=(("lib/lib.cpp", "//"), ("include/header.h", "//")),
            allowlist=frozenset(), repo_root=tmp,
        )
        assert failures == [], f"a split of a still-live symbol must not redden: {failures}"


def test_real_tree_check_symbol_integrity_reddens_when_gpu_port_h_s_own_split_is_restored_and_renamed():
    # The real-tree reproduction of the corrected claim, in scratch-equivalent form: `gpu_port.h`'s
    # own pre-T-2088 split of `check_build_bat_defines_o11_gate` restored, WHILE the real function
    # is renamed in a scratch copy of this module -- the exact class M1 named ("the next rename of
    # it is the fifth instance, by construction").
    with tempfile.TemporaryDirectory() as tmp:
        gph_dir = os.path.join(tmp, "include")
        os.makedirs(gph_dir, exist_ok=True)
        gph_path = os.path.join(gph_dir, "gpu_port.h")
        with open(gph_path, "w", encoding="utf-8") as f:
            f.write(
                "// pinned assertion in `check.py` (`check_build_bat_defines_o11_\n"
                "// gate`), not this sentence alone.\n"
            )
        py_dir = os.path.join(tmp, "py")
        os.makedirs(py_dir, exist_ok=True)
        py_path = os.path.join(py_dir, "check.py")
        with open(py_path, "w", encoding="utf-8") as f:
            f.write("def check_build_bat_defines_the_o11_gate(x):\n    return x\n")
        failures = chk.check_symbol_integrity(
            scanned_files=(("include/gpu_port.h", "//"), ("py/check.py", "#")),
            allowlist=frozenset(), repo_root=tmp,
        )
        assert failures, "a rename that misses gpu_port.h's own split occurrence must redden"
        assert "check_build_bat_defines_o11_gate" in failures[0]


# --- parse_lwuws_path_counts / derive_lwuws_before_decision_count / derive_lwuws_after_decision_
# count / check_lwuws_path_count_claim (T-2094, S3's own structural remedy, build log §28): the
# "15-path count saga" closed structurally -- BOTH sides read fresh (the prose's own number-words,
# RunLayerLoopGpu's own real body), neither a re-typed constant. ---

_LWUWS_SENTENCE_FIXTURE = """\
// ...the recording-window catch, fourteen
// paths in all, none of which...
// ...the sticky-tag-decoded path alike, fifteen paths' own destination in
// total across the function...
"""


def test_parse_lwuws_path_counts_extracts_both_number_words():
    assert chk.parse_lwuws_path_counts(_LWUWS_SENTENCE_FIXTURE) == (14, 15)


def test_parse_lwuws_path_counts_raises_when_before_sentence_absent():
    try:
        chk.parse_lwuws_path_counts("nothing relevant here")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "paths in all" in str(e)


def test_parse_lwuws_path_counts_raises_when_total_sentence_absent():
    try:
        chk.parse_lwuws_path_counts("// catch, fourteen\n// paths in all, none of which")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "own destination in total" in str(e)


def test_derive_lwuws_before_decision_count_matches_the_gpu_fixture():
    # _GPU_FIXTURE has three ladder returns (InvalidLayerBudget, InvalidContextCap,
    # HeadDimGeometryMismatch) plus one device-capability rejection (!dev.available) before
    # `weights_resident` -- no catch/ternary in this small fixture, so no +1.
    assert chk.derive_lwuws_before_decision_count(_GPU_FIXTURE) == 4


def test_derive_lwuws_after_decision_count_is_one_on_the_gpu_fixture():
    assert chk.derive_lwuws_after_decision_count(_GPU_FIXTURE) == 1


def test_check_lwuws_path_count_claim_passes_when_the_words_match_real_structure():
    gph = "// catch, four\n// paths in all\n// alike, five paths' own destination in\n// total, ..."
    assert chk.check_lwuws_path_count_claim(gph, _GPU_FIXTURE) == []


def test_check_lwuws_path_count_claim_reddens_when_the_before_word_is_wrong():
    gph = "// catch, three\n// paths in all\n// alike, five paths' own destination in\n// total, ..."
    failures = chk.check_lwuws_path_count_claim(gph, _GPU_FIXTURE)
    assert failures, "a wrong before-decision word must redden"
    assert "'3'" in failures[0] and "4" in failures[0]


def test_check_lwuws_path_count_claim_reddens_when_the_total_word_is_wrong():
    gph = "// catch, four\n// paths in all\n// alike, six paths' own destination in\n// total, ..."
    failures = chk.check_lwuws_path_count_claim(gph, _GPU_FIXTURE)
    assert failures, "a wrong total word must redden"
    assert "'6'" in failures[0]


def test_check_lwuws_path_count_claim_reddens_when_only_the_english_word_is_wrong_structure_untouched():
    # The class this remedy specifically closes (T-2091's own gap, D-SLM3277 S3): the structural
    # side (RunLayerLoopGpu's own real body) is UNCHANGED and correct; only the ENGLISH WORD in the
    # sentence is wrong -- T-2091's own check (comparing two hardcoded constants) could not see
    # this at all, because neither side it compared was the sentence itself.
    gph = "// catch, NINETEEN\n// paths in all\n// alike, TWENTY paths' own destination in\n// total, ..."
    failures = chk.check_lwuws_path_count_claim(gph, _GPU_FIXTURE)
    assert failures, "a corrupted English word, with the real structure untouched, must redden"


def test_real_tree_check_lwuws_path_count_claim_passes_today():
    with open(chk.GPU_PORT_H, "r", encoding="utf-8") as f:
        gph = f.read()
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        gpu = f.read()
    assert chk.check_lwuws_path_count_claim(gph, gpu) == []


# --- derive_execution_scope / check_execution_scope_waivers (T-2091, class-B sweep, O30's own
# generalized closure, build log §27): a pin protecting cells a pipeline never runs is hollow; a
# hollow scope with no named waiver fails the build. ---

_BUILD_BAT_SCOPE_FIXTURE = """\
cl /nologo /std:c++20 /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION /DSUPERSLM_O11_ALLOC_INJECTION ^
    src\\gpu\\superslm_gpu.cpp tests\\test_main.cpp /Fe:out\\superslm_tests.exe
cl /nologo /std:c++20 /DSUPERSLM_ENABLE_BAD_ALLOC_INJECTION ^
    src\\gpu\\superslm_gpu.cpp tools\\t2039_c5_harness.cpp /Fe:out\\t2039_c5_harness.exe
"""

_CMAKE_SCOPE_FIXTURE_HOLLOW = """\
set(SUPERSLM_CORE_SOURCES
	src/artifact.cpp
	src/forward/forward_sites.cpp
)
add_library(superslm_test_injection STATIC ${SUPERSLM_CORE_SOURCES})
add_executable(superslm_tests tests/test_main.cpp)
target_link_libraries(superslm_tests PRIVATE superslm_test_injection)
"""

_CMAKE_SCOPE_FIXTURE_NOT_HOLLOW = _CMAKE_SCOPE_FIXTURE_HOLLOW.replace(
    "src/forward/forward_sites.cpp\n", "src/forward/forward_sites.cpp\n\tsrc/gpu/superslm_gpu.cpp\n"
)


def test_derive_execution_scope_finds_build_bat_s_two_targets_clean():
    scopes = chk.derive_execution_scope(_BUILD_BAT_SCOPE_FIXTURE, _CMAKE_SCOPE_FIXTURE_NOT_HOLLOW)
    by_name = {s.name: s for s in scopes}
    assert not by_name["build.bat: test binary (out\\superslm_tests.exe)"].is_hollow
    assert not by_name["build.bat: C5 harness (out\\t2039_c5_harness.exe)"].is_hollow


def test_derive_execution_scope_flags_cmake_target_hollow_when_gpu_source_missing():
    scopes = chk.derive_execution_scope(_BUILD_BAT_SCOPE_FIXTURE, _CMAKE_SCOPE_FIXTURE_HOLLOW)
    by_name = {s.name: s for s in scopes}
    assert by_name["CMake: superslm_tests (GitHub Actions)"].is_hollow


def test_derive_execution_scope_cmake_target_not_hollow_when_gpu_source_present():
    scopes = chk.derive_execution_scope(_BUILD_BAT_SCOPE_FIXTURE, _CMAKE_SCOPE_FIXTURE_NOT_HOLLOW)
    by_name = {s.name: s for s in scopes}
    assert not by_name["CMake: superslm_tests (GitHub Actions)"].is_hollow


def test_check_execution_scope_waivers_reddens_on_an_unwaived_hollow_scope():
    with tempfile.TemporaryDirectory() as tmp:
        bb_path = os.path.join(tmp, "build.bat")
        cm_path = os.path.join(tmp, "CMakeLists.txt")
        with open(bb_path, "w", encoding="utf-8") as f:
            f.write(_BUILD_BAT_SCOPE_FIXTURE)
        with open(cm_path, "w", encoding="utf-8") as f:
            f.write(_CMAKE_SCOPE_FIXTURE_HOLLOW)
        failures = chk.check_execution_scope_waivers(build_bat_path=bb_path, cmake_path=cm_path, waivers={})
        assert failures, "an unwaived hollow scope must redden"
        assert "hollow execution scope" in failures[0]


def test_check_execution_scope_waivers_passes_when_the_hollow_scope_is_named():
    with tempfile.TemporaryDirectory() as tmp:
        bb_path = os.path.join(tmp, "build.bat")
        cm_path = os.path.join(tmp, "CMakeLists.txt")
        with open(bb_path, "w", encoding="utf-8") as f:
            f.write(_BUILD_BAT_SCOPE_FIXTURE)
        with open(cm_path, "w", encoding="utf-8") as f:
            f.write(_CMAKE_SCOPE_FIXTURE_HOLLOW)
        failures = chk.check_execution_scope_waivers(
            build_bat_path=bb_path, cmake_path=cm_path,
            waivers={"CMake: superslm_tests (GitHub Actions)": "waived for this cell"},
        )
        assert failures == [], f"a named waiver must silence the hollow-scope finding: {failures}"


def test_real_tree_check_execution_scope_waivers_passes_today():
    # The real tree's own CMake target IS hollow (confirmed by
    # test_derive_execution_scope_flags_cmake_target_hollow_when_gpu_source_missing's own real-file
    # sibling below) -- this must still pass because EXECUTION_SCOPE_WAIVERS names it.
    assert chk.check_execution_scope_waivers() == []


def test_real_tree_cmake_superslm_tests_target_is_hollow_today():
    # Confirms the finding this round's waiver exists FOR is real, not hypothetical -- if a future
    # round adds GPU sources to CMakeLists.txt, THIS cell goes red first, which is the signal to
    # delete the now-stale waiver rather than leave it describing a gap that no longer exists.
    with open(chk.BUILD_BAT, "r", encoding="utf-8") as f:
        build_bat_text = f.read()
    with open(chk.CMAKE_LISTS, "r", encoding="utf-8") as f:
        cmake_text = f.read()
    scopes = chk.derive_execution_scope(build_bat_text, cmake_text)
    by_name = {s.name: s for s in scopes}
    assert by_name["CMake: superslm_tests (GitHub Actions)"].is_hollow, (
        "if this is no longer true, EXECUTION_SCOPE_WAIVERS's own entry for it is stale -- delete it"
    )


# --- Wiring-vitality registry (T-2091, Minor 2; T-2094, S2's own rebuild, D-SLM3277): MUT-6's own
# falsifying case (wiring blocks replaced with `if False: pass`, nothing reddened) turned into a
# standing, table-driven proof for every check `run_all_checks` can individually disable -- for
# each, a fixture built to defeat EXACTLY that check reddens when the check is enabled (the
# default) AND stops reddening FOR THAT REASON when explicitly disabled, THE SAME DEFEATING INPUT
# THREADED IN BOTH RUNS. The T-2091 registry proved only half of this for five of its eleven cells
# (asserting "the standalone function discriminates" and "the fully-disabled run is clean"
# separately, never the conjunction against a live defeating input run THROUGH `run_all_checks`
# itself) -- D-SLM3277's own S2 finding, reproduced by its MUT-G (five blocks neutered, 98 passed,
# OK, nothing noticed). Every cell below passes the SAME on/off pair the same defeating fixture, so
# neutering the wiring block it exercises must flip its own `enabled` assertion, by name. Four of
# the old population's cells (`self_citations`, `prose_citations`, `check_self_citation_population`,
# `check_self_line_count`) no longer exist -- T-2094's substrate removal folded the first three into
# `check_marked_citation_scan` (one whole-file scan replaces three chunk/table/population checks)
# and deleted the fourth's own claim entirely (`gpu_port.h`'s "N lines" self-citation sentence was
# removed, not converted -- nothing claims it any more, so nothing needs pinning). ---

def _real_tree_kwargs_with(**overrides):
    kwargs = dict(
        forward_sites_path=chk.FORWARD_SITES_CPP,
        superslm_gpu_path=chk.SUPERSLM_GPU_CPP,
        guards_def_path=chk.GUARDS_DEF,
        repo_root=chk._REPO_ROOT,
        gpu_port_h_path=None,
        check_decode_sticky_tag=False,
        build_bat_path=None,
        check_marked_citation_scan=False,
        run_symbol_integrity_scan=False,
        check_lwuws_path_count=False,
        cmake_path=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_wiring_vitality_check_decode_sticky_tag_disable_stops_catching_a_shrunk_switch():
    with tempfile.TemporaryDirectory() as tmp:
        with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
            real_text = f.read()
        mutated = real_text.replace("\t\tcase 13: return S::SiluCompositionScaleOutOfDomain;\n", "", 1)
        assert mutated != real_text, "sanity: case 13 must exist verbatim in the real file to mutate"
        gpu_path = os.path.join(tmp, "shrunk_decode_gpu.cpp")
        with open(gpu_path, "w", encoding="utf-8") as f:
            f.write(mutated)
        on = chk.check_decode_sticky_tag_range(mutated)
        assert on, "sanity: deleting one interior DecodeStickyTag case must defeat the standalone check"
        # superslm_gpu_path points at the mutated COPY (only DecodeStickyTag's own body changed --
        # RunLayerLoopGpu's own ladder is untouched, so the unconditional CPU/GPU/.def three-way
        # comparison stays clean and this proof isolates check_decode_sticky_tag alone).
        enabled = chk.run_all_checks(**_real_tree_kwargs_with(
            superslm_gpu_path=gpu_path, check_decode_sticky_tag=True,
        ))
        disabled = chk.run_all_checks(**_real_tree_kwargs_with(
            superslm_gpu_path=gpu_path, check_decode_sticky_tag=False,
        ))
        assert enabled, "a shrunk DecodeStickyTag switch must be caught when check_decode_sticky_tag is enabled"
        assert disabled == [], "check_decode_sticky_tag=False must remove the catch with the same mutated file threaded"


def test_wiring_vitality_build_bat_path_disable_stops_catching_the_missing_flag():
    with tempfile.TemporaryDirectory() as tmp:
        bb_path = os.path.join(tmp, "build.bat")
        with open(bb_path, "w", encoding="utf-8") as f:
            f.write(_BUILD_BAT_SCOPE_FIXTURE.replace(" /DSUPERSLM_O11_ALLOC_INJECTION", ""))
        on = chk.check_build_bat_defines_o11_gate(open(bb_path, encoding="utf-8").read())
        assert on, "sanity: the mutated fixture must defeat the standalone check"
        enabled = chk.run_all_checks(**_real_tree_kwargs_with(build_bat_path=bb_path))
        disabled = chk.run_all_checks(**_real_tree_kwargs_with(build_bat_path=None))
        assert enabled, "the missing flag must be caught when build_bat_path is threaded"
        assert disabled == [], "disabling build_bat_path must remove the catch -- proves the kwarg is load-bearing"


def test_wiring_vitality_check_marked_citation_scan_disable_stops_catching_a_stale_citation():
    with tempfile.TemporaryDirectory() as tmp:
        fake_path = os.path.join(tmp, "fake.cpp")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write("// `TotallyMadeUpSymbolThatDoesNotExistXyz` (check_gpu_guard_status_parity.py)\n")
        scanned = ((fake_path, "//"),)
        on = chk.check_marked_citations(scanned_files=scanned, repo_root=chk._REPO_ROOT)
        assert on, "sanity: a made-up symbol name must defeat the standalone check"
        enabled = chk.run_all_checks(**_real_tree_kwargs_with(
            check_marked_citation_scan=True, marked_citation_scanned_files=scanned,
        ))
        disabled = chk.run_all_checks(**_real_tree_kwargs_with(
            check_marked_citation_scan=False, marked_citation_scanned_files=scanned,
        ))
        assert enabled, "a stale marked citation must be caught when check_marked_citation_scan is enabled"
        assert disabled == [], (
            "check_marked_citation_scan=False must remove the catch even with the same defeating "
            "scanned_files threaded"
        )


def test_wiring_vitality_run_symbol_integrity_scan_disable_stops_catching_a_split():
    with tempfile.TemporaryDirectory() as tmp:
        fake_path = os.path.join(tmp, "fake.cpp")
        with open(fake_path, "w", encoding="utf-8") as f:
            f.write("// `SomeRenamedSymbol\n// ThatNoLongerExists` split\n")
        scanned = ((fake_path, "//"),)
        on = chk.check_symbol_integrity(scanned_files=scanned, allowlist=frozenset(), repo_root=chk._REPO_ROOT)
        assert on, "sanity: the split fixture must defeat the standalone check"
        enabled = chk.run_all_checks(**_real_tree_kwargs_with(
            run_symbol_integrity_scan=True, symbol_integrity_scanned_files=scanned,
        ))
        disabled = chk.run_all_checks(**_real_tree_kwargs_with(
            run_symbol_integrity_scan=False, symbol_integrity_scanned_files=scanned,
        ))
        assert enabled, "a split identifier must be caught when run_symbol_integrity_scan is enabled"
        assert disabled == [], (
            "run_symbol_integrity_scan=False must remove the catch even with the same defeating "
            "scanned_files threaded"
        )


def test_wiring_vitality_check_lwuws_path_count_disable_stops_catching_a_corrupted_word():
    with tempfile.TemporaryDirectory() as tmp:
        with open(chk.GPU_PORT_H, "r", encoding="utf-8") as f:
            real_text = f.read()
        corrupted = real_text.replace("catch, fourteen\n", "catch, nineteen\n", 1)
        assert corrupted != real_text, "sanity: the exact wrapped phrase must exist in the real file"
        gph_path = os.path.join(tmp, "corrupted_before_word_gpu_port.h")
        with open(gph_path, "w", encoding="utf-8") as f:
            f.write(corrupted)
        with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
            gpu_text = f.read()
        on = chk.check_lwuws_path_count_claim(corrupted, gpu_text)
        assert on, "sanity: the corrupted number-word must defeat the standalone check"
        enabled = chk.run_all_checks(**_real_tree_kwargs_with(
            gpu_port_h_path=gph_path, check_lwuws_path_count=True,
        ))
        disabled = chk.run_all_checks(**_real_tree_kwargs_with(
            gpu_port_h_path=gph_path, check_lwuws_path_count=False,
        ))
        assert enabled, "a corrupted number-word must be caught when check_lwuws_path_count is enabled"
        assert disabled == [], (
            "check_lwuws_path_count=False must remove the catch even with the same corrupted file threaded"
        )


def test_wiring_vitality_gpu_port_h_path_disable_stops_catching_a_corrupted_word():
    with tempfile.TemporaryDirectory() as tmp:
        with open(chk.GPU_PORT_H, "r", encoding="utf-8") as f:
            real_text = f.read()
        corrupted = real_text.replace("alike, fifteen", "alike, twenty", 1)
        assert corrupted != real_text, "sanity: the exact phrase must exist in the real file"
        gph_path = os.path.join(tmp, "corrupted_total_word_gpu_port.h")
        with open(gph_path, "w", encoding="utf-8") as f:
            f.write(corrupted)
        with_path = chk.run_all_checks(**_real_tree_kwargs_with(
            gpu_port_h_path=gph_path, check_lwuws_path_count=True,
        ))
        without_path = chk.run_all_checks(**_real_tree_kwargs_with(
            gpu_port_h_path=None, check_lwuws_path_count=True,
        ))
        assert with_path, "a corrupted total word must be caught when gpu_port_h_path is threaded"
        assert without_path == [], "gpu_port_h_path=None must remove the catch"


def test_wiring_vitality_cmake_path_disable_stops_catching_an_unwaived_hollow_scope(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        bb_path = os.path.join(tmp, "build.bat")
        cm_path = os.path.join(tmp, "CMakeLists.txt")
        with open(bb_path, "w", encoding="utf-8") as f:
            f.write(_BUILD_BAT_SCOPE_FIXTURE)
        with open(cm_path, "w", encoding="utf-8") as f:
            f.write(_CMAKE_SCOPE_FIXTURE_HOLLOW)
        # The fixture's own hollow scope shares its NAME with the real, production-waived one
        # (EXECUTION_SCOPE_WAIVERS names "CMake: superslm_tests (GitHub Actions)") -- silenced via
        # monkeypatch to an empty dict for this cell only, so this proves the cmake_path KWARG's
        # own wiring, not whether the real waiver happens to cover this fixture's scope name too.
        monkeypatch.setattr(chk, "EXECUTION_SCOPE_WAIVERS", {})
        with_path = chk.run_all_checks(**_real_tree_kwargs_with(build_bat_path=bb_path, cmake_path=cm_path))
        without_path = chk.run_all_checks(**_real_tree_kwargs_with(build_bat_path=bb_path, cmake_path=None))
        assert with_path, "an unwaived hollow CMake scope must be caught when cmake_path is threaded"
        assert without_path == [], "cmake_path=None must remove the catch"


# ===========================================================================
# T-2098 -- Claude/Poirot/152035b-gpu-serial-port-substrate-removal-review.md
# (D-SLM3291). One cell per remedy this round actually LANDED, derived from
# the production diff rather than from the review's finding list, per
# `Claude/CLAUDE.md`'s own standing rule that a fix round pins its own
# changes in the same round. Every mutation named below is the reviewing
# seat's own, re-run against the repair.
# ===========================================================================

# --- S1: the scan population's own vitality (the structural half) ---

def test_marked_citation_population_is_live_on_the_real_tree():
    # The round-15 state this closes, in one assertion: three files were named and one returned
    # results (21, 0, 0). A member contributing nothing is indistinguishable from one that works.
    assert chk.check_marked_citation_population_is_live() == []


def test_every_real_scanned_file_yields_at_least_one_marked_citation():
    # The reviewing seat's own distinguishing test, kept as a standing cell: run the mechanism over
    # its own declared population and count what it returns.
    for rel_path, marker in chk.MARKED_CITATION_SCANNED_FILES:
        with open(os.path.join(chk._REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
            found = chk.find_marked_citations(f.read(), marker)
        assert found, f"{rel_path} is scanned for marked citations and yields none"


def test_marked_citation_population_is_live_reddens_on_a_silent_file():
    with tempfile.TemporaryDirectory() as tmp:
        rel = "silent.h"
        with open(os.path.join(tmp, rel), "w", encoding="utf-8") as f:
            f.write("// prose naming RunLayerLoopImpl and forward_sites.cpp, but not as a citation\n")
        failures = chk.check_marked_citation_population_is_live(
            scanned_files=((rel, "//"),), repo_root=tmp
        )
        assert failures and "yields ZERO marked citations" in failures[0]


def test_forward_sites_cpp_is_a_recognized_citation_target():
    # MUT-C, re-run against the repair. Round 15: writing a citation in the round's OWN marked form,
    # naming the file seven of nine converted citations point at, turned CI RED ("not a recognized
    # citation target") -- so the convention forbade the correct form and prose was the only option.
    assert chk.check_marked_citation("RunLayerLoopImpl", "forward_sites.cpp") is None
    assert chk.check_marked_citation("RunLayerLoop", "forward_sites.h") is None
    assert chk.check_marked_citation("CheckBiasAccumulateMagnitudeDomain", "checked_chain_funnel.cpp") is None
    assert chk.check_marked_citation("kComposedResourceBindingCount", "d3d12_harness.h") is None


def test_a_stale_citation_into_forward_sites_cpp_reddens():
    # The other direction: widening the resolvable population must not make it accept anything.
    failure = chk.check_marked_citation("RunLayerLoopImplRenamedAway", "forward_sites.cpp")
    assert failure is not None and "citation is stale" in failure


# --- S2: the four refreshed citations, pinned at source rather than by line ---

def test_the_module_own_device_capability_citations_resolve():
    # The four stale citations sat in THIS module and cited `superslm_gpu.cpp:643, 655, 1276-1277`
    # while the real sites were `:777`, `:788`, `:1471` -- six rounds, through three rounds
    # commissioned to close citation staleness, because the module was in its own scan population
    # and the scan returned nothing for it. In the marked form there is no line number to go stale.
    for name in ("dev.available", "MapModelGpuResidencyTierCheck", "device_removed_reason"):
        assert chk.check_marked_citation(name, "superslm_gpu.cpp") is None, name


# --- S3: the derivation cuts on the comment-stripped body, at a complete statement ---

def test_derive_before_count_is_insensitive_to_an_unrelated_comment(monkeypatch):
    # MUT-N2 / MUT-D, re-run against the repair. Round 15 cut on the UN-stripped body, so a comment
    # merely CONTAINING `weights_resident;` decided the boundary: rewording an unrelated comment
    # moved the count, and a comment naming the fragment above the ladder collapsed it to 1.
    baseline = chk.derive_lwuws_before_decision_count(_GPU_FIXTURE)
    reworded = _GPU_FIXTURE.replace(
        "// the real function ends via",
        "// mentions g_last_weight_upload_was_skipped = weights_resident; harmlessly\n\t// the real function ends via",
    )
    assert chk.derive_lwuws_before_decision_count(reworded) == baseline
    above_the_ladder = _GPU_FIXTURE.replace(
        "\tif (x == 0) return",
        "\t// a comment naming g_last_weight_upload_was_skipped = weights_resident; up here\n\tif (x == 0) return",
    )
    assert chk.derive_lwuws_before_decision_count(above_the_ladder) == baseline


def test_derive_before_count_sees_a_new_rejecting_return_before_the_residency_write():
    # MUT-N, re-run against the repair: round 15 left this at 14, checker OK, exit 0 -- a fifteenth
    # path before the residency decision that the check existing to pin exactly that number missed.
    baseline = chk.derive_lwuws_before_decision_count(_GPU_FIXTURE)
    mutated = _GPU_FIXTURE.replace(
        "\tconst bool weights_resident",
        "\tif (x == 99) return superslm::SslmForwardStatus::KvPrecisionUnsupported;\n\tconst bool weights_resident",
    )
    assert chk.derive_lwuws_before_decision_count(mutated) == baseline + 1


def test_derive_before_count_raises_when_the_residency_statement_is_absent():
    # Round 15 fell back to the whole body on a missing marker, silently measuring a different
    # region. A check that cannot find the anchor it verifies must not pass.
    without = _GPU_FIXTURE.replace("\tg_last_weight_upload_was_skipped = weights_resident;\n", "")
    try:
        chk.derive_lwuws_before_decision_count(without)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no boundary to cut on" in str(e)


# --- M1: catch ternaries are COUNTED, not detected ---

def test_count_status_return_ternaries_counts_each_occurrence():
    one = "return device_removed_reason != S_OK ? A::X : A::Y;"
    assert chk.count_status_return_ternaries(one) == 1
    assert chk.count_status_return_ternaries(one + "\n" + one) == 2
    assert chk.count_status_return_ternaries("// " + one) == 0


def test_a_second_catch_ternary_moves_the_derived_count():
    # Round 15 asked whether `device_removed_reason` appeared anywhere and added exactly 1, so a
    # SECOND ternary-returned rejection -- the shape T-2059 added once already -- was free.
    with_one = _GPU_FIXTURE.replace(
        "\treturn DecodeStickyTag(sticky_tag);",
        "\tif (x == 7) { return device_removed_reason != S_OK ? superslm::SslmForwardStatus::GpuDeviceRemoved\n"
        "\t                                                   : superslm::SslmForwardStatus::GpuAllocationFailed; }\n"
        "\treturn DecodeStickyTag(sticky_tag);",
    )
    with_two = with_one.replace(
        "\treturn DecodeStickyTag(sticky_tag);",
        "\tif (x == 8) { return device_removed_reason != S_OK ? superslm::SslmForwardStatus::GpuDeviceRemoved\n"
        "\t                                                   : superslm::SslmForwardStatus::GpuAllocationFailed; }\n"
        "\treturn DecodeStickyTag(sticky_tag);",
    )
    assert chk.derive_lwuws_before_decision_count(with_two) == \
        chk.derive_lwuws_before_decision_count(with_one) + 1


def test_the_real_tree_catch_ternary_is_counted_from_the_whole_body():
    # The one asymmetry in the derivation, pinned so it cannot be "tidied" into a positional cut:
    # `gpu_port.h`'s own fourteen enumerates the ladder, the two device-capability rejections AND
    # the recording-window catch, whose ternary sits BELOW the residency write. Cutting both terms
    # at the same index reads 13 against a prose 14.
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        gpu_text = f.read()
    body = chk.strip_comments(chk.extract_function_body(gpu_text, chk.GPU_FUNC_SIGNATURE, label="x"))
    before = body[:body.find(chk._LWUWS_RESIDENCY_WRITE_STATEMENT)]
    assert chk.count_status_return_ternaries(before) == 0
    assert chk.count_status_return_ternaries(body) == 1
    assert chk.derive_lwuws_before_decision_count(gpu_text) == 14


# --- M2: O34's successor residual is a MEASURED property, not a claim about one ---

def test_a_citation_naming_a_different_real_symbol_is_not_caught():
    # MUT-O, kept as a standing cell exactly as this module already keeps its set-equality residual:
    # the existence check cannot distinguish a correct citation from a citation to a different real
    # thing. This asserts the DOCUMENTED weakness, so a future change that closes it fails here and
    # routes someone to the docstring rather than leaving the disclosure silently stale.
    assert chk.check_marked_citation("WorkspaceTooSmall", "superslm_gpu.cpp") is None
    assert chk.check_marked_citation("InvalidLayerBudget", "superslm_gpu.cpp") is None


# --- O3: main() answers an unreadable source file in its own failure shape ---

def test_main_reports_a_missing_scanned_file_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        chk, "MARKED_CITATION_SCANNED_FILES", (("does/not/exist.h", "//"),)
    )
    rc = chk.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "check_gpu_guard_status_parity.py: FAILED" in err
    assert "source file unreadable" in err


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
