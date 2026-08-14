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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
        assert failures, "a .def row naming a status neither ladder returns must fail the check"


# --- Mutation C: a guard removed from the GPU ladder only -- must redden
# (GPU set now disagrees with both CPU and .def).

def test_mutation_c_guard_removed_from_gpu_ladder_only_reddens():
    gpu = _GPU_FIXTURE.replace(
        'if (x < 1) return superslm::SslmForwardStatus::InvalidContextCap;  // ContextCapNonPositive\n', ''
    )
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp, gpu=gpu)
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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
        failures = chk.run_all_checks(cpu_path, gpu_path, def_path, repo_root=tmp, prose_citations=())
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


# --- ProseCitation / check_prose_citation / check_all_prose_citations
# (T-2075, Structural remedy) ---

def test_check_prose_citation_passes_for_a_real_matching_line():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, _, _ = _write_fixture_tree(tmp)
        citation = chk.ProseCitation("fixture label", "src/forward/forward_sites.cpp", 4, "InvalidLayerBudget")
        assert chk.check_prose_citation(citation, repo_root=tmp) is None


def test_check_prose_citation_fails_when_needle_absent_from_cited_line():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        citation = chk.ProseCitation("fixture label", "src/forward/forward_sites.cpp", 4, "SomethingNotThere")
        failure = chk.check_prose_citation(citation, repo_root=tmp)
        assert failure is not None
        assert "citation is stale" in failure


def test_check_prose_citation_fails_when_line_out_of_range():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_tree(tmp)
        citation = chk.ProseCitation("fixture label", "src/forward/forward_sites.cpp", 9999, "x")
        failure = chk.check_prose_citation(citation, repo_root=tmp)
        assert failure is not None
        assert "out of range" in failure


def test_check_prose_citation_fails_when_file_missing():
    citation = chk.ProseCitation("fixture label", "nonexistent/file.cpp", 1, "x")
    failure = chk.check_prose_citation(citation, repo_root=os.getcwd())
    assert failure is not None
    assert "file not found" in failure


def test_check_all_prose_citations_reddens_when_one_of_several_shifts():
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, _, _ = _write_fixture_tree(tmp)
        good = chk.ProseCitation("good", "src/forward/forward_sites.cpp", 4, "InvalidLayerBudget")
        bad = chk.ProseCitation("bad (shifted)", "src/forward/forward_sites.cpp", 4, "NeverThere")
        failures = chk.check_all_prose_citations(citations=(good, bad), repo_root=tmp)
        assert len(failures) == 1
        assert "bad (shifted)" in failures[0]


def test_real_tree_gpu_port_h_lwuws_citations_all_resolve_today():
    # Every citation in gpu_port.h's own LastWeightUploadWasSkipped
    # paragraph, re-derived fresh at T-2075, resolves against the real file
    # today -- the exact property M1 found false for the T-2069 correction's
    # own citations one round after they were written.
    failures = chk.check_all_prose_citations()
    assert failures == [], f"gpu_port.h's own citations should all resolve today: {failures}"


def test_real_tree_prose_citation_population_is_the_named_nineteen():
    assert len(chk.GPU_PORT_H_LWUWS_CITATIONS) == 19


# --- Structural red proof (T-2075): shifting a cited line reddens run_all_checks. ---

def test_mutation_f_shifting_a_cited_gpu_line_reddens_via_prose_citation_check():
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        real_gpu_text = f.read()
    with tempfile.TemporaryDirectory() as tmp:
        cpu_path, gpu_path, def_path = _write_fixture_tree(tmp)
        # Overwrite the fixture tree's GPU file with a shifted copy of the
        # REAL file (one blank line inserted before the first ladder
        # citation's own line, 640) -- proves the structural machinery
        # against the real citation population, not only the small fixture
        # population used elsewhere in this file.
        real_lines = real_gpu_text.splitlines(keepends=True)
        shifted = real_lines[:639] + ["\n"] + real_lines[639:]
        with open(gpu_path, "w", encoding="utf-8") as f:
            f.writelines(shifted)
        failures = chk.check_all_prose_citations(repo_root=tmp)
        # citations relative to repo_root resolve against tmp's OWN tree
        # layout (src/gpu/superslm_gpu.cpp under tmp), which now holds the
        # shifted content at the real citations' own line numbers.
        assert failures, "a shifted cited line must redden the prose-citation check"
        assert any("citation is stale" in f for f in failures)


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


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
