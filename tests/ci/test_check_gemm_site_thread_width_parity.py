"""Red suite for tests/ci/check_gemm_site_thread_width_parity.py -- mirrors
test_check_gpu_guard_status_parity.py's own two-tier convention: fixture-driven mechanism cells
first, then cells against the real repo tree proving the real population passes today and the
wiring (real paths, real anchors) is not vacuous.

T-2125: re-derived for the checker module's own T-2113 (B4) re-derivation. The pre-B4 contract
this file used to test -- `parse_hlsl_thread_width` returning a 4-number
`(numthreads, stride_add, stride_div, stride_mul)` tuple, and a stride-formula self-consistency
check -- no longer exists in the module under test (see that module's own header comment): under
the transposed partition every GEMM site's `numthreads` is uniformly 256 and the per-site lane
split is host-supplied runtime data, not a compile-time formula a shader could independently
re-derive. What the module checks now is `(numthreads, calls_coalesced_gemm)` parity plus a
did-this-site-actually-adopt-the-transposed-primitive check, and this file is rewritten to that
contract, including the `KvProj` split site T-2113 (B4, D-SLM3341) added to `GEMM_SPLIT_SITES`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_gemm_site_thread_width_parity as chk  # noqa: E402


_CPP_FIXTURE = """\
namespace superslm_gpu {
GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site, uint32_t hidden_size,
                                                  uint32_t kv_out_channels,
                                                  uint32_t intermediate_size) {
	GpuGemmSiteGroupPlan plan;
	switch (site) {
		case GpuGemmSplitSite::QProj:
		case GpuGemmSplitSite::OProj:
		case GpuGemmSplitSite::DownProj:
			plan.out_channels = hidden_size;
			plan.threads_per_group = 64u;
			break;
		case GpuGemmSplitSite::KvProj:
			plan.out_channels = kv_out_channels;
			plan.threads_per_group = 64u;
			break;
		case GpuGemmSplitSite::GateProj:
		case GpuGemmSplitSite::UpProj:
			plan.out_channels = intermediate_size;
			plan.threads_per_group = 256u;
			break;
		default:
			throw GpuGemmGroupArithmeticError("unhandled");
	}
	plan.groups = ComputeGpuGemmGroupCount(plan.out_channels, plan.threads_per_group);
	return plan;
}
}  // namespace superslm_gpu
"""


def _hlsl_fixture(numthreads: int, call: str = "GemmCoalescedGpu(gtid.x, gid.x);") -> str:
    return (
        f"[numthreads({numthreads}, 1, 1)]\n"
        "void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)\n"
        "{\n"
        f"    {call}\n"
        "}\n"
    )


_HLSL_FIXTURES_CONSISTENT = {
    "q_proj_gemm_site.hlsl": _hlsl_fixture(64),
    "o_proj_gemm_site.hlsl": _hlsl_fixture(64),
    "kv_proj_gemm_site.hlsl": _hlsl_fixture(64, call="GemmCoalescedGpuAt(t, j, gtid.x, gid.x);"),
    "gate_proj_gemm_site.hlsl": _hlsl_fixture(256),
    "up_proj_gemm_site.hlsl": _hlsl_fixture(256),
    "down_proj_gemm_site.hlsl": _hlsl_fixture(64),
}


# --- strip_comments ---

def test_strip_comments_removes_line_and_block_comments():
    text = "a // numthreads(999,1,1)\nb /* block\ncomment */ c"
    out = chk.strip_comments(text)
    assert "999" not in out
    assert "a " in out and "b " in out and " c" in out


# --- parse_hlsl_thread_width ---

def test_parse_hlsl_thread_width_extracts_numthreads_and_coalesced_flag():
    got = chk.parse_hlsl_thread_width(_hlsl_fixture(64), label="fixture")
    assert got == (64, True)


def test_parse_hlsl_thread_width_reports_false_when_body_has_no_coalesced_call():
    text = _hlsl_fixture(64, call="GemmParallelGpu(gtid.x);")
    got = chk.parse_hlsl_thread_width(text, label="fixture")
    assert got == (64, False)


def test_parse_hlsl_thread_width_raises_when_numthreads_absent():
    try:
        chk.parse_hlsl_thread_width("void main() { GemmCoalescedGpu(0, 0); }", label="fixture")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "numthreads" in str(e)


# --- parse_cpp_threads_per_group_by_site ---

def test_parse_cpp_threads_per_group_by_site_attributes_grouped_case_labels():
    got = chk.parse_cpp_threads_per_group_by_site(_CPP_FIXTURE)
    assert got == {
        "QProj": 64, "OProj": 64, "DownProj": 64, "KvProj": 64,
        "GateProj": 256, "UpProj": 256,
    }


# --- check_gemm_site_thread_width_parity: the mechanism ---

def test_check_passes_when_everything_agrees():
    assert chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, _HLSL_FIXTURES_CONSISTENT) == []


def test_check_reddens_on_a_numthreads_host_shader_mismatch():
    # The reviewer's own named residual, reproduced at fixture scale: the shader says 128, the host
    # (ComputeGpuGemmSiteGroupPlan) says 64 for the same site -- a real host/shader divergence.
    mismatched = dict(_HLSL_FIXTURES_CONSISTENT)
    mismatched["q_proj_gemm_site.hlsl"] = _hlsl_fixture(128)
    failures = chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, mismatched)
    assert any("q_proj_gemm_site.hlsl" in f and "128" in f and "64" in f for f in failures), failures


def test_check_reddens_when_a_site_still_calls_legacy_parallel_gemm():
    # numthreads agrees, but the shader body never adopted T-2113 (B4)'s transposed-partition
    # primitive -- silently correct at fixture scale, silently wrong relative to the group count
    # ComputeGpuGemmSiteGroupPlan now computes for the transposed geometry.
    reverted = dict(_HLSL_FIXTURES_CONSISTENT)
    reverted["q_proj_gemm_site.hlsl"] = _hlsl_fixture(64, call="GemmParallelGpu(gtid.x);")
    failures = chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, reverted)
    assert any("q_proj_gemm_site.hlsl" in f and "still" in f for f in failures), failures


def test_check_reddens_when_a_site_calls_both_partitions():
    # A partial edit that adds the new call without removing the old one.
    partial = dict(_HLSL_FIXTURES_CONSISTENT)
    partial["q_proj_gemm_site.hlsl"] = _hlsl_fixture(
        64, call="GemmCoalescedGpu(gtid.x, gid.x); GemmParallelGpu(gtid.x);"
    )
    failures = chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, partial)
    assert any("q_proj_gemm_site.hlsl" in f and "BOTH" in f for f in failures), failures


def test_check_reddens_when_a_shader_file_is_missing():
    missing = dict(_HLSL_FIXTURES_CONSISTENT)
    del missing["down_proj_gemm_site.hlsl"]
    failures = chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, missing)
    assert any("down_proj_gemm_site.hlsl" in f and "not found" in f for f in failures), failures


# --- against the real repo tree ---

def test_real_tree_run_all_checks_passes_today():
    assert chk.run_all_checks() == []


def test_real_tree_falsification_a_mutated_numthreads_reddens():
    # The falsification the dispatch itself specified: edit one shader's own numthreads, confirm
    # the check reddens, without touching the real file (read, mutate in memory, check).
    real_path = os.path.join(chk.SHADERS_DIR, "q_proj_gemm_site.hlsl")
    with open(real_path, "r", encoding="utf-8") as f:
        real_text = f.read()
    mutated_text = real_text.replace("[numthreads(256, 1, 1)]", "[numthreads(128, 1, 1)]", 1)
    assert mutated_text != real_text, "sanity: the real file must contain the exact text mutated"
    with open(chk.SUPERSLM_GPU_CPP, "r", encoding="utf-8") as f:
        cpp_text = f.read()
    hlsl_texts = {}
    for site_name, hlsl_filename in chk.GEMM_SPLIT_SITES:
        path = os.path.join(chk.SHADERS_DIR, hlsl_filename)
        with open(path, "r", encoding="utf-8") as f:
            hlsl_texts[hlsl_filename] = f.read()
    hlsl_texts["q_proj_gemm_site.hlsl"] = mutated_text
    failures = chk.check_gemm_site_thread_width_parity(cpp_text, hlsl_texts)
    assert failures, "a mutated numthreads on the real tree must redden the real-tree check"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
