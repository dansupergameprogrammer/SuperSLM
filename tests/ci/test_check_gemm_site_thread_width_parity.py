"""Red suite for tests/ci/check_gemm_site_thread_width_parity.py -- mirrors
test_check_gpu_guard_status_parity.py's own two-tier convention: fixture-driven mechanism cells
first, then cells against the real repo tree proving the real population passes today and the
wiring (real paths, real anchors) is not vacuous.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_gemm_site_thread_width_parity as chk  # noqa: E402


_CPP_FIXTURE = """\
namespace superslm_gpu {
GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site, uint32_t hidden_size,
                                                  uint32_t intermediate_size) {
	GpuGemmSiteGroupPlan plan;
	switch (site) {
		case GpuGemmSplitSite::QProj:
		case GpuGemmSplitSite::OProj:
		case GpuGemmSplitSite::DownProj:
			plan.out_channels = hidden_size;
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


def _hlsl_fixture(numthreads: int, stride_add: int, stride_div: int, stride_mul: int) -> str:
    return (
        f"[numthreads({numthreads}, 1, 1)]\n"
        "void main(uint3 dtid : SV_DispatchThreadID)\n"
        "{\n"
        f"    int stride = ((out_channels + {stride_add}) / {stride_div}) * {stride_mul};\n"
        "}\n"
    )


_HLSL_FIXTURES_CONSISTENT = {
    "q_proj_gemm_site.hlsl": _hlsl_fixture(64, 63, 64, 64),
    "o_proj_gemm_site.hlsl": _hlsl_fixture(64, 63, 64, 64),
    "gate_proj_gemm_site.hlsl": _hlsl_fixture(256, 255, 256, 256),
    "up_proj_gemm_site.hlsl": _hlsl_fixture(256, 255, 256, 256),
    "down_proj_gemm_site.hlsl": _hlsl_fixture(64, 63, 64, 64),
}


# --- strip_comments ---

def test_strip_comments_removes_line_and_block_comments():
    text = "a // numthreads(999,1,1)\nb /* block\ncomment */ c"
    out = chk.strip_comments(text)
    assert "999" not in out
    assert "a " in out and "b " in out and " c" in out


# --- parse_hlsl_thread_width ---

def test_parse_hlsl_thread_width_extracts_all_four_numbers():
    got = chk.parse_hlsl_thread_width(_hlsl_fixture(64, 63, 64, 64), label="fixture")
    assert got == (64, 63, 64, 64)


def test_parse_hlsl_thread_width_raises_when_numthreads_absent():
    try:
        chk.parse_hlsl_thread_width("int stride = ((out_channels + 63) / 64) * 64;", label="fixture")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "numthreads" in str(e)


def test_parse_hlsl_thread_width_raises_when_stride_absent():
    try:
        chk.parse_hlsl_thread_width("[numthreads(64, 1, 1)]", label="fixture")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "stride" in str(e)


# --- parse_cpp_threads_per_group_by_site ---

def test_parse_cpp_threads_per_group_by_site_attributes_grouped_case_labels():
    got = chk.parse_cpp_threads_per_group_by_site(_CPP_FIXTURE)
    assert got == {
        "QProj": 64, "OProj": 64, "DownProj": 64,
        "GateProj": 256, "UpProj": 256,
    }


# --- check_gemm_site_thread_width_parity: the mechanism ---

def test_check_passes_when_everything_agrees():
    assert chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, _HLSL_FIXTURES_CONSISTENT) == []


def test_check_reddens_on_a_numthreads_host_shader_mismatch():
    # The reviewer's own named residual, reproduced at fixture scale: the shader says 128, the host
    # (ComputeGpuGemmSiteGroupPlan) says 64 for the same site -- a real host/shader divergence.
    mismatched = dict(_HLSL_FIXTURES_CONSISTENT)
    mismatched["q_proj_gemm_site.hlsl"] = _hlsl_fixture(128, 127, 128, 128)
    failures = chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, mismatched)
    assert any("q_proj_gemm_site.hlsl" in f and "128" in f and "64" in f for f in failures), failures


def test_check_reddens_on_a_stride_formula_self_inconsistency():
    # numthreads and the host both agree at 64, but this shader's OWN stride formula uses a
    # different width internally -- a copy-paste-style defect the numthreads check alone misses.
    inconsistent = dict(_HLSL_FIXTURES_CONSISTENT)
    inconsistent["q_proj_gemm_site.hlsl"] = _hlsl_fixture(64, 255, 256, 256)
    failures = chk.check_gemm_site_thread_width_parity(_CPP_FIXTURE, inconsistent)
    assert any("self-consistent" in f for f in failures), failures


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
    mutated_text = real_text.replace("[numthreads(64, 1, 1)]", "[numthreads(128, 1, 1)]", 1)
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
