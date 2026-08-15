"""check_gemm_site_thread_width_parity.py -- the reviewer's own named residual (code review
6d9e04e-t2101-gpu-throughput-review.md, second confirmation pass, "the reviewer's residual"): the
shader half of the original S3 class my own prescribed remedy did not reach.

T-2101's own multi-group GEMM fix (build log Sec34) made the HOST side of the group-arithmetic a
single source of truth: `ComputeGpuGemmSiteGroupPlan` (`src/gpu/superslm_gpu.cpp`) is the ONE place
that decides how many threads per group each of the five split GEMM sites launches, and
`RunLayerLoopGpu`'s own dispatch call reads that value directly. But the SHADER side of the same
fact -- each `*_gemm_site.hlsl`'s own `[numthreads(N, 1, 1)]` attribute, and its own stride formula
`((out_channels + N - 1) / N) * N` -- is still five hand-written, independent copies of the SAME
number, with nothing that checks them against the host's own value or against each other. A shader
whose `numthreads` diverges from what the host computes its dispatch grid size against produces a
silent wrong answer at real dimensions: some output channels get computed by two threads (redundant,
survivable) or by none at all (silently wrong, exactly the class T-2101's own standing runtime guard
exists to catch on the HOST side -- the guard cannot see a shader-side divergence at all, since it
checks the host's own arithmetic, never what the shader actually declares).

This module closes it the same way `check_gpu_guard_status_parity.py` (this directory) closes the
CPU/GPU/`.def` three-way guard-status parity: read all THREE sources at their own real anchors (the
five `.hlsl` files' own `numthreads` and stride constants; `superslm_gpu.cpp`'s own
`ComputeGpuGemmSiteGroupPlan` switch), never re-derive a fourth number by hand, and fail the build on
any disagreement.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

SUPERSLM_GPU_CPP = os.path.join(_REPO_ROOT, "src", "gpu", "superslm_gpu.cpp")
SHADERS_DIR = os.path.join(_REPO_ROOT, "src", "gpu", "shaders")

# The five split GEMM sites, mapped to their own `GpuGemmSplitSite` enumerator name (as it appears
# in `superslm_gpu.cpp`'s own switch) and their own `.hlsl` file name -- the SAME five names
# `ComputeGpuGemmSiteGroupPlan`'s own header comment and `superslm_gpu.cpp`'s dispatch loop use.
GEMM_SPLIT_SITES: tuple[tuple[str, str], ...] = (
    ("QProj", "q_proj_gemm_site.hlsl"),
    ("OProj", "o_proj_gemm_site.hlsl"),
    ("GateProj", "gate_proj_gemm_site.hlsl"),
    ("UpProj", "up_proj_gemm_site.hlsl"),
    ("DownProj", "down_proj_gemm_site.hlsl"),
)


def strip_comments(text: str) -> str:
    """// line comments and /* block */ comments removed, strings left alone -- shared shape with
    `check_gpu_guard_status_parity.py`'s own `strip_comments`, reimplemented here rather than
    imported so this module has no import-time dependency on that one (the two check different
    things and should be able to fail, or be run, independently)."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        two = text[i:i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if two == "/*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        c = text[i]
        if c == '"' or c == "'":
            quote = c
            out.append(c)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


_NUMTHREADS_RE = re.compile(r"\[numthreads\(\s*(\d+)\s*,\s*1\s*,\s*1\s*\)\]")
_STRIDE_RE = re.compile(
    r"int\s+stride\s*=\s*\(\(\s*out_channels\s*\+\s*(\d+)\s*\)\s*/\s*(\d+)\s*\)\s*\*\s*(\d+)\s*;"
)


def parse_hlsl_thread_width(hlsl_text: str, *, label: str) -> tuple[int, int, int, int]:
    """`(numthreads, stride_add, stride_div, stride_mul)` parsed directly from a `*_gemm_site.hlsl`
    file's own text (comments stripped first) -- `numthreads` from its `[numthreads(N, 1, 1)]`
    attribute, the other three from its own `int stride = ((out_channels + A) / D) * M;` line
    (`stride_add`/`stride_div`/`stride_mul` respectively). Raises `ValueError` if either anchor is
    missing -- a shader that has been restructured so this module cannot find its own thread-width
    declaration must fail loudly, not report a stale or default number."""
    stripped = strip_comments(hlsl_text)
    nt_m = _NUMTHREADS_RE.search(stripped)
    if nt_m is None:
        raise ValueError(f"{label}: no [numthreads(N, 1, 1)] attribute found")
    stride_m = _STRIDE_RE.search(stripped)
    if stride_m is None:
        raise ValueError(f"{label}: no 'int stride = ((out_channels + A) / D) * M;' line found")
    return (
        int(nt_m.group(1)),
        int(stride_m.group(1)),
        int(stride_m.group(2)),
        int(stride_m.group(3)),
    )


# Matches one `case GpuGemmSplitSite::<Name>:` label -- one or more of these, consecutive, share
# the SAME `plan.out_channels =`/`plan.threads_per_group =` pair that follows the last one, up to
# the next `break;`. Mirrors the real C++ switch's own fallthrough-by-design shape
# (`ComputeGpuGemmSiteGroupPlan`, `superslm_gpu.cpp`) -- QProj/OProj/DownProj share one pair,
# GateProj/UpProj share the other.
_CASE_LABEL_RE = re.compile(r"case\s+GpuGemmSplitSite::(\w+)\s*:")
_THREADS_PER_GROUP_RE = re.compile(r"plan\.threads_per_group\s*=\s*(\d+)u?\s*;")


def extract_compute_gpu_gemm_site_group_plan_switch(cpp_text: str) -> str:
    """The `switch (site) { ... }` body of `ComputeGpuGemmSiteGroupPlan` (`superslm_gpu.cpp`),
    found by locating the function's own signature and the FIRST `switch (site) {` inside it, then
    real brace-depth counting to that switch's own matching close -- not the whole function body,
    so a change elsewhere in the function (the post-switch validation, the return) cannot be
    mistaken for a case label. Raises `ValueError` if the function or its switch cannot be found."""
    sig = "GpuGemmSiteGroupPlan ComputeGpuGemmSiteGroupPlan(GpuGemmSplitSite site,"
    fn_at = cpp_text.find(sig)
    if fn_at == -1:
        raise ValueError(f"superslm_gpu.cpp: signature not found: {sig!r}")
    switch_at = cpp_text.find("switch (site) {", fn_at)
    if switch_at == -1:
        raise ValueError("superslm_gpu.cpp: 'switch (site) {' not found inside ComputeGpuGemmSiteGroupPlan")
    open_brace = cpp_text.find("{", switch_at)
    depth = 0
    i = open_brace
    n = len(cpp_text)
    while i < n:
        c = cpp_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return cpp_text[open_brace + 1:i]
        i += 1
    raise ValueError("superslm_gpu.cpp: ComputeGpuGemmSiteGroupPlan's own switch has no matching close brace")


def parse_cpp_threads_per_group_by_site(cpp_text: str) -> dict[str, int]:
    """`{site_name: threads_per_group}` for every `GpuGemmSplitSite` enumerator
    `ComputeGpuGemmSiteGroupPlan`'s own switch names, parsed directly from its real switch body
    (`extract_compute_gpu_gemm_site_group_plan_switch`) -- every `case` label between one
    `plan.threads_per_group = N;` assignment and the next is attributed that same N, matching C++'s
    own fallthrough semantics for grouped case labels. Comments stripped first."""
    switch_body = strip_comments(extract_compute_gpu_gemm_site_group_plan_switch(cpp_text))
    result: dict[str, int] = {}
    pending_sites: list[str] = []
    pos = 0
    # Walk case labels and threads_per_group assignments in source order, attributing each
    # pending group of case labels to the NEXT threads_per_group assignment found after it.
    tokens: list[tuple[int, str, str]] = []
    for m in _CASE_LABEL_RE.finditer(switch_body):
        tokens.append((m.start(), "case", m.group(1)))
    for m in _THREADS_PER_GROUP_RE.finditer(switch_body):
        tokens.append((m.start(), "assign", m.group(1)))
    tokens.sort(key=lambda t: t[0])
    for _, kind, value in tokens:
        if kind == "case":
            pending_sites.append(value)
        else:
            for site in pending_sites:
                result[site] = int(value)
            pending_sites = []
    return result


def check_gemm_site_thread_width_parity(
    cpp_text: str,
    hlsl_texts_by_filename: dict[str, str],
    *,
    sites: tuple[tuple[str, str], ...] = GEMM_SPLIT_SITES,
) -> list[str]:
    """[] iff, for every one of the five split GEMM sites: (1) the shader's own `numthreads` equals
    the host's own `ComputeGpuGemmSiteGroupPlan`-derived `threads_per_group` for that site, and
    (2) the shader's own stride formula is internally self-consistent with its own `numthreads`
    (`stride_add + 1 == stride_div == stride_mul == numthreads`) -- the ceiling-division shape
    `ComputeGpuGemmGroupCount` uses host-side, reproduced independently in HLSL since a shader
    cannot call the host's own function."""
    cpp_by_site = parse_cpp_threads_per_group_by_site(cpp_text)
    failures: list[str] = []
    for site_name, hlsl_filename in sites:
        if site_name not in cpp_by_site:
            failures.append(
                f"ComputeGpuGemmSiteGroupPlan's own switch (superslm_gpu.cpp) has no "
                f"threads_per_group assignment reachable from case GpuGemmSplitSite::{site_name}"
            )
            continue
        if hlsl_filename not in hlsl_texts_by_filename:
            failures.append(f"{hlsl_filename}: not found among the scanned shader files")
            continue
        try:
            numthreads, stride_add, stride_div, stride_mul = parse_hlsl_thread_width(
                hlsl_texts_by_filename[hlsl_filename], label=hlsl_filename
            )
        except ValueError as e:
            failures.append(str(e))
            continue
        cpp_threads_per_group = cpp_by_site[site_name]
        if numthreads != cpp_threads_per_group:
            failures.append(
                f"{hlsl_filename}: [numthreads({numthreads}, 1, 1)] does not match "
                f"ComputeGpuGemmSiteGroupPlan's own threads_per_group={cpp_threads_per_group} for "
                f"GpuGemmSplitSite::{site_name} -- the host's own Dispatch(groups, 1, 1) grid size "
                f"and this shader's own per-group thread count would disagree at real dimensions"
            )
        if stride_add + 1 != numthreads or stride_div != numthreads or stride_mul != numthreads:
            failures.append(
                f"{hlsl_filename}: its own stride formula '((out_channels + {stride_add}) / "
                f"{stride_div}) * {stride_mul}' is not self-consistent with its own "
                f"[numthreads({numthreads}, 1, 1)] -- expected "
                f"'((out_channels + {numthreads - 1}) / {numthreads}) * {numthreads}'"
            )
    return failures


def run_all_checks(
    *,
    superslm_gpu_cpp_path: str = SUPERSLM_GPU_CPP,
    shaders_dir: str = SHADERS_DIR,
    sites: tuple[tuple[str, str], ...] = GEMM_SPLIT_SITES,
) -> list[str]:
    try:
        with open(superslm_gpu_cpp_path, "r", encoding="utf-8") as f:
            cpp_text = f.read()
    except OSError as e:
        return [f"cannot read {superslm_gpu_cpp_path}: {e}"]
    hlsl_texts: dict[str, str] = {}
    for _site_name, hlsl_filename in sites:
        path = os.path.join(shaders_dir, hlsl_filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                hlsl_texts[hlsl_filename] = f.read()
        except OSError as e:
            return [f"cannot read {path}: {e}"]
    try:
        return check_gemm_site_thread_width_parity(cpp_text, hlsl_texts, sites=sites)
    except ValueError as e:
        return [f"parity check could not run: {e}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    failures = run_all_checks()
    if failures:
        print("check_gemm_site_thread_width_parity.py: FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "check_gemm_site_thread_width_parity.py: OK -- every split GEMM site's own [numthreads] "
        "matches ComputeGpuGemmSiteGroupPlan's own threads_per_group, and every stride formula is "
        "self-consistent with its own numthreads."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
