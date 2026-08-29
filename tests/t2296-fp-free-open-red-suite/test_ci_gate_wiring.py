"""T-2366 (Curie), D-SLM5001 item (6) -- CI reachability for the FP-free
ship gate.

Design Sec4.1 (fold round 34, D-SLM4856) already states the law this file
pins: "a scan this design specifies and nothing in the pipeline runs is not
a gate, whatever its own verdicts say." At the time this file was authored,
`.github/workflows/tests.yml` wired only `tests/ci/run_fp_free_scan_real_corpus.py`
-- the OLD driver, retired as the gate's own mechanism at fold round 39
(D-SLM4981) -- into a job whose own last line was a hardcoded `exit 0`,
forcing the step to succeed regardless of what the scan reported (that
job's own comment: "NON-GATING, deliberately"), and `tests/ci/scan_build_output.py`,
design Sec4.1's own current production driver, was referenced nowhere
under `.github/workflows/`. T-2367 (fold round 39) closed that gap: the
`fp-free-scan-gate` job now invokes `scan_build_output.py` directly, gating
on its exit code, with no `continue-on-error` and no trailing `exit 0`.

`tools/ci/check_tests_have_build_recipe.py` (T-2314) is this repo's own
precedent for a static-text CI-wiring check (a suite directory referenced by
neither build.bat, CMakeLists.txt, nor this workflow file is orphaned). This
ticket's own writable scope is `tests/t2296-fp-free-open-red-suite/`, not
`tools/ci/`, so this check is filed here rather than there -- a sibling test
file, per this ticket's own contract ("tests/t2296-fp-free-open-red-suite/
and any new test file it needs").

Every cell below is collected by the same directory-level pytest invocation
`test_check_fp_free_scan.py` documents and build.bat now runs.

T-2371 (Brunel), D-SLM5018 O1. Poirot's review found the other half of this
same asymmetry: CI gated on scan_build_output.py but never collected THIS
suite (the one that commissions that driver), and build.bat ran this suite
but never gated locally on scan_build_output.py's own CLI/exit code -- "a
local build.bat cannot fail on the ship gate, and CI cannot fail on the
suite that commissions it." The cells below pin both halves of the closed
gap: the workflow's `fp-free-scan-gate` job now also runs this suite,
gating, reusing the build the job's own prior steps already produced; and
build.bat now also invokes scan_build_output.py against the real corpus
this suite's own `real_build_dir` fixture built, gating.
"""
from __future__ import annotations

import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_WORKFLOW = os.path.join(_ENGINE_ROOT, ".github", "workflows", "tests.yml")
_BUILD_BAT = os.path.join(_ENGINE_ROOT, "build.bat")


def _read_workflow():
    if not os.path.exists(_WORKFLOW):
        pytest.skip("workflow file not present in this checkout: {}".format(_WORKFLOW))
    with open(_WORKFLOW, encoding="utf-8") as f:
        return f.read()


def _read_build_bat():
    if not os.path.exists(_BUILD_BAT):
        pytest.skip("build.bat not present in this checkout: {}".format(_BUILD_BAT))
    with open(_BUILD_BAT, encoding="utf-8") as f:
        return f.read()


def _job_blocks(text):
    """Splits the workflow's own `jobs:` section into per-job text blocks,
    keyed by job id -- a narrow, line-indentation-based split (every job id
    in this file starts at exactly two spaces of indentation under `jobs:`,
    confirmed at source), sufficient for this file's own existing shape
    without taking on a full YAML-parser dependency this repo's own CI
    checks do not otherwise require (mirrors
    tools/ci/check_tests_have_build_recipe.py's own plain-text-scan
    precedent)."""
    lines = text.splitlines()
    jobs_start = next(i for i, l in enumerate(lines) if l.strip() == "jobs:")
    job_re = re.compile(r"^  (\S+):\s*$")
    blocks = {}
    current_id = None
    current_lines = []
    for line in lines[jobs_start + 1:]:
        m = job_re.match(line)
        if m:
            if current_id is not None:
                blocks[current_id] = "\n".join(current_lines)
            current_id = m.group(1)
            current_lines = []
        else:
            current_lines.append(line)
    if current_id is not None:
        blocks[current_id] = "\n".join(current_lines)
    return blocks


def test_workflow_invokes_the_current_production_driver():
    """The gate's own current production entry point (design Sec4.1, fold
    round 39, D-SLM4981) is `tests/ci/scan_build_output.py` -- the driver
    named in every present-tense sentence design Sec4.1/Sec5.5 write about
    "the gate." At the time this cell was authored it was absent from the
    workflow -- only the retired driver, `run_fp_free_scan_real_corpus.py`,
    was invoked; T-2367 wired it in, and this cell now confirms that stays
    true.
    """
    text = _read_workflow()
    assert "scan_build_output.py" in text, (
        "no step under .github/workflows/tests.yml invokes tests/ci/"
        "scan_build_output.py -- design Sec4.1's own current production "
        "driver (fold round 39, D-SLM4981). \"A scan this design specifies "
        "and nothing in the pipeline runs is not a gate\" (design Sec4.1, "
        "D-SLM4856), whatever its own verdicts say."
    )


def test_workflow_job_invoking_the_driver_is_not_forced_to_pass():
    """The step invoking scan_build_output.py must not sit in a job whose
    own exit is forced to succeed regardless of the scan's own verdict -- a
    hardcoded `exit 0` after the invocation, or a `continue-on-error: true`
    on the step/job -- the non-gating shape the now-retired
    fp-free-scan-report job used for the OLD driver before T-2367 removed
    that job. A CI job that always reports success is not a gate.
    """
    text = _read_workflow()
    if "scan_build_output.py" not in text:
        pytest.fail(
            "scan_build_output.py is not referenced anywhere in the "
            "workflow -- see test_workflow_invokes_the_current_production_"
            "driver for the primary failure; this cell cannot evaluate "
            "gating status for a step that does not exist."
        )
    blocks = _job_blocks(text)
    hosting_jobs = [jid for jid, body in blocks.items() if "scan_build_output.py" in body]
    assert hosting_jobs, (
        "internal inconsistency: scan_build_output.py is present in the "
        "workflow text but not inside any job block this parser found -- "
        "check _job_blocks's own two-space job-id convention against the "
        "workflow's current indentation"
    )
    for jid in hosting_jobs:
        body = blocks[jid]
        assert "continue-on-error: true" not in body, (
            "job {!r} invokes scan_build_output.py under "
            "continue-on-error: true -- its own failure can never fail the "
            "workflow, which is not a gate".format(jid)
        )
        after_driver = body[body.index("scan_build_output.py"):]
        assert not re.search(r"^\s*exit 0\s*$", after_driver, re.MULTILINE), (
            "job {!r} invokes scan_build_output.py but then unconditionally "
            "runs `exit 0` -- the step's own exit code is forced to succeed "
            "regardless of the scan's own verdict, the identical "
            "non-gating shape the fp-free-scan-report job already uses for "
            "the OLD driver".format(jid)
        )


def test_fp_free_scan_gate_job_also_collects_this_suite():
    """D-SLM5018 O1: the same job that gates on scan_build_output.py must
    also collect `tests/t2296-fp-free-open-red-suite/` -- the suite that
    commissions that driver (this file's own two cells above, the
    enumerator-completeness pin, and every real-corpus cell in
    test_check_fp_free_scan.py). Before this round, no CI job anywhere ran
    this directory; `bad-alloc-membership-check` collects `tests/ci/` only.
    """
    text = _read_workflow()
    blocks = _job_blocks(text)
    assert "fp-free-scan-gate" in blocks, (
        "internal inconsistency: no job named fp-free-scan-gate found -- "
        "check _job_blocks's own two-space job-id convention against the "
        "workflow's current indentation"
    )
    body = blocks["fp-free-scan-gate"]
    assert "t2296-fp-free-open-red-suite" in body, (
        "the fp-free-scan-gate job does not invoke pytest over "
        "tests/t2296-fp-free-open-red-suite -- CI gates on "
        "scan_build_output.py's own exit code but never runs the suite "
        "that commissions it"
    )
    after_pytest = body[body.index("t2296-fp-free-open-red-suite"):]
    assert "continue-on-error: true" not in body, (
        "the fp-free-scan-gate job runs under continue-on-error: true -- "
        "its own failure can never fail the workflow"
    )
    assert not re.search(r"^\s*exit 0\s*$", after_pytest, re.MULTILINE), (
        "the fp-free-scan-gate job's pytest step is followed by an "
        "unconditional exit 0 -- forced to succeed regardless of the "
        "suite's own verdict"
    )


def test_fp_free_scan_gate_job_pytest_step_reuses_the_jobs_own_build():
    """The pytest step added for O1 sets SUPERSLM_FP_SCAN_BUILD_DIR to the
    same build directory this job's own `cmake -B build`/`cmake --build`
    steps already produce (conftest.py's own documented override) --
    proving the suite grades the SAME build scan_build_output.py just
    graded, and that CI pays for one configure+build of the target, not
    two.
    """
    text = _read_workflow()
    blocks = _job_blocks(text)
    body = blocks.get("fp-free-scan-gate", "")
    assert "SUPERSLM_FP_SCAN_BUILD_DIR" in body, (
        "the fp-free-scan-gate job's pytest step does not set "
        "SUPERSLM_FP_SCAN_BUILD_DIR -- conftest.py's real_build_dir "
        "fixture will pay for a second configure+build of the identical "
        "superslm target instead of reusing this job's own build"
    )


def test_build_bat_also_invokes_scan_build_output_gating():
    """D-SLM5018 O1: build.bat must invoke scan_build_output.py against a
    real corpus and treat a nonzero exit as a build failure -- before this
    round, build.bat ran this suite (which builds a real corpus via
    conftest.py's own fixture) and separately ran the retired
    run_fp_free_scan_real_corpus.py driver non-gating, but never invoked
    the actual production driver CI gates on, so a local build.bat could
    pass while the real ship gate would have failed.
    """
    text = _read_build_bat()
    assert "scan_build_output.py" in text, (
        "build.bat does not invoke tests/ci/scan_build_output.py anywhere "
        "-- a local build.bat cannot fail on the ship gate CI actually runs"
    )
    after_driver = text[text.index("scan_build_output.py"):]
    assert re.search(r"exit\s*/b\s*1", after_driver), (
        "build.bat invokes scan_build_output.py but nothing after it exits "
        "the build with a nonzero code on failure -- the invocation is "
        "present but not gating"
    )
