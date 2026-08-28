"""T-2366 (Curie), D-SLM5001 item (6) -- CI reachability for the FP-free
ship gate.

Design Sec4.1 (fold round 34, D-SLM4856) already states the law this file
pins: "a scan this design specifies and nothing in the pipeline runs is not
a gate, whatever its own verdicts say." `.github/workflows/tests.yml` today
wires `tests/ci/run_fp_free_scan_real_corpus.py` -- the OLD driver, retired
as the gate's own mechanism at fold round 39 (D-SLM4981) -- into a job whose
own last line is a hardcoded `exit 0`, forcing the step to succeed
regardless of what the scan reports (that job's own comment: "NON-GATING,
deliberately"). `tests/ci/scan_build_output.py`, design Sec4.1's own current
production driver, is referenced nowhere under `.github/workflows/`.

`tools/ci/check_tests_have_build_recipe.py` (T-2314) is this repo's own
precedent for a static-text CI-wiring check (a suite directory referenced by
neither build.bat, CMakeLists.txt, nor this workflow file is orphaned). This
ticket's own writable scope is `tests/t2296-fp-free-open-red-suite/`, not
`tools/ci/`, so this check is filed here rather than there -- a sibling test
file, per this ticket's own contract ("tests/t2296-fp-free-open-red-suite/
and any new test file it needs").

Both cells below are collected by the same directory-level pytest invocation
`test_check_fp_free_scan.py` documents and build.bat now runs.
"""
from __future__ import annotations

import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE_ROOT = os.path.dirname(os.path.dirname(_HERE))
_WORKFLOW = os.path.join(_ENGINE_ROOT, ".github", "workflows", "tests.yml")


def _read_workflow():
    if not os.path.exists(_WORKFLOW):
        pytest.skip("workflow file not present in this checkout: {}".format(_WORKFLOW))
    with open(_WORKFLOW, encoding="utf-8") as f:
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
    "the gate." Confirmed absent from the workflow today: only the retired
    driver, `run_fp_free_scan_real_corpus.py`, is invoked.
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
    on the step/job -- the exact non-gating shape the fp-free-scan-report
    job uses today for the OLD driver. A CI job that always reports success
    is not a gate.
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
