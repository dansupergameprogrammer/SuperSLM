"""S1/S2 offline attachment contract, authored before the command exists.

Environment paths point to complete, independently verified .sslm artifacts and
their exact and same-vocabulary-size wrong tokenizer checkpoints. The runner
supplies these under D:/_artifacts/SuperSLM/t3094. Missing production inputs are
explicit red-unimplemented, never counted as a guard or tokenizer mutant kill.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from resource_pairs import DIAGNOSTICS, pair

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOL = ROOT / "tools/sslm_attach_schema.py"
SCHEMA = HERE / "command.schema.json"


def _inputs():
    assert TOOL.is_file(), "red-unimplemented: supported offline schema attachment command is absent"
    names = ("T3094_COMPLETE_ARTIFACT", "T3094_MATCHING_CHECKPOINT", "T3094_SAME_SIZE_WRONG_CHECKPOINT")
    paths = [Path(os.environ[name]) for name in names if name in os.environ]
    assert len(paths) == 3 and all(path.exists() for path in paths), (
        "red-unimplemented: complete artifact and matched/mismatched tokenizer fixtures are not installed")
    return paths


def _run(base, checkpoint, schema, output, *extra):
    return subprocess.run([sys.executable, str(TOOL), "--base", str(base), "--ckpt", str(checkpoint),
                           "--schema", f"commands:{schema}", "--out", str(output), *extra],
                          text=True, capture_output=True, timeout=120)


def test_attachment_emits_named_schema_and_preserves_base(tmp_path):
    base, matching, _wrong = _inputs()
    output = tmp_path / "attached.sslm"
    result = _run(base, matching, SCHEMA, output)
    assert result.returncode == 0, result.stderr
    assert output.is_file() and output.stat().st_size > base.stat().st_size
    # The independent loader/section verifier in S2 checks every carried
    # section/flag byte, SCM1, map/bind/generate and both backends. File size
    # alone is only the S1 authoring-surface red cell.


def test_same_size_wrong_tokenizer_refuses_without_output(tmp_path):
    base, _matching, wrong = _inputs()
    output = tmp_path / "wrong.sslm"
    result = _run(base, wrong, SCHEMA, output)
    assert result.returncode != 0 and "tokenizer" in result.stderr.lower()
    assert not output.exists(), "refusal left a partial artifact"


@pytest.mark.parametrize("bad_name", ("", "commands"))
def test_duplicate_or_zero_hash_name_refuses_without_output(tmp_path, bad_name):
    base, matching, _wrong = _inputs()
    output = tmp_path / "names.sslm"
    args = ("--schema", f"{bad_name}:{SCHEMA}")
    result = _run(base, matching, SCHEMA, output, *args)
    assert result.returncode != 0 and "name" in result.stderr.lower()
    assert not output.exists(), "refusal left a partial artifact"


@pytest.mark.parametrize("guard", tuple(DIAGNOSTICS))
def test_isolated_resource_refusal_leaves_no_partial_artifact(tmp_path, guard):
    base, matching, _wrong = _inputs()
    under, over, below, above = pair(guard)
    assert below[guard] < above[guard]
    limits = {name: 2**63 - 1 for name in DIAGNOSTICS}
    limits[guard] = below[guard]
    limit_file = tmp_path / "limits.json"
    limit_file.write_text(json.dumps(limits), encoding="utf-8")
    under_schema, over_schema = tmp_path / "under.json", tmp_path / "over.json"
    under_schema.write_text(json.dumps(under), encoding="utf-8")
    over_schema.write_text(json.dumps(over), encoding="utf-8")
    healthy = tmp_path / "healthy.sslm"
    good = _run(base, matching, under_schema, healthy, "--limits-json", str(limit_file))
    assert good.returncode == 0 and healthy.exists(), good.stderr
    refused = tmp_path / "refused.sslm"
    bad = _run(base, matching, over_schema, refused, "--limits-json", str(limit_file))
    assert bad.returncode != 0 and DIAGNOSTICS[guard].lower() in bad.stderr.lower()
    assert not refused.exists(), f"{guard} refusal left a partial artifact"
