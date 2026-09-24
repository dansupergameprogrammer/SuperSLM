"""Validation of check_gpu_status_site_census.py against an independently found population
(StandardsDocument.md Sec4: a new structure is validated against a population found by someone other
than its author, before any fix lands).

THE POPULATION (plan `te421-slm172-host-oom.md` Sec3.5 R11). The sites the plan's first draft left out,
found by the plan reviewer (TE-422 S-2), the coverage auditor (TE-423 F-4, F-5) and the planner's own
census (TE-424 P-1) -- not by this check's author:
  - TE-422 S-2 site 1: InvokeGpuApiBoundary's catch clauses (T11).
  - TE-422 S-2 site 2: SubmitAdmittedChunkForG5Bridge's swallowing std::runtime_error clause (T10).
  - TE-422 S-2 site 3: RunDeviceLogits's catch clauses (T2).
  - TE-423 F-5: the two guard-refusal exclusion lists -- the final sub-chunk's in
    SubmitAdmittedChunkForG5Bridge, the non-final sub-chunks' in SubmitChunkToFullDepthForG5Bridge.
  - TE-424 P-1: RunLayerLoopGpuFinish's null-token return (a caller error returning a forward status).
Deleting any of them from the committed list must turn the check red, naming the site. Each member is
selected by what it IS (its function, and its kind or statement shape), not by line or exact text, so
the validation still finds it after the fix re-dispositions the site.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("census", HERE / "check_gpu_status_site_census.py")
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)

LIST = census.DEFAULT_LIST


def rows():
    out = []
    for line in LIST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out.append(line)
    return out


def fields(line):
    kind, rel, func, count, block, disposition, token = line.split("\t")
    return kind, rel, func, token


POPULATION = {
    "TE-422 S-2 site 1 (InvokeGpuApiBoundary, T11)":
        lambda k, f, fn, t: k == "catch" and fn.endswith("InvokeGpuApiBoundary"),
    "TE-422 S-2 site 2 (SubmitAdmittedChunkForG5Bridge's runtime_error clause, T10)":
        lambda k, f, fn, t: k == "catch" and fn.endswith("SubmitAdmittedChunkForG5Bridge") and t == "std::runtime_error",
    "TE-422 S-2 site 3 (RunDeviceLogits, T2)":
        lambda k, f, fn, t: k == "catch" and fn.endswith("RunDeviceLogits"),
    "TE-423 F-5 (final sub-chunk exclusion list)":
        lambda k, f, fn, t: k == "status" and fn.endswith("SubmitAdmittedChunkForG5Bridge"),
    "TE-423 F-5 (non-final sub-chunk exclusion list)":
        lambda k, f, fn, t: k == "status" and fn.endswith("SubmitChunkToFullDepthForG5Bridge"),
    "TE-424 P-1 (RunLayerLoopGpuFinish's null-token return)":
        lambda k, f, fn, t: k == "status" and fn.endswith("RunLayerLoopGpuFinish") and re.fullmatch(r"return \w+", t),
}


def test_must_accept_the_committed_list_matches_the_source(capsys):
    assert census.main(["--list", str(LIST)]) == 0, capsys.readouterr().out


@pytest.mark.parametrize("member", sorted(POPULATION))
def test_must_reject_each_population_member_deleted(member, tmp_path, capsys):
    pick = POPULATION[member]
    kept, dropped = [], []
    for line in rows():
        (dropped if pick(*fields(line)) else kept).append(line)
    assert dropped, f"the population member {member} is not in the list -- the validation cannot run"
    trimmed = tmp_path / "sites.txt"
    trimmed.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert census.main(["--list", str(trimmed)]) == 1
    out = capsys.readouterr().out
    for line in dropped:
        kind, rel, func, token = fields(line)
        assert f"UNLISTED {kind} site: {rel} :: {func} :: {token}" in out, out


def copy_sources(dst: Path) -> None:
    shutil.copytree(REPO / "src" / "gpu", dst / "src" / "gpu", ignore=shutil.ignore_patterns("*.hlsl*", "*.cso"))
    for rel in census.STATUS_EXTRA:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst / rel)


def test_vitality_a_new_catch_clause_is_unlisted(tmp_path, capsys):
    copy_sources(tmp_path)
    target = tmp_path / "src" / "gpu" / "gpu_1p0.cpp"
    target.write_text(target.read_text(encoding="utf-8") +
                      "\nint Te425VitalityProbe() {\n\ttry {\n\t\treturn 1;\n\t} catch (const std::bad_alloc&) {\n"
                      "\t\treturn 0;\n\t}\n}\n", encoding="utf-8")
    assert census.main(["--root", str(tmp_path), "--list", str(LIST)]) == 1
    assert "UNLISTED catch site: src/gpu/gpu_1p0.cpp :: Te425VitalityProbe :: std::bad_alloc" in capsys.readouterr().out


def test_vitality_a_new_status_site_is_unlisted(tmp_path, capsys):
    copy_sources(tmp_path)
    target = tmp_path / "src" / "gpu" / "superslm_gpu.cpp"
    target.write_text(target.read_text(encoding="utf-8") +
                      "\nsuperslm::SslmForwardStatus Te425VitalityProbe2() {\n"
                      "\treturn superslm::SslmForwardStatus::GpuAllocationFailed;\n}\n", encoding="utf-8")
    assert census.main(["--root", str(tmp_path), "--list", str(LIST)]) == 1
    assert "UNLISTED status site: src/gpu/superslm_gpu.cpp :: Te425VitalityProbe2 :: return GpuAllocationFailed" in \
        capsys.readouterr().out


def test_a_status_named_only_in_a_comment_or_string_is_not_a_site(tmp_path, capsys):
    copy_sources(tmp_path)
    target = tmp_path / "src" / "gpu" / "gpu_1p0.cpp"
    target.write_text(target.read_text(encoding="utf-8") +
                      "\n// GpuAllocationFailed in prose\n/* GpuDeviceRemoved */\n"
                      "static const char* kTe425Probe = \"GpuOperationFailed catch (int)\";\n", encoding="utf-8")
    assert census.main(["--root", str(tmp_path), "--list", str(LIST)]) == 0, capsys.readouterr().out


def test_a_row_the_source_does_not_have_is_stale(tmp_path, capsys):
    extra = "\t".join(["catch", "src/gpu/gpu_1p0.cpp", "Te425NoSuchFunction", "1", "T0", "none", "std::bad_alloc"])
    padded = tmp_path / "sites.txt"
    padded.write_text("\n".join(rows() + [extra]) + "\n", encoding="utf-8")
    assert census.main(["--list", str(padded)]) == 1
    assert "STALE catch row: src/gpu/gpu_1p0.cpp :: Te425NoSuchFunction :: std::bad_alloc" in capsys.readouterr().out


def test_a_row_without_a_disposition_is_refused(tmp_path, capsys):
    first = rows()[0].split("\t")
    first[5] = ""
    bare = tmp_path / "sites.txt"
    bare.write_text("\n".join(["\t".join(first)] + rows()[1:]) + "\n", encoding="utf-8")
    assert census.main(["--list", str(bare)]) == 1
    assert "without a block and a disposition" in capsys.readouterr().out
