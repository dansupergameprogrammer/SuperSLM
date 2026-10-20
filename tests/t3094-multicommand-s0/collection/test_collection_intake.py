"""T-3103 collection intake cells. Mechanism controls pass now; the frozen-collection cell is
red until independent writers submit and the condition revision (D-SLM8075) lands."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collection_intake  # noqa: E402

SUBMISSIONS = HERE / "submissions"


def _row(i, **over):
    row = {"row_id": f"r{i}", "utterance": f"bravo hold the ford number {i}", "writer_id": "w1",
           "session_id": f"s{i // 10}", "speaker_id": f"p{i // 10}", "source_kind": "synthetic_writer",
           "commission": "parallel", "writer_flag": False, "writer_note": ""}
    row.update(over)
    return row


def _write(tmp_path, rows, name="w.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_intake_accepts_clean_rows(tmp_path):
    assert collection_intake.intake([_write(tmp_path, [_row(i) for i in range(12)])], quotas=False) == []


@pytest.mark.parametrize("mutation", ["label_key", "missing_key", "dup_text", "flagged", "split_session",
                                      "two_speakers", "bad_commission", "empty_utterance"])
def test_intake_refuses(tmp_path, mutation):
    rows = [_row(i) for i in range(12)]
    if mutation == "label_key":
        rows[0]["condition"] = "beacon lit"
    elif mutation == "missing_key":
        del rows[0]["speaker_id"]
    elif mutation == "dup_text":
        rows[1]["utterance"] = "  Bravo HOLD the ford number 0 "
    elif mutation == "flagged":
        rows[2]["writer_flag"] = True
    elif mutation == "split_session":
        rows[3]["commission"] = "development"
    elif mutation == "two_speakers":
        rows[4]["speaker_id"] = "other"
    elif mutation == "bad_commission":
        rows[5]["commission"] = "pc2_new_watch"
    elif mutation == "empty_utterance":
        rows[6]["utterance"] = "   "
    assert collection_intake.intake([_write(tmp_path, rows)], quotas=False)


def test_quota_mode_refuses_shortfall(tmp_path):
    errors = collection_intake.intake([_write(tmp_path, [_row(i) for i in range(12)])], quotas=True)
    assert any(e.startswith("quota") for e in errors)


def test_collected_population_meets_quotas_red():
    files = sorted(SUBMISSIONS.glob("*.jsonl")) if SUBMISSIONS.is_dir() else []
    assert files, "red-owed: no independent writer submission yet (collection-spec.md §4)"
    assert collection_intake.intake(files, quotas=True) == []
