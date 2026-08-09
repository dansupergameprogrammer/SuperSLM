"""t1872_probe.common -- shared helpers for the T-1872 portable feasibility probe.

Every stage script is a standalone process (see orchestrator.py's design note for why:
a hang inside torch/HIP is only killable from outside the hung process). This module
holds the small amount of code every stage shares: locating the bundle root and the
local model directory relative to THIS script's own location (never a hardcoded drive
letter, so the bundle runs from F:, E:, or any other drive Windows assigns it), and a
uniform JSON-line result format each stage prints on its last line of stdout.
"""
import json
import sys
from pathlib import Path


def bundle_root() -> Path:
    """The bundle's own root directory, wherever it was copied to."""
    # tools/t1872_probe/common.py -> bundle_root/probe/common.py at deploy time
    # (see assemble scripts: this file is copied to <bundle>/probe/common.py)
    return Path(__file__).resolve().parent.parent


def model_dir() -> Path:
    return bundle_root() / "model"


def prompts_path() -> Path:
    return Path(__file__).resolve().parent / "prompts.jsonl"


def load_prompts(limit: int | None = None) -> list[dict]:
    out = []
    with open(prompts_path(), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def emit(result: dict) -> None:
    """Print the stage's result as the LAST line of stdout, always valid JSON.

    The orchestrator parses only the last line; everything printed before it is
    human-readable progress that a stage is free to emit for Dan's benefit while
    it runs, without breaking the machine-readable contract.
    """
    print("RESULT_JSON:" + json.dumps(result))
    sys.stdout.flush()


def fail(stage: str, reason: str, detail: dict | None = None) -> dict:
    d = detail or {}
    d["reason"] = reason
    return {"stage": stage, "pass": False, "detail": d}


def ok(stage: str, detail: dict | None = None) -> dict:
    return {"stage": stage, "pass": True, "detail": detail or {}}
