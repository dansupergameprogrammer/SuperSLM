#!/usr/bin/env python3
"""T-1810 -- re-grade the SLM (decode) calibration oracle's engine-int8 arm
against the float32 reference instead of the filed bfloat16 reference.

WHAT THIS DOES. Re-scores data ALREADY ON DISK -- no re-generation, no GPU.
The engine's (int8, `qwen2.5-1.5b-instruct.sslm`, "healthy_maxabs_baseline"
arm) per-position output token ids are read from T-1800's own filed baseline
(`D:\\SuperSLM\\.worktrees\\t1800-dual-oracle-baseline\\out\\t1800\\
decode_agreement_healthy.json`). The float32 reference's per-position output
token ids for the SAME 32-prompt population, SAME max_new=8, are read from
T-1800's own bf16-vs-fp32 self-consistency floor run
(`...\\out\\t1800\\bf16_fp32_floor.json`, `fp32_output_ids` field per
prompt) -- that script already ran `float_reference_generate.py --dtype
float32` over this exact population as part of measuring the reference's
own noise floor, so the float32 generation this ticket needs already
exists, confirmed identical in provenance (same 32 questions, same order,
same max_new=8) and byte-identical on the bf16 side to
`decode_agreement_healthy.json`'s own bf16 arm (cross-checked: 0/32
mismatches), which establishes the float reference generation is
reproducible run-to-run at this project's determinism floor before this
script trusts a comparison across the two files.

Uses the SAME per-position agreement definition and SAME Wilson 95% CI
half-width construction as `t1800_decode_agreement.py`
(StandardsDocument.md 5.4).

This script performs no generation and touches no GPU; it reads two
existing JSON files and recomputes an aggregate over them.

Usage:
    python tools\\t1810_slm_engine_vs_fp32.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

T1800_DIR = Path(r"D:\SuperSLM\.worktrees\t1800-dual-oracle-baseline\out\t1800")
HEALTHY_PATH = T1800_DIR / "decode_agreement_healthy.json"
FLOOR_PATH = T1800_DIR / "bf16_fp32_floor.json"


def wilson_halfwidth(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (hi - lo) / 2


def main() -> int:
    healthy = json.loads(HEALTHY_PATH.read_text())
    floor = json.loads(FLOOR_PATH.read_text())

    hp = healthy["per_prompt"]
    fp = floor["per_prompt"]
    assert len(hp) == len(fp) == 32, f"expected 32 prompts each, got {len(hp)}/{len(fp)}"
    assert [p["question"] for p in hp] == [p["question"] for p in fp], \
        "prompt order mismatch between decode_agreement_healthy.json and bf16_fp32_floor.json"

    # Provenance cross-check: the bf16 arm recorded in decode_agreement_healthy.json
    # must match the bf16 arm recorded in bf16_fp32_floor.json byte-for-byte -- both
    # are the same reference, same dtype, same prompts, run in two separate sessions.
    bf16_mismatches = sum(
        1 for i in range(32) if hp[i]["float_output_ids"] != fp[i]["bf16_output_ids"]
    )
    if bf16_mismatches:
        raise SystemExit(
            f"REFUSING: {bf16_mismatches}/32 bf16 cross-check mismatches between "
            f"{HEALTHY_PATH} and {FLOOR_PATH} -- the two files' bf16 arms disagree, "
            f"so the fp32 arm's provenance against the int8 arm's session is not "
            f"established as comparable"
        )

    total_positions = 0
    total_agree = 0
    full_sequence_matches = 0
    per_prompt_out = []

    for i in range(32):
        int8_ids = hp[i]["int8_output_ids"]
        fp32_ids = fp[i]["fp32_output_ids"]
        n = min(len(int8_ids), len(fp32_ids))
        n_agree = sum(1 for p in range(n) if int8_ids[p] == fp32_ids[p])
        total_positions += n
        total_agree += n_agree
        seq_match = (int8_ids == fp32_ids)
        full_sequence_matches += int(seq_match)
        per_prompt_out.append({
            "index": i,
            "question": hp[i]["question"],
            "int8_output_ids": int8_ids,
            "fp32_output_ids": fp32_ids,
            "n_positions_compared": n,
            "n_agree": n_agree,
            "full_sequence_match": seq_match,
        })

    rate = total_agree / total_positions if total_positions else float("nan")
    hw = wilson_halfwidth(total_agree, total_positions)
    seq_rate = full_sequence_matches / 32

    print("=" * 100)
    print("T-1810 SLM-side re-grade: engine (int8) vs FLOAT32 reference")
    print(f"source (engine arm):      {HEALTHY_PATH}")
    print(f"source (fp32 reference):  {FLOOR_PATH}")
    print(f"bf16 cross-check mismatches: {bf16_mismatches}/32 (0 required to trust this comparison)")
    print("=" * 100)
    print(f"cell: 32 prompts x max_new=8, positions compared N={total_positions}, "
          f"model={healthy['model']}, reference=float32 Qwen2.5-1.5B-Instruct greedy decode")
    print(f"per-position token agreement: {total_agree}/{total_positions} = {rate:.4f} "
          f"(95% Wilson half-width = {hw:.4f})")
    print(f"full-sequence exact match: {full_sequence_matches}/32 = {seq_rate:.4f}")

    result = {
        "label": healthy["label"] + "_vs_fp32",
        "model": healthy["model"],
        "reference_model": healthy["reference_model"],
        "reference_dtype": "float32 (torch_dtype=float32, resolved_dtype=" + floor["resolved_dtype_fp32_side"] + ")",
        "max_new": 8,
        "n_prompts": 32,
        "total_positions": total_positions,
        "total_agree": total_agree,
        "agreement_rate": rate,
        "wilson_95_halfwidth": hw,
        "full_sequence_match_count": full_sequence_matches,
        "full_sequence_match_rate": seq_rate,
        "bf16_crosscheck_mismatches": bf16_mismatches,
        "per_prompt": per_prompt_out,
    }
    out_path = Path(__file__).resolve().parent.parent / "out" / "t1810" / "slm_engine_vs_fp32.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nfull result set written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
