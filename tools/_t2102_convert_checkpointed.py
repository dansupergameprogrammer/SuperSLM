"""T-2102: a checkpointed wrapper around tools/sslm_convert_adapter.py's own main() logic.

The plain CLI (`python sslm_convert_adapter.py --adapter ... --base ... --out ...`) was killed
twice by the harness's background-task lifetime limit (~55-60 minutes), each time partway through
the real 196-pair sweep (once at layer18/28). `build_runtime_additive_sections` already accepts a
`checkpoint_path` parameter (its own docstring: built for exactly this -- "a real 196-pair sweep
was killed mid-run by an external process-management event, not a crash"); the CLI's own argparse
just never exposes it. This wrapper calls the SAME function the CLI calls, with checkpointing wired,
and reproduces the CLI's own dispatch/write logic afterward -- no algorithm, threshold, or gate is
changed; only the resumability the function already supported is turned on.
"""
import os
import sys
import time
from pathlib import Path

BUILD_TOOLS = Path(r"D:\SuperSLM\.worktrees\t2021-build\tools")
sys.path.insert(0, str(BUILD_TOOLS))
import sslm_convert_adapter as A  # noqa: E402

ADAPTER_DIR = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v2\final")
BASE_SSLM = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-instruct.sslm")
OUT_PATH = Path(r"D:\hf_cache\superslm_artifacts\qwen2.5-1.5b-shopkeeper-lora-v2-t2102-runtime.sslm")
# T-2104 (Poirot 8e07d0c review, Minor 5): write to a temp path and rename on success -- the
# ORIGINAL script unlinked OUT_PATH before an hour of compute that had already been killed twice by
# the harness's own background-task limit, which meant a THIRD kill mid-write could have left the
# destination worse than it found it (a deleted-but-not-yet-replaced artifact). Nothing was actually
# lost by this build (the name was new), but the ordering was backwards regardless. `os.replace` is
# atomic on the same filesystem, so the destination is either the old file (untouched) or the
# complete new one, never a partial write or a gap.
OUT_TMP_PATH = OUT_PATH.with_name(OUT_PATH.name + ".tmp")
CHECKPOINT_PATH = Path(__file__).parent / "_t2102_convert_checkpoint.jsonl"
PROGRESS_PATH = Path(__file__).parent / "t2102_convert_progress.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    t0 = time.time()
    log("T-2102 checkpointed real conversion starting")

    sections, verdict, _round_trip = A.build_runtime_additive_sections(
        ADAPTER_DIR, BASE_SSLM, checkpoint_path=CHECKPOINT_PATH, verbose=True)

    log(f"build_runtime_additive_sections done in {time.time() - t0:.0f}s: "
        f"domain_trip={verdict['domain_trip']} margin_exceeded={verdict['margin_exceeded']} "
        f"saturation_elevated={verdict['saturation_elevated']}")

    branch, outcome = A.dispatch_conversion_outcome(
        domain_trip=verdict["domain_trip"], margin_exceeded=verdict["margin_exceeded"],
        saturation_elevated=verdict["saturation_elevated"], fallback_flag_present=False)

    if outcome == A.ArtifactOutcome.RUNTIME_ADDITIVE:
        if OUT_TMP_PATH.exists():
            OUT_TMP_PATH.unlink()
        fingerprint = A.sf.write_artifact(str(OUT_TMP_PATH), sections)
        os.replace(OUT_TMP_PATH, OUT_PATH)  # atomic same-filesystem rename -- no partial destination
        log(f"wrote {OUT_PATH}")
        log(f"fingerprint {fingerprint}")
        log(f"sections {len(sections)}: " + ", ".join(str(s.type) for s in sections))
    else:
        log(f"REJECTED: {branch.name} -- no runtime-additive artifact emitted "
            f"(domain_trip={verdict['domain_trip']} margin_exceeded={verdict['margin_exceeded']} "
            f"saturation_elevated={verdict['saturation_elevated']})")
        if verdict["margin_exceeded"] and verdict["pooled"] is not None:
            p = verdict["pooled"]
            log(f"  pooled B3 gate (design Sec25.5, whole-adapter, n_pairs={p['n_pairs']}): "
                f"composed_mean_accepts={p['composed_mean_accepts']} "
                f"composed_tail_accepts={p['composed_tail_accepts']} "
                f"effect_mean_accepts={p['effect_mean_accepts']} "
                f"effect_tail_accepts={p['effect_tail_accepts']}")
            flagged = [d["name"] for d in p["per_pair_diagnostics"] if d["flagged"]]
            log(f"  {len(flagged)} pair(s) flagged for review (diagnostic only, never gates)")

    log(f"DONE in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
