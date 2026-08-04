#!/usr/bin/env python3
"""T-1704 Part 2 -- the invariance obligation for all-position site-dump
capture, executed (not argued).

Runs `tools/sslm_layer_trace.exe` three times per prompt over the campaign's
own established 9-prompt population (`kv_saturation_report.POPULATION`):

  1. capture off        -- no --site-dump at all
  2. last-token capture  -- --site-dump (default, unchanged since T-1691)
  3. all-position capture -- --site-dump --site-dump-all-positions (T-1704)

and confirms the `--dump` file (the 29-row hidden-state body PLUS the T-1689/
T-1691 trailers -- the tool's own record of what the engine actually
computed, independent of whether/how any site-dump was captured) is
byte-identical across all three, for every prompt. `--dump`'s own body never
depends on `site_ctx`/site-dump capture at all (WriteDump's own arguments
carry no site-dump state), so this is a direct, executed check that adding
capture -- at any granularity -- does not perturb what the engine computes,
not an inference from the hook's "read-only by construction" contract.

Also runs the T-1704 fault-injection instrument (--corrupt-site-hook,
requires --site-dump-all-positions): confirms the tool's own self-check
(production vs. manual-replay, unconditional on every invocation) FAILS when
a prefill-position record is deliberately corrupted -- proving the
invariance check has power specifically at the NEW prefill-capture path, not
merely re-using power already proven for the last-token path.

Usage: python tools\\t1704_invariance_check.py
"""
from __future__ import annotations

import filecmp
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_saturation_report as KSR  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DRIVER_EXE = os.path.join(REPO_ROOT, "out", "sslm_layer_trace.exe")
OUT_DIR = os.path.join(REPO_ROOT, "out", "t1704_invariance")


def run(args, timeout=600):
    proc = subprocess.run([str(DRIVER_EXE), str(KSR.MODEL_PATH), str(KSR.TOKENIZER_PATH)] + args,
                           capture_output=True, text=True, timeout=timeout)
    return proc


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isfile(DRIVER_EXE):
        print(f"FAILED: {DRIVER_EXE} not built", file=sys.stderr)
        return 1

    all_match = True
    rows = []
    for label, question, role in KSR.POPULATION:
        prompt = KSR.build_prompt(question)
        dump_off = os.path.join(OUT_DIR, f"{label}.off.dump")
        dump_last = os.path.join(OUT_DIR, f"{label}.last.dump")
        dump_all = os.path.join(OUT_DIR, f"{label}.all.dump")
        site_last = os.path.join(OUT_DIR, f"{label}.last.sitedump")
        site_all = os.path.join(OUT_DIR, f"{label}.all.sitedump")

        p_off = run([prompt, "--dump", dump_off])
        p_last = run([prompt, "--dump", dump_last, "--site-dump", site_last])
        p_all = run([prompt, "--dump", dump_all, "--site-dump", site_all,
                     "--site-dump-all-positions"])

        for tag, p in (("off", p_off), ("last", p_last), ("all", p_all)):
            if p.returncode != 0:
                print(f"FAILED: prompt={label} mode={tag} exit={p.returncode}\n{p.stdout}\n{p.stderr}",
                      file=sys.stderr)
                return 1
            if "self_check: production and manual-replay paths agree" not in p.stdout:
                print(f"FAILED: prompt={label} mode={tag} self_check line missing", file=sys.stderr)
                return 1

        off_vs_last = filecmp.cmp(dump_off, dump_last, shallow=False)
        off_vs_all = filecmp.cmp(dump_off, dump_all, shallow=False)
        last_vs_all = filecmp.cmp(dump_last, dump_all, shallow=False)
        ok = off_vs_last and off_vs_all and last_vs_all
        all_match = all_match and ok
        rows.append((label, role, off_vs_last, off_vs_all, last_vs_all,
                      os.path.getsize(dump_off), os.path.getsize(site_last), os.path.getsize(site_all)))
        print(f"{label:24s} role={role:8s} off==last={off_vs_last} off==all={off_vs_all} "
              f"last==all={last_vs_all}  dump_bytes={os.path.getsize(dump_off)} "
              f"sitedump_last_bytes={os.path.getsize(site_last)} "
              f"sitedump_all_bytes={os.path.getsize(site_all)}")

    print()
    if all_match:
        print("RESULT: --dump body byte-identical across capture-off / last-token-capture / "
              "all-position-capture, all 9 prompts. Adding capture (any granularity) does not "
              "perturb the computed decode.")
    else:
        print("RESULT: MISMATCH -- capture mode changed the computed decode. STOP.", file=sys.stderr)
        return 1

    # --- Fault injection: prove the check can fail, specifically at the new
    # prefill-capture path. ---------------------------------------------------
    label, question, role = KSR.POPULATION[0]
    prompt = KSR.build_prompt(question)
    corrupt_dump = os.path.join(OUT_DIR, "corrupt_probe.dump")
    corrupt_site = os.path.join(OUT_DIR, "corrupt_probe.sitedump")
    p_corrupt = run([prompt, "--dump", corrupt_dump, "--site-dump", corrupt_site,
                     "--site-dump-all-positions", "--corrupt-site-hook"])
    print()
    print(f"--corrupt-site-hook probe (prompt={label}): exit={p_corrupt.returncode}")
    print(p_corrupt.stdout[-800:])
    print(p_corrupt.stderr[-800:])
    if p_corrupt.returncode == 0:
        print("FAILED: --corrupt-site-hook probe exited 0 -- the corruption did not reach the "
              "self-check; the invariance check's power at the prefill-capture path is NOT proven.",
              file=sys.stderr)
        return 1
    if "FAILED at stage=self_check" not in p_corrupt.stdout and \
       "FAILED at stage=self_check" not in p_corrupt.stderr and \
       "FAILED at stage=interior_row_oracle" not in p_corrupt.stdout and \
       "FAILED at stage=interior_row_oracle" not in p_corrupt.stderr:
        print("FAILED: --corrupt-site-hook probe failed, but not at the expected self_check/"
              "interior_row_oracle stage -- inspect stdout/stderr above.", file=sys.stderr)
        return 1
    print("fault injection confirmed: a corrupted PREFILL-position record made the self-check fail "
          "(no dump written) -- the invariance check has power at the new capture path.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
