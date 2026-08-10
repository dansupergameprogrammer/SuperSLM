#!/bin/bash
# T-1881 -- sequential width-1 captures of the buildable (finite-stored-precision) Option E
# sidecar at site 4, plus the null/base anchors, all through T-1859's own tool (extended with
# a --k-cap CLI parameter rather than a hardcoded module constant, T-1881's only code change).
# One process per config, batch width exactly 1 (T-1867/T-1879's shipping anchor), checkpointed:
# a config whose pooled/<name>.npz already exists is skipped rather than recaptured, so a
# restart after an interruption costs nothing already paid for. Runs are serialized (one GPU
# process at a time, StandardsDocument.md's "serialize heavy GPU work" convention).
set -uo pipefail
cd "$(dirname "$0")"

SHARED=out/t1881_capture/shared
DOCS=/d/SuperSLM/.worktrees/t1777-retrieval-agreement/out/t1777_corpus/docs.jsonl
T1777_TOOLS=/d/SuperSLM/.worktrees/t1777-retrieval-agreement/tools
T1835_TOOLS=/d/SuperSLM/.worktrees/t1835-t1836/tools
LOG=out/t1881_progress.log

mkdir -p "$SHARED/pooled" out/logs
mkdir -p out/t1881_capture/kcap3/pooled out/t1881_capture/kcap8/pooled out/t1881_capture/kcap31/pooled

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "$(ts) run_t1881_captures.sh started (pid $$)" >> "$LOG"

run_config() {
  local ARM="$1" OUTDIR="$2" KCAP="$3" TAG="$4"
  if [ -f "$OUTDIR/pooled/$ARM.npz" ]; then
    echo "$(ts) SKIP $TAG (already captured at $OUTDIR/pooled/$ARM.npz)" >> "$LOG"
    return 0
  fi
  echo "$(ts) START $TAG (arm=$ARM k_cap=$KCAP out=$OUTDIR)" >> "$LOG"
  python tools/t1859_option_e_measure.py \
    --docs "$DOCS" \
    --out-dir "$OUTDIR" \
    --t1777-tools "$T1777_TOOLS" \
    --t1835-tools "$T1835_TOOLS" \
    --single-arm "$ARM" \
    --k-cap "$KCAP" \
    > "out/logs/$TAG.log" 2>&1
  RC=$?
  if [ $RC -eq 0 ] && [ -f "$OUTDIR/pooled/$ARM.npz" ]; then
    LASTLINE=$(tail -1 "out/logs/$TAG.log")
    echo "$(ts) DONE $TAG rc=$RC -- $LASTLINE" >> "$LOG"
  else
    echo "$(ts) FAILED $TAG rc=$RC -- see out/logs/$TAG.log" >> "$LOG"
    exit 1
  fi
}

# null/base do not touch site 19/20 -- k_cap is a no-op for them, captured once, shared
# across all three precision arms.
run_config "null" "$SHARED" 8 "null"
run_config "base" "$SHARED" 8 "base"

# The buildable sidecar at site 4 (E_recovery_site4_only), swept at three stored precisions:
# coarser (k_cap=3, 2-bit field), M1's own (k_cap=8, T-1859's already-measured ceiling
# construction, 4-bit field generously stored as a byte), finer (k_cap=31, 5-bit field).
run_config "E_recovery_site4_only" "out/t1881_capture/kcap3"  3  "kcap3"
run_config "E_recovery_site4_only" "out/t1881_capture/kcap8"  8  "kcap8"
run_config "E_recovery_site4_only" "out/t1881_capture/kcap31" 31 "kcap31"

echo "$(ts) run_t1881_captures.sh: all configs complete" >> "$LOG"
exit 0
