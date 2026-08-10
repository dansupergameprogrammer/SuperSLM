#!/bin/bash
# T-1879 -- captures the four remaining width-1 configs sequentially, one process per
# config, checkpointing each config's pooled output before the next begins. Resumable: a
# config whose pooled/<name>.npz already exists is skipped rather than recaptured, so a
# restart after an interruption costs nothing already paid for. Waits for the separately
# started `null` capture to finish before starting the first of the four (their contrasts
# all read against `null` or against each other, but running while `null` is still writing
# to the same GPU would contend for it, and the instrument is meant to run one process at
# a time -- StandardsDocument.md's "serialize heavy GPU work" convention this campaign
# already follows).
set -uo pipefail
cd "$(dirname "$0")"

OUT=out/t1879_capture
LOG=$OUT/progress.log
DOCS=/d/SuperSLM/.worktrees/t1777-retrieval-agreement/out/t1777_corpus/docs.jsonl
TOOLS=/d/SuperSLM/.worktrees/t1777-retrieval-agreement/tools

mkdir -p "$OUT/pooled" out/logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(ts) run_remaining_configs.sh started (pid $$)" >> "$LOG"

if [ -f "$OUT/pooled/null.npz" ]; then
  echo "$(ts) null.npz already present -- not waiting" >> "$LOG"
else
  echo "$(ts) waiting for null.npz to appear (null capture running separately)..." >> "$LOG"
  while [ ! -f "$OUT/pooled/null.npz" ]; do
    sleep 30
  done
  echo "$(ts) null.npz confirmed present" >> "$LOG"
fi

CONFIGS="only04_k_proj_landing only05_v_proj_landing onlyK_vscale onlyV_kscale"

for CFG in $CONFIGS; do
  if [ -f "$OUT/pooled/$CFG.npz" ]; then
    echo "$(ts) SKIP $CFG (already captured)" >> "$LOG"
    continue
  fi
  echo "$(ts) START $CFG" >> "$LOG"
  python tools/t1835_site_toggle_dump.py \
    --docs "$DOCS" \
    --out-dir "$OUT" \
    --t1777-tools "$TOOLS" \
    --single-arm "$CFG" --no-drift \
    > "out/logs/$CFG.log" 2>&1
  RC=$?
  if [ $RC -eq 0 ] && [ -f "$OUT/pooled/$CFG.npz" ]; then
    LASTLINE=$(tail -1 "out/logs/$CFG.log")
    echo "$(ts) DONE $CFG rc=$RC -- $LASTLINE" >> "$LOG"
  else
    echo "$(ts) FAILED $CFG rc=$RC -- see out/logs/$CFG.log" >> "$LOG"
    exit 1
  fi
done

echo "$(ts) run_remaining_configs.sh: all remaining configs complete" >> "$LOG"
exit 0
