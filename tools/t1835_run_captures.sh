#!/bin/sh
# T-1835 -- every GPU capture this ticket takes, strictly serial (one process at a time).
#
#   gate_base / gate_null : batch 1, one configuration, through the identical code, so the
#                           dumps can be compared BIT-FOR-BIT against T-1809's committed
#                           arm C and arm A. This is the null gate: all-toggles-on must
#                           reproduce the instrument it extends, exactly.
#   cellA / cellB         : every arm, at two different batch widths. The width selects the
#                           GEMM kernel, so the two runs are different numeric realizations
#                           of the same 56 configurations; the spread between them is the
#                           run-to-run resolving power of every per-site figure.
set -e
cd "$(dirname "$0")/.."
T1777=D:\\SuperSLM\\.worktrees\\t1777-retrieval-agreement\\tools
DOCS=/d/SuperSLM/.worktrees/t1777-retrieval-agreement/out/t1777_corpus/docs.jsonl

echo "=== gate_base (B=1, all 18 sites on, 24 documents) ==="
python tools/t1835_site_toggle_dump.py --docs "$DOCS" --out-dir out/t1835_gate_base \
  --t1777-tools "$T1777" --single-arm base --full-dump base --no-drift --limit 24

echo "=== gate_null (B=1, all 18 sites off, 24 documents) ==="
python tools/t1835_site_toggle_dump.py --docs "$DOCS" --out-dir out/t1835_gate_null \
  --t1777-tools "$T1777" --single-arm null --full-dump null --no-drift --limit 24

echo "=== cell A (B=59: 56 arms + 3 base duplicates, 239 documents) ==="
python tools/t1835_site_toggle_dump.py --docs "$DOCS" --out-dir out/t1835_cellA \
  --t1777-tools "$T1777" --dup-base 3 --full-dump base --full-dump null

echo "=== cell B (B=56: 56 arms, 239 documents) ==="
python tools/t1835_site_toggle_dump.py --docs "$DOCS" --out-dir out/t1835_cellB \
  --t1777-tools "$T1777" --full-dump base --full-dump null

echo "=== all captures done ==="
