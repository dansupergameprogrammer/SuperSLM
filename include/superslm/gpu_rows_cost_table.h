// Multi-row GPU plan rev 1.3.2, slice MR4: the cost table in C18 (plan §4.7, §5.1, cell 10.2; rulings item 12).
//
// Host-only data, no code and no device types: the plugin vendors this header and its integer chooser (MR6) prices a
// chunk from it. Integers only (§5.1, integer pricing). Names and layout: the MR4 suite's contract
// (tests/multirow-gpu/mr4_table.cpp).
//
//   r(M) = C(M) / C(1)  in per-mille: one whole-layer pass of an M-row chunk, relative to one row.
//   g(M) = G(M) / C(1)  in per-mille: the per-row logits stage of an M-row chunk (Exact), relative to one row.
// Index M - 1 holds M's entry, M = 1..8 (SSLM_GPU_MAX_ROWS).
//
// Absolute costs are r(M) x GpuRowCostMs / 1000. `GpuRowCostMs` keeps its name and default (6.2 ms) and is the table's
// M = 1 entry, so r(1) is 1000 by definition. C(M) is taken at its MAXIMUM over the timed runs (MR-D15), and ratios
// round UP (costs round up, §5.1).
//
// MEASURED (ruling 37; MB4/MB5, box record box/mb-workhorse-7900xt-2026-10-26.md):
//   card    AMD Radeon RX 7900 XT, driver 32.0.21030.2001, Windows 10.0.26300
//   engine  SuperSLM 9493b927 (claude/multirow-gpu-r17), tools/multirow_cost_bench, 30 timed runs + 1 warm-up per M,
//           p0 = 16; kit Wizard c0dad9a76, results at Wizard 8147eea0 (Claude/Brunel/multirow-mb/results/2026-10-26/
//           table-mb.csv), graded by tests/multirow-gpu/mr4_box.py check102 (PASS)
//   models  Qwen2.5-0.5B-Instruct (sha256 9ccf7e37..., the A-EX pin) and Qwen2.5-1.5B-Instruct (sha256 040be03e...)
//   r(M) = ceil(1000 x maxC(M) / maxC(1)), C the recorder's GPU time (logits = 0 rows).
//   g(M) = ceil(1000 x S(M) / maxC(1)), S the widened logits stage's own GPU bracket at its max (rulings 18a, 27).
//          g(1) is this table's: the run at 9493b927 timed a one-row stage, which the engine no longer runs (ruling 37:
//          a one-row close keeps Off's one-row logits call), so no later bench measures it.
// The shipped arrays are the per-M MAXIMUM of the two models (the 1.5B prices every r(M) above the 0.5B; g(M) mixes);
// each model's own values are kept below for the record. MR7's re-measure on the R-20 build takes the per-M maximum
// across every certified card (§5.1).
#ifndef SUPERSLM_GPU_ROWS_COST_TABLE_H
#define SUPERSLM_GPU_ROWS_COST_TABLE_H

#include <cstdint>

namespace superslm_gpu {

constexpr bool kGpuRowsCostTableMeasured = true;    // r(M), measured (above)
constexpr bool kGpuRowsLogitsRatioMeasured = true;  // g(M), measured (above)
constexpr uint32_t kGpuRowCostUs = 6200;           // GpuRowCostMs's default (6.2 ms) in us: C(1), the M = 1 entry

// r(M), per-mille of C(1); index M - 1. The per-M maximum of the 0.5B and the 1.5B.
constexpr uint32_t kGpuRowsCostRatioPerMille[8] = {
    1000,  // M = 1: GpuRowCostMs, by definition
    1140,  // M = 2: 1.5B (0.5B 1004)
    1604,  // M = 3: 1.5B (0.5B 1221)
    1618,  // M = 4: 1.5B (0.5B 1228)
    2479,  // M = 5: 1.5B (0.5B 1704)
    2491,  // M = 6: 1.5B (0.5B 1711)
    2503,  // M = 7: 1.5B (0.5B 1712)
    2521,  // M = 8: 1.5B (0.5B 1717)
};

// g(M), per-mille of C(1): the per-row logits stage (Exact); index M - 1. The per-M maximum of the two models.
constexpr uint32_t kGpuRowsLogitsRatioPerMille[8] = {
    48,   // M = 1: 0.5B (1.5B 39), from this table (ruling 37)
    95,   // M = 2: 0.5B (1.5B 81)
    122,  // M = 3: 1.5B (0.5B 116)
    162,  // M = 4: 1.5B (0.5B 154)
    200,  // M = 5: 1.5B (0.5B 192)
    238,  // M = 6: 1.5B (0.5B 231)
    275,  // M = 7: 1.5B (0.5B 272)
    313,  // M = 8: 1.5B (0.5B 312)
};

// Each model's own measurement on this card (the record; the chooser reads the arrays above).
constexpr uint32_t kGpuRowsCostRatioPerMille05B[8] = {1000, 1004, 1221, 1228, 1704, 1711, 1712, 1717};
constexpr uint32_t kGpuRowsLogitsRatioPerMille05B[8] = {48, 95, 116, 154, 192, 231, 272, 312};
constexpr uint32_t kGpuRowsCostRatioPerMille15B[8] = {1000, 1140, 1604, 1618, 2479, 2491, 2503, 2521};
constexpr uint32_t kGpuRowsLogitsRatioPerMille15B[8] = {39, 81, 122, 162, 200, 238, 275, 313};

}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_ROWS_COST_TABLE_H
