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
// Provenance. kGpuRowsCostTableMeasured is false while r(M) is an ESTIMATE, true once tools/multirow_cost_bench has
// measured it on a box from the recorder's GPU time (it writes this whole header in this layout). g(M) stays an estimate
// in either case until MR5 (kGpuRowsLogitsRatioMeasured):
//   - r(1), r(2), r(4), r(8): COST's spike table [A], C(M) in us {6,200; 7,900; 11,500; 18,600}, on another model;
//   - r(3), r(5), r(6), r(7): linear interpolation of C between those [I];
//   - g(M): unknown until MR5's widened logits stage (§15 item 10); estimated at gamma = G / C = 0.25, the plan's
//     pricier column (§4.7), so g(M) = ceil(r(M) / 4) [I].
// The shipped table is MR7's re-measure on the R-20 build, the per-M maximum across every certified card (§5.1).
#ifndef SUPERSLM_GPU_ROWS_COST_TABLE_H
#define SUPERSLM_GPU_ROWS_COST_TABLE_H

#include <cstdint>

namespace superslm_gpu {

constexpr bool kGpuRowsCostTableMeasured = false;    // estimates (above)
constexpr bool kGpuRowsLogitsRatioMeasured = false;  // g(M): estimates until MR5's widened logits stage (ruling 18a)
constexpr uint32_t kGpuRowCostUs = 6200;            // GpuRowCostMs's default (6.2 ms) in us: C(1), the M = 1 entry

// r(M), per-mille of C(1); index M - 1.
constexpr uint32_t kGpuRowsCostRatioPerMille[8] = {
    1000,  // M = 1: GpuRowCostMs, by definition
    1275,  // M = 2: [A] estimate
    1565,  // M = 3: [I] estimate
    1855,  // M = 4: [A] estimate
    2142,  // M = 5: [I] estimate
    2428,  // M = 6: [I] estimate
    2714,  // M = 7: [I] estimate
    3000,  // M = 8: [A] estimate
};

// g(M), per-mille of C(1): the per-row logits stage (Exact); index M - 1.
constexpr uint32_t kGpuRowsLogitsRatioPerMille[8] = {
    250,  // M = 1: [I] estimate, gamma = 0.25
    319,  // M = 2: [I] estimate
    392,  // M = 3: [I] estimate
    464,  // M = 4: [I] estimate
    536,  // M = 5: [I] estimate
    607,  // M = 6: [I] estimate
    679,  // M = 7: [I] estimate
    750,  // M = 8: [I] estimate
};

}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_ROWS_COST_TABLE_H
