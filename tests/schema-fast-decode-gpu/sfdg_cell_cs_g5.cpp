// Engine 1.19 clean stop, cell G5 (sfdg_cleanstop.h has the contract): G3's EXACT_DRAFT chunks whose last row is the
// closing token (sfdg_cell_cs_g3.cpp), closed in one-layer slices (sslm_decode_step_gpu at one layer's dispatch budget,
// drained after each): at least as many slices as layers, and the same end as G3 -- rows_produced -2 at SSLM_OK,
// last_rows_committed M, accepting 1, ready re-armed, the Off run's blob. Device head. Box only.
// Run: sfdg_cell_cs_g5.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) { return EndOnLastRow(argc, argv, /*sliced=*/true); }
