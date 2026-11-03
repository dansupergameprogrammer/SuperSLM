// Engine 1.19 clean stop, cell G3 (sfdg_cleanstop.h has the contract): a multi-row EXACT_DRAFT chunk whose
// last row is the closing token. With p = o_j in hand (j = e - M, the Off stream o_0 .. o_(e-1) ending at its -2),
// the rows [o_j .. o_(e-1)] are all admitted; the close finishes row M - 1 and gets -2, so it commits all M rows
// (CommitPrefix(M, M - 1): last_rows_committed M) and sslm_gpu_seq_rows_produced returns -2 at SSLM_OK. Accepting is
// 1 on both queries, the walk is the closing token's state, the ready flag is re-armed, the context is the prompt plus
// e, and the blob equals the Off run's after its -2. One more decode-bridge call gives -2 with nothing changed.
// G3 closes in one slice of the full budget; G5 (sfdg_cell_cs_g5.cpp) runs the same chunks closed in one-layer slices.
// M in {1, 3, 8}; dedicated sequences on the model mapped with the device head (EXACT_DRAFT needs it). Box only.
// Run: sfdg_cell_cs_g3.exe [--fixture-dir=DIR]
#include "sfdg_cleanstop.h"

using namespace sfdg;
using namespace sfdg::cs;

int main(int argc, char** argv) { return EndOnLastRow(argc, argv, /*sliced=*/false); }
