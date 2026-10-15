// GPU paged-KV plan (rev 16), step G3: cell 3.6h in superslm_tests (§7: "The cell is in
// superslm_tests, and it runs in the linux-x64-tsan job"; §4.4: G3's ThreadSanitizer evidence is a
// local run of superslm_tests built with g++ 13.3 and -fsanitize=thread). test_main.cpp calls
// RunGpuLedgerG3Cells and adds this unit's counts to its own totals, the pattern
// test_slm18x_saturation_census.cpp set. The cells themselves are tests/gpu-g3-core's (the same
// source superslm_g3_core_tests runs); superslm_tests links g3_cells_ledger.cpp and runs the cells
// whose ids start "3.6h" and "G3/ledger". SUPERSLM_LEDGER_YIELD_HOOK is defined on the whole
// superslm_tests target, so every translation unit sees one definition of the ledger.
#include "g3_harness.h"

#include <cstdio>
#include <string>
#include <vector>

void RunGpuLedgerG3Cells(int& checks, int& failures) {
	g3::RunState& s = g3::State();
	const int c0 = s.checks, f0 = s.failures;
	int unmatched = 0;
	const int ran = g3::RunCells({"3.6h", "G3/ledger"}, &unmatched);
	const int c = s.checks - c0, f = s.failures - f0 + unmatched + (ran == 0 ? 1 : 0);
	std::printf("GPU paged-KV G3 ledger cells (3.6h, G3/ledger): %d cells, %d checks, %d failures\n", ran, c, f);
	checks += c;
	failures += f;
}
