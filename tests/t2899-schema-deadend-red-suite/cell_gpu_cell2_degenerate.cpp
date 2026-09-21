// T-2900 (Curie) -- plan Sec3.10.2 Cell 2: the degenerate-row dead end at a reachable
// non-accepting, non-S_c state, through the named seam `ArmGpuFinishDegenerateLogitRowInjection()`
// (`gpu_1p0.cpp:2406-2410` at Stage 1, compiled under `SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION`).
//
// NOT YET AUTHORABLE AS A RUNNABLE CELL, for the same reason CPU's own twin
// (`cell_cpu_deadend_retry_reset.cpp`'s `CpuCell2DegenerateRowTwin`) is not: the seam itself does
// not exist anywhere -- not in tracked source, not in EITHER of the plan's own cumulative
// reference-fix files (`Claude/Vitruvius/t2895-probe/gpu_1p0_v5.cpp`/`superslm_gpu_v5.cpp`,
// confirmed absent by grep) -- because it is production code the builder owes (plan Sec3.10.2:
// "Not yet built; owed to Sec3.10.4's builder alongside the CPU fix"), not this seat's to add.
// Curie writes tests, never implementation; authoring the seam herself -- even in a scratch
// reference-fix copy -- would make her the author of the mechanism her own cell is meant to gate,
// which is exactly the seam the test-author/implementer split exists to keep separate.
//
// This declaration reserves the call so the cell is RED BY LINK the moment the builder adds the
// production seam -- the SAME pattern `sslm_gpu_seq_read_prefill_final_hidden` used before
// Sec3.1's builder built it (`tests/t2791-gpu-prefill-read-red-suite/fixture_common.h`'s own
// header comment), and the SAME pattern CPU's own twin already carries. Compiling under the macro
// but failing to LINK (`LNK2019: unresolved external symbol
// ArmGpuFinishDegenerateLogitRowInjection`) is the correct, intended state; it becomes a real cell
// -- drive a real schema-bound sequence to a reachable interior state via real transitions, arm
// the seam, present the degenerate row, assert `-2`/`SSLM_OK` -- the moment the symbol exists.
//
// A real, reachable, non-accepting, non-start interior state IS available on this machine to
// drive to once the seam lands: the G5 production-scale fixture's own `shopkeeper_intent_extraction`
// schema (`t2132_g5_fixture_1p5b.sslm`, 594 states) reaches many such states along
// `cell_gpu_cell1_realschema.cpp`'s own 251-token census-discovered path before its terminal dead
// end at state 592 -- for example the state reached after the first 50 tokens of that path. The
// C39 synthetic (`g5_minimal_one_field`) cannot serve this cell: it compiles to 2 states and 1
// transition (`tests/t2791-gpu-prefill-read-red-suite/make_g5an_fixture.py`'s own header), so its
// only non-start state IS the accepting state Cell 1 already dead-ends at, with no interior state
// distinct from it.
#if defined(SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION)
extern "C" void ArmGpuFinishDegenerateLogitRowInjection();
#endif

#include <cstdio>

int main() {
#if defined(SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION)
	ArmGpuFinishDegenerateLogitRowInjection();  // TODO(builder): exercise once the seam exists.
	std::printf("checks=0 failures=0 skips=0\n");
	return 0;
#else
	std::printf("SKIP cell_gpu_cell2_degenerate -- the injection seam does not exist yet; "
	            "compile with -DSUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION to confirm red-by-link\n");
	std::printf("checks=0 failures=0 skips=1\n");
	return 0;
#endif
}
