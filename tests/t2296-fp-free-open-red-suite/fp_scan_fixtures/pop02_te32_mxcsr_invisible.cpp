// T-2333 (Curie) -- Sec7 dimension 11's SECOND commissioning population:
// TE-32's own "four always-invisible-to-MXCSR classes" plus the compiled
// BodyUnderTest() artifact from Part E/E2.
//
// PROVENANCE: scope GRANTED for T-2333 (Claude/Loki/te32-probe/ was denied to
// T-2326; the coordinator's own brief, item 5, grants it). Reproduced
// verbatim from Claude/Loki/te32-probe/te32_fp_observability_cell.cpp:
//
//   - `RunOp`'s own switch cases 5 (i64->f64), 8 (ordered compare), 9
//     (abs/negate), 10 (min) -- identified as the FOUR classes TE-32's own
//     executed Part D census reports "INVISIBLE on both operand sets" (exact
//     AND inexact), confirmed at source: Claude/Loki/te32-probe/
//     output-te32.txt lines 170/177/178/179 ("i64 -> f64 ... NONE NONE
//     INVISIBLE on both operand sets"; "ordered compare ... NONE NONE";
//     "abs / negate ... NONE NONE"; "min ... NONE NONE") -- the only four of
//     twelve rows with that verdict; every other row (add/sub/mul/div/sqrt/
//     f64->f32/f64->i64-trunc/composed) flips to visible (PE) on the inexact
//     operand set and is therefore NOT this population's own concern (a
//     runtime observer already catches those eight; only the four
//     always-invisible ones are what a byte-level scan must catch where a
//     trap-based liveness control structurally cannot, at ANY operand set).
//   - `BodyUnderTest()` verbatim (Part E/E2's own compiled artifact), plus
//     `kCorpusA` (the first of TE-32's own ten corpora, "8 powers of two").
//
// Operand values reproduced verbatim from TE-32's own `kOps` table (its own
// "exact" column, since all four classes are invisible regardless of exact/
// inexact -- the exact column is reproduced here for simplicity, matching
// TE-32's own row: i64->f64 uses 7.0; ordered compare/abs-negate/min use
// (1.0,2.0)/(-3.0,0.0)/(3.0,4.0) respectively).
//
// This population's own falsifying claim: every one of these four bodies,
// though invisible to MXCSR under BOTH operand sets (confirmed by TE-32's own
// execution, not re-derived here), must still REJECT under the deciding
// instrument's check (A) -- a vector/FP register touched by an instruction
// not on VEC_MOVE_ALLOW. A liveness control alone (Sec4.2/Sec4.3) would see
// nothing on any of these four bodies at any operand set; the byte-level scan
// must.
//
// Build: MSVC cl.exe (/std:c++20 /O2 /EHsc), matching TE-32's own build.bat.

#include <cstdint>

namespace {
volatile double g_a = 0, g_b = 0;
volatile double g_dsink = 0;
volatile int64_t g_isink = 0;
volatile int g_bsink = 0;
}

// RunOp case 5: i64 -> f64 conversion. TE-32's own exact operand: a=7.0.
extern "C" __declspec(dllexport) void Te32OpI64ToF64() {
	g_a = 7.0;
	g_dsink = static_cast<double>(static_cast<int64_t>(g_a));
}

// RunOp case 8: ordered compare. TE-32's own exact operands: a=1.0, b=2.0.
extern "C" __declspec(dllexport) void Te32OpOrderedCompare() {
	g_a = 1.0; g_b = 2.0;
	g_bsink = (g_a < g_b) ? 1 : 0;
}

// RunOp case 9: abs/negate. TE-32's own exact operand: a=-3.0.
extern "C" __declspec(dllexport) void Te32OpAbsNegate() {
	g_a = -3.0;
	g_dsink = (g_a < 0) ? -g_a : g_a;
}

// RunOp case 10: min. TE-32's own exact operands: a=3.0, b=4.0.
extern "C" __declspec(dllexport) void Te32OpMin() {
	g_a = 3.0; g_b = 4.0;
	g_dsink = (g_a < g_b) ? g_a : g_b;
}

// Part E/E2's own BodyUnderTest(), verbatim, plus one corpus (kCorpusA).
volatile double g_corpus_buf[16];
volatile int g_corpus_n = 0;

void MaterialiseCorpus(const double* d, int n) {
	for (int i = 0; i < n; ++i) g_corpus_buf[i] = d[i];
	g_corpus_n = n;
}

extern "C" __declspec(dllexport) double BodyUnderTest() {
	const int n = g_corpus_n;
	double acc = 0.0;
	for (int i = 0; i < n; ++i) acc = acc + g_corpus_buf[i];
	double scaled = acc / static_cast<double>(n);
	return scaled;
}

static const double kCorpusA[8] = {2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0};

extern "C" __declspec(dllexport) double RunBodyUnderTestOnCorpusA() {
	MaterialiseCorpus(kCorpusA, 8);
	return BodyUnderTest();
}
