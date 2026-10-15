// Paged-KV plan (rev 16.2) §7 cell 10.3 [G6]: "GPU: 10.1 on both certified GPUs, with tokens equal to CPU."
// Verdict provenance (§7): tokens equal CPU (itself pinned by 10.1); operand: the CPU run against the R0 reference.
// Group g1e (one cell per executable), written at G1 (§9's G1 row lists 10.3), bracketed G6.
//
// 10.1's construction on the GPU (tests/paged-kv/c6_dim10_cohort.cpp Cell101On, GPU verbs; g1e_cohort.h): a page
// pool of exactly the cohort's 1,842 pages (§5), the world, 10 personas by begin_from, the world released, then 50
// budget-512 sequences, each adopting its persona and running to its budget before the next is created.
//   - every create, begin_from and adopt is admitted in the 1,842-page pool;
//   - every sequence emits the §3.4 count, 1 + 512 = 513 tokens, then SSLM_GPU_KV_BUDGET_EXCEEDED;
//   - every sequence's tokens equal the CPU twin's for its persona, and the CPU twin's equal the v1.11.0
//     no-sharing record (the plan's "v1.9.0", re-anchored), so GPU == CPU == the no-sharing run.
// Kills: anything GPU sharing changes in what a user receives -- a shared page read through the table wrongly, a
// begin_from child's tail page copied short, a persona's ready flag or residual not carried by adopt -- and any
// GPU page draw beyond §3.4's (an adopt that draws, a begin_from that copies the world, a decode that maps a page
// outside the reservation: the 50th create is refused in the exact pool).
//
// Legs (g1e_cohort.h): --leg=fixture (pkv_def; runs on the software GPU) and --leg=box (the real 0.5B; box-only).
// Run once per certified GPU (SSLM_GPU_ADAPTER_INDEX=0, then 1) with SSLM_GPU_ENABLE_DEBUG_LAYER=1.
//
// Run: g1e_cell_10_3.exe --leg=fixture --fixture-dir=DIR [--pkv-reference=DIR]
//      g1e_cell_10_3.exe --leg=box --model05=PATH [--pkv-reference=DIR]
#include "g1e_cohort.h"

using namespace g1e;
using namespace g1e::cohort;

namespace {

void Cell103(const Args& a) {
	const std::string path = ModelPath(a, "10.3");
	if (path.empty()) return;
	Env e;
	if (!e.Open(path)) return;
	const Figures* f = FiguresFor(e.geo, "10.3");
	if (!f) return;
	const std::string stem = Stem(path);
	const CpuTwin cpu = CpuTwinPinned(a, stem, e.fx.bytes, "10.3");
	const CohortRun r = RunCohort(e, static_cast<uint32_t>(f->pool), /*drive=*/true, /*keep_blobs=*/false, "10.3");
	// kills: any draw beyond §3.4's in the exact pool
	CHECK_MSG(AdmissionAsExpected(r, -1), "10.3 %s: a pool of %lld pages: built %d, %d creates refused (the first at %d, %s), "
	                                      "%d adopts refused; the cohort must be admitted whole",
	          stem.c_str(), static_cast<long long>(f->pool), r.built ? 1 : 0, r.creates_refused, r.first_refused,
	          StatusName(r.refusal), r.adopts_refused);
	CHECK_MSG(r.tokens.size() == static_cast<size_t>(kSeqs), "10.3 %s: %zu sequences ran, not 50", stem.c_str(), r.tokens.size());
	int equal = 0;
	for (size_t i = 0; i < r.tokens.size(); ++i) {
		const int p = PersonaOf(static_cast<int>(i));
		CHECK_MSG(r.tokens[i].size() == kTokens && r.end[i] == GS::SSLM_GPU_KV_BUDGET_EXCEEDED,
		          "10.3 %s: sequence %zu emitted %zu tokens, ending with %s (want 513, then SSLM_GPU_KV_BUDGET_EXCEEDED)",
		          stem.c_str(), i, r.tokens[i].size(), StatusName(r.end[i]));
		if (!cpu.ok) continue;
		const std::vector<int32_t>& want = cpu.persona[static_cast<size_t>(p)];
		const bool same = r.tokens[i] == want;
		equal += same ? 1 : 0;
		// kills: anything sharing changes in what a user receives, on the GPU
		CHECK_MSG(same, "10.3 %s: GPU sequence %zu (persona %d) [%s] differs from the CPU twin [%s]", stem.c_str(), i, p,
		          Ids(r.tokens[i]).c_str(), Ids(want).c_str());
	}
	CHECK_MSG(cpu.ok, "10.3 %s: the CPU twin did not run (see FAIL above): GPU tokens are ungraded", stem.c_str());
	std::printf("10.3 %s: %zu GPU sequences in a %lld-page pool, %d equal to the CPU twin\n", stem.c_str(), r.tokens.size(),
	            static_cast<long long>(f->pool), equal);
}

}  // namespace

int main(int argc, char** argv) {
	const Args a = ParseArgs(argc, argv);
	Cell103(a);
	return FinishSuite(a.leg == "box" ? "g1e_cell_10_3/box" : "g1e_cell_10_3/fixture");
}
