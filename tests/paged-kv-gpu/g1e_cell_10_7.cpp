// Paged-KV plan (rev 16.2) §7 cell 10.7 [G6]: "Rev 3; rev 4 grading. GPU twin of 10.6 and 10.8 on both certified
// GPUs, graded by GPU admission." Verdict provenance (§7): GPU statuses; tokens. Group g1e (one cell per
// executable), written at G1 (§9's G1 row lists 10.7), bracketed G6. The CPU twins are
// tests/paged-kv/c6_dim10_cohort.cpp Cell106On and Cell108On.
//
// The source run: 10.3's cohort in the exact 1,842-page pool (g1e_cohort.h RunCohort), every sequence saved at
// 1,456 (mid-budget, after 257 tokens) and again at the limit (1,712), each blob a budget SLM6 (self-checked:
// kv_mode budget, budget 512, origin 1,200, its context length). Its tokens are the "unsaved run", and they are
// checked equal to the CPU twin (pinned to the v1.11.0 no-sharing record), so "tokens equal the unsaved run's (and
// v1.9.0's)" is closed in this process. Then, for each pass (1,456, the limit):
//
//   10.6's twin -- restore WITH handles. The world and the 10 personas are rebuilt in a fresh pool, the world
//   handle released, and all 50 blobs restored into it (sslm_gpu_seq_restore_into_pool) with their persona handles.
//     - a pool of 1,842 admits all 50 restores; every restored sequence continues to its limit with tokens equal
//       to the unsaved run's continuation, then SSLM_GPU_KV_BUDGET_EXCEEDED (at the limit pass: no token, the
//       refusal at once);
//     - a pool of 1,841 refuses exactly the 50th restore, with SSLM_GPU_KV_POOL_EXHAUSTED.
//     Kills: a restore that does not re-share (needs the 75 private pages, refused early at 1,842), one drawing
//     less than R(512) (admitted at 1,841), one that loses the budget or the origin (runs past or short of the
//     limit), one that shares a page it should not (tokens differ).
//   10.8's twin -- restore WITHOUT handles (the no-handle recovery):
//     - a pool of 5,592 admits all 50 restores; then every restored sequence is reset (sslm_gpu_seq_reset) and
//       re-adopts its persona, and exactly 113 further budget-512 creates are admitted, the 114th refused with
//       SSLM_GPU_KV_POOL_EXHAUSTED;
//     - a pool of 5,591 refuses exactly the 50th restore, exhausted.
//     Kills: a private restore drawing more than R + E (refused early at 5,592) or less (admitted at 5,591);
//     materialized pages kept or sent to the reserve at the reset (fewer than 113 further creates).
// *out_shared_pages is summed and printed as a diagnostic only (§7 10.6). Nothing reads sslm_gpu_kv_pool_stats.
//
// Legs (g1e_cohort.h): --leg=fixture (pkv_def; runs on the software GPU) and --leg=box (the real 0.5B; box-only).
// Run once per certified GPU (SSLM_GPU_ADAPTER_INDEX=0, then 1) with SSLM_GPU_ENABLE_DEBUG_LAYER=1.
//
// Run: g1e_cell_10_7.exe --leg=fixture --fixture-dir=DIR [--pkv-reference=DIR]
//      g1e_cell_10_7.exe --leg=box --model05=PATH [--pkv-reference=DIR]
#include "g1e_cohort.h"

using namespace g1e;
using namespace g1e::cohort;

namespace {

const char* PassName(int pass) { return pass == 0 ? "1,456" : "the limit"; }

void Twin106(const Env& e, const Figures& f, const CohortRun& run, int pass, const std::string& stem) {
	const std::vector<std::vector<uint8_t>>& blobs = pass == 0 ? run.mid : run.lim;
	const std::vector<GSeq>& like = pass == 0 ? run.mid_state : run.lim_state;
	const size_t from = pass == 0 ? static_cast<size_t>(kMidAt - kOrigin + 1) : kTokens;
	int continued = 0;
	const RestoreRun ok = RestoreCohort(e, static_cast<uint32_t>(f.pool), blobs, like, /*handles=*/true,
	                                    [&](Rig&, const GWorld&, std::vector<GSeq>& seqs) {
		                                    for (size_t i = 0; i < seqs.size(); ++i) {
			                                    if (!seqs[i].h) continue;
			                                    GS end = GS::SSLM_OK;
			                                    const std::vector<int32_t> t = RunToRefusal(e, seqs[i], &end);
			                                    const std::vector<int32_t>& all = run.tokens[i];
			                                    const std::vector<int32_t> want(all.begin() + static_cast<std::ptrdiff_t>(std::min(from, all.size())),
			                                                                    all.end());
			                                    const bool same = t == want && end == GS::SSLM_GPU_KV_BUDGET_EXCEEDED;
			                                    continued += same ? 1 : 0;
			                                    // kills: a restore that loses the budget or the origin, or shares a page
			                                    // it should not
			                                    CHECK_MSG(same, "10.7/10.6 %s (%s): restored sequence %zu continues with %zu tokens [%s] "
			                                                    "(%s); the unsaved run with %zu [%s], then SSLM_GPU_KV_BUDGET_EXCEEDED",
			                                              stem.c_str(), PassName(pass), i, t.size(), Ids(t).c_str(), StatusName(end),
			                                              want.size(), Ids(want).c_str());
		                                    }
	                                    });
	// kills: a restore that does not re-share (refused early at 1,842)
	CHECK_MSG(ok.first_refused < 0 && ok.refused == 0, "10.7/10.6 %s (%s): a pool of %lld refused %d restores, the first at %d (%s)",
	          stem.c_str(), PassName(pass), static_cast<long long>(f.pool), ok.refused, ok.first_refused, StatusName(ok.refusal));
	std::printf("10.7/10.6 %s (%s): %d of 50 restored sequences continued equal to the unsaved run; *out_shared_pages summed "
	            "= %llu (diagnostic)\n",
	            stem.c_str(), PassName(pass), continued, static_cast<unsigned long long>(ok.shared_diag));
	const RestoreRun short1 = RestoreCohort(e, static_cast<uint32_t>(f.pool - 1), blobs, like, true,
	                                        [](Rig&, const GWorld&, std::vector<GSeq>&) {});
	// kills: a shared restore drawing less than R(512) (admitted at 1,841)
	CHECK_MSG(short1.first_refused == kSeqs - 1 && short1.refused == 1 && short1.refusal == GS::SSLM_GPU_KV_POOL_EXHAUSTED,
	          "10.7/10.6 %s (%s): a pool of %lld refused %d restores, the first at %d (%s); want exactly the 50th, exhausted",
	          stem.c_str(), PassName(pass), static_cast<long long>(f.pool - 1), short1.refused, short1.first_refused,
	          StatusName(short1.refusal));
}

void Twin108(const Env& e, const Figures& f, const CohortRun& run, int pass, const std::string& stem) {
	const std::vector<std::vector<uint8_t>>& blobs = pass == 0 ? run.mid : run.lim;
	const std::vector<GSeq>& like = pass == 0 ? run.mid_state : run.lim_state;
	int admitted = -1;
	GS refusal = GS::SSLM_OK;
	int readopt_failed = 0;
	const RestoreRun ok = RestoreCohort(e, static_cast<uint32_t>(f.private_pool), blobs, like, /*handles=*/false,
	                                    [&](Rig& rig, const GWorld& w, std::vector<GSeq>& seqs) {
		                                    for (size_t i = 0; i < seqs.size(); ++i) {
			                                    if (!seqs[i].h) continue;
			                                    const GS rs = sslm_gpu_seq_reset(rig.ctx, seqs[i].h);
			                                    const GS ad = rs == GS::SSLM_OK
			                                                      ? Adopt(rig, seqs[i], w.personas[static_cast<size_t>(PersonaOf(static_cast<int>(i)))], kOrigin)
			                                                      : rs;
			                                    readopt_failed += (rs != GS::SSLM_OK || ad != GS::SSLM_OK) ? 1 : 0;
			                                    CHECK_MSG(rs == GS::SSLM_OK && ad == GS::SSLM_OK,
			                                              "10.7/10.8 %s (%s): restored sequence %zu: reset %s, re-adopt %s", stem.c_str(),
			                                              PassName(pass), i, StatusName(rs), StatusName(ad));
		                                    }
		                                    std::vector<SslmGpuSequenceHandle*> extra;
		                                    for (int guard = 0; guard < 100000; ++guard) {
			                                    SslmGpuSequenceHandle* h = nullptr;
			                                    refusal = sslm_gpu_seq_create_budgeted(rig.ctx, rig.model, rig.pool, kBudget, &h);
			                                    if (refusal != GS::SSLM_OK) break;
			                                    extra.push_back(h);
		                                    }
		                                    admitted = static_cast<int>(extra.size());
		                                    for (auto it = extra.rbegin(); it != extra.rend(); ++it) sslm_gpu_seq_release(rig.ctx, *it);
	                                    });
	// kills: a private restore drawing more than R + E (refused early at 5,592)
	CHECK_MSG(ok.first_refused < 0 && ok.refused == 0, "10.7/10.8 %s (%s): a pool of %lld refused %d restores, the first at %d (%s)",
	          stem.c_str(), PassName(pass), static_cast<long long>(f.private_pool), ok.refused, ok.first_refused,
	          StatusName(ok.refusal));
	// kills: materialized pages kept, or sent to the reserve, at the reset
	CHECK_MSG(admitted == f.further && refusal == GS::SSLM_GPU_KV_POOL_EXHAUSTED && readopt_failed == 0,
	          "10.7/10.8 %s (%s): %d further budget-512 creates admitted after the re-adopts (refusal %s, %d re-adopts failed); "
	          "expected %lld, then exhausted",
	          stem.c_str(), PassName(pass), admitted, StatusName(refusal), readopt_failed, static_cast<long long>(f.further));
	const RestoreRun short1 = RestoreCohort(e, static_cast<uint32_t>(f.private_pool - 1), blobs, like, false,
	                                        [](Rig&, const GWorld&, std::vector<GSeq>&) {});
	// kills: a private restore drawing less than R + E (admitted at 5,591)
	CHECK_MSG(short1.first_refused == kSeqs - 1 && short1.refused == 1 && short1.refusal == GS::SSLM_GPU_KV_POOL_EXHAUSTED,
	          "10.7/10.8 %s (%s): a pool of %lld refused %d restores, the first at %d (%s); want exactly the 50th, exhausted",
	          stem.c_str(), PassName(pass), static_cast<long long>(f.private_pool - 1), short1.refused, short1.first_refused,
	          StatusName(short1.refusal));
}

void Cell107(const Args& a) {
	const std::string path = ModelPath(a, "10.7");
	if (path.empty()) return;
	Env e;
	if (!e.Open(path)) return;
	const Figures* f = FiguresFor(e.geo, "10.7");
	if (!f) return;
	const std::string stem = Stem(path);
	const CpuTwin cpu = CpuTwinPinned(a, stem, e.fx.bytes, "10.7");
	const CohortRun run = RunCohort(e, static_cast<uint32_t>(f->pool), /*drive=*/true, /*keep_blobs=*/true, "10.7 source run");
	const bool built = AdmissionAsExpected(run, -1) && run.tokens.size() == static_cast<size_t>(kSeqs) &&
	                   run.mid.size() == static_cast<size_t>(kSeqs) && run.lim.size() == static_cast<size_t>(kSeqs);
	CHECK_MSG(built, "10.7 %s: the source cohort run in %lld pages did not save 50 sequences (%zu run, %d creates refused, the "
	                 "first at %d, %s)",
	          stem.c_str(), static_cast<long long>(f->pool), run.tokens.size(), run.creates_refused, run.first_refused,
	          StatusName(run.refusal));
	if (!built) return;
	for (size_t i = 0; i < run.tokens.size(); ++i) {
		CHECK_MSG(run.tokens[i].size() == kTokens && run.end[i] == GS::SSLM_GPU_KV_BUDGET_EXCEEDED,
		          "10.7 %s: unsaved sequence %zu emitted %zu tokens (%s)", stem.c_str(), i, run.tokens[i].size(), StatusName(run.end[i]));
		if (cpu.ok)
			CHECK_MSG(run.tokens[i] == cpu.persona[static_cast<size_t>(PersonaOf(static_cast<int>(i)))],
			          "10.7 %s: the unsaved run's sequence %zu differs from the CPU twin (and so from v1.11.0)", stem.c_str(), i);
		Slm6SelfCheck(run.mid[i], e.geo, kBudget, kOrigin, kMidAt, "10.7 save at 1,456");
		Slm6SelfCheck(run.lim[i], e.geo, kBudget, kOrigin, kLimit, "10.7 save at the limit");
	}
	CHECK_MSG(cpu.ok, "10.7 %s: the CPU twin did not run (see FAIL above)", stem.c_str());
	for (int pass = 0; pass < 2; ++pass) {
		Twin106(e, *f, run, pass, stem);
		Twin108(e, *f, run, pass, stem);
	}
}

}  // namespace

int main(int argc, char** argv) {
	const Args a = ParseArgs(argc, argv);
	Cell107(a);
	return FinishSuite(a.leg == "box" ? "g1e_cell_10_7/box" : "g1e_cell_10_7/fixture");
}
