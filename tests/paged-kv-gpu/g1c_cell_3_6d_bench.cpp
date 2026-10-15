// Paged-KV plan (rev 16.2) §7 cell 3.6d [G3], the BENCH PREFILL's run and its precedence run, group g1c.
// Built ONLY in the bench configuration build (SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING on the engine and the
// cells, as build.bat:184 compiles its one tool): "The bench prefill runs in the bench configuration build of
// the GPU suite" (§7 3.6d); §4.4's derived table: SslmGpuSeqPrefillPromptPreBatchingBenchOnly, singleton, by
// holder, guarded.
//
//   main run: g1c_guard_runs.h's GUARDED RUN, A pooled on B's pool (AKind::PooledSamePool), B fresh: SSLM_BUSY,
//             B's mirror and the fill probe unchanged, SslmGpuSeqSetSchemaForG5Bridge(B) after the drain SSLM_OK
//             (as for the prompt prefill bridge: a pre-guard reset-only placement revokes bind_eligible), then
//             B proceeds with the unsaved twin's status and its decode tokens.
//   precedence: count == 0 -- "in scope SSLM_BUSY, out of scope SSLM_OK; guard-below mutant in scope SSLM_OK"
//             (§7 3.6d precedence, rev 8).
//
// RED NOW (63e7b0e): BY LINK -- sslm_gpu_kv_pool_create and sslm_gpu_seq_create_budgeted (G3); also no bench
// configuration build of the suite exists (G3 delivers it).
// GREEN AT G3 when both runs pass on both GPUs in the bench configuration build.
#ifndef SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING
#error "g1c_cell_3_6d_bench.cpp belongs to the bench configuration build (SUPERSLM_ENABLE_GPU_BENCH_PRE_BATCHING)"
#endif
#include "g1c_guard_runs.h"

namespace {
using namespace g1c;

GuardedSpec SpecBenchPrefill() {
	GuardedSpec g;
	g.name = "3.6d/G3 SslmGpuSeqPrefillPromptPreBatchingBenchOnly";
	g.a = AKind::PooledSamePool;
	g.build = [](World& w) {
		w.toks = Toks(w.s.art, 20, 9);
		return BuildFreshB(w);
	};
	g.call = [](World& w) {
		return SslmGpuSeqPrefillPromptPreBatchingBenchOnly(w.s.ctx, w.B, w.toks.data(),
		                                                    static_cast<int32_t>(w.toks.size()), w.s.one_layer);
	};
	g.output = [](World& w, SslmGpuStatus) { return DecodeRun(w.s, w.B, 4, 1); };
	g.schema_operand = SetSchemaB;
	g.has_expect_schema = true;
	g.expect_schema = SSLM_OK;
	g.resave_holder = nullptr;  // the re-save is G5's; the bench prefill has no G5 part in §7
	return g;
}

void G3BenchPrefill() { GuardedRun(SpecBenchPrefill()); }

void G3BenchPrefillCount0() {
	PrecedenceSpec p;
	p.name = "3.6d/G3 bench prefill, count == 0";
	p.build = [](World& w) {
		w.toks = Toks(w.s.art, 4, 47);
		return BuildFreshB(w);
	};
	p.call = [](World& w) {
		return SslmGpuSeqPrefillPromptPreBatchingBenchOnly(w.s.ctx, w.B, w.toks.data(), 0, w.s.one_layer);
	};
	p.expect_idle = {SSLM_OK};
	PrecedenceRun(p);
}
}  // namespace

G1C_CELL("3.6d/G3:bench_prefill", "G3", G3BenchPrefill);
G1C_CELL("3.6d/G3:precedence:bench_count0", "G3", G3BenchPrefillCount0);
