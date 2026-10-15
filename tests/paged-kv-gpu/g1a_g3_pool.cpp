// Paged-KV plan (rev 16.2) step G1, group g1a: the G3 cells and cell parts (the GPU page pool and
// budgeted GPU sequences).
//
//   1.7 [G3 part]     GPU neighbour invariance through the GPU mirror peek.
//   4.8 [G3 part]     the matrix's paged column (B = 16) and a pool at the largest admissible size.
//   6.5b [G3 part]    GPU paged tokens and K/V byte-equal CPU paged, on both vendors (no adopt).
//   8.13 [G3]         paging x dispatch budget.
//   8.14 [G3]         paging x a batch decode mixing a dedicated and a pooled sequence.
//   11.7 [G3]         the GPU coverage guard: a table one page short is refused before any dispatch.
//   11.9 [G3 part]    guard 5 at a multi-page view, the GPU twin.
//
// RED NOW (LINK): every cell calls sslm_gpu_kv_pool_create, sslm_gpu_seq_create_budgeted,
// sslm_gpu_kv_pool_destroy and sslm_gpu_test_only_peek_pool_mirror_bytes (declared, undefined until
// G3; gpu_1p0.h), and the G3 seams of g1a_helpers.h (sslm_gpu_test_only_seq_table_entry; 11.7 and
// 11.9 also sslm_gpu_test_only_arm_view_fault and sslm_gpu_test_only_last_ladder_status).
// GREEN AT G3: each cell's checks pass on both GPUs with the debug layer on
// (SSLM_GPU_ENABLE_DEBUG_LAYER=1, a clean info queue is G3's exit).

#include "g1a_constructions.h"

namespace {

using namespace g1a;

// ---- 1.7 [G3] -------------------------------------------------------------------------------------
// B's next 16 tokens and B's host-mirror rows, read through the GPU mirror peek, byte-equal a run
// where A never existed. The separating mutant is the reset that fills the whole pool mirror (today's
// dedicated reset, G33, applied to the pool's mirror): it zeroes B's rows [0, 20) in the mirror, which
// the device-side tokens alone cannot show (B decodes from the device buffer) and the row comparison
// does. kills also: a pooled reset or release that returns or poisons one of B's pages.
void Cell17G3() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	const Run17Out with_a = Run17(fx, true, false);
	const Run17Out alone = Run17(fx, false, false);
	if (!with_a.ok || !alone.ok) return;
	G1A_CHECK_EQ(with_a.L, 35);  // 20 prefilled + 15 written by 16 decodes (the first is the ready token)
	G1A_CHECK_MSG(with_a.tokens == alone.tokens, "1.7: B's next 16 tokens differ from the run where A never existed");
	G1A_CHECK_MSG(with_a.rows == alone.rows, "1.7: B's host-mirror rows differ from the run where A never existed");
}

// ---- 4.8 [G3] -------------------------------------------------------------------------------------
// The matrix's paged column: on this GPU, a budgeted sequence (B = 16) on each geometry runs the same
// short scenario as the legacy column and equals a CPU budget holder's tokens, context length and rows.
// Then a pool at the largest admissible size on each real artifact: the arithmetic ceiling of §4 item
// 3 (43,690 pages at 0.5B, 18,724 at 1.5B, from page_bytes computed here), or, when this GPU refuses
// it with SSLM_GPU_ALLOCATION_FAILED, the largest page count it admits, found by bisection on SSLM_OK
// against SSLM_GPU_ALLOCATION_FAILED (11.4's rule; SSLM_DEVICE_LOST fails the cell). On that pool a
// sequence A is created first, the pool is filled, two 2-page fillers are released, and a sequence Z
// is created last; both prefill across a page and decode, and each equals its CPU twin. Whatever
// order the allocator hands out pages, one of the first and the last draws sits at the high end of
// the pool, so a page at a byte offset above 2^31 is driven (checked through the table seam: the
// cell's own vacuity guard).
void ShortRun(SRun& d) {
	void* s = nullptr;
	if (!d.Create(static_cast<int32_t>(d.b.fx.geo.context_cap), &s)) return;
	if (d.Prefill(s, pkv::Stream(81, 100, d.b.fx.vocab)) && d.Mark("prefill100", s) && d.Decode(s, 16))
		d.Mark("prefill100+decode16", s);
	d.Release(s);
}

uint32_t LargestAdmittedPool(const Fx& fx) {
	const uint32_t ceiling = fx.CeilingPages();
	auto try_create = [&](uint32_t pages, SslmGpuStatus* st) {
		SslmGpuKvPoolHandle* p = nullptr;
		*st = sslm_gpu_kv_pool_create(Ctx(), fx.gpu, pages, &p);
		if (p) G1A_CHECK_GPU(sslm_gpu_kv_pool_destroy(Ctx(), p), GS::SSLM_OK);
		return *st == GS::SSLM_OK;
	};
	SslmGpuStatus st = GS::SSLM_OK;
	if (try_create(ceiling, &st)) return ceiling;
	G1A_CHECK_MSG(st == GS::SSLM_GPU_ALLOCATION_FAILED, "4.8 %s: the ceiling create (%u pages) -> %s", fx.name.c_str(), ceiling,
	              GpuStatusName(st));
	if (st != GS::SSLM_GPU_ALLOCATION_FAILED) return 0;
	uint32_t lo = 1, hi = ceiling;  // lo admitted, hi refused
	if (!try_create(lo, &st)) return 0;
	while (hi - lo > 1) {
		const uint32_t mid = lo + (hi - lo) / 2;
		if (try_create(mid, &st)) lo = mid;
		else if (st == GS::SSLM_GPU_ALLOCATION_FAILED) hi = mid;
		else {
			G1A_CHECK_MSG(false, "4.8 %s: bisection create of %u pages -> %s", fx.name.c_str(), mid, GpuStatusName(st));
			return 0;
		}
	}
	std::printf("  4.8 %s: this GPU refuses the arithmetic ceiling %u; largest admitted pool %u pages\n", fx.name.c_str(),
	            ceiling, lo);
	return lo;
}

void LargestPool(const Fx& fx, const std::string& adapter) {
	const uint32_t pages = LargestAdmittedPool(fx);
	if (pages == 0) return;
	GpuPooledBackend gpu(fx, pages);
	if (!gpu.pool) return;
	SRun d(gpu, true);
	void* a = nullptr;
	void* z = nullptr;
	std::vector<void*> fillers2, fillers;
	G1A_CHECK(d.Create(64, &a));
	for (;;) {  // whole-cap fillers until the pool refuses one
		void* f = nullptr;
		if (gpu.Create(static_cast<int32_t>(fx.geo.context_cap), &f) != 0) break;
		fillers.push_back(f);
	}
	for (;;) {  // then 2-page fillers until it refuses one (at most one page is left)
		void* f = nullptr;
		if (gpu.Create(BudgetForPages(fx, 2), &f) != 0) break;
		fillers2.push_back(f);
	}
	G1A_CHECK_MSG(fillers2.size() >= 2, "4.8 %s: the fill left fewer than two 2-page fillers", fx.name.c_str());
	for (int i = 0; i < 2 && !fillers2.empty(); ++i) {
		gpu.Release(fillers2.back());
		fillers2.pop_back();
	}
	G1A_CHECK(d.Create(17, &z));  // R(17) = 3 pages, drawn last
	const int32_t V = fx.vocab;
	const bool ok = a && z && d.Prefill(a, pkv::Stream(91, 40, V)) && d.Decode(a, 8) && d.Mark("a", a) &&
	                d.Prefill(z, pkv::Stream(92, 12, V)) && d.Decode(z, 5) && d.Mark("z", z);
	G1A_CHECK_MSG(ok, "4.8 %s largest pool: %s", fx.name.c_str(), d.fail.c_str());
	// The CPU twins, on a small CPU page pool.
	CpuPagedBackend cpu(fx, 16);
	SRun c(cpu, true);
	void* ca = nullptr;
	void* cz = nullptr;
	(void)(c.Create(64, &ca) && c.Create(17, &cz) && c.Prefill(ca, pkv::Stream(91, 40, V)) && c.Decode(ca, 8) &&
	       c.Mark("a", ca) && c.Prefill(cz, pkv::Stream(92, 12, V)) && c.Decode(cz, 5) && c.Mark("z", cz));
	ExpectSameRecords(d.records, c.records, Cmp::kTokensAndRows,
	                  "4.8 " + fx.name + " pool of " + std::to_string(pages) + " pages " + adapter);
	// The vacuity guard: one of A's or Z's mapped pages ends above 2^31 bytes into the pool.
	uint64_t top = 0;
	for (void* s : {a, z})
		if (s)
			for (uint32_t pg : GpuTable(static_cast<SslmGpuSequenceHandle*>(s)))
				top = std::max<uint64_t>(top, (uint64_t{pg} + 1) * fx.page_bytes);
	G1A_CHECK_MSG(top > (uint64_t{1} << 31),
	              "4.8 %s: construction: neither the first nor the last draw maps a page above 2^31 bytes (top %llu)",
	              fx.name.c_str(), static_cast<unsigned long long>(top));
	if (ca) c.Release(ca);
	if (cz) c.Release(cz);
	for (void* f : fillers) gpu.Release(f);
	for (void* f : fillers2) gpu.Release(f);
	if (a) d.Release(a);
	if (z) d.Release(z);
}

void Cell48Paged() {
	const std::string adapter = AdapterTag();
	const Fx* fxs[] = {&PkvFixture("pkv_def"), &PkvFixture("pkv_qk"), &Real05(), &Real15()};
	for (const Fx* f : fxs) {
		const Fx& fx = *f;
		if (!fx.ok) continue;
		GpuPooledBackend gpu(fx, static_cast<uint32_t>(fx.CapPages()));
		CpuPagedBackend cpu(fx, static_cast<uint32_t>(fx.CapPages()));
		SRun g(gpu, true), c(cpu, true);
		ShortRun(g);
		ShortRun(c);
		G1A_CHECK_MSG(g.fail.empty() && c.fail.empty(), "4.8 paged %s: %s%s", fx.name.c_str(), g.fail.c_str(), c.fail.c_str());
		ExpectSameRecords(g.records, c.records, Cmp::kTokensAndRows, "4.8 paged B=16 " + fx.name + " " + adapter);
	}
	for (const Fx* f : {&Real05(), &Real15()})
		if (f->ok) LargestPool(*f, adapter);
}

// ---- 6.5b [G3] ------------------------------------------------------------------------------------
// GPU paged (budgeted sequences, B = 16) on R0's long_prefill and widths: tokens, context length and
// K/V rows [0, L) through the GPU mirror peek equal the CPU paged twin (budget holders in a CPU page
// pool, rows from the holder's SSB6 save) and the R0 reference, on pkv_def and pkv_qk, on this GPU.
// kills: a page-table address off by a page or a row on either vendor; a page run in the attention
// loop that reads the next table entry early (the widths at 15, 16 and 17); a cap-width view (4096).
void Cell65bG3() {
	const std::string adapter = AdapterTag();
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const Fx& fx = PkvFixture(stem);
		if (!fx.ok) continue;
		const uint32_t pages = static_cast<uint32_t>(fx.CapPages()) + 4;
		GpuPooledBackend gpu(fx, pages);
		CpuPagedBackend cpu(fx, pages);
		for (int scen = 0; scen < 2; ++scen) {
			const char* name = scen == 0 ? "long_prefill" : "widths";
			SRun g(gpu, true), c(cpu, true);
			if (scen == 0) {
				LongPrefill(g);
				LongPrefill(c);
			} else {
				Widths(g);
				Widths(c);
			}
			G1A_CHECK_MSG(g.fail.empty() && c.fail.empty(), "6.5b %s %s: %s%s", stem, name, g.fail.c_str(), c.fail.c_str());
			const std::string what = std::string("6.5b ") + stem + " " + name + " " + adapter;
			ExpectSameRecords(g.records, c.records, Cmp::kTokensAndRows, what + " vs CPU paged");
			ExpectReference("v1.11.0", fx, name, g.records, Cmp::kTokensAndRows, what + " vs R0");
		}
	}
}

// ---- 8.13 [G3] paging x dispatch budget ----------------------------------------------------------------
// A pooled GPU sequence decodes through the asynchronous verbs by hand with a dispatch budget of one
// layer (each token split over separate submissions, the page mapped at its first layer and read at
// its last) and of full depth, and through the bridge decode at one layer; on long_prefill (decode
// across the page boundary at 1,504) and widths (the boundary at 16). Each run's tokens and rows equal
// the R0 reference. kills: a table update recorded only at a token's first submission and lost by the
// next (the one-layer split), or recorded twice; a page mapped per submission instead of per token.
void Cell813() {
	for (const char* stem : {"pkv_def", "pkv_qk"}) {
		const Fx& fx = PkvFixture(stem);
		if (!fx.ok) continue;
		struct Mode {
			GpuDecodeMode mode;
			uint32_t budget;
			const char* name;
		};
		const Mode modes[] = {{GpuDecodeMode::kAsyncByHand, fx.one_layer, "by hand, one layer"},
		                      {GpuDecodeMode::kAsyncByHand, fx.full_depth, "by hand, full depth"},
		                      {GpuDecodeMode::kBridge, fx.one_layer, "bridge, one layer"}};
		for (const Mode& m : modes) {
			GpuPooledBackend gpu(fx, static_cast<uint32_t>(fx.CapPages()) + 4);
			gpu.mode = m.mode;
			gpu.decode_budget = m.budget;
			SRun lp(gpu, true), wd(gpu, true);
			LongPrefill(lp);
			Widths(wd);
			const std::string what = std::string("8.13 ") + stem + " " + m.name;
			G1A_CHECK_MSG(lp.fail.empty() && wd.fail.empty(), "%s: %s%s", what.c_str(), lp.fail.c_str(), wd.fail.c_str());
			ExpectReference("v1.11.0", fx, "long_prefill", lp.records, Cmp::kTokensAndRows, what);
			ExpectReference("v1.11.0", fx, "widths", wd.records, Cmp::kTokensAndRows, what);
		}
	}
}

// ---- 8.14 [G3] paging x batch decode (dedicated + pooled in one call) -------------------------------------
// A dedicated sequence D (prefill 30) and a pooled sequence P (prefill 14, so its decode crosses the
// page boundary at 16) decode 12 steps, each step one sslm_decode_step_batch_gpu over both (the order
// alternates, D first on even steps), then each finishes its token. Each sequence's tokens and rows
// equal the same sequence decoded alone, and its CPU twin's (a no-budget holder for D, a budget holder
// for P). kills: the pool latch and the per-handle latch confused within one batch (P's table read
// under D's binding, or D's buffer addressed through P's table); a batch that maps P's page for the
// wrong slot.
std::vector<int32_t> Solo(Backend& be, int32_t budget, const std::vector<int32_t>& prompt, int32_t n, std::vector<uint8_t>* rows,
                          int64_t L) {
	SRun r(be, false);
	void* s = nullptr;
	std::vector<int32_t> toks;
	if (r.Create(budget, &s) && r.Prefill(s, prompt) && r.Decode(s, n)) toks = r.pending;
	G1A_CHECK_MSG(r.fail.empty(), "8.14 solo %s: %s", be.Name(), r.fail.c_str());
	if (s) {
		G1A_CHECK_MSG(be.Rows(s, L, rows), "8.14 solo %s: rows unreadable", be.Name());
		r.Release(s);
	}
	return toks;
}

void Cell814() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	const int32_t V = fx.vocab, kSteps = 12;
	const std::vector<int32_t> pd = pkv::Stream(141, 30, V), pp = pkv::Stream(142, 14, V);
	GpuPooledBackend pooled(fx, 16);
	if (!pooled.pool) return;
	SslmGpuSequenceHandle* D = nullptr;
	SslmGpuSequenceHandle* P = nullptr;
	G1A_CHECK_GPU(sslm_gpu_seq_create(Ctx(), fx.gpu, fx.geo.context_cap, &D), GS::SSLM_OK);
	G1A_CHECK_GPU(sslm_gpu_seq_create_budgeted(Ctx(), fx.gpu, pooled.pool, 64, &P), GS::SSLM_OK);
	if (!D || !P) return;
	G1A_CHECK_GPU(SslmGpuSeqPrefillPromptForG5Bridge(Ctx(), D, pd.data(), 30, fx.one_layer), GS::SSLM_OK);
	G1A_CHECK_GPU(SslmGpuSeqPrefillPromptForG5Bridge(Ctx(), P, pp.data(), 14, fx.one_layer), GS::SSLM_OK);
	std::vector<int32_t> td, tp;
	int32_t t = -1;
	G1A_CHECK_GPU(SslmGpuSeqFinishTokenForG5Bridge(Ctx(), D, &t), GS::SSLM_OK);  // the ready tokens
	td.push_back(t);
	G1A_CHECK_GPU(SslmGpuSeqFinishTokenForG5Bridge(Ctx(), P, &t), GS::SSLM_OK);
	tp.push_back(t);
	for (int step = 0; step < kSteps; ++step) {
		G1A_CHECK_GPU(sslm_gpu_seq_embed_token(Ctx(), D, td.back()), GS::SSLM_OK);
		G1A_CHECK_GPU(sslm_gpu_seq_embed_token(Ctx(), P, tp.back()), GS::SSLM_OK);
		SslmGpuSequenceHandle* order[2] = {D, P};
		if (step % 2) std::swap(order[0], order[1]);
		for (int guard = 0; guard < 64 && (*SslmGpuSeqHandleLayerIndexForBench(D) < fx.geo.layers ||
		                                   *SslmGpuSeqHandleLayerIndexForBench(P) < fx.geo.layers); ++guard) {
			SslmGpuStatus st[2] = {GS::SSLM_DEVICE_LOST, GS::SSLM_DEVICE_LOST};
			G1A_CHECK_GPU(sslm_decode_step_batch_gpu(Ctx(), order, nullptr, 2, 2 * fx.full_depth, st), GS::SSLM_OK);
			G1A_CHECK_MSG(st[0] == GS::SSLM_OK && st[1] == GS::SSLM_OK, "8.14 step %d: batch statuses %s, %s", step,
			              GpuStatusName(st[0]), GpuStatusName(st[1]));
			G1A_CHECK_GPU(DrainGpu(D), GS::SSLM_OK);
			G1A_CHECK_GPU(DrainGpu(P), GS::SSLM_OK);
			if (st[0] != GS::SSLM_OK || st[1] != GS::SSLM_OK) break;
		}
		G1A_CHECK_GPU(SslmGpuSeqFinishTokenForG5Bridge(Ctx(), D, &t), GS::SSLM_OK);
		td.push_back(t);
		G1A_CHECK_GPU(SslmGpuSeqFinishTokenForG5Bridge(Ctx(), P, &t), GS::SSLM_OK);
		tp.push_back(t);
	}
	const int64_t LD = 30 + kSteps, LP = 14 + kSteps;
	std::vector<uint8_t> rd, rp;
	G1A_CHECK_MSG(GpuDedicatedRows(fx, D, LD, &rd), "8.14: D's rows unreadable");
	G1A_CHECK_MSG(GpuPooledRows(fx, pooled.pool, P, LP, &rp), "8.14: P's rows unreadable");
	sslm_gpu_seq_release(Ctx(), D);
	sslm_gpu_seq_release(Ctx(), P);
	// Each alone, on the GPU, and on the CPU.
	std::vector<uint8_t> rd_solo, rp_solo, rd_cpu, rp_cpu;
	GpuDedicatedBackend ded(fx);
	GpuPooledBackend pool2(fx, 16);
	CpuPagedBackend cpu_legacy(fx, static_cast<uint32_t>(fx.CapPages()), true);
	CpuPagedBackend cpu_budget(fx, 16);
	const std::vector<int32_t> td_solo = Solo(ded, 0, pd, kSteps + 1, &rd_solo, LD);
	const std::vector<int32_t> tp_solo = Solo(pool2, 64, pp, kSteps + 1, &rp_solo, LP);
	const std::vector<int32_t> td_cpu = Solo(cpu_legacy, 0, pd, kSteps + 1, &rd_cpu, LD);
	const std::vector<int32_t> tp_cpu = Solo(cpu_budget, 64, pp, kSteps + 1, &rp_cpu, LP);
	G1A_CHECK_MSG(td == td_solo && rd == rd_solo, "8.14: D in the mixed batch differs from D decoded alone");
	G1A_CHECK_MSG(tp == tp_solo && rp == rp_solo, "8.14: P in the mixed batch differs from P decoded alone");
	G1A_CHECK_MSG(td == td_cpu && rd == rd_cpu, "8.14: D differs from its CPU twin");
	G1A_CHECK_MSG(tp == tp_cpu && rp == rp_cpu, "8.14: P differs from its CPU twin");
}

// ---- 11.7 [G3] the GPU coverage guard ----------------------------------------------------------------
// A pooled sequence V (prefill 17, ready token taken, so its next row 17 is on its second page) beside
// a neighbour N (prefill 40). The table-one-short fault is armed on V and its next token submitted:
// sslm_decode_step_gpu returns SSLM_SEQUENCE_REJECTED, the ladder's rung is KvPageUnmapped, V is not
// left Submitted, and the whole GPU mirror is unchanged through the GPU mirror peek; N's next tokens
// equal a twin that never saw V's refused call. The mutant with the guard removed submits: the status
// is SSLM_OK (and the row lands through the missing entry), so the refusal is absent (the registry's
// arithmetic entry).
void Cell117() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	const uint32_t kPages = 16;
	std::vector<int32_t> n_tokens[2];
	for (int run = 0; run < 2; ++run) {
		GpuPooledBackend gpu(fx, kPages);
		if (!gpu.pool) return;
		SRun d(gpu, false);
		void* n = nullptr;
		void* v = nullptr;
		G1A_CHECK(d.Create(64, &n) && d.Create(64, &v) && d.Prefill(n, pkv::Stream(175, 40, fx.vocab)) &&
		          d.Prefill(v, pkv::Stream(176, 17, fx.vocab)) && d.Decode(v, 1));
		if (!n || !v) return;
		SslmGpuSequenceHandle* V = static_cast<SslmGpuSequenceHandle*>(v);
		if (run == 0) {
			const std::vector<uint8_t> before = WholeMirror(fx, gpu.pool, kPages);
			G1A_CHECK_GPU(sslm_gpu_seq_embed_token(Ctx(), V, d.pending.back()), GS::SSLM_OK);
			G1A_CHECK_GPU(sslm_gpu_test_only_arm_view_fault(V, kViewFaultTableOneShort), GS::SSLM_OK);
			G1A_CHECK_GPU(sslm_decode_step_gpu(Ctx(), V, nullptr, fx.full_depth), GS::SSLM_SEQUENCE_REJECTED);
			int32_t rung = -1;
			G1A_CHECK_GPU(sslm_gpu_test_only_last_ladder_status(V, &rung), GS::SSLM_OK);
			G1A_CHECK_EQ(rung, static_cast<int32_t>(superslm::SslmForwardStatus::KvPageUnmapped));
			int32_t ready = 0;
			SslmGpuStatus out_status = GS::SSLM_OK;
			G1A_CHECK_GPU(sslm_gpu_ready(Ctx(), V, 0, &ready, &out_status), GS::SSLM_OK);
			G1A_CHECK_MSG(ready == 1, "11.7: the refused sequence was left in flight");
			G1A_CHECK_MSG(WholeMirror(fx, gpu.pool, kPages) == before, "11.7: the refused submission changed the GPU mirror");
		}
		d.pending.clear();
		G1A_CHECK(d.Decode(n, 6));
		n_tokens[run] = d.pending;
		d.Release(v);
		d.Release(n);
	}
	G1A_CHECK_MSG(n_tokens[0] == n_tokens[1] && n_tokens[0].size() == 6, "11.7: the neighbour's tokens moved");
}

// ---- 11.9 [G3] guard 5 at a multi-page view, the GPU twin -------------------------------------------------
// B = 16 < cap = 4096. A pooled sequence V (prefill 16, ready token taken, next row 16 opens its second
// page) beside a neighbour N (prefill 40). Each fault in turn, on a fresh pool: page_bytes one byte
// short -> WorkspaceTooSmall; a null pool base -> WorkspaceTooSmall; an overflowing per-page product ->
// InvalidContextCap. Each: sslm_decode_step_gpu returns SSLM_SEQUENCE_REJECTED before any write, the
// ladder's rung is the one named, and the neighbouring pages' mirror rows (the whole mirror) are
// unchanged through the GPU mirror peek. The mutant with the check removed writes with the short
// stride, so the last row's write crosses into the next page and the peek sees it (the registry's
// arithmetic entry, 11.9's GPU half); in the overflow case a flat-product-only check behaves the same
// (the registry's arithmetic entry, stated, not claimed as isolation).
void Cell119G3() {
	const Fx& fx = PkvFixture("pkv_def");
	if (!fx.ok) return;
	G1A_CHECK_MSG(fx.B < fx.geo.context_cap, "11.9: construction: the view is not multi-page");
	struct Case {
		uint32_t fault;
		superslm::SslmForwardStatus want;
		const char* name;
	};
	const Case cases[] = {{kViewFaultPageBytesShort, superslm::SslmForwardStatus::WorkspaceTooSmall, "page_bytes - 1"},
	                      {kViewFaultNullPoolBase, superslm::SslmForwardStatus::WorkspaceTooSmall, "null pool base"},
	                      {kViewFaultProductOverflow, superslm::SslmForwardStatus::InvalidContextCap, "overflowing product"}};
	const uint32_t kPages = 16;
	for (const Case& c : cases) {
		GpuPooledBackend gpu(fx, kPages);
		if (!gpu.pool) return;
		SRun d(gpu, false);
		void* n = nullptr;
		void* v = nullptr;
		G1A_CHECK(d.Create(64, &n) && d.Create(64, &v) && d.Prefill(n, pkv::Stream(195, 40, fx.vocab)) &&
		          d.Prefill(v, pkv::Stream(196, 16, fx.vocab)) && d.Decode(v, 1));
		if (!n || !v) return;
		SslmGpuSequenceHandle* V = static_cast<SslmGpuSequenceHandle*>(v);
		const std::vector<uint8_t> before = WholeMirror(fx, gpu.pool, kPages);
		G1A_CHECK_GPU(sslm_gpu_seq_embed_token(Ctx(), V, d.pending.back()), GS::SSLM_OK);
		G1A_CHECK_GPU(sslm_gpu_test_only_arm_view_fault(V, c.fault), GS::SSLM_OK);
		G1A_CHECK_GPU(sslm_decode_step_gpu(Ctx(), V, nullptr, fx.full_depth), GS::SSLM_SEQUENCE_REJECTED);
		int32_t rung = -1;
		G1A_CHECK_GPU(sslm_gpu_test_only_last_ladder_status(V, &rung), GS::SSLM_OK);
		G1A_CHECK_MSG(rung == static_cast<int32_t>(c.want), "11.9 %s: ladder rung %d, want %d", c.name, rung,
		              static_cast<int>(c.want));
		G1A_CHECK_MSG(WholeMirror(fx, gpu.pool, kPages) == before, "11.9 %s: a write reached the mirror", c.name);
		d.Release(v);
		d.Release(n);
	}
}

}  // namespace

G1A_CELL("1.7/G3", "G3", Cell17G3);
G1A_CELL("4.8/G3", "G3", Cell48Paged);
G1A_CELL("6.5b/G3", "G3", Cell65bG3);
G1A_CELL("8.13", "G3", Cell813);
G1A_CELL("8.14", "G3", Cell814);
G1A_CELL("11.7", "G3", Cell117);
G1A_CELL("11.9/G3", "G3", Cell119G3);
