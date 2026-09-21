// T-2900 (Curie) -- plan Sec3.10.1/Sec3.10.3 row 9/Sec3.10.6: the GPU 'SLM5' save/restore cell --
// a restored copy equal to the original at every reachable save point, over the schema-binding
// triple (`bound_schema_index`, `dfa_walk_state`, `ready_for_logits`) T-2895's own blob format
// adds. Closes TE-362's own fracture (`Claude/Loki/te362-gpu-plan-restrike-2026-09-19.md`): before
// T-2895, `sslm_gpu_seq_save`/`sslm_gpu_seq_restore` dropped the triple entirely, so a restored,
// schema-bound sequence decoded outside its own grammar.
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED. Driving logic (five save points, the two control decodes
// after restore, the rebind-repair leg) is `Claude/Loki/te362-probe/te362_gpu.cpp` -- already
// executed against the real shipped GPU library and cited by the plan (Sec3.10.1/Sec3.10.3 row 9)
// as the authored-cell source. Converted from printf-verdict probing into CHECK-asserted cells,
// dropping the CPU-side census this file's own probe carried alongside it (CPU's own save/restore
// format is unaffected by T-2895's GPU-only delta, and CPU's own twin is
// `cell_cpu_deadend_retry_reset.cpp`'s `CheckSaveRestoreRoundTrip`); the calls, save points and
// fixture are unchanged.
//
// FIXTURE. --model=PATH, the C39 synthetic (`t2199_s8_fixture.sslm`), schema
// `g5_minimal_one_field`.
//
// SAVE POINTS (identical to TE-362's own construction):
//   P0 -- bound, prompt only (walk unused/0, no schema-content progress yet).
//   P1 -- accepting, token pending (route D after decode #1: the accepting token was just
//         produced but the walk has not yet advanced past it in this construction).
//   P2 -- dead end, route R.
//   P3 -- dead end, route D.
//   P4 -- dead end, route C (at the context cap).
//
// RED AT a3f89cb (pristine, pre-T-2895): the schema-binding triple is dropped on save/restore
// entirely -- the restored copy decodes as if UNBOUND, producing a token the original's own
// schema would refuse at its own walk state (TE-362's own finding: 0 of 5 save points match, 7 of
// 7 out-of-grammar decodes on the restored copies, 8 of 8 restored tokens not admitted by the
// schema). This cell's own AS_BUILT run confirms it.
//
// Run: cell_gpu_slm5_saverestore.exe --model=PATH
#include "fixture_common.h"

namespace {
struct GDec {
	SslmGpuStatus st = SSLM_OK;
	int32_t out = 12345;
	uint32_t walk = 0;
	long long ctx = 0;
};
GDec Decode(const GpuModelFixture& fx, SslmGpuSequenceHandle* s, int32_t tok) {
	GDec d;
	d.st = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, s, tok, fx.one_layer_budget, &d.out);
	d.walk = SslmGpuSeqWalkStateForG5Bridge(s);
	d.ctx = static_cast<long long>(ContextLength(s));
	return d;
}

// Builds the original at save point `pt`; *last receives the most recently produced token (or -1).
SslmGpuSequenceHandle* Build(const GpuModelFixture& fx, int pt, const std::vector<int32_t>& P, int32_t t0,
                             int32_t other, int32_t* last) {
	SslmGpuSequenceHandle* s = nullptr;
	sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &s);
	*last = -1;
	auto step = [&]() {
		GDec d = Decode(fx, s, *last >= 0 ? *last : other);
		if (d.st == SSLM_OK && d.out >= 0) *last = d.out;
	};
	int32_t consumed = -1;
	if (pt == 4) {
		std::vector<int32_t> fill;
		for (int64_t i = 0; i < fx.model_cap - 1; ++i) fill.push_back(P[static_cast<size_t>(i) % P.size()]);
		Prefill(fx, s, fill);
		SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, s, 0);
		SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, s, &t0, 1, fx.one_layer_budget, &consumed);
		step();
		step();
		return s;
	}
	Prefill(fx, s, P);
	SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, s, 0);
	if (pt == 0) return s;
	if (pt == 1) {
		step();
		return s;
	}
	if (pt == 2) {
		SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, s, &t0, 1, fx.one_layer_budget, &consumed);
		step();
		step();
		return s;
	}
	if (pt == 3) {
		step();
		step();
		step();
		return s;
	}
	return s;
}

SslmGpuSequenceHandle* SaveRestore(const GpuModelFixture& fx, SslmGpuSequenceHandle* o, SslmGpuStatus* sv,
                                   SslmGpuStatus* rs) {
	size_t need = 0;
	sslm_gpu_seq_save(fx.ctx, o, nullptr, &need);
	std::vector<uint8_t> blob(need);
	size_t n = need;
	*sv = sslm_gpu_seq_save(fx.ctx, o, blob.data(), &n);
	SslmGpuSequenceHandle* r = nullptr;
	*rs = sslm_gpu_seq_restore(fx.ctx, fx.model, blob.data(), n, &r);
	return r;
}

const char* kPointName[5] = {"P0 bound, prompt only (walk 0)", "P1 accepting, token pending (route D after #1)",
                             "P2 dead end, route R", "P3 dead end, route D", "P4 dead end, route C (at cap)"};
}  // namespace

int main(int argc, char** argv) {
	std::string model_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
	}
	if (model_path.empty()) {
		std::printf("SKIP cell_gpu_slm5_saverestore -- needs --model=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	GpuModelFixture fx;
	CHECK_MSG(fx.Open(model_path, ctx), "GPU open failed");
	IndependentSchema ds;
	CHECK(ds.Build(fx.bytes, "g5_minimal_one_field"));
	if (GFailures > 0 || !ds.entry) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}
	const int32_t t0 = ds.FirstLegal(0);
	const std::vector<int32_t> P = fx.TokensP();
	const int32_t other = fx.OtherToken();

	int gpu_match = 0, gpu_rebind_match = 0;
	for (int pt = 0; pt < 5; ++pt) {
		std::printf("\n[GPU] %s\n", kPointName[pt]);
		int32_t last = -1;
		SslmGpuSequenceHandle* o = Build(fx, pt, P, t0, other, &last);
		SslmGpuStatus sv, rs;
		SslmGpuSequenceHandle* r = SaveRestore(fx, o, &sv, &rs);
		CHECK_MSG(sv == SSLM_OK, "[pt%d] save returned %s", pt, StatusName(sv));
		CHECK_MSG(rs == SSLM_OK && r, "[pt%d] restore returned %s", pt, StatusName(rs));
		if (!r) {
			sslm_gpu_seq_release(fx.ctx, o);
			continue;
		}
		bool same = true;
		int32_t lo = last, lr = last;
		for (int k = 1; k <= 2; ++k) {
			const int32_t to = lo >= 0 ? lo : other, tr = lr >= 0 ? lr : other;
			const uint32_t wb = SslmGpuSeqWalkStateForG5Bridge(o);
			GDec a = Decode(fx, o, to), c = Decode(fx, r, tr);
			if (a.st == SSLM_OK && a.out >= 0) lo = a.out;
			if (c.st == SSLM_OK && c.out >= 0) lr = c.out;
			// A token the ORIGINAL's schema would not admit at its own walk state before this
			// call, produced by the RESTORED copy at SSLM_OK -- decoding out of grammar, TE-362's
			// own headline defect.
			if (c.st == SSLM_OK && c.out >= 0 && a.out < 0) {
				uint32_t nx = 0;
				CHECK_MSG(false,
				          "[pt%d] call #%d: restored copy produced %d at SSLM_OK where the original dead-ended "
				          "(-- out-of-grammar decode)",
				          pt, k, c.out);
				(void)nx;
			}
			if (a.st != c.st || a.out != c.out || a.walk != c.walk || a.ctx != c.ctx) same = false;
		}
		CHECK_MSG(same, "[pt%d] restored copy diverges from the un-saved original over the next two decode calls",
		          pt);
		if (same) ++gpu_match;
		sslm_gpu_seq_release(fx.ctx, r);

		// Leg 3: the caller's only repair before T-2895 -- re-bind the schema on a second restored
		// copy, saved from a FRESH original at the same save point.
		//
		// T-2903 fix. T-2900's original assertion (`rb == SSLM_OK`, unconditionally) assumed the
		// AS_BUILT shape: a restored copy that dropped its schema binding entirely, so rebinding
		// it is an ordinary bind-onto-unbound call. Under FIXED that assumption is exactly what
		// T-2895 closes -- the restored copy already carries the ORIGINAL's own binding and walk
		// state through the round trip -- so the SAME call is now a rebind ATTEMPT on an
		// ALREADY-bound sequence, and the plan (Sec3.10.1) states the disposition explicitly: "a
		// re-bind on an already-bound restored sequence now correctly refuses
		// SSLM_SEQUENCE_REJECTED, mirroring the CPU ABI's own 'no mid-generation re-binding' rule."
		// That CPU rule (Sec3.10.5) allows a rebind only when the sequence is unbound or its own
		// walk sits at the schema's start; it refuses once real schema-content progress exists.
		//
		// The two variants differ in EXACTLY the fact under test (whether restore preserves the
		// binding at all), so the expected disposition cannot be hardcoded by `pt` alone -- it
		// would need to already know which variant is linked. Instead it is read directly off
		// `r2`'s own observable walk state right after restore, before the rebind call: the
		// "unused" sentinel means unbound (AS_BUILT, every point, by T-2895's own 0-of-5 finding);
		// the schema's own start state (0) means bound but not yet progressed (FIXED, pt0 only,
		// where Build() only prefills the prompt and binds -- no schema-content token consumed
		// yet); anything else means bound with real progress (FIXED, pt1-4: an accepting token
		// consumed or a dead end reached). This is a live read of the precondition the rule is
		// stated over, not a per-point guess -- and it makes the same assertion correct under
		// both AS_BUILT (unbound -> OK, all 5 points) and FIXED (start-state -> OK at pt0,
		// progressed -> REJECTED at pt1-4) without knowing which is linked.
		int32_t last2 = -1;
		SslmGpuSequenceHandle* o2 = Build(fx, pt, P, t0, other, &last2);
		SslmGpuSequenceHandle* r2 = SaveRestore(fx, o2, &sv, &rs);
		const uint32_t wr2_pre_rebind = SslmGpuSeqWalkStateForG5Bridge(r2);
		const bool r2_unbound = (wr2_pre_rebind == 0xFFFFFFFFu);
		const bool r2_bound_progressed = !r2_unbound && (wr2_pre_rebind != 0u);
		const SslmGpuStatus rb = SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, r2, 0);
		if (r2_bound_progressed) {
			CHECK_MSG(rb == SSLM_SEQUENCE_REJECTED,
			          "[pt%d] rebind on an already-bound, schema-content-progressed restored copy "
			          "(walk=%u) should be REFUSED (no mid-generation re-binding, mirroring the CPU "
			          "ABI's own rule) -- got %s",
			          pt, wr2_pre_rebind, StatusName(rb));
		} else {
			CHECK_MSG(rb == SSLM_OK,
			          "[pt%d] rebind on a restored copy that is unbound or at its own schema start "
			          "(walk=%u, not mid-generation) should succeed -- got %s",
			          pt, wr2_pre_rebind, StatusName(rb));
		}
		bool same2 = true;
		int32_t lo2 = last2, lr2 = last2;
		for (int k = 1; k <= 2; ++k) {
			const int32_t to = lo2 >= 0 ? lo2 : other, tr = lr2 >= 0 ? lr2 : other;
			GDec a = Decode(fx, o2, to), c = Decode(fx, r2, tr);
			if (a.st == SSLM_OK && a.out >= 0) lo2 = a.out;
			if (c.st == SSLM_OK && c.out >= 0) lr2 = c.out;
			if (a.st != c.st || a.out != c.out || a.walk != c.walk || a.ctx != c.ctx) same2 = false;
		}
		CHECK_MSG(same2, "[pt%d] rebound copy diverges from a fresh original over the next two decode calls", pt);
		if (same2) ++gpu_rebind_match;
		sslm_gpu_seq_release(fx.ctx, r2);
		sslm_gpu_seq_release(fx.ctx, o2);
		sslm_gpu_seq_release(fx.ctx, o);
	}
	std::printf("SUMMARY: restored copy == un-saved original at %d of 5 save points; re-bound restored copy == "
	            "original at %d of 5 save points\n",
	            gpu_match, gpu_rebind_match);

	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
