// T-2900 (Curie) -- plan `Claude/Plans/te266-gpu-path.md` Sec3.10.1/Sec3.10.2/Sec3.10.4: GPU
// Cell 1, the short-schema dead end plus post-dead-end resumability and resettability, driven
// through the REAL GPU finish bridge on real GPU hardware -- the item T-2899 (Curie) named as
// not authored in its own pass (`Claude/Curie/t2899-stage1-red-suite-2026-09-21.md` Sec5).
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED. The driving logic (routes R, D, C; presenting a DIFFERENT
// caller-supplied token on each repeat call, `caller + k`) is TE-338's own
// `Claude/Loki/te338-probe/probe_gpu.cpp` (the dead-end/resumability legs) and TE-361's own
// `Claude/Loki/te361-probe/probe_gpu_recover.cpp` (the reset/recover/reuse leg, `Recover()`) --
// both already executed against the real shipped GPU library and cited by the plan
// (Sec3.10.1/Sec3.10.4) as the authored-cell source. Converted from print-and-eyeball probes
// into CHECK-asserted cells (this suite's own convention, `cell_cpu_deadend_retry_reset.cpp`);
// the underlying calls, routes and fixture are unchanged. `IndependentSchema` (IndependentSchema
// parse of the SchemaMasks section, not the GPU model's own table) derives t0 exactly as the
// adopted probes did.
//
// FIXTURE. --model=PATH, the C39 synthetic (`tests/t2199-damped-greedy-red-suite`'s own
// `t2199_s8_fixture.sslm`), schema `g5_minimal_one_field` (index 0) -- the same fixture and
// schema TE-338/T-2866's own GPU probe used, and CPU Cell 1's own twin
// (`cell_cpu_deadend_retry_reset.cpp`) uses for the identical construction on the other backend.
//
// ROUTES (identical to TE-338/TE-361's own construction, and to the CPU twin's):
//   R  -- prompt P, then a schema-content prefill of {t0} reaches the dead end from the READY
//         branch (layer_index == 0 already, GPU's own post-prefill convention).
//   D  -- prompt P only; the FIRST decode call itself produces t0 (the accepting token), so the
//         SECOND decode call reaches the dead end from the EMBED branch.
//   C  -- prompt fills context_cap-1, schema-content {t0} fills the cap exactly (the dead end
//         coincides with a saturated context).
//
// ORACLE. The specification (plan Sec3.10.1), not "whatever the shipped GPU already does" --
// TE-336's own census found the unfixed GPU processes a schema dead end as a silent, bogus
// produced token (the finish bridge's own unconditional `next_state` write bug); this cell's own
// acceptance bar is the promise text.
//
// RED AT a3f89cb (pristine, GPU): every repeat-decode call after the first dead end embeds
// the NEW caller-supplied token and advances `dfa_walk_state`/`layer_index` instead of
// reproducing the miss (Cell 1 (i)-(iii)) -- confirmed by this cell's own AS_BUILT run.
//
// (iv) resettability is a CONFIRMING assertion, not a red one: the plan states GPU "already met
// this by construction" (`sslm_gpu_seq_resetImpl` carries no `layer_index` precondition at all,
// unlike CPU's pre-T-2894 wedge) -- this cell executes it on both AS_BUILT and FIXED and expects
// it green on both, which the run below confirms rather than assumes.
//
// GUARD VITALITY, two independent single-line mutants against the reference fix
// (`Claude/Vitruvius/t2895-probe/gpu_1p0_v5.cpp`, T-2900's own `gpu_1p0_mut_*.cpp` copies):
//   MUT_CHECKEDRETURN : the checked return (Sec3.10.1) reverted to a3f89cb's own unconditional
//                        `next_state` write -- must turn Cell 1's own first dead-end assertion
//                        red (a genuine dead end is silently processed as a produced token).
//   MUT_NOREARM       : the checked return stays, `ready_for_logits` re-arm dropped -- must turn
//                        (iii)'s own resumability assertion red while the FIRST dead-end call
//                        stays green (the two assertions are independent, exactly the shape
//                        TE-361/T-2894 found unpinned on the CPU side).
//
// Run: cell_gpu_cell1_shortschema.exe --model=PATH
#include "fixture_common.h"

namespace {

// T-2909 (TE-365 M3): `sslm_gpu_seq_save`'s own byte layout (src/gpu/superslm_gpu.cpp's
// GpuSeqBlobHeader + T-2895's own 12-byte 'SLM5' tail) is private to that translation unit --
// no header declares it, and no bench accessor exposes `ready_for_logits` directly (unlike
// `layer_index`/`live_state.layer_index`, both read below via their own existing ForBench
// accessors). This mirrors the CPU twin's own established technique in this exact suite
// (`cell_cpu_deadend_retry_reset.cpp`'s `BlobContextLength`/`BlobDfaWalkState`): read the field
// back through the PUBLIC `sslm_gpu_seq_save` API's own documented byte layout (the header
// comment above `GpuSeqBlobHeader`/`kGpuSeqBlobV5TailBytes`), never a new production accessor.
// Header = magic(4)+layer_index(4)+hidden_scale_m(8)+hidden_scale_e(8)+kv_saturation_count(8)+
// context_length(8)+hidden_codes_size(8)+workspace_size(8)+model_content_hash(32)+
// kv_landing_saturation_count(8)+k_channel_landing_saturation_count(8)+
// rope_q_saturation_count(8)+rope_k_saturation_count(8) = 120 bytes (every member already
// lands on its own natural alignment boundary, so no compiler padding). The v5 tail then adds
// bound_schema_index(4)/dfa_walk_state(4)/ready_for_logits-as-LE32(4) immediately after --
// `ready_for_logits` is the LE32 at byte offset 128. Pinned by a SETUP self-check against the
// real returned blob size, below, rather than assumed.
constexpr size_t kGpuBlobHeaderBytes = 120;
constexpr size_t kGpuBlobV5TailBytes = 12;
constexpr size_t kGpuBlobReadyOffset = kGpuBlobHeaderBytes + 4 + 4;  // 128

bool ReadGpuReadyForLogits(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, bool* out_ready) {
	size_t need = 0;
	sslm_gpu_seq_save(ctx, seq, nullptr, &need);
	std::vector<uint8_t> blob(need);
	size_t n = need;
	if (sslm_gpu_seq_save(ctx, seq, blob.data(), &n) != SSLM_OK) return false;
	// SETUP self-check: a lower bound only -- the blob's own total length also carries
	// hidden_codes_size and workspace_size (host_kv_mirror, sized to the KV cache capacity at
	// creation and never zero), neither of which this helper's own offset arithmetic depends
	// on. Guards against the header/tail layout moving without pinning an unrelated size.
	if (n < kGpuBlobReadyOffset + 4) return false;
	uint32_t word = 0;
	std::memcpy(&word, blob.data() + kGpuBlobReadyOffset, sizeof(word));
	*out_ready = word != 0;
	return true;
}

// Cell 1 (i): drive `route` to its first dead end. Returns the live handle plus the
// context_length/walk/layer_index observed right after the first -2.
struct DeadEndState {
	SslmGpuSequenceHandle* seq = nullptr;
	int64_t ctx0 = -1;
	uint32_t walk0 = 0xFFFFFFFFu;
	uint32_t layer0 = 0xFFFFFFFFu;
	uint32_t live0 = 0xFFFFFFFFu;
};

DeadEndState ReachDeadEnd(const GpuModelFixture& fx, const std::string& route, const std::vector<int32_t>& P,
                          int32_t t0, int32_t caller) {
	DeadEndState d;
	sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &d.seq);
	CHECK_MSG(d.seq != nullptr, "[%s] sslm_gpu_seq_create failed", route.c_str());
	if (!d.seq) return d;
	CHECK_MSG(SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, d.seq, 0) == SSLM_OK, "[%s] bind schema 0", route.c_str());

	if (route == "R") {
		CHECK_MSG(Prefill(fx, d.seq, P) == SSLM_OK, "[R] prompt prefill");
		int32_t consumed = -1;
		CHECK_MSG(SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, d.seq, &t0, 1, fx.one_layer_budget,
		                                                    &consumed) == SSLM_OK &&
		              consumed == 1,
		          "[R] schema-content prefill {t0}");
		int32_t out = 12345;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, d.seq, caller, fx.one_layer_budget, &out);
		CHECK_MSG(st == SSLM_OK && out == -2, "[R] first decode after the accepting prefill: st=%s out=%d, want -2/OK",
		          StatusName(st), out);
	} else if (route == "D") {
		CHECK_MSG(Prefill(fx, d.seq, P) == SSLM_OK, "[D] prompt prefill");
		int32_t out1 = 12345;
		const SslmGpuStatus st1 = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, d.seq, caller, fx.one_layer_budget, &out1);
		CHECK_MSG(st1 == SSLM_OK && out1 == t0, "[D] first decode should PRODUCE the accepting token %d: st=%s out=%d",
		          t0, StatusName(st1), out1);
		// Must-accept neighbour (plan Sec3.10.2): a produced (non-dead-end) token resets BOTH
		// layer_index mirrors to 0 and does NOT re-arm ready_for_logits -- the caller must still
		// embed the next token itself on the FOLLOWING call, unlike the dead-end's own resting
		// shape. Observed here on route D specifically, the only route where `ReachDeadEnd`
		// itself passes through a genuine production before reaching the dead end.
		CHECK_MSG(LayerIndex(d.seq) == 0, "[D] must-accept neighbour: layer_index is %u after a produced token, want 0",
		          LayerIndex(d.seq));
		CHECK_MSG(SslmGpuSeqHandleLiveStateForBench(d.seq)->layer_index == 0,
		          "[D] must-accept neighbour: live_state.layer_index is %u after a produced token, want 0",
		          SslmGpuSeqHandleLiveStateForBench(d.seq)->layer_index);
		bool ready_after_produce = true;
		CHECK_MSG(ReadGpuReadyForLogits(fx.ctx, d.seq, &ready_after_produce), "[D] must-accept neighbour: save failed");
		CHECK_MSG(!ready_after_produce,
		          "[D] must-accept neighbour: ready_for_logits is armed after a produced token -- the caller's "
		          "next decode call would skip embedding it");
		int32_t out2 = 12345;
		const SslmGpuStatus st2 = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, d.seq, caller + 1, fx.one_layer_budget, &out2);
		CHECK_MSG(st2 == SSLM_OK && out2 == -2, "[D] second decode should dead-end: st=%s out=%d", StatusName(st2), out2);
	} else {  // C
		std::vector<int32_t> fill;
		for (int64_t i = 0; i < fx.model_cap - 1; ++i) fill.push_back(P[static_cast<size_t>(i) % P.size()]);
		CHECK_MSG(Prefill(fx, d.seq, fill) == SSLM_OK, "[C] cap-filling prompt prefill");
		int32_t consumed = -1;
		CHECK_MSG(SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, d.seq, &t0, 1, fx.one_layer_budget,
		                                                    &consumed) == SSLM_OK &&
		              consumed == 1,
		          "[C] schema-content prefill {t0} at the cap");
		int32_t out = 12345;
		const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, d.seq, caller, fx.one_layer_budget, &out);
		CHECK_MSG(st == SSLM_OK && out == -2, "[C] first decode at the cap: st=%s out=%d, want -2/OK", StatusName(st),
		          out);
	}
	d.ctx0 = ContextLength(d.seq);
	d.walk0 = SslmGpuSeqWalkStateForG5Bridge(d.seq);
	d.layer0 = LayerIndex(d.seq);
	d.live0 = SslmGpuSeqHandleLiveStateForBench(d.seq)->layer_index;
	return d;
}

}  // namespace

int main(int argc, char** argv) {
	std::string model_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
	}
	if (model_path.empty()) {
		std::printf("SKIP cell_gpu_cell1_shortschema -- needs --model=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK) {
		std::printf("FAIL cell_gpu_cell1_shortschema -- sslm_gpu_context_create failed\n");
		return 1;
	}
	GpuModelFixture fx;
	if (!fx.Open(model_path, ctx)) {
		std::printf("FAIL cell_gpu_cell1_shortschema -- fixture open failed\n");
		return 1;
	}
	IndependentSchema ds;
	CHECK(ds.Build(fx.bytes, "g5_minimal_one_field"));
	if (!ds.entry) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}
	const int32_t t0 = ds.FirstLegal(0);
	CHECK_MSG(t0 >= 0, "SETUP: no schema-admitted token found at state 0");
	std::printf("schema 'g5_minimal_one_field': states=%u t0=%d\n", ds.entry->state_count, t0);

	const std::vector<int32_t> P = fx.TokensP();
	const int32_t caller = fx.OtherToken();

	// Cell 1(iii) and Cell 1(iv) are checked on INDEPENDENT sequence instances, each freshly
	// driven to the SAME dead end -- identical discipline to the CPU twin
	// (`cell_cpu_deadend_retry_reset.cpp`): (iii)'s own extra decode calls must not be allowed to
	// move the sequence further before (iv) asks whether reset succeeds "after the dead end."
	for (const std::string& route : {std::string("R"), std::string("D"), std::string("C")}) {
		DeadEndState d_iii = ReachDeadEnd(fx, route, P, t0, caller);
		if (d_iii.seq) {
			for (int call = 2; call <= 3; ++call) {
				const int32_t feed = caller + 100 + call;  // a genuinely DIFFERENT token every call
				int32_t out = 12345;
				const SslmGpuStatus st =
				    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, d_iii.seq, feed, fx.one_layer_budget, &out);
				const int64_t ctx = ContextLength(d_iii.seq);
				const uint32_t walk = SslmGpuSeqWalkStateForG5Bridge(d_iii.seq);
				CHECK_MSG(st == SSLM_OK && out == -2,
				          "[%s] Cell1(iii) repeat call #%d (fed a DIFFERENT token %d): st=%s out=%d, want "
				          "-2/SSLM_OK -- a repeat call must dead-end identically, never embed the new token",
				          route.c_str(), call, feed, StatusName(st), out);
				CHECK_MSG(ctx == d_iii.ctx0,
				          "[%s] Cell1(iii) repeat call #%d: context_length moved %lld -> %lld",
				          route.c_str(), call, static_cast<long long>(d_iii.ctx0), static_cast<long long>(ctx));
				CHECK_MSG(walk == d_iii.walk0,
				          "[%s] Cell1(iii) repeat call #%d: dfa_walk_state moved %u -> %u", route.c_str(), call,
				          d_iii.walk0, walk);
				// Cell 1 (ii): a dead-end call -- the first one (captured as d_iii.layer0/live0
				// right after ReachDeadEnd returns) and every repeat call alike -- leaves BOTH
				// layer_index mirrors unchanged.
				CHECK_MSG(LayerIndex(d_iii.seq) == d_iii.layer0,
				          "[%s] Cell1(ii) repeat call #%d: layer_index moved %u -> %u on a dead-end call",
				          route.c_str(), call, d_iii.layer0, LayerIndex(d_iii.seq));
				CHECK_MSG(SslmGpuSeqHandleLiveStateForBench(d_iii.seq)->layer_index == d_iii.live0,
				          "[%s] Cell1(ii) repeat call #%d: live_state.layer_index moved %u -> %u on a dead-end call",
				          route.c_str(), call, d_iii.live0, SslmGpuSeqHandleLiveStateForBench(d_iii.seq)->layer_index);
			}
			sslm_gpu_seq_release(fx.ctx, d_iii.seq);
		}

		DeadEndState d_iv = ReachDeadEnd(fx, route, P, t0, caller);
		if (d_iv.seq) {
			// Cell 1(iv): resettability -- a CONFIRMING assertion (plan Sec3.10.2: "GPU already met
			// this by construction"), executed on every route rather than assumed. Mirrors TE-361's
			// own `Recover()`: reset, unbind, prompt prefill, decode.
			const SslmGpuStatus rst = sslm_gpu_seq_reset(fx.ctx, d_iv.seq);
			CHECK_MSG(rst == SSLM_OK, "[%s] Cell1(iv): sslm_gpu_seq_reset after the dead end returned %s",
			          route.c_str(), StatusName(rst));
			const SslmGpuStatus ub = SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, d_iv.seq, -1);
			CHECK_MSG(ub == SSLM_OK, "[%s] Cell1(iv): unbind after reset returned %s", route.c_str(),
			          StatusName(ub));
			if (rst == SSLM_OK) {
				const SslmGpuStatus pp = Prefill(fx, d_iv.seq, P);
				CHECK_MSG(pp == SSLM_OK, "[%s] Cell1(iv): prompt prefill after reset returned %s", route.c_str(),
				          StatusName(pp));
				int32_t tok = 12345;
				const SslmGpuStatus ds2 =
				    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, d_iv.seq, caller, fx.one_layer_budget, &tok);
				CHECK_MSG(ds2 == SSLM_OK && tok >= 0,
				          "[%s] Cell1(iv): decode after reset+reuse: st=%s out=%d, want a real, non-negative token",
				          route.c_str(), StatusName(ds2), tok);
			}
			sslm_gpu_seq_release(fx.ctx, d_iv.seq);
		}
	}

	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
