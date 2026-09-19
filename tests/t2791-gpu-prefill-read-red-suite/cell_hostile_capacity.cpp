// T-2791 (Curie) -- plan Sec3.4 row 2 (trust boundaries and hostile inputs), row 4's capacity
// matrix, row 5's "every status in Sec3.1 reached by construction" and its check ORDER, and
// row 7's "nothing written on a short buffer" (Sec2.6 M19); plus the width query.
//
// ORACLES. Status and no-write expectations are plan Sec3.1's contract text, check by check:
//   1 malformed handle or out-pointer -> SSLM_SEQUENCE_KV_BUFFER_MISMATCH, writes nothing
//     (row 2: "each refuses by the status in Sec3.1 and writes nothing"); `out_codes` may be null
//     only when `out_capacity == 0`; the width query refuses a null model or out-pointer the same way;
//   2 a Submitted sequence -> SSLM_BUSY;
//   3 no snapshot -> SSLM_PREFILL_HIDDEN_UNAVAILABLE;
//   4 `*out_required = hidden_size` from here on; capacity < hidden -> SSLM_OUTPUT_BUFFER_TOO_SMALL,
//     nothing written to out_codes, out_scale_m, out_scale_e;
//   6 success writes exactly hidden_size codes and the scale.
// The frame a successful read must return is final_norm of the live residual immediately after
// the prefill (fixture_common.h, OracleFromLive); the width is the artifact's own
// config.hidden_size from this file's independent parse. Neither comes from the code under test.
//
// *out_required (plan Sec3.4 row 7, T-2806 N4): H1-H3 and H5-H7 pass a sentinel-seeded *out_required
// and assert it unchanged (check 1 writes nothing; H4 passes it null). O1, O3 and O5 assert the same for
// checks 3 and 2. Plan Sec3.1 check 4 writes *out_required only once checks 1-3 pass.
//
// Every malformed-input cell runs against a sequence that HAS a snapshot, so an implementation
// cannot pass it by refusing for a different reason (no snapshot); every precedence cell pairs two
// violated checks so only the Sec3.1 order yields the asserted status.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden and
// sslm_gpu_model_hidden_size.
//
// Run: cell_hostile_capacity.exe [--qwen3=PATH] [--synthetic=PATH] [--g5fixture=PATH]
#include "fixture_common.h"

namespace {

struct Out {
	std::vector<int8_t> codes;
	int64_t m = kScaleMSentinel, e = kScaleESentinel;
	size_t required = kRequiredSentinel;
	explicit Out(size_t n) : codes(n, kCodeSentinel) {}
	bool CodesUntouched() const {
		for (int8_t c : codes) if (c != kCodeSentinel) return false;
		return true;
	}
	bool ScaleUntouched() const { return m == kScaleMSentinel && e == kScaleESentinel; }
};

void RunOn(const std::string& path, const char* flag, SslmGpuContext* ctx) {
	if (path.empty()) {
		SKIP_MSG("hostile/capacity: %s not supplied", flag);
		return;
	}
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "could not open %s", path.c_str());
		return;
	}
	const char* tag = path.c_str();
	const size_t H = fx.hidden;
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, std::min<int64_t>(512, fx.model_cap), &seq) == SSLM_OK && seq);
	if (!seq) { fx.Close(); return; }
	CHECK(sslm_gpu_seq_reset(ctx, seq) == SSLM_OK);
	CHECK_MSG(Prefill(fx, seq, fx.TokensP()) == SSLM_OK, "[%s] SETUP: base prefill", tag);
	const Frame FP = OracleFromLive(fx, seq);

	// ---- width query ------------------------------------------------------------------------
	{
		uint32_t w = 0xDEADBEEFu;
		CHECK_MSG(sslm_gpu_model_hidden_size(fx.model, &w) == SSLM_OK && w == H,
		          "[%s] W1: sslm_gpu_model_hidden_size must return SSLM_OK and config.hidden_size %zu (got %u)", tag,
		          H, w);
		CHECK_MSG(sslm_gpu_model_hidden_size(nullptr, &w) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		          "[%s] W2: a null model must return SSLM_SEQUENCE_KV_BUFFER_MISMATCH", tag);
		CHECK_MSG(sslm_gpu_model_hidden_size(fx.model, nullptr) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		          "[%s] W3: a null out_hidden_size must return SSLM_SEQUENCE_KV_BUFFER_MISMATCH", tag);
		uint32_t w2 = 0xDEADBEEFu;
		(void)sslm_gpu_model_hidden_size(nullptr, &w2);
		CHECK_MSG(w2 == 0xDEADBEEFu, "[%s] W2: a refused width query wrote *out_hidden_size", tag);
	}

	// ---- check 1: malformed handles and out-pointers (row 2) --------------------------------
	auto expect_malformed = [&](const char* cell, SslmGpuContext* c, SslmGpuSequenceHandle* s, int8_t* codes,
	                            size_t cap, size_t* req, int64_t* m, int64_t* e, const Out& o) {
		const SslmGpuStatus st = sslm_gpu_seq_read_prefill_final_hidden(c, s, codes, cap, req, m, e);
		CHECK_MSG(st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "[%s] %s: returned %s, want SSLM_SEQUENCE_KV_BUFFER_MISMATCH",
		          tag, cell, StatusName(st));
		CHECK_MSG(o.CodesUntouched() && o.ScaleUntouched() && o.required == kRequiredSentinel,
		          "[%s] %s: a malformed call wrote an out-parameter", tag, cell);
	};
	{ Out o(H); expect_malformed("H1 null ctx", nullptr, seq, o.codes.data(), H, &o.required, &o.m, &o.e, o); }
	{ Out o(H); expect_malformed("H2 null seq", ctx, nullptr, o.codes.data(), H, &o.required, &o.m, &o.e, o); }
	{ Out o(H); expect_malformed("H4 null out_required", ctx, seq, o.codes.data(), H, nullptr, &o.m, &o.e, o); }
	{ Out o(H); expect_malformed("H5 null out_scale_m", ctx, seq, o.codes.data(), H, &o.required, nullptr, &o.e, o); }
	{ Out o(H); expect_malformed("H6 null out_scale_e", ctx, seq, o.codes.data(), H, &o.required, &o.m, nullptr, o); }
	{ Out o(H); expect_malformed("H7 null out_codes, capacity > 0", ctx, seq, nullptr, H, &o.required, &o.m, &o.e, o); }
	{
		// H3: a sequence read through a DIFFERENT, live context.
		SslmGpuContext* other = nullptr;
		CHECK_MSG(sslm_gpu_context_create(GpuContextConfig{}, &other) == SSLM_OK && other,
		          "[%s] H3 SETUP: second context", tag);
		if (other) {
			Out o(H);
			expect_malformed("H3 sequence from another context", other, seq, o.codes.data(), H, &o.required, &o.m,
			                 &o.e, o);
			sslm_gpu_context_destroy(other);
		}
	}
	// The must-accept neighbour of every malformed cell: the same sequence, well-formed.
	{
		const Frame f = ReadVerb(fx, seq);
		CHECK_MSG(f.status == SSLM_OK && f.SameFrame(FP, H) && f.required == H,
		          "[%s] H0 must-accept: a well-formed read of the same sequence returns F_P (%s)", tag, StatusName(f.status));
	}

	// ---- check 4: capacity matrix (row 4) and no-write on a short buffer (row 7, M19) ---------
	{
		const Frame f = ReadVerb(ctx, seq, H - 1);
		CHECK_MSG(f.status == kOutputBufferTooSmall, "[%s] C1 capacity hidden-1: returned %s, want SSLM_OUTPUT_BUFFER_TOO_SMALL",
		          tag, StatusName(f.status));
		CHECK_MSG(f.required == H, "[%s] C1: *out_required=%zu, want %zu", tag, f.required, H);
		CHECK_MSG(f.OutputsUntouched(), "[%s] C1 (M19): a short buffer wrote codes or scale", tag);
	}
	{
		const Frame f = ReadVerb(ctx, seq, 1);
		CHECK_MSG(f.status == kOutputBufferTooSmall && f.required == H && f.OutputsUntouched(),
		          "[%s] C1b capacity 1: %s required=%zu untouched=%d", tag, StatusName(f.status), f.required,
		          f.OutputsUntouched() ? 1 : 0);
	}
	{
		// The query form: out_codes == nullptr with out_capacity == 0.
		Out o(0);
		const SslmGpuStatus st = sslm_gpu_seq_read_prefill_final_hidden(ctx, seq, nullptr, 0, &o.required, &o.m, &o.e);
		CHECK_MSG(st == kOutputBufferTooSmall && o.required == H && o.ScaleUntouched(),
		          "[%s] C2 query form (null, 0): %s required=%zu scale untouched=%d; want SSLM_OUTPUT_BUFFER_TOO_SMALL, "
		          "required=%zu, nothing else written",
		          tag, StatusName(st), o.required, o.ScaleUntouched() ? 1 : 0, H);
	}
	{
		const Frame f = ReadVerb(ctx, seq, H);
		CHECK_MSG(f.status == SSLM_OK && f.SameFrame(FP, H) && f.required == H,
		          "[%s] C3 capacity == hidden (must-accept): %s", tag, StatusName(f.status));
	}
	{
		const Frame f = ReadVerb(ctx, seq, H + 1);
		CHECK_MSG(f.status == SSLM_OK && f.SameFrame(FP, H) && f.required == H,
		          "[%s] C4 capacity hidden+1: %s", tag, StatusName(f.status));
		CHECK_MSG(f.codes.size() == H + 1 && f.codes[H] == kCodeSentinel,
		          "[%s] C4: the byte past hidden_size was written (success writes exactly hidden_size codes)", tag);
	}
	{
		// No alignment requirement on out_codes (Sec3.1 properties): an odd-offset buffer.
		std::vector<int8_t> raw(H + 8, kCodeSentinel);
		int8_t* odd = raw.data() + 1;
		size_t req = 0;
		int64_t m = 0, e = 0;
		const SslmGpuStatus st = sslm_gpu_seq_read_prefill_final_hidden(ctx, seq, odd, H, &req, &m, &e);
		CHECK_MSG(st == SSLM_OK && std::equal(odd, odd + H, FP.codes.begin()) && m == FP.m && e == FP.e &&
		              raw[0] == kCodeSentinel && raw[H + 1] == kCodeSentinel,
		          "[%s] C5 unaligned out_codes: %s", tag, StatusName(st));
	}

	// ---- check order (row 5): each pair of violated checks resolves to the earlier one -------
	{
		// 3 before 4: no snapshot AND a short buffer -> UNAVAILABLE.
		SslmGpuSequenceHandle* fresh = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &fresh) == SSLM_OK && fresh);
		if (fresh) {
			const Frame f = ReadVerb(ctx, fresh, H - 1);
			CHECK_MSG(f.status == kPrefillHiddenUnavailable && f.OutputsUntouched() && f.required == kRequiredSentinel,
			          "[%s] O1 no snapshot + short buffer: %s (required %zu), want SSLM_PREFILL_HIDDEN_UNAVAILABLE and "
			          "*out_required unwritten",
			          tag, StatusName(f.status), f.required);
			// 1 before 3: malformed AND no snapshot -> MISMATCH.
			Out o(H);
			const SslmGpuStatus st =
			    sslm_gpu_seq_read_prefill_final_hidden(ctx, fresh, o.codes.data(), H, &o.required, nullptr, &o.e);
			CHECK_MSG(st == SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "[%s] O2 malformed + no snapshot: %s", tag, StatusName(st));
			// 2 before 3: Submitted AND no snapshot -> BUSY.
			CHECK(sslm_gpu_seq_embed_token(ctx, fresh, fx.SomeToken()) == SSLM_OK);
			CHECK(sslm_decode_step_gpu(ctx, fresh, nullptr, fx.one_layer_budget) == SSLM_OK);
			const Frame b = ReadVerb(ctx, fresh, H);
			CHECK_MSG(b.status == SSLM_BUSY && b.OutputsUntouched() && b.required == kRequiredSentinel,
			          "[%s] O3 Submitted + no snapshot: %s (required %zu), want SSLM_BUSY and *out_required unwritten", tag,
			          StatusName(b.status), b.required);
			// 1 before 2: malformed AND Submitted -> MISMATCH.
			Out o2(H);
			const SslmGpuStatus st2 =
			    sslm_gpu_seq_read_prefill_final_hidden(ctx, fresh, nullptr, H, &o2.required, &o2.m, &o2.e);
			CHECK_MSG(st2 == SSLM_SEQUENCE_KV_BUFFER_MISMATCH, "[%s] O4 malformed + Submitted: %s", tag, StatusName(st2));
			// 2 before 4: Submitted AND short buffer -> BUSY.
			const Frame b2 = ReadVerb(ctx, fresh, H - 1);
			CHECK_MSG(b2.status == SSLM_BUSY && b2.required == kRequiredSentinel,
			          "[%s] O5 Submitted + short buffer: %s (required %zu), want SSLM_BUSY and *out_required unwritten", tag,
			          StatusName(b2.status), b2.required);
			CHECK(Drain(ctx, fresh) == SSLM_OK);
			sslm_gpu_seq_release(ctx, fresh);
		}
	}

	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	RunOn(g_qwen3_path, "--qwen3", ctx);
	RunOn(g_synthetic_path, "--synthetic", ctx);
	RunOn(g_g5_path, "--g5fixture", ctx);
	sslm_gpu_context_destroy(ctx);
	return FinishSuite("cell_hostile_capacity");
}
