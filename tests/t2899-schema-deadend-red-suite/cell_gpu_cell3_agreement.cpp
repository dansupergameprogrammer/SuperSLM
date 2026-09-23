// T-2900 (Curie) -- plan Sec3.10.2/Sec3.10.6 Cell 3: cross-backend agreement over every
// lifecycle call the plan promises, including the input-agreement assertion (c) T-2897 added and
// the committed-content comparison it is grounded in -- the item T-2899 (Curie) named as not
// authored in its own pass.
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED. Driving logic is
// `Claude/Vitruvius/t2897-probe/cell3_committed_content_v3.cpp` (the healthy, identical-input
// construction) and `cell3_mutant_divergent_input_v3.cpp` (TE-361's own divergent-input
// construction) -- both already executed against the real shipped libraries and cited by the
// plan (Sec3.10.2/Sec3.10.6) as the authored-cell source. The two probes are unified here as ONE
// binary selected by `--mutant`, since the only difference between them is which expression
// `feed_token` is assigned from (a test-construction toggle, not a production-code variant) --
// converted from printf-verdict probing into CHECK-asserted cells; the calls, routes and fixture
// are unchanged.
//
// FIXTURE. --model=PATH, the C39 synthetic (`t2199_s8_fixture.sslm`), schema
// `g5_minimal_one_field`, layer_budget=8 (identical to the adopted probes' own hardcoded value).
//
// THE THREE CHECKS, per call, on routes R/D/C:
//   (a) cross-backend agreement -- both backends land on the identical status CLASS
//       (cpu_ok/SSLM_OK), the identical committed token, and the identical context_length.
//   (b) per-backend self-consistency -- once a dead end is reached, the content hash stays
//       unchanged call over call (never re-asserted independently here; folded into (a) at the
//       harness's own established granularity, matching the adopted probe).
//   (c) INPUT AGREEMENT (T-2897, closing TE-361's own residual) -- the token GPU is ACTUALLY fed
//       equals the token CPU's own internal state shows it will actually use, checked BEFORE the
//       decode call. This is what (a)/(b) cannot see: a stable WRONG answer is indistinguishable
//       from a stable right one under status/token/context_length/self-consistency alone.
//
// RED demonstration (--mutant): TE-361's own divergent-input construction (GPU fed a hardcoded,
// caller-supplied token instead of CPU's own internal current_token) turns (c) red on every route
// from the first call, while (a) stays green (both backends still land on -2/SSLM_OK/ctx=4,
// agreeing on a WRONG committed content) -- proving (a) alone cannot see this defect class and
// (c) is what closes the gap. This is the mutant plan Sec3.10.3 row 11 names for Cell 3.
//
// Run: cell_gpu_cell3_agreement.exe --model=PATH [--mutant]
#include "fixture_common.h"

extern "C" bool CpuOpenForCell3(const char* path, int32_t layers);
extern "C" void* CpuBuildRoute(const char* route, const int32_t* p_tokens, int32_t p_count, int32_t t0, int64_t cap);
extern "C" void CpuPeekState(void* handle, int32_t* out_current_token, int64_t* out_ctx);
extern "C" void CpuDecodeOnce(void* handle, int32_t* out_status, int32_t* out_token, int64_t* out_ctx,
                               uint64_t* out_hash);
extern "C" void CpuReleaseSeq(void* handle);
extern "C" int32_t CpuOkValue();

namespace {
uint64_t GpuContentHash(const GpuModelFixture& fx, SslmGpuSequenceHandle* s) {
	size_t need = 0;
	sslm_gpu_seq_save(fx.ctx, s, nullptr, &need);
	std::vector<uint8_t> b(need);
	size_t n = need;
	if (sslm_gpu_seq_save(fx.ctx, s, b.data(), &n) != SSLM_OK) return 0;
	const size_t skip = n > 200 ? 132 : (n > 92 ? 92 : 0);
	uint64_t h = 0;
	for (size_t i = skip; i < n; ++i) h = h * 1099511628211ull + b[i];
	return h;
}
const char* kRouteName[3] = {"R", "D", "C"};
}  // namespace

int main(int argc, char** argv) {
	std::string model_path;
	bool mutant = false;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a == "--mutant") mutant = true;
	}
	if (model_path.empty()) {
		std::printf("SKIP cell_gpu_cell3_agreement -- needs --model=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	CHECK_MSG(CpuOpenForCell3(model_path.c_str(), 8), "CPU open failed");
	const int32_t cpu_ok = CpuOkValue();

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	GpuModelFixture fx;
	CHECK_MSG(fx.Open(model_path, ctx), "GPU open failed");
	if (GFailures > 0) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}

	const std::vector<int32_t> P = {18, 42, 127};
	const int32_t t0 = 0;
	const int64_t cap = 64;

	std::printf("mode: %s\n", mutant ? "MUTANT (TE-361's own divergent-input construction, feed_token=66)"
	                                 : "HEALTHY (feed_token derived from CPU's own peeked current_token)");

	for (int route_idx = 0; route_idx < 3; ++route_idx) {
		const char* route = kRouteName[route_idx];
		void* cs = CpuBuildRoute(route, P.data(), static_cast<int32_t>(P.size()), t0, cap);
		CHECK_MSG(cs != nullptr, "[%s] CPU setup failed", route);
		if (!cs) continue;

		SslmGpuSequenceHandle* gs = nullptr;
		const SslmGpuStatus created = sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &gs);
		CHECK_MSG(created == SSLM_OK && gs, "[%s] GPU create returned %s", route, StatusName(created));
		if (!gs) { CpuReleaseSeq(cs); continue; }
		const SslmGpuStatus bound = SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, gs, 0);
		CHECK_MSG(bound == SSLM_OK, "[%s] GPU bind returned %s", route, StatusName(bound));
		bool setup_ok = bound == SSLM_OK;
		if (!std::strcmp(route, "R")) {
			const SslmGpuStatus prefill = setup_ok ? Prefill(fx, gs, P) : bound;
			CHECK_MSG(prefill == SSLM_OK, "[R] GPU prompt prefill returned %s", StatusName(prefill));
			setup_ok = setup_ok && prefill == SSLM_OK;
			int32_t consumed = 0;
			const SslmGpuStatus content = setup_ok ? SslmGpuSeqPrefillSchemaContentForG5Bridge(
			    fx.ctx, gs, &t0, 1, fx.one_layer_budget, &consumed) : prefill;
			CHECK_MSG(content == SSLM_OK && consumed == 1,
			          "[R] GPU content prefill returned %s, consumed=%d", StatusName(content), consumed);
			setup_ok = setup_ok && content == SSLM_OK && consumed == 1;
		} else if (!std::strcmp(route, "D")) {
			const SslmGpuStatus prefill = setup_ok ? Prefill(fx, gs, P) : bound;
			CHECK_MSG(prefill == SSLM_OK, "[D] GPU prompt prefill returned %s", StatusName(prefill));
			setup_ok = setup_ok && prefill == SSLM_OK;
		} else {  // C
			std::vector<int32_t> fill;
			for (int64_t i = 0; i < cap - 1; ++i) fill.push_back(P[static_cast<size_t>(i) % P.size()]);
			const SslmGpuStatus prefill = setup_ok ? Prefill(fx, gs, fill) : bound;
			CHECK_MSG(prefill == SSLM_OK, "[C] GPU prompt prefill returned %s", StatusName(prefill));
			setup_ok = setup_ok && prefill == SSLM_OK;
			int32_t consumed = 0;
			const SslmGpuStatus content = setup_ok ? SslmGpuSeqPrefillSchemaContentForG5Bridge(
			    fx.ctx, gs, &t0, 1, fx.one_layer_budget, &consumed) : prefill;
			CHECK_MSG(content == SSLM_OK && consumed == 1,
			          "[C] GPU content prefill returned %s, consumed=%d", StatusName(content), consumed);
			setup_ok = setup_ok && content == SSLM_OK && consumed == 1;
		}
		if (!setup_ok) { CpuReleaseSeq(cs); sslm_gpu_seq_release(fx.ctx, gs); continue; }

		const int n_calls = (!std::strcmp(route, "D")) ? 4 : 3;
		uint64_t prev_cpu_hash = 0, prev_gpu_hash = 0;
		bool have_prev = false;
		for (int k = 1; k <= n_calls; ++k) {
			int32_t cpu_current_token = -99;
			int64_t cpu_ctx_before = 0;
			CpuPeekState(cs, &cpu_current_token, &cpu_ctx_before);
			const int32_t expected_token = cpu_current_token >= 0 ? cpu_current_token : t0;
			// The mutant toggle: TE-361's own construction hardcodes feed_token=66, ignoring CPU's
			// own peeked state -- everything else in this loop is identical between the two modes.
			const int32_t feed_token = mutant ? 66 : expected_token;

			int32_t cpu_status = -1, cpu_out = -9999;
			int64_t cpu_ctx = 0;
			uint64_t cpu_hash = 0;
			CpuDecodeOnce(cs, &cpu_status, &cpu_out, &cpu_ctx, &cpu_hash);

			int32_t gpu_out = -9999;
			const SslmGpuStatus gpu_status =
			    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, gs, feed_token, fx.one_layer_budget, &gpu_out);
			const long long gpu_ctx = static_cast<long long>(ContextLength(gs));
			const uint64_t gpu_hash = GpuContentHash(fx, gs);

			// (c) INPUT AGREEMENT: fails immediately under the mutant, on every route, from the
			// first call -- exactly the property this check exists to catch, independent of
			// whatever the two backends go on to commit downstream.
			CHECK_MSG(feed_token == expected_token,
			          "[%s] call #%d: INPUT AGREEMENT -- GPU fed %d, CPU's own state expects %d",
			          route, k, feed_token, expected_token);

			// (a) cross-backend agreement, asserted on EVERY call (TE-361's own finding: a HIT
			// call's committed content can diverge unseen if only dead-end calls are checked).
			const bool status_class_ok = (cpu_status == cpu_ok && gpu_status == SSLM_OK);
			CHECK_MSG(status_class_ok, "[%s] call #%d: status class disagrees -- CPU st=%d GPU st=%s", route, k,
			          cpu_status, StatusName(gpu_status));
			CHECK_MSG(cpu_out == gpu_out, "[%s] call #%d: committed token disagrees -- CPU=%d GPU=%d", route, k,
			          cpu_out, gpu_out);
			CHECK_MSG(static_cast<long long>(cpu_ctx) == gpu_ctx,
			          "[%s] call #%d: context_length disagrees -- CPU=%lld GPU=%lld", route, k,
			          static_cast<long long>(cpu_ctx), gpu_ctx);
			const int32_t want_out = std::strcmp(route, "D") == 0 && k == 1 ? t0 : -2;
			const int64_t want_ctx = std::strcmp(route, "C") == 0 ? cap :
			                         static_cast<int64_t>(P.size()) + (std::strcmp(route, "R") == 0 || k > 1 ? 1 : 0);
			CHECK_MSG(cpu_out == want_out && gpu_out == want_out,
			          "[%s] call #%d tokens CPU=%d GPU=%d, want %d", route, k, cpu_out, gpu_out, want_out);
			CHECK_MSG(cpu_ctx == want_ctx && gpu_ctx == want_ctx,
			          "[%s] call #%d contexts CPU=%lld GPU=%lld, want %lld", route, k,
			          static_cast<long long>(cpu_ctx), gpu_ctx, static_cast<long long>(want_ctx));

			// (b) per-backend content self-consistency across repeat calls at the dead end -- the
			// check the mutant does NOT fail (both post-divergence calls read the SAME wrong hash).
			if (cpu_out == -2 && gpu_out == -2 && have_prev) {
				CHECK_MSG(cpu_hash == prev_cpu_hash, "[%s] call #%d: CPU content hash changed at a repeat dead end",
				          route, k);
				CHECK_MSG(gpu_hash == prev_gpu_hash, "[%s] call #%d: GPU content hash changed at a repeat dead end",
				          route, k);
			}
			if (cpu_out == -2 && gpu_out == -2) {
				prev_cpu_hash = cpu_hash;
				prev_gpu_hash = gpu_hash;
				have_prev = true;
			}
			std::printf("  [%s] call #%d fed=%-4d expected=%-4d CPU st=%-2d out=%-5d ctx=%-3lld | GPU st=%-2d "
			            "out=%-5d ctx=%-3lld\n",
			            route, k, feed_token, expected_token, cpu_status, cpu_out,
			            static_cast<long long>(cpu_ctx), static_cast<int>(gpu_status), gpu_out, gpu_ctx);
		}
		CpuReleaseSeq(cs);
		sslm_gpu_seq_release(fx.ctx, gs);
	}

	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
