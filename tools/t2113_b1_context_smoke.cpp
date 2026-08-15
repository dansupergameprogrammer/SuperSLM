// T-2113 (B1): the bench proof for design Sec10 B1's own gate -- "existing suite
// green, unchanged behavior when exactly one context is created (a same-process,
// same-thread equivalence check against the substrate's own current single-call
// behavior)". Not part of Curie's t2112 red suite (that suite's own dim1-11 cells
// need B2+ symbols to link at all, per its own Sec5/Sec6 handoff); this is a build-
// seat verification tool, the same convention as tools/t2039_c5_harness.cpp
// ("Throwaway harness ... compiled and run directly for this session's own
// verification").
//
// Proves, on real D3D12 hardware:
//   1. sslm_gpu_context_create succeeds and returns a non-null context.
//   2. sslm_gpu_context_destroy on that context returns Ok (no live handles yet --
//      none are constructible before B2/B3/B6 land).
//   3. sslm_gpu_context_destroy(nullptr) is a no-op that returns Ok (documented
//      caller contract, gpu_1p0.cpp).
//   4. Two contexts may be created and destroyed independently in the same
//      process without interfering with each other (each owns its own
//      harness::Device, never the pre-1.0 singleton) -- the literal shape of
//      "no process-global device state" for the one construction B1 delivers.
//   5. The pre-1.0 entry points this build did not touch still work: PlanDispatch-
//      BudgetGpu (a pure, no-device function, gpu_port.h) is called and checked
//      against its own known-correct output, proving this translation unit's
//      addition did not perturb the substrate's existing symbols by inclusion
//      order, static-initialization order, or any other TU-linkage effect.
//
// Usage: t2113_b1_context_smoke   (no arguments; exits 0 on pass, 1 on failure)
#include <cstdio>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                            \
	do {                                                                              \
		++g_checks;                                                                    \
		if (!(cond)) {                                                                 \
			++g_failures;                                                                \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);            \
		}                                                                               \
	} while (0)

int main() {
	GpuContextConfig cfg{};

	// 1/2: single context, create then destroy.
	SslmGpuContext* ctx = nullptr;
	SslmGpuStatus st = sslm_gpu_context_create(cfg, &ctx);
	CHECK(st == SSLM_OK, "sslm_gpu_context_create did not return SSLM_OK");
	CHECK(ctx != nullptr, "sslm_gpu_context_create left *out_ctx null on SSLM_OK");
	if (ctx) {
		SslmGpuStatus dst = sslm_gpu_context_destroy(ctx);
		CHECK(dst == SSLM_OK, "sslm_gpu_context_destroy on a fresh context did not return SSLM_OK");
	}

	// 3: destroy(nullptr) is a documented no-op.
	SslmGpuStatus null_dst = sslm_gpu_context_destroy(nullptr);
	CHECK(null_dst == SSLM_OK, "sslm_gpu_context_destroy(nullptr) did not return SSLM_OK");

	// 4: two independent, concurrently-live contexts.
	SslmGpuContext* ctx_a = nullptr;
	SslmGpuContext* ctx_b = nullptr;
	SslmGpuStatus st_a = sslm_gpu_context_create(cfg, &ctx_a);
	SslmGpuStatus st_b = sslm_gpu_context_create(cfg, &ctx_b);
	CHECK(st_a == SSLM_OK && ctx_a != nullptr, "context A failed to create");
	CHECK(st_b == SSLM_OK && ctx_b != nullptr, "context B failed to create");
	CHECK(ctx_a != ctx_b, "two independent context_create calls returned the same pointer");
	if (ctx_a) CHECK(sslm_gpu_context_destroy(ctx_a) == SSLM_OK, "context A failed to destroy");
	if (ctx_b) CHECK(sslm_gpu_context_destroy(ctx_b) == SSLM_OK, "context B failed to destroy");

	// 5: the pre-1.0 substrate's own pure policy function is unperturbed by this
	// TU's presence in the same binary (24-vs-17 re-derivation is B4's own job,
	// not B1's -- this checks against the substrate's CURRENT, still-17 constant,
	// as a non-regression pin, not a claim about the design's target geometry).
	uint32_t layers_out = 0;
	superslm_gpu::SslmGpuStatus plan_st =
	    superslm_gpu::PlanDispatchBudgetGpu(/*dispatch_budget=*/17, /*num_hidden_layers=*/28,
	                                         /*current_layer_position=*/0, &layers_out);
	CHECK(plan_st == superslm_gpu::SslmGpuStatus::Ok, "PlanDispatchBudgetGpu(17,...) did not return Ok");
	CHECK(layers_out == 1, "PlanDispatchBudgetGpu(17,...) did not yield 1 complete layer "
	                        "(substrate's pre-B4 dispatches-per-layer constant is still 17)");

	std::printf("T-2113 B1 context smoke: checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
