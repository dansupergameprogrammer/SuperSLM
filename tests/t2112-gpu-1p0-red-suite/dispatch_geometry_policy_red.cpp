// T-2112 (Curie) -- supplementary REAL, EXECUTING red cell (not one of the Coverage Model's own
// 37 §11 cells -- a pin on design Sec6.1's own build-decomposition claim, "re-derived against 24,
// not the stale 17"). Unlike every other file in this suite, this one links against the REAL,
// ALREADY-SHIPPED `PlanDispatchBudgetGpu` (include/superslm/gpu_port.h:474, defined
// src/gpu/superslm_gpu.cpp:2049) -- pure policy arithmetic, no device required, already built on
// main@495fbb4. Its own comment (gpu_port.h:467-469) states the CURRENT, PRE-B4 constant:
// `complete_layers = min(dispatch_budget / 17, ...)`. Design Sec6.1: the 1.0 build re-derives this
// against 24 (T-2105's own dispatches/layer count), not 17. This cell is RED BY ASSERTION, not
// red by link -- it compiles AND links today, and FAILS because the shipped constant is still 17.
// It goes green the moment design Sec10 B4 lands (the constant becomes 24), without needing any
// new symbol -- a real, minimal, already-executable regression pin for one piece of B4's own gate.
#include <cstdio>
#include "superslm/gpu_port.h"

static int GChecks = 0;
static int GFailures = 0;
#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

int main() {
	// A budget of 40 discriminates the two constants: floor(40/24) = 1 complete layer (the design
	// Sec6.1 claim this cell pins); floor(40/17) = 2 complete layers (the stale, currently-shipped
	// constant) -- the two readings differ, so this cell is genuinely red against today's source,
	// not merely differently-labeled.
	uint32_t layers = 0xFFFFFFFFu;
	const superslm_gpu::SslmGpuStatus st = superslm_gpu::PlanDispatchBudgetGpu(/*dispatch_budget=*/40u,
	                                                /*num_hidden_layers=*/28u,
	                                                /*current_layer_position=*/0u, &layers);
	CHECK_MSG(st == superslm_gpu::SslmGpuStatus::Ok, "superslm_gpu::PlanDispatchBudgetGpu(40, 28, 0) must succeed -- got "
	          "status %d", (int)st);
	CHECK_MSG(layers == 1, "design Sec6.1 (T-2105's own 24 dispatches/layer, re-derived from the "
	          "substrate's stale 17): superslm_gpu::PlanDispatchBudgetGpu(dispatch_budget=40) must yield exactly "
	          "1 complete layer (40/24=1). Got %u -- if this reads 2, the shipped constant is "
	          "still the pre-T-2105 17 (40/17=2), the exact staleness design Sec6.1 names and "
	          "design Sec10 B4 is gated on repairing.", layers);
	std::printf("checks=%d failures=%d\n", GChecks, GFailures);
	return GFailures ? 1 : 0;
}
