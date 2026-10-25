// Multi-row GPU plan (rev 1.3.2) §5.1 and §4.7 at MR4, rulings item 12: "the table in C18". SuperSLM ships the M-row cost
// table as data, r(M) = C(M) / C(1) and g(M) = G(M) / C(1) as per-mille integers in a host-only header, with
// GpuRowCostMs (6.2 ms) the M = 1 entry, estimates until MB4's 10.2 measures them. The integer chooser and cell 10.6
// are MR6's (the plugin); this checks the data only. Built natively by mr4_wine.sh (host-only: no device types).
//
// CONTRACT FOR THE BUILDER (the test author's proposal; one edit here for other names):
//   include/superslm/gpu_rows_cost_table.h, namespace superslm_gpu:
//     constexpr uint32_t kGpuRowCostUs = 6200;                       // GpuRowCostMs's default, in µs (§5.1's pinned conversion)
//     constexpr uint32_t kGpuRowsCostRatioPerMille[SSLM_GPU_MAX_ROWS];    // r(M) for M = 1..8 (index M - 1); r(1) = 1000
//     constexpr uint32_t kGpuRowsLogitsRatioPerMille[SSLM_GPU_MAX_ROWS];  // g(M), the per-row logits column (§4.7)
//     constexpr bool kGpuRowsCostTableMeasured;                      // false while the entries are estimates (until MB4)
//   It includes nothing that needs a device (the plugin vendors it at MR6).
//
// Graded: r(1) = 1000 (the budget's own unit, §5.1); every r(M) and g(M) positive; r nondecreasing in M (an M-row
// chunk never prices below an (M-1)-row one); r(M) <= 1000 * M (an M-row chunk never prices above M one-row tokens:
// that would make multi-row pointless, and 10.2's port gate is 600 at M = 4); g nondecreasing. Whether the values are
// measured is printed, not graded (10.2 at MB4 files them).
#include <cstdint>
#include <cstdio>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_rows_cost_table.h"

int main() {
	using namespace superslm_gpu;
	int bad = 0;
	auto check = [&](bool ok, const char* what, int m) {
		if (!ok) {
			std::printf("RED table: %s (M = %d)\n", what, m);
			++bad;
		}
	};
	static_assert(sizeof(kGpuRowsCostRatioPerMille) / sizeof(kGpuRowsCostRatioPerMille[0]) == SSLM_GPU_MAX_ROWS, "r(M) for M = 1..8");
	static_assert(sizeof(kGpuRowsLogitsRatioPerMille) / sizeof(kGpuRowsLogitsRatioPerMille[0]) == SSLM_GPU_MAX_ROWS, "g(M) for M = 1..8");
	check(kGpuRowCostUs == 6200u, "GpuRowCostMs's default is not 6,200 us (1.17's 6.2 ms)", 1);
	check(kGpuRowsCostRatioPerMille[0] == 1000u, "r(1) is not 1000 (the budget's unit)", 1);
	for (int m = 1; m <= SSLM_GPU_MAX_ROWS; ++m) {
		const uint32_t r = kGpuRowsCostRatioPerMille[m - 1], g = kGpuRowsLogitsRatioPerMille[m - 1];
		check(r > 0 && g > 0, "an entry is zero", m);
		check(r <= 1000u * static_cast<uint32_t>(m), "r(M) prices above M one-row tokens", m);
		if (m > 1) {
			check(r >= kGpuRowsCostRatioPerMille[m - 2], "r(M) below r(M - 1)", m);
			check(g >= kGpuRowsLogitsRatioPerMille[m - 2], "g(M) below g(M - 1)", m);
		}
	}
	std::printf("table: r =");
	for (uint32_t r : kGpuRowsCostRatioPerMille) std::printf(" %u", r);
	std::printf("; g =");
	for (uint32_t g : kGpuRowsLogitsRatioPerMille) std::printf(" %u", g);
	std::printf("; %s\n", kGpuRowsCostTableMeasured ? "measured" : "estimates (until MB4's 10.2)");
	std::printf("%s table\n", bad ? "RED" : "PASS");
	return bad ? 1 : 0;
}
