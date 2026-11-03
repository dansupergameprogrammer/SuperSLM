// Schema fast-decode plan rev 1.4.1 §11.2, cell G0.1, its host half (slice G1-F): the meaning of
// SSLM_GPU_ENABLE_DEBUG_LAYER and the banner for values unset, 0, 1 and 2, from src/gpu/debug_layer_mode.h,
// the one place the harness's Init() reads them from. No device: this half runs in the cloud and in CI.
// The device half (tests/schema-fast-decode-gpu/sfdg_cell_g0_1.cpp) runs on the box at 0 and 2 and reads
// what Init() did; value 1 is never run on the box, whose 2080 SUPER never runs GPU-based validation (§10).
//
// Build and run: g++ -std=c++20 -Isrc/gpu tests/schema-fast-decode-gpu/host/g0_1_debug_layer_mode.cpp -o g0_1 && ./g0_1
#include "debug_layer_mode.h"

#include <cstdio>
#include <string>

namespace {
int g_fail = 0;
void Check(bool ok, const char* what) {
	std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
	if (!ok) ++g_fail;
}
bool Has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }
}  // namespace

int main() {
	namespace dl = superslm_gpu::debug_layer;
	// The modes.
	Check(dl::ParseMode(nullptr) == 0, "[G0.1] unset is mode 0");
	Check(dl::ParseMode("0") == 0, "[G0.1] 0 is mode 0");
	Check(dl::ParseMode("1") == 1, "[G0.1] 1 is mode 1");
	Check(dl::ParseMode("2") == 2, "[G0.1] 2 is mode 2");
	Check(dl::ParseMode("3") == 0 && dl::ParseMode("12") == 0 && dl::ParseMode("") == 0,
	      "[G0.1] any other value is mode 0");
	// GPU-based validation: value 1 only.
	Check(!dl::RequestsGpuBasedValidation(0), "[G0.1] mode 0 asks for no GPU-based validation");
	Check(dl::RequestsGpuBasedValidation(1), "[G0.1] mode 1 asks for GPU-based validation");
	Check(!dl::RequestsGpuBasedValidation(2), "[G0.1] mode 2 makes no SetEnableGPUBasedValidation call");
	// The banners.
	Check(dl::Banner(nullptr, 0, false, false).empty(), "[G0.1] unset prints no banner");
	const std::string b0 = dl::Banner("0", 0, false, false);
	Check(Has(b0, "=0: no D3D12 debug layer"), "[G0.1] the banner at 0 reports no layer");
	const std::string b1 = dl::Banner("1", 1, true, true);
	Check(Has(b1, "=1: D3D12 debug layer + GPU-based validation enabled"),
	      "[G0.1] the banner at 1 reports the layer plus GPU-based validation");
	const std::string b1n = dl::Banner("1", 1, true, false);
	Check(Has(b1n, "GPU-based validation unavailable"), "[G0.1] the banner at 1 without ID3D12Debug1 says so");
	const std::string b2 = dl::Banner("2", 2, true, false);
	Check(Has(b2, "=2: D3D12 debug layer enabled, GPU-based validation off"),
	      "[G0.1] the banner at 2 reports the layer only");
	Check(!Has(b2, "+ GPU-based validation"), "[G0.1] the banner at 2 never claims GPU-based validation");
	const std::string b2f = dl::Banner("2", 2, false, false);
	Check(Has(b2f, "WITHOUT the debug layer"), "[G0.1] a failed debug interface at 2 says the layer is off");
	std::printf("g0_1_debug_layer_mode: %s (%d failed)\n", g_fail == 0 ? "GREEN" : "RED", g_fail);
	return g_fail == 0 ? 0 : 1;
}
