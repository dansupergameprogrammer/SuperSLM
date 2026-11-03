// Schema fast-decode plan rev 1.4.1 §11.2, cell G0.1, its device half (slice G1-F; rev 1.4.2, strike N3).
// What the harness's Init() did with SSLM_GPU_ENABLE_DEBUG_LAYER, read through the test-only accessor
// ReadDebugLayerReportForTest (src/gpu/d3d12_harness.h). The parent runs itself twice as a child, with the
// variable at 0 and at 2, because the device reads it once per process:
//   0  no layer, no GPU-based validation request, the "no D3D12 debug layer" banner;
//   2  the layer on, no SetEnableGPUBasedValidation call, the info queue present, the "layer only" banner.
// Value 1 is pinned by the host half (host/g0_1_debug_layer_mode.cpp) and never run here: the box's 2080
// SUPER never runs GPU-based validation (§10). This is the one G binary that sets the variable, and only in
// its children.
//
// Needs: the test-seam configuration (SUPERSLM_GPU_TEST_SEAMS) and a D3D12 device with the Windows
// "Graphics Tools" feature. Box only. Run: sfdg_cell_g0_1.exe
#include "d3d12_harness.h"

#include <process.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
int g_fail = 0;
void Check(bool ok, const char* what) {
	std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
	if (!ok) ++g_fail;
}

int Child(const char* value) {
	namespace h = superslm_gpu::harness;
	const h::DebugLayerReportForTest r = h::ReadDebugLayerReportForTest();
	Check(r.device_available, "[G0.1] the process device is available");
	if (!r.device_available) return 1;
	const int want = superslm_gpu::debug_layer::ParseMode(value);
	Check(r.mode == want, "[G0.1] Init() read the mode the variable selects");
	Check(!r.gpu_based_validation_requested, "[G0.1] no SetEnableGPUBasedValidation call at 0 or 2");
	Check(r.banner == superslm_gpu::debug_layer::Banner(value, want, want != 0, false),
	      "[G0.1] the banner is debug_layer_mode.h's for this value");
	if (want == 0) {
		Check(!r.layer_enabled, "[G0.1] at 0 the layer is off");
		Check(r.banner.find("no D3D12 debug layer") != std::string::npos, "[G0.1] at 0 the banner reports no layer");
	} else {
		Check(r.layer_enabled, "[G0.1] at 2 the layer is on (needs the Graphics Tools feature)");
		Check(r.info_queue_present, "[G0.1] at 2 the harness's debug_info_queue exists");
		Check(r.banner.find("GPU-based validation off") != std::string::npos, "[G0.1] at 2 the banner reports the layer only");
	}
	std::printf("[G0.1] child at %s: mode %d, layer %d, queue %d, banner: %s", value, r.mode, r.layer_enabled ? 1 : 0,
	            r.info_queue_present ? 1 : 0, r.banner.empty() ? "(none)\n" : r.banner.c_str());
	return g_fail == 0 ? 0 : 1;
}
}  // namespace

int main(int argc, char** argv) {
	if (argc > 1 && std::strncmp(argv[1], "--child=", 8) == 0) return Child(argv[1] + 8);
	int bad = 0;
	for (const char* v : {"0", "2"}) {
		_putenv_s("SSLM_GPU_ENABLE_DEBUG_LAYER", v);
		const std::string arg = std::string("--child=") + v;
		const intptr_t rc = _spawnl(_P_WAIT, argv[0], argv[0], arg.c_str(), static_cast<const char*>(nullptr));
		std::printf("[G0.1] value %s: child exit %lld\n", v, static_cast<long long>(rc));
		if (rc != 0) ++bad;
	}
	_putenv_s("SSLM_GPU_ENABLE_DEBUG_LAYER", "");
	std::printf("sfdg_cell_g0_1: %s\n", bad == 0 ? "GREEN" : "RED");
	return bad == 0 ? 0 : 1;
}
