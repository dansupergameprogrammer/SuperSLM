#ifndef SUPERSLM_GPU_DEBUG_LAYER_MODE_H
#define SUPERSLM_GPU_DEBUG_LAYER_MODE_H
// Schema fast-decode plan rev 1.4.1 §7.3 (slice G1-F, cell G0.1): the meaning of
// SSLM_GPU_ENABLE_DEBUG_LAYER, kept apart from d3d12_harness.h so it has no Windows dependency and
// a host cell can pin it in the cloud.
//
//   unset        no layer, and no banner (zero behavioural change, as before)
//   1            the D3D12 debug layer plus GPU-based validation, when the SDK supports it (as before)
//   2            the debug layer only: no SetEnableGPUBasedValidation call. The box driver's value (§10)
//   anything else  no layer, and a banner that says so
//
// Init() (d3d12_harness.h) asks these functions what to do and what to print, so the banner a host
// cell reads here is the one the device prints.
#include <cstdio>
#include <string>

namespace superslm_gpu {
namespace debug_layer {

// The mode a raw environment value selects; `value` null means unset.
inline int ParseMode(const char* value) noexcept {
	if (value == nullptr) return 0;
	if ((value[0] == '1' || value[0] == '2') && value[1] == '\0') return value[0] - '0';
	return 0;
}

// Only value 1 asks for GPU-based validation.
inline bool RequestsGpuBasedValidation(int mode) noexcept { return mode == 1; }

// The banner Init() prints to stderr. `value` is the raw environment value (null: unset, which prints
// nothing). `interface_ok`: D3D12GetDebugInterface succeeded. `gbv_available`: the ID3D12Debug1
// interface exists (read only at mode 1, the one mode that uses it).
inline std::string Banner(const char* value, int mode, bool interface_ok, bool gbv_available) {
	char b[256] = {0};
	if (value == nullptr) return std::string();
	if (mode == 0) {
		std::snprintf(b, sizeof(b), "# SSLM_GPU_ENABLE_DEBUG_LAYER=%.16s: no D3D12 debug layer\n", value);
	} else if (!interface_ok) {
		std::snprintf(b, sizeof(b),
		              "# SSLM_GPU_ENABLE_DEBUG_LAYER=%d requested but D3D12GetDebugInterface failed -- the Windows "
		              "'Graphics Tools' optional feature is likely not installed on this machine; proceeding "
		              "WITHOUT the debug layer\n",
		              mode);
	} else if (mode == 1) {
		std::snprintf(b, sizeof(b), "# SSLM_GPU_ENABLE_DEBUG_LAYER=1: D3D12 debug layer %s\n",
		              gbv_available ? "+ GPU-based validation enabled"
		                            : "enabled (GPU-based validation unavailable)");
	} else {
		std::snprintf(b, sizeof(b),
		              "# SSLM_GPU_ENABLE_DEBUG_LAYER=2: D3D12 debug layer enabled, GPU-based validation off\n");
	}
	return std::string(b);
}

}  // namespace debug_layer
}  // namespace superslm_gpu

#endif  // SUPERSLM_GPU_DEBUG_LAYER_MODE_H
