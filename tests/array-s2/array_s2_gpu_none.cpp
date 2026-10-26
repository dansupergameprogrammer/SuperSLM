// array_s2_gpu_none.cpp -- the GPU side of the S2 array harness for a build with no GPU backend
// (Linux, CPU only: build_harness_linux.sh). Every GPU mode reports a setup error, exit 2.
#include "array_s2_shared.h"

struct GpuSession {};

GpuSession* GpuOpen(const uint8_t*, size_t, const std::string&, MapResult* map, const GpuOptions&) {
	map->ok = false;
	map->error = "this harness was built without the GPU backend (array_s2_gpu_none.cpp)";
	return nullptr;
}
void GpuClose(GpuSession*) {}
WalkResult GpuWalk(GpuSession*, const std::vector<int32_t>&) { return {}; }
GenResult GpuGen(GpuSession*, const std::vector<int32_t>&, int32_t) { return {}; }
std::vector<RestoreResult> GpuRestores(GpuSession*, const std::vector<int32_t>&, const std::vector<int32_t>&,
                                       const std::vector<int32_t>&, const RestoreOptions&, Trace*,
                                       std::string* error) {
	*error = "no GPU backend";
	return {};
}
std::vector<RestoreResult> GpuTfRestores(GpuSession*, const std::vector<int32_t>&, const std::vector<int32_t>&,
                                         const std::vector<int32_t>&, const RestoreOptions&, Trace*,
                                         std::string* error) {
	*error = "no GPU backend";
	return {};
}
