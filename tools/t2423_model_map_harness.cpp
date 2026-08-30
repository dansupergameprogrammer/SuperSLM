// t2423_model_map_harness.cpp -- disposable spike harness (T-2423). Loads a real .sslm
// artifact through sslm_model_map -- the full runtime path (SslmModel::Load, then
// BuildEngineCache's per-layer MarshalLayer call) -- rather than SslmModel::Load alone
// (sslm_verify.cpp's own narrower container/schema/join check). This is what actually
// exercises layer_marshal.h's MarshalProjectionFold channel-count check (Track A step 4,
// GS-08) at load time. Not part of the build.bat/CMake build graph; disposable, per
// T-2423's own brief.
//
// Usage: t2423_model_map_harness <artifact.sslm>
#include <cstdio>
#include <fstream>
#include <string>

#include "superslm/sslm_abi.h"

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: t2423_model_map_harness <artifact.sslm>\n");
		return 2;
	}
	const char* path = argv[1];

	std::ifstream f(path, std::ios::binary);
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", path);
		return 2;
	}
	std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	std::fprintf(stdout, "artifact: %s (%zu bytes)\n", path, bytes.size());

	sslm_model model = nullptr;
	const sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	std::fprintf(stdout, "sslm_model_map status = %d\n", static_cast<int>(st));

	if (st == SSLM_OK) {
		std::fprintf(stdout, "sslm_model_map: OK -- model handle mapped (MarshalLayer succeeded "
		                     "for every layer)\n");
		sslm_model_unmap(model);
		return 0;
	}
	std::fprintf(stdout, "sslm_model_map: REJECTED (status=%d) -- this ABI's 3-argument shape "
	                     "carries no diagnostic out-parameter, so no further detail is available "
	                     "from this call alone\n", static_cast<int>(st));
	return 1;
}
