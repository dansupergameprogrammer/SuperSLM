// t2139_c6_smoke_negative.cpp -- Gate B's MUST-REJECT half for C6 (S9, see
// t2139_c2_smoke_negative.cpp's own header comment for this construction's shape). The hostile
// call: sslm_adapter_map against a base model the adapter was never compiled for
// (SSLM_ADAPTER_MODEL_MISMATCH) -- the same real op-set/dims/base-hash mismatch class dim2_M6 of
// the red suite names, driven here as its own standalone Gate B construction.
//
// Usage: t2139_c6_smoke_negative.exe <base.sslm> <adapter.sslm> <foreign-base.sslm>
#include <cstdio>
#include <fstream>
#include <vector>

#include "superslm/sslm_abi.h"

namespace {
bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	if (size < 0) return false;
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out->data()), size);
	return static_cast<bool>(f) || f.eof();
}
}  // namespace

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: %s <base.sslm> <adapter.sslm> <foreign-base.sslm>\n", argv[0]);
		return 1;
	}
	std::vector<uint8_t> adapter_bytes, foreign_bytes;
	if (!ReadFile(argv[2], &adapter_bytes) || !ReadFile(argv[3], &foreign_bytes)) {
		std::fprintf(stderr, "FAIL: could not read adapter or foreign-base artifact\n");
		return 1;
	}
	sslm_model foreign_base = nullptr;
	if (sslm_model_map(foreign_bytes.data(), foreign_bytes.size(), &foreign_base) != SSLM_OK ||
	    !foreign_base) {
		std::fprintf(stderr, "FAIL: sslm_model_map(foreign_base) returned non-OK\n");
		return 1;
	}
	// The hostile call: an adapter compiled against argv[1]'s own base, mapped over the
	// DIFFERENT-shaped foreign base.
	sslm_adapter adapter = nullptr;
	const sslm_status st =
	    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), foreign_base, &adapter);
	sslm_model_unmap(foreign_base);
	if (st != SSLM_ADAPTER_MODEL_MISMATCH || adapter != nullptr) {
		std::fprintf(stderr,
		             "GATE B MUST-REJECT FAILED: sslm_adapter_map(foreign base) returned %d, "
		             "expected SSLM_ADAPTER_MODEL_MISMATCH\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("t2139_c6_smoke_negative: PASS (foreign-base adapter mismatch correctly "
	            "rejected)\n");
	return 0;
}
