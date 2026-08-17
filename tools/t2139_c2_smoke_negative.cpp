// t2139_c2_smoke_negative.cpp -- Gate B's MUST-REJECT half for C1/C2 (S9, Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md: "Gate B has no must-reject construction at any slot ...
// the one gate whose subject is linkability and runtime behaviour is the one never shown able to
// fail"). Unlike Gate A/C's negative siblings (which MUST FAIL TO COMPILE), Gate B's subject is
// runtime behaviour over a real, linked artifact -- so this file's own gate is: it MUST COMPILE,
// LINK, AND RUN, and its own exit code is 0 ONLY IF the hostile call it drives was correctly
// REJECTED. An implementation that silently accepts a hostile sslm_workspace_create call (the
// exact class the positive smoke's own hostile-input block already isolates, tools/
// t2139_c2_smoke.cpp) makes THIS file's own final check fail, exiting 1 -- proving this gate can
// itself fail, not merely assert that it can.
//
// Usage: t2139_c2_smoke_negative.exe <path-to-real.sslm>
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
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <path-to-real.sslm>\n", argv[0]);
		return 1;
	}
	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], &bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}
	sslm_model model = nullptr;
	if (sslm_model_map(bytes.data(), bytes.size(), &model) != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned non-OK\n");
		return 1;
	}
	// The hostile call this gate exists to prove REJECTED: an all-zero sslm_config (design
	// Sec7.1's own hostile-input domain) reaching sslm_workspace_create over an otherwise
	// perfectly real, valid model and buffer.
	sslm_config hostile{};
	uint8_t buf[4096];
	sslm_workspace ws = nullptr;
	const sslm_status st = sslm_workspace_create(model, &hostile, buf, sizeof(buf), &ws);
	sslm_model_unmap(model);
	if (st != SSLM_INVALID_ARGUMENT || ws != nullptr) {
		std::fprintf(stderr,
		             "GATE B MUST-REJECT FAILED: sslm_workspace_create(hostile config) returned "
		             "%d, expected SSLM_INVALID_ARGUMENT -- this IS the regression this gate "
		             "exists to catch\n",
		             static_cast<int>(st));
		return 1;
	}
	std::printf("t2139_c2_smoke_negative: PASS (hostile config correctly rejected)\n");
	return 0;
}
