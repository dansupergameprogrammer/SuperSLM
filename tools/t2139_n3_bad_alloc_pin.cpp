// t2139_n3_bad_alloc_pin.cpp -- N3 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3):
// proves sslm_model_map returns SSLM_ALLOCATION_FAILED, not undefined behaviour, when
// SslmModel::Load hits a genuine std::bad_alloc -- via the SAME test-only fault-injection seam
// (tests/support/bad_alloc_injection.h, src/bad_alloc_wrap.h) S-HARDEN-7's own population
// already trusts, not a synthetic OOM condition. Requires SUPERSLM_ENABLE_BAD_ALLOC_INJECTION
// (the test-injection build only -- see src/bad_alloc_wrap.h's own header comment for why a
// release build never references this seam at all).
//
// Usage: t2139_n3_bad_alloc_pin.exe <path-to-real.sslm>
#include <cstdio>
#include <fstream>
#include <vector>

#include "superslm/sslm_abi.h"

#include "support/bad_alloc_injection.h"

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

	// Baseline, seam disarmed: sslm_model_map succeeds normally against the real artifact --
	// proves the seam's own presence (a no-op call to MaybeThrowInjectedBadAllocFault() at the
	// top of sslm_model_map's own try block) does not perturb the ordinary path.
	{
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
		if (st != SSLM_OK || !model) {
			std::fprintf(stderr, "FAIL: baseline sslm_model_map (seam disarmed) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(model);
		std::printf("baseline sslm_model_map (seam disarmed): PASS\n");
	}

	// Armed: the SAME real artifact, the SAME call -- but MaybeThrowInjectedBadAllocFault()
	// throws a genuine std::bad_alloc at the exact point sslm_model_map's own new try/catch
	// (N3, src/sslm_abi.cpp) is meant to catch it.
	{
		superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kBadAlloc);
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
		superslm_test::DisarmInjectedFault();
		if (st != SSLM_ALLOCATION_FAILED || model != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_model_map(injected bad_alloc) returned %d, expected "
			             "SSLM_ALLOCATION_FAILED -- N3's own fix did not hold\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("N3 pin (sslm_model_map, injected bad_alloc): PASS (SSLM_ALLOCATION_FAILED "
		            "fired as designed, no crash, no UB)\n");
	}

	// The seam being disarmed again: a SECOND ordinary call still succeeds, proving the injected
	// fault was single-shot and left no residual state behind (bad_alloc_injection.h's own
	// "single-shot" contract) and that sslm_model_map itself is still fully usable afterward.
	{
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
		if (st != SSLM_OK || !model) {
			std::fprintf(stderr,
			             "FAIL: post-injection sslm_model_map (seam disarmed again) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(model);
		std::printf("post-injection sslm_model_map (seam disarmed again): PASS\n");
	}

	std::printf("t2139_n3_bad_alloc_pin: PASS\n");
	return 0;
}
