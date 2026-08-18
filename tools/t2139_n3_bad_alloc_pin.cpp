// t2139_n3_bad_alloc_pin.cpp -- N3/P1 pin (Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec6.3
// / Sec7.3): proves sslm_model_map returns SSLM_ALLOCATION_FAILED, not undefined behaviour, when
// a genuine std::bad_alloc reaches it -- via the SAME test-only fault-injection seam (tests/
// support/bad_alloc_injection.h, src/bad_alloc_wrap.h) S-HARDEN-7's own population already
// trusts, not a synthetic OOM condition. Requires SUPERSLM_ENABLE_BAD_ALLOC_INJECTION (the
// test-injection build only -- see src/bad_alloc_wrap.h's own header comment for why a release
// build never references this seam at all).
//
// P1 (third confirmation pass), AND A REAL LIMITATION FOUND WHILE CLOSING IT: BuildEngineCache
// (src/sslm_abi.cpp) now has its own MaybeThrowInjectedBadAllocFault() consultation at its own
// entry, matching the S-HARDEN-7 *Impl convention. It is NOT independently exercisable through
// THIS pin's own armed sslm_model_map() call, and that is worth stating plainly rather than
// implying otherwise: SslmModel::Load (model.cpp) is ITSELF an S-HARDEN-7 site whose own *Impl
// consults the SAME shared, single-shot, thread_local seam as its own first statement -- and
// Load always runs BEFORE BuildEngineCache inside sslm_model_map. Arming the seam and calling
// sslm_model_map therefore ALWAYS trips Load's own pre-existing consultation first (confirmed by
// instrumentation during this round: BuildEngineCache's own consultation is never reached while
// the seam is armed through this call path), which sslm_model_map's own try/catch around Load
// catches -- proving the OVERALL catch-and-return contract end to end, but not isolating
// BuildEngineCache's own wrap specifically from Load's. Closing that isolation gap for real needs
// either a site-specific arming mechanism in the shared seam (tests/support/
// bad_alloc_injection.h, S-HARDEN-7's own infrastructure, out of this round's scope) or a
// test-only exported hook directly onto BuildEngineCache -- neither built here; reported rather
// than papered over with a pin that would look isolated without being isolated.
// BuildEngineCache's own wrap is instead verified by construction: the SAME CatchAllocationFailure
// helper this whole sweep applies at every other identified site in this file (src/sslm_abi.cpp's
// own P1 sweep-method comment lists them), proven correct end-to-end by the armed test below
// (Load's own catch) and by code inspection (the wrap's own try/catch shape is identical
// everywhere it is used), and confirmed to compile with zero C4297 diagnostics either way.
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
	// proves the seam's own presence (a no-op call to MaybeThrowInjectedBadAllocFault() at both
	// SslmModel::Load's own entry, model.cpp, and BuildEngineCache's own entry, src/sslm_abi.cpp)
	// does not perturb the ordinary path.
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

	// Armed: the SAME real artifact, the SAME call -- MaybeThrowInjectedBadAllocFault() throws a
	// genuine std::bad_alloc at the FIRST consultation point this call chain reaches, which is
	// SslmModel::Load's own (model.cpp, an S-HARDEN-7 site that runs before BuildEngineCache is
	// ever called) -- see this file's own header comment for why the shared, single-shot seam
	// cannot isolate BuildEngineCache's own, later consultation point from this one. Still real,
	// still proves sslm_model_map's own end-to-end catch-and-return contract holds under a
	// genuine injected fault, not a synthetic one.
	{
		superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kBadAlloc);
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
		superslm_test::DisarmInjectedFault();
		if (st != SSLM_ALLOCATION_FAILED || model != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_model_map(injected bad_alloc) returned %d, expected "
			             "SSLM_ALLOCATION_FAILED -- N3's own end-to-end fix did not hold\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("N3 pin (sslm_model_map, injected bad_alloc via SslmModel::Load's own S-HARDEN-7 "
		            "seam): PASS (SSLM_ALLOCATION_FAILED fired as designed, no crash, no UB)\n");
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
