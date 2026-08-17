// t2139_d3464_foreignfault_pin.cpp -- the same-round pin D-SLM3464 owes (Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec6, fold 2026-08-17 second pass, "Same-round pin
// owed, in addition to the one above"): a fault-injection cell proving a non-std::exception-
// derived throw raised beneath sslm_model_map/sslm_adapter_map SPECIFICALLY (not only the
// standalone CatchAllocationFailure construction pin, tools/t2139_f2_catchall_construction_pin.cpp,
// which does not exercise the real SslmModel::Load-fronted call path) reaches the extern "C"
// boundary's own catch(...) arm and returns SSLM_ARTIFACT_REJECTED.
//
// WHY THE EXISTING SEAM COULD NOT DO THIS (tools/t2139_f2_length_error_pin.cpp's own header,
// Sec1d of Claude/Curie/t2139-abi-suite-fix-round-2026-08-17.md): every InjectThrowKind before
// this pin (kLengthError, kRuntimeError, kBadAlloc) is std::exception-derived, and
// SslmModel::Load's own public entry, internal::WrapBadAllocContract (src/bad_alloc_wrap.h),
// narrows every non-bad_alloc std::exception to std::bad_alloc BEFORE either Load-fronted verb's
// own try/catch ever sees it -- proven end to end by that pin's own kLengthError/kRuntimeError
// cells, both of which return SSLM_ALLOCATION_FAILED, never SSLM_ARTIFACT_REJECTED. The one class
// WrapBadAllocContract's own two catch clauses (catch (const std::bad_alloc&), catch (const
// std::exception&)) do NOT narrow is a throw not derived from std::exception at all -- which is
// exactly D-SLM3464's own ruled carve-out for sslm_model_map's/sslm_adapter_map's catch(...) arm.
// tests/support/bad_alloc_injection.h gained a new InjectThrowKind, kForeignFault, and a matching
// ForeignFault type (deliberately NOT std::exception-derived) for exactly this cell -- the
// extension the derived-pin-list item itself anticipated ("extend it with a non-std::exception
// fault kind if that is the clean mechanism").
//
// Needs SUPERSLM_ENABLE_BAD_ALLOC_INJECTION + /Itests, matching every other seam-based pin's own
// build convention (see the C5 block / N3 pin block in build.bat).
//
// Usage: t2139_d3464_foreignfault_pin.exe <path-to-real.sslm> [path-to-real-adapter.sslm]
// The adapter argument is optional -- when supplied, the identical cell shape is also run through
// sslm_adapter_map (D-SLM3464's own ruling names both Load-fronted verbs); when omitted, only
// sslm_model_map is exercised and the tool still proves the pin's own literal claim.
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
		std::fprintf(stderr, "usage: %s <path-to-real.sslm> [path-to-real-adapter.sslm]\n", argv[0]);
		return 1;
	}
	std::vector<uint8_t> model_bytes;
	if (!ReadFile(argv[1], &model_bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}

	// --- sslm_model_map: the pin's own literal, explicitly-named target. ---

	// Baseline, seam disarmed: sslm_model_map succeeds normally.
	{
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &model);
		if (st != SSLM_OK || !model) {
			std::fprintf(stderr, "FAIL: baseline sslm_model_map (seam disarmed) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(model);
		std::printf("baseline sslm_model_map (seam disarmed): PASS\n");
	}

	// Armed kForeignFault: a throw NOT derived from std::exception, raised at LoadImpl's own
	// entry (model.cpp), propagates through WrapBadAllocContract unmolested (neither of its own
	// two catch clauses matches ForeignFault) and reaches sslm_model_map's own catch(...) arm.
	{
		superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kForeignFault);
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &model);
		superslm_test::DisarmInjectedFault();
		if (st != SSLM_ARTIFACT_REJECTED || model != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_model_map(injected ForeignFault, non-std::exception) returned "
			             "%d, expected SSLM_ARTIFACT_REJECTED -- D-SLM3464's own carve-out did not "
			             "hold\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("D-SLM3464 pin (sslm_model_map, injected ForeignFault -- NOT std::exception-"
		            "derived -- via SslmModel::Load's own S-HARDEN-7 seam, propagating through "
		            "WrapBadAllocContract unmolested): PASS (SSLM_ARTIFACT_REJECTED fired as ruled, "
		            "no crash, no UB)\n");
	}

	// Seam disarmed again -- single-shot, no residual state, sslm_model_map still fully usable.
	{
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &model);
		if (st != SSLM_OK || !model) {
			std::fprintf(stderr,
			             "FAIL: post-injection sslm_model_map (seam disarmed again) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(model);
		std::printf("post-injection sslm_model_map (seam disarmed again): PASS\n");
	}

	// --- sslm_adapter_map: D-SLM3464's own ruling names this verb too (identical structural
	// reason -- also Load-fronted). Optional: only run when a real adapter artifact is supplied. ---
	if (argc >= 3) {
		std::vector<uint8_t> adapter_bytes;
		if (!ReadFile(argv[2], &adapter_bytes)) {
			std::fprintf(stderr, "FAIL: could not read %s\n", argv[2]);
			return 1;
		}
		sslm_model base = nullptr;
		{
			const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &base);
			if (st != SSLM_OK || !base) {
				std::fprintf(stderr, "FAIL: sslm_model_map (base, for adapter cell) returned %d\n",
				             static_cast<int>(st));
				return 1;
			}
		}

		// Baseline, seam disarmed.
		{
			sslm_adapter adapter = nullptr;
			const sslm_status st =
			    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
			if (st != SSLM_OK || !adapter) {
				std::fprintf(stderr, "FAIL: baseline sslm_adapter_map (seam disarmed) returned %d\n",
				             static_cast<int>(st));
				sslm_model_unmap(base);
				return 1;
			}
			sslm_adapter_release(adapter);
			std::printf("baseline sslm_adapter_map (seam disarmed): PASS\n");
		}

		// Armed kForeignFault -- same shape as the sslm_model_map cell above.
		{
			superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kForeignFault);
			sslm_adapter adapter = nullptr;
			const sslm_status st =
			    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
			superslm_test::DisarmInjectedFault();
			if (st != SSLM_ARTIFACT_REJECTED || adapter != nullptr) {
				std::fprintf(stderr,
				             "FAIL: sslm_adapter_map(injected ForeignFault) returned %d, expected "
				             "SSLM_ARTIFACT_REJECTED -- D-SLM3464's own carve-out did not hold for "
				             "this verb\n",
				             static_cast<int>(st));
				sslm_model_unmap(base);
				return 1;
			}
			std::printf("D-SLM3464 pin (sslm_adapter_map, injected ForeignFault): PASS "
			            "(SSLM_ARTIFACT_REJECTED fired as ruled, no crash, no UB)\n");
		}

		// Seam disarmed again.
		{
			sslm_adapter adapter = nullptr;
			const sslm_status st =
			    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
			if (st != SSLM_OK || !adapter) {
				std::fprintf(stderr,
				             "FAIL: post-injection sslm_adapter_map (seam disarmed again) returned "
				             "%d\n",
				             static_cast<int>(st));
				sslm_model_unmap(base);
				return 1;
			}
			sslm_adapter_release(adapter);
			std::printf("post-injection sslm_adapter_map (seam disarmed again): PASS\n");
		}

		sslm_model_unmap(base);
	} else {
		std::printf("t2139_d3464_foreignfault_pin: sslm_adapter_map cell NOT run (pass a real "
		            "adapter .sslm as argv[2] to run it)\n");
	}

	std::printf("t2139_d3464_foreignfault_pin: PASS\n");
	return 0;
}
