// t2139_d3466_postload_region_pin.cpp -- D-SLM3466's owed pin (Claude/Poirot/
// 3bcbe43-t2139-fourth-confirmation-review.md S2/S3): proves that a NON-allocation cause raised in
// sslm_model_map's/sslm_adapter_map's own POST-Load construction step (BuildEngineCache /
// PopulateAdapterFromView, src/sslm_abi.cpp) returns SSLM_ARTIFACT_REJECTED, not
// SSLM_ALLOCATION_FAILED -- the opposite region from the one include/superslm/
// sslm_abi_functions_g5_comparable.inc's own per-declaration comment documents for the
// SslmModel::Load call itself (tools/t2139_f2_length_error_pin.cpp's own kRuntimeError cell).
//
// THE OBSTACLE THIS PIN CLOSES (tools/t2139_n3_bad_alloc_pin.cpp's own disclosed limitation,
// carried forward at F3/P1 in this file's own sslm_abi.cpp comments until this round): the shared
// fault-injection seam (tests/support/bad_alloc_injection.h) was single-shot and thread_local, and
// SslmModel::Load's own *Impl consults it FIRST on both Load-fronted verbs' own call paths -- an
// armed fault always tripped Load's own consultation and never reached BuildEngineCache's or
// PopulateAdapterFromView's own, later consultation point.
//
// THE FIX: tests/support/bad_alloc_injection.h gained a SECOND, INDEPENDENT single-shot slot
// (g_inject_throw_post_load_region / MaybeThrowInjectedFaultPostLoadRegion()), consulted ONLY by
// BuildEngineCache's and PopulateAdapterFromView's own entry (via
// superslm::internal::MaybeThrowInjectedBadAllocFaultPostLoadRegion(), src/bad_alloc_wrap.h) --
// NEVER by SslmModel::Load's own *Impl sites, which continue to consult the ORIGINAL slot
// exclusively. Because the two slots are independent, arming THIS pin's own slot and calling
// sslm_model_map/sslm_adapter_map is not consumed by Load's own (untouched) consultation -- the
// fault survives past Load and fires for real at the post-Load region this pin targets.
//
// kRuntimeError is used deliberately (not kLengthError/kBadAlloc): it is std::exception-derived
// but NOT one of CatchAllocationFailure's own two named allocation causes
// (std::bad_alloc/std::length_error), so it must fall through to CatchAllocationFailure's own
// final catch(...) arm and return SSLM_ARTIFACT_REJECTED -- proving the .inc header's own
// corrected claim (S2's fix) that the post-Load region follows CatchAllocationFailure's general
// three-way split, not WrapBadAllocContract's narrower one.
//
// Needs SUPERSLM_ENABLE_BAD_ALLOC_INJECTION + /Itests, matching every other seam-based pin's own
// build convention (see the N3 / D-SLM3464 pin blocks in build.bat).
//
// Usage: t2139_d3466_postload_region_pin.exe <path-to-real.sslm> [path-to-real-adapter.sslm]
// The adapter argument is optional -- when supplied, the identical cell shape also runs through
// sslm_adapter_map's own PopulateAdapterFromView step; when omitted, only sslm_model_map's own
// BuildEngineCache step is exercised and the tool still proves the pin's own literal claim.
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

	// --- sslm_model_map / BuildEngineCache: the pin's own literal, explicitly-named target. ---

	// Baseline, both seams disarmed: sslm_model_map succeeds normally.
	{
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &model);
		if (st != SSLM_OK || !model) {
			std::fprintf(stderr, "FAIL: baseline sslm_model_map (both seams disarmed) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(model);
		std::printf("baseline sslm_model_map (both seams disarmed): PASS\n");
	}

	// Armed on the POST-LOAD-REGION slot only (the plain slot Load's own *Impl consults stays
	// disarmed throughout this cell): a std::runtime_error -- std::exception-derived, but not one
	// of CatchAllocationFailure's own two named allocation causes -- raised at BuildEngineCache's
	// own entry, past Load's own untouched consultation, must fall to CatchAllocationFailure's own
	// final catch(...) arm and return SSLM_ARTIFACT_REJECTED.
	{
		superslm_test::ArmInjectedFaultPostLoadRegion(superslm_test::InjectThrowKind::kRuntimeError);
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &model);
		superslm_test::DisarmInjectedFaultPostLoadRegion();
		if (st != SSLM_ARTIFACT_REJECTED || model != nullptr) {
			std::fprintf(stderr,
			             "FAIL: sslm_model_map(injected post-load-region runtime_error) returned %d, "
			             "expected SSLM_ARTIFACT_REJECTED -- the post-Load region did not follow "
			             "CatchAllocationFailure's own general catch(...) rule\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("D-SLM3466 pin (sslm_model_map / BuildEngineCache, injected runtime_error via "
		            "the SITE-SPECIFIC post-load-region seam, isolated from SslmModel::Load's own "
		            "consultation): PASS (SSLM_ARTIFACT_REJECTED fired as CatchAllocationFailure's "
		            "own general rule requires, no crash, no UB)\n");
	}

	// Both seams disarmed again -- single-shot, no residual state on either slot, sslm_model_map
	// still fully usable.
	{
		sslm_model model = nullptr;
		const sslm_status st = sslm_model_map(model_bytes.data(), model_bytes.size(), &model);
		if (st != SSLM_OK || !model) {
			std::fprintf(stderr,
			             "FAIL: post-injection sslm_model_map (both seams disarmed again) returned "
			             "%d\n",
			             static_cast<int>(st));
			return 1;
		}
		sslm_model_unmap(model);
		std::printf("post-injection sslm_model_map (both seams disarmed again): PASS\n");
	}

	// --- sslm_adapter_map / PopulateAdapterFromView: the same region-scoped rule applies to this
	// verb's own post-Load construction step (D-SLM3464's ruling names both Load-fronted verbs
	// symmetrically; S2 does the same). Optional: only run when a real adapter artifact is
	// supplied. ---
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

		// Baseline, both seams disarmed.
		{
			sslm_adapter adapter = nullptr;
			const sslm_status st =
			    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
			if (st != SSLM_OK || !adapter) {
				std::fprintf(stderr,
				             "FAIL: baseline sslm_adapter_map (both seams disarmed) returned %d\n",
				             static_cast<int>(st));
				sslm_model_unmap(base);
				return 1;
			}
			sslm_adapter_release(adapter);
			std::printf("baseline sslm_adapter_map (both seams disarmed): PASS\n");
		}

		// Armed on the post-load-region slot only -- same shape as the sslm_model_map cell above,
		// reaching PopulateAdapterFromView's own consultation point instead of BuildEngineCache's.
		{
			superslm_test::ArmInjectedFaultPostLoadRegion(
			    superslm_test::InjectThrowKind::kRuntimeError);
			sslm_adapter adapter = nullptr;
			const sslm_status st =
			    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
			superslm_test::DisarmInjectedFaultPostLoadRegion();
			if (st != SSLM_ARTIFACT_REJECTED || adapter != nullptr) {
				std::fprintf(stderr,
				             "FAIL: sslm_adapter_map(injected post-load-region runtime_error) "
				             "returned %d, expected SSLM_ARTIFACT_REJECTED -- the post-Load region "
				             "did not follow CatchAllocationFailure's own general catch(...) rule "
				             "for this verb\n",
				             static_cast<int>(st));
				sslm_model_unmap(base);
				return 1;
			}
			std::printf("D-SLM3466 pin (sslm_adapter_map / PopulateAdapterFromView, injected "
			            "runtime_error via the site-specific post-load-region seam): PASS "
			            "(SSLM_ARTIFACT_REJECTED fired as CatchAllocationFailure's own general rule "
			            "requires, no crash, no UB)\n");
		}

		// Both seams disarmed again.
		{
			sslm_adapter adapter = nullptr;
			const sslm_status st =
			    sslm_adapter_map(adapter_bytes.data(), adapter_bytes.size(), base, &adapter);
			if (st != SSLM_OK || !adapter) {
				std::fprintf(stderr,
				             "FAIL: post-injection sslm_adapter_map (both seams disarmed again) "
				             "returned %d\n",
				             static_cast<int>(st));
				sslm_model_unmap(base);
				return 1;
			}
			sslm_adapter_release(adapter);
			std::printf("post-injection sslm_adapter_map (both seams disarmed again): PASS\n");
		}

		sslm_model_unmap(base);
	} else {
		std::printf("t2139_d3466_postload_region_pin: sslm_adapter_map cell NOT run (pass a real "
		            "adapter .sslm as argv[2] to run it)\n");
	}

	std::printf("t2139_d3466_postload_region_pin: PASS\n");
	return 0;
}
