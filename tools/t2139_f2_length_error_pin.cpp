// t2139_f2_length_error_pin.cpp -- derived pin list item 2 (Claude/Brunel/
// t2139-abi-build-2026-08-16.md Sec22, "A cell confirming std::length_error also narrows to
// SSLM_ALLOCATION_FAILED -- the ratified family now names two exception types (bad_alloc,
// length_error); only bad_alloc has a pin"), against the FOLD RULING on F2 (Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec6, design commit dated 2026-08-17; D-SLM3462):
// catch (const std::bad_alloc&) / catch (const std::length_error&) both -> SSLM_ALLOCATION_FAILED.
//
// Reuses the SAME real, test-only fault-injection seam (tests/support/bad_alloc_injection.h,
// src/bad_alloc_wrap.h) tools/t2139_n3_bad_alloc_pin.cpp already trusts -- InjectThrowKind
// already carries kLengthError and kRuntimeError alongside kBadAlloc (this file's own header
// comment names kLengthError "F5's second confirmed exception type" and kRuntimeError "an
// arbitrary std::exception-derived type... for the 'anything else narrows to the promise'
// branch"), so no new injection-seam fault kind needed here, contrary to the derived pin list's
// own "needs either a new injection-seam fault kind... or a dedicated test-only throw site" --
// both kinds already existed; this pin uses them as-is.
//
// A REAL FINDING, discovered while authoring this pin, stated plainly rather than routed around
// (StandardsDocument Sec5.6): armed through sslm_model_map's own call path, BOTH kLengthError and
// kRuntimeError are consulted at SslmModel::Load's own S-HARDEN-7 *Impl entry (model.cpp,
// LoadImpl's own first statement) -- and SslmModel::Load's own public wrapper is
// internal::WrapBadAllocContract([&]{ return LoadImpl(...); }) (src/model.cpp:1411-1414).
// WrapBadAllocContract's own shape (src/bad_alloc_wrap.h) is `catch (const std::bad_alloc&)
// { throw; }` then `catch (const std::exception&) { throw std::bad_alloc{}; }` -- EVERY
// std::exception subtype that is not bad_alloc itself, INCLUDING std::length_error AND
// std::runtime_error, is narrowed to a bare std::bad_alloc BEFORE it ever leaves Load's own
// public entry point. By the time sslm_model_map's own bare try/catch (src/sslm_abi.cpp,
// F2's own three-way split) sees the exception, it is ALWAYS std::bad_alloc, regardless of which
// kind was armed -- sslm_model_map's own catch(const std::length_error&) and catch(...) clauses
// are therefore UNREACHABLE via this call path specifically, for the identical single-shot-seam-
// ordering reason tools/t2139_n3_bad_alloc_pin.cpp's own header already names for
// BuildEngineCache. This does NOT falsify the F2 ruling's own claim about the boundary helper in
// isolation (tools/t2139_f2_catchall_construction_pin.cpp proves that directly, by construction,
// since no real call path reaches CatchAllocationFailure's own catch(...) arm with an unnarrowed
// exception type either -- see that file's own header for the full accounting). What this pin
// below DOES prove, for real, against the real landed sslm_model_map: the OBSERVABLE, end-to-end
// contract "a length_error raised beneath sslm_model_map's own Load call narrows to
// SSLM_ALLOCATION_FAILED" holds -- which is exactly pin item 2's own literal claim -- even though
// the INTERNAL mechanism that delivers it is WrapBadAllocContract's narrowing, not
// sslm_model_map's own dedicated length_error clause. The kRuntimeError cell below is included as
// the honest companion measurement: it proves that a non-allocation std::exception raised beneath
// sslm_model_map ALSO currently narrows to SSLM_ALLOCATION_FAILED (via the identical
// WrapBadAllocContract mechanism) rather than reaching SSLM_ARTIFACT_REJECTED -- a real,
// execution-confirmed divergence from the F2 ruling's own per-helper claim, named here rather
// than silently assumed swept, and left for the design/build authority to rule on (whether
// WrapBadAllocContract's own S-HARDEN-7 narrowing is itself in scope for a Sec6 update, or
// whether F2's "non-allocation std::exception -> SSLM_ARTIFACT_REJECTED" claim is scoped to
// verbs whose CatchAllocationFailure wrap is the FIRST consultation, which sslm_model_map/
// sslm_adapter_map -- both Load-fronted -- structurally never are).
//
// Usage: t2139_f2_length_error_pin.exe <path-to-real.sslm>
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

bool CallOnce(const std::vector<uint8_t>& bytes, sslm_status* out_status) {
	sslm_model model = nullptr;
	*out_status = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (*out_status == SSLM_OK && model) {
		sslm_model_unmap(model);
	}
	return true;
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

	// Baseline, seam disarmed.
	{
		sslm_status st = SSLM_OK;
		CallOnce(bytes, &st);
		if (st != SSLM_OK) {
			std::fprintf(stderr, "FAIL: baseline sslm_model_map (seam disarmed) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("baseline sslm_model_map (seam disarmed): PASS\n");
	}

	// Cell 1 -- armed kLengthError: the FOLD RULING's own second covered exception type.
	{
		superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kLengthError);
		sslm_status st = SSLM_OK;
		CallOnce(bytes, &st);
		superslm_test::DisarmInjectedFault();
		if (st != SSLM_ALLOCATION_FAILED) {
			std::fprintf(stderr,
			             "FAIL: sslm_model_map(injected length_error) returned %d, expected "
			             "SSLM_ALLOCATION_FAILED\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("length_error pin (sslm_model_map, injected std::length_error via "
		            "SslmModel::Load's own S-HARDEN-7 seam, narrowed by WrapBadAllocContract "
		            "before sslm_model_map's own catch runs): PASS "
		            "(SSLM_ALLOCATION_FAILED fired, no crash, no UB)\n");
	}

	// Cell 2 -- armed kRuntimeError: the honest companion measurement (see file header). Asserts
	// the REAL, currently-observed behavior (SSLM_ALLOCATION_FAILED via WrapBadAllocContract's own
	// narrowing), not the isolated-helper claim (SSLM_ARTIFACT_REJECTED) -- a pin that asserted the
	// isolated-helper claim here would be RED against the real landed code and misreport what this
	// call path actually does.
	{
		superslm_test::ArmInjectedFault(superslm_test::InjectThrowKind::kRuntimeError);
		sslm_status st = SSLM_OK;
		CallOnce(bytes, &st);
		superslm_test::DisarmInjectedFault();
		if (st != SSLM_ALLOCATION_FAILED) {
			std::fprintf(stderr,
			             "FAIL: sslm_model_map(injected runtime_error) returned %d, expected "
			             "SSLM_ALLOCATION_FAILED (the REAL, WrapBadAllocContract-narrowed "
			             "behavior of this call path today -- see file header)\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("runtime_error companion measurement (sslm_model_map, injected "
		            "std::runtime_error): PASS -- narrows to SSLM_ALLOCATION_FAILED via "
		            "WrapBadAllocContract, NOT SSLM_ARTIFACT_REJECTED, because SslmModel::Load's "
		            "own S-HARDEN-7 wrap is the first consultation point on this call path. See "
		            "this file's own header for the routed finding this measurement supports.\n");
	}

	// Seam disarmed again -- single-shot, no residual state, handle still fully usable.
	{
		sslm_status st = SSLM_OK;
		CallOnce(bytes, &st);
		if (st != SSLM_OK) {
			std::fprintf(stderr,
			             "FAIL: post-injection sslm_model_map (seam disarmed again) returned %d\n",
			             static_cast<int>(st));
			return 1;
		}
		std::printf("post-injection sslm_model_map (seam disarmed again): PASS\n");
	}

	std::printf("t2139_f2_length_error_pin: PASS\n");
	return 0;
}
