// t2139_f2_catchall_construction_pin.cpp -- derived pin list item 1 (Claude/Brunel/
// t2139-abi-build-2026-08-16.md Sec22): "A fault-injection cell for the widened catch (...) arm
// -- prove a non-std::exception-derived throw (a custom type, or a non-class throw like
// throw int) raised beneath one of the CatchAllocationFailure-wrapped verbs ... is caught at the
// extern "C" boundary and returns SSLM_ARTIFACT_REJECTED -- not SSLM_ALLOCATION_FAILED, not a
// crash, not UB."
//
// WHY THIS IS A CONSTRUCTION CELL, NOT A FAULT-INJECTION CELL AGAINST A LIVE CALL PATH (stated
// plainly, per this arc's own culture of naming a limitation rather than silently working around
// it -- see tools/t2139_n3_bad_alloc_pin.cpp's own header for the precedent this follows):
// CatchAllocationFailure (src/sslm_abi.cpp) has internal linkage (defined inside an anonymous
// namespace, lines ~250-634) and is not callable from any other translation unit. Every REAL
// call path that reaches it either (a) is fronted by SslmModel::Load, whose own public wrapper
// is internal::WrapBadAllocContract([&]{ return LoadImpl(...); }) (src/model.cpp:1411-1414) --
// and WrapBadAllocContract's own catch (const std::exception&) clause (src/bad_alloc_wrap.h)
// narrows EVERY non-bad_alloc std::exception to std::bad_alloc before the exception ever reaches
// sslm_model_map's/sslm_adapter_map's own outer catch, so their own catch(...) arm is structurally
// unreachable via injected faults today (see tools/t2139_f2_length_error_pin.cpp's own header for
// the full, executed measurement of this); or (b) is BuildEngineCache's own direct seam
// consultation (src/sslm_abi.cpp:446), which the SAME single-shot thread_local seam makes
// unreachable because SslmModel::Load's own consultation always runs first within one
// sslm_model_map call (tools/t2139_n3_bad_alloc_pin.cpp's own header names this exact gap); or
// (c) a sizing/construction verb (sslm_workspace_create, sslm_kv_pool_create, sslm_prefix_begin,
// sslm_seq_create, ...) whose own CatchAllocationFailure-wrapped lambda calls no
// MaybeThrowInjectedBadAllocFault() consultation at all (grep-confirmed absent from every one of
// these bodies), so the seam has nothing to trigger there regardless of arming.
//
// The derived pin list itself anticipated exactly this: "needs either a new injection-seam fault
// kind ... or a dedicated test-only throw site." Both existing InjectThrowKind values already
// cover length_error/runtime_error (no new kind needed -- see the length_error pin), but no
// dedicated test-only throw site reaches CatchAllocationFailure's own catch(...) arm with an
// unnarrowed type through any currently-built call path, and adding one is a production-code
// change (src/sslm_abi.cpp) outside this suite's own writable scope (Curie owns
// tests/t2130-g5-red-suite/** and tests/t2138-abi-red-suite/** only). The invocation for this
// pin item explicitly allows a "fault-injection OR construction cell" for exactly this reason.
//
// CONSTRUCTION METHOD: CatchAllocationFailure is transcribed here VERBATIM from src/sslm_abi.cpp
// as it stands at this round's own commit (the three-way catch landed by the F2 fold ruling,
// Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md Sec6, design commit dated 2026-08-17)
// -- the SAME "same helper shape, same flags" method Poirot's own F2 finding used
// (Claude/Poirot/4466666-t2139-third-confirmation-review.md Sec3, F2: "Executed probe, same
// helper shape, same flags"). CURRENT() below is this transcription; OLD_CATCH_EXCEPTION_ONLY()
// is the pre-fix shape (catch (const std::exception&) only, the exact defect F2 found) --
// reproduced for the mutation comparison this pin's own exit condition requires ("discrimination
// verified by mutation where cheap").
#include <cstdio>
#include <stdexcept>
#include <type_traits>

// A minimal stand-in for sslm_status -- only the three values this pin needs, values chosen to
// match the real header's own ordinals so a reader can cross-reference directly (not load-bearing
// for the proof itself, which only compares relative outcomes).
enum class ProbeStatus : int {
	kOk = 0,
	kAllocationFailed = 25,
	kArtifactRejected = 4,
};

namespace current_shape {
// Verbatim transcription of src/sslm_abi.cpp's CatchAllocationFailure, post F2 fold ruling.
template <typename Fn>
ProbeStatus CatchAllocationFailure(Fn&& fn) {
	try {
		return fn();
	} catch (const std::bad_alloc&) {
		return ProbeStatus::kAllocationFailed;
	} catch (const std::length_error&) {
		return ProbeStatus::kAllocationFailed;
	} catch (...) {
		return ProbeStatus::kArtifactRejected;
	}
}
}  // namespace current_shape

namespace old_shape {
// The PRE-fix shape F2 found defective: catch (const std::exception&) only -- both under- and
// over-reaches (Claude/Poirot/4466666-t2139-third-confirmation-review.md F2's own compiled
// probe). Kept here for the mutation comparison, not as a claim about any file in the tree today.
template <typename Fn>
ProbeStatus CatchAllocationFailure(Fn&& fn) {
	try {
		return fn();
	} catch (const std::exception&) {
		return ProbeStatus::kAllocationFailed;
	}
}
}  // namespace old_shape

// A non-std::exception-derived type -- the class of throw F2's own probe showed crossing the
// extern "C" boundary as real, observed undefined behavior under the old shape.
struct ForeignError {
	int code = 7;
};

namespace {
int g_fail = 0;
void Check(bool cond, const char* what) {
	if (cond) {
		std::printf("  PASS: %s\n", what);
	} else {
		std::printf("  FAIL: %s\n", what);
		++g_fail;
	}
}
}  // namespace

int main() {
	std::printf("=== t2139_f2_catchall_construction_pin ===\n");

	// --- Cell 1: CURRENT shape, a non-allocation std::exception (std::runtime_error) is caught by
	// catch (...) and returns kArtifactRejected, NOT kAllocationFailed. This is the F2 ruling's own
	// "no wrapper for causes that already have their own status" claim, at the helper level. ---
	{
		const ProbeStatus st = current_shape::CatchAllocationFailure([]() -> ProbeStatus {
			throw std::runtime_error("non-allocation cause");
		});
		Check(st == ProbeStatus::kArtifactRejected,
		      "CURRENT shape: std::runtime_error -> kArtifactRejected (not kAllocationFailed)");
	}

	// --- Cell 2: CURRENT shape, a genuinely non-std::exception-derived throw (ForeignError) is
	// caught by catch (...) and returns kArtifactRejected -- no crash, no UB, no propagation past
	// this boundary. This is the UB-closing half of F2 (the previous catch (const std::exception&)
	// let exactly this class of throw escape). ---
	{
		bool escaped = true;
		ProbeStatus st = ProbeStatus::kOk;
		try {
			st = current_shape::CatchAllocationFailure([]() -> ProbeStatus {
				throw ForeignError{};
			});
			escaped = false;
		} catch (const ForeignError&) {
			escaped = true;
		}
		Check(!escaped, "CURRENT shape: ForeignError (non-std::exception) does NOT escape the boundary");
		Check(!escaped && st == ProbeStatus::kArtifactRejected,
		      "CURRENT shape: ForeignError -> kArtifactRejected");
	}

	// --- Cell 3: CURRENT shape, a non-class throw (throw int) is ALSO caught by catch (...) --
	// the widest form of the UB class, since it is not even an object with a vtable/type_info the
	// old catch (const std::exception&) could ever have matched under any circumstance. ---
	{
		bool escaped = true;
		ProbeStatus st = ProbeStatus::kOk;
		try {
			st = current_shape::CatchAllocationFailure([]() -> ProbeStatus {
				throw 42;
			});
			escaped = false;
		} catch (int) {
			escaped = true;
		}
		Check(!escaped, "CURRENT shape: throw int (non-class) does NOT escape the boundary");
		Check(!escaped && st == ProbeStatus::kArtifactRejected, "CURRENT shape: throw int -> kArtifactRejected");
	}

	// --- Cell 4: CURRENT shape, std::bad_alloc and std::length_error are unaffected by the
	// narrowing to catch (...) -- both still resolve to kAllocationFailed, confirming the fix did
	// not regress the family it is required to keep (design Sec6's own ratified meaning). ---
	{
		const ProbeStatus st_ba = current_shape::CatchAllocationFailure(
		    []() -> ProbeStatus { throw std::bad_alloc(); });
		Check(st_ba == ProbeStatus::kAllocationFailed, "CURRENT shape: std::bad_alloc -> kAllocationFailed (unchanged)");
		const ProbeStatus st_le = current_shape::CatchAllocationFailure(
		    []() -> ProbeStatus { throw std::length_error("x"); });
		Check(st_le == ProbeStatus::kAllocationFailed, "CURRENT shape: std::length_error -> kAllocationFailed (unchanged)");
	}

	// --- MUTATION -- discrimination proof (StandardsDocument Sec5.4: "state the wrong version,
	// show the cells fail"). Re-run cell 1's own std::runtime_error input against the OLD,
	// pre-fix shape: it must produce the WRONG answer (kAllocationFailed, the mis-attribution F2
	// found), proving cell 1 above discriminates the fix rather than passing under both shapes. ---
	{
		const ProbeStatus st = old_shape::CatchAllocationFailure([]() -> ProbeStatus {
			throw std::runtime_error("non-allocation cause");
		});
		Check(st == ProbeStatus::kAllocationFailed,
		      "MUTATION (old shape): std::runtime_error mis-attributed as kAllocationFailed "
		      "(reproduces F2's own defect -- proves cell 1 above discriminates the fix)");
	}

	// --- MUTATION -- discrimination proof, UB half: the OLD shape's own catch (const
	// std::exception&) does not match ForeignError at all, so it propagates PAST this function
	// entirely (the exact undefined-behavior-at-the-extern-"C"-boundary class F2 exists to close).
	// Caught one level up here ONLY so this pin's own process does not terminate -- in the real
	// extern "C" boundary this crossing is undefined behavior, not a recoverable catch. ---
	{
		bool old_shape_let_it_escape = false;
		try {
			old_shape::CatchAllocationFailure([]() -> ProbeStatus { throw ForeignError{}; });
		} catch (const ForeignError&) {
			old_shape_let_it_escape = true;
		}
		Check(old_shape_let_it_escape,
		      "MUTATION (old shape): ForeignError propagates PAST the boundary uncaught "
		      "(reproduces the UB class F2 found -- proves cell 2 above discriminates the fix)");
	}

	if (g_fail != 0) {
		std::printf("t2139_f2_catchall_construction_pin: FAIL (%d)\n", g_fail);
		return 1;
	}
	std::printf("t2139_f2_catchall_construction_pin: PASS\n");
	return 0;
}
