// Test-only fault-injection seam for the "throws only std::bad_alloc" contract
// (S-HARDEN-7, F5; Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_Design-2026-07-23.md
// §3.1, §3.2). Header-only (inline thread_locals; the project's CMakeLists.txt
// builds a single test translation unit, tests/test_main.cpp, so this needs no
// new build-system entry).
//
// §3.1 commissions "a test-only seam (a function pointer or thread-local
// override, compiled only under the test build, guarded by a macro so it does
// not exist in a release binary) that lets each site's *Impl body be swapped
// for a stub that throws the target type." This header is that seam's public
// half: the flags a test sets before calling a wrapped site's real public
// entry point, and the marker the shared wrap helper's catch clauses are
// expected to set once built.
//
// THE SEAM IS NOT YET WIRED INTO PRODUCTION. MaybeThrowInjectedFault() below
// is dead code today -- nothing in src/*.cpp calls it. S-HARDEN-7's
// rename-and-wrap (design §3.1: each public entry point's body is renamed to
// an internal *Impl, and the seam check is added at its entry) is what wires
// it in. Until then, every red test in this file's companion cells in
// tests/test_main.cpp calls a real, unwrapped production function while this
// seam is armed, observes that the production function ran its real logic and
// never consulted the seam (so no exception crosses at all), and fails for
// that documented reason -- proving the contract does not yet hold, which is
// the point of a red-first test.
#ifndef SUPERSLM_TESTS_SUPPORT_BAD_ALLOC_INJECTION_H
#define SUPERSLM_TESTS_SUPPORT_BAD_ALLOC_INJECTION_H

#include <stdexcept>

namespace superslm_test {

// T-2133 fold, second pass (D-SLM3464, Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec6): a std::exception-derived type (this seam's
// three existing kinds below) is exactly the class WrapBadAllocContract (src/bad_alloc_wrap.h)
// narrows to std::bad_alloc BEFORE the T-2139 ABI's own extern "C" boundary can classify it --
// none of the existing kinds can exercise that boundary's own catch(...) arm through a real
// Load-fronted call path (sslm_model_map/sslm_adapter_map). A throw NOT derived from
// std::exception at all matches neither of WrapBadAllocContract's own two catch clauses
// (catch (const std::bad_alloc&), catch (const std::exception&)) and propagates through it
// unmolested -- ForeignFault below is exactly that: a minimal, deliberately NOT
// std::exception-derived type, for the one class this seam previously had no way to inject.
struct ForeignFault {};

// Which non-bad_alloc exception type (or bad_alloc itself, for the
// passthrough-branch cell) the seam should throw the next time a wrapped
// site's *Impl body consults it. Single-shot: MaybeThrowInjectedFault()
// resets this to kNone after throwing (or after finding it already kNone).
enum class InjectThrowKind {
	kNone = 0,
	kLengthError,   // std::length_error -- F5's second confirmed exception type
	kRuntimeError,  // an arbitrary std::exception-derived type, for the "anything
	                // else narrows to the promise" branch (design §3.1's wrap code)
	kBadAlloc,      // std::bad_alloc itself -- the permitted-exception passthrough cell
	kForeignFault,  // ForeignFault, above -- NOT std::exception-derived, T-2133 D-SLM3464:
	                // the one class WrapBadAllocContract's own two catch clauses do not narrow,
	                // for proving the T-2139 ABI boundary's own catch(...) arm through a real
	                // Load-fronted call path (sslm_model_map/sslm_adapter_map).
};

inline thread_local InjectThrowKind g_inject_throw = InjectThrowKind::kNone;

// Which catch clause of the shared wrap helper most recently ran. Set by the
// wrap helper itself (once built) so a test can assert the bad_alloc-specific
// clause fired, not merely that the caller observed a std::bad_alloc that
// could equally have been reconstructed by the general clause (design §3.2,
// "New cell -- force a genuine std::bad_alloc through the wrap").
enum class LastWrapClause {
	kNone = 0,
	kBadAllocClause,  // catch (const std::bad_alloc&) { throw; }
	kGeneralClause,   // catch (const std::exception&) { throw std::bad_alloc{}; }
};

inline thread_local LastWrapClause g_last_wrap_clause = LastWrapClause::kNone;

// Consulted at the entry of each wrapped site's *Impl body, once S-HARDEN-7's
// rename-and-wrap lands. Throws the armed exception type (resetting the seam
// to kNone first, so a second call in the same test does not re-throw) if one
// is armed; otherwise returns immediately. A no-op today because nothing
// calls it -- see the file header comment above.
inline void MaybeThrowInjectedFault() {
	InjectThrowKind kind = g_inject_throw;
	g_inject_throw = InjectThrowKind::kNone;
	switch (kind) {
		case InjectThrowKind::kNone:
			return;
		case InjectThrowKind::kLengthError:
			throw std::length_error("superslm_test: injected fault (length_error)");
		case InjectThrowKind::kRuntimeError:
			throw std::runtime_error("superslm_test: injected fault (runtime_error)");
		case InjectThrowKind::kBadAlloc:
			throw std::bad_alloc();
		case InjectThrowKind::kForeignFault:
			throw ForeignFault{};
	}
}

// Test-side convenience: arm the seam and reset the clause marker, so a test
// cannot read a stale marker left by an earlier test.
inline void ArmInjectedFault(InjectThrowKind kind) {
	g_inject_throw = kind;
	g_last_wrap_clause = LastWrapClause::kNone;
}

// Belt-and-braces disarm, called after every cell whether or not the expected
// exception crossed, so a test that fails early never leaves the seam armed
// for the next test in the same binary.
inline void DisarmInjectedFault() {
	g_inject_throw = InjectThrowKind::kNone;
}

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SUPPORT_BAD_ALLOC_INJECTION_H
