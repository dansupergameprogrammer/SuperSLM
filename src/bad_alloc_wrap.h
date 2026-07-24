// S-HARDEN-7 (Claude/Vitruvius/SuperSLM_SHARDEN678_Bundle_Design-2026-07-23.md
// Sec3.1; T-411): the shared catch-and-rethrow boundary every public entry
// point in the "throws only std::bad_alloc" contract's derived population
// wraps itself with. std::bad_alloc crosses unchanged; every other
// std::exception-derived type narrows to std::bad_alloc. This is a
// rename-and-wrap, not a rewrite -- the eighteen sites' own validation logic
// is unchanged; only renamed to a *Impl-suffixed inner call and wrapped.
//
// tools/ci/check_bad_alloc_contract.py's "check wrapping" step looks for
// exactly this shape (a call to WrapBadAllocContract wrapping a *Impl call)
// in each derived member's definition -- the convention here is what that
// checker verifies, so every wrapped site follows it uniformly.
//
// The test-only fault-injection seam (tests/support/bad_alloc_injection.h)
// is consulted at the entry of each *Impl body via
// internal::MaybeThrowInjectedBadAllocFault() below, and this helper's catch
// clauses additionally record which one fired -- but ONLY when
// SUPERSLM_ENABLE_BAD_ALLOC_INJECTION is defined. This header does not
// itself define that macro; CMakeLists.txt defines it for the
// superslm_test_injection library target only (the target superslm_tests
// links against), never for the production superslm library sslm_verify and
// every other tool link against. Without the macro, neither the include
// below nor the marker-setting code is compiled: a release build of this
// library never references tests/support/* at all, and the seam check
// compiles to nothing (Layer 1 stays independently embeddable, D-SLM13).
#ifndef SUPERSLM_SRC_BAD_ALLOC_WRAP_H
#define SUPERSLM_SRC_BAD_ALLOC_WRAP_H

#include <stdexcept>
#include <utility>

#ifdef SUPERSLM_ENABLE_BAD_ALLOC_INJECTION
#include "support/bad_alloc_injection.h"
#endif

namespace superslm {
namespace internal {

// Wraps `fn` (typically a lambda calling one site's *Impl body) so that a
// thrown std::bad_alloc crosses unchanged and every other
// std::exception-derived throw narrows to std::bad_alloc. Templated on the
// callable's return type via `decltype(fn())`, which covers `void` the same
// way it covers every other return type (a `return` statement of a void
// expression inside a void-returning function is well-formed) -- one shape
// for all eighteen sites' return types (SslmStatus, SslmModelStatus, bool,
// std::string, std::vector<T>, void), matching design Sec3.1's "two handling
// shapes... covering all eighteen sites with one helper".
template <typename Fn>
auto WrapBadAllocContract(Fn&& fn) -> decltype(fn()) {
	try {
		return fn();
	} catch (const std::bad_alloc&) {
#ifdef SUPERSLM_ENABLE_BAD_ALLOC_INJECTION
		superslm_test::g_last_wrap_clause = superslm_test::LastWrapClause::kBadAllocClause;
#endif
		throw;  // the one permitted exception crosses unchanged
	} catch (const std::exception&) {
#ifdef SUPERSLM_ENABLE_BAD_ALLOC_INJECTION
		superslm_test::g_last_wrap_clause = superslm_test::LastWrapClause::kGeneralClause;
#endif
		throw std::bad_alloc{};  // length_error and anything else narrows to the promise
	}
}

// Consulted at the entry of each wrapped site's *Impl body (design Sec3.1's
// "a test-only seam... that lets each site's *Impl body be swapped for a
// stub that throws the target type"). A no-op unless
// SUPERSLM_ENABLE_BAD_ALLOC_INJECTION is defined (the test build only).
inline void MaybeThrowInjectedBadAllocFault() {
#ifdef SUPERSLM_ENABLE_BAD_ALLOC_INJECTION
	superslm_test::MaybeThrowInjectedFault();
#endif
}

}  // namespace internal
}  // namespace superslm

#endif  // SUPERSLM_SRC_BAD_ALLOC_WRAP_H
