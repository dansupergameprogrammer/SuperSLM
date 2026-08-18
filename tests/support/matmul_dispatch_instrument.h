// T-2158 test-only seam for T-2149's dispatch-cache instrumentation
// (Claude/Vitruvius/t2149-avx-kernel-design-2026-08-18.md §10 dimensions 1 and 3;
// Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md). Header-only, mirroring
// tests/support/bad_alloc_injection.h's own shape exactly (a single test-only translation
// unit, tests/test_main.cpp, includes every seam header directly -- no new build-system
// entry is needed for the header itself).
//
// THE COUNTER IS NOT YET WIRED INTO PRODUCTION as of this header's authoring -- the same
// gap tests/support/bad_alloc_injection.h's own header comment named before S-HARDEN-7's
// rename-and-wrap landed. `src/matmul.cpp`'s `DetectBestDotRowTier()` (design §6.2, not yet
// built) is expected to increment `superslm_test::g_dot_row_tier_probe_invocations` exactly
// once, from inside its own C++11 magic-static initializer body -- so the increment runs
// under the same happens-before guarantee the initializer itself carries, which is the
// property design §10 dimension 1's "selected exactly once per process" claim and dimension
// 3's first-call-race cell both need observed -- and ONLY when
// SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT is defined. CMakeLists.txt defines that macro
// (alongside SUPERSLM_T2149_AVX_TIERS_BUILT) for the `superslm_test_injection` library target
// only, under the `SUPERSLM_T2149_AVX_TIERS_BUILT` option (OFF by default) -- never for the
// production `superslm` library, `sslm_verify`, or any other consumer -- matching
// SUPERSLM_ENABLE_BAD_ALLOC_INJECTION's own isolation (src/bad_alloc_wrap.h's header
// comment): a release build never references tests/support/* and the seam compiles to
// nothing (Layer 1 stays independently embeddable, D-SLM13).
//
// Until DetectBestDotRowTier() exists and is wired to this seam, every red-suite cell built
// against this header is itself gated behind `#if defined(SUPERSLM_T2149_AVX_TIERS_BUILT)`
// (tests/test_main.cpp) -- the option stays OFF until the design's §11 build-decomposition
// step 1 lands, so this header, the cells that include it, and this seam all compile to
// nothing today, keeping the existing 34184/0 battery (D-SLM3496) unaffected.
#ifndef SUPERSLM_TESTS_SUPPORT_MATMUL_DISPATCH_INSTRUMENT_H
#define SUPERSLM_TESTS_SUPPORT_MATMUL_DISPATCH_INSTRUMENT_H

#include <atomic>

namespace superslm_test {

// Atomic, not a plain counter: design §10 dimension 3's entire cell is N threads racing the
// very first call to the magic-static initializer that is expected to perform this
// increment. A non-atomic counter observed under that same contention would itself be a data
// race the TSan leg (tests/test_main.cpp:5433-5544's crash-probe children run under
// `linux-x64-tsan`, .github/workflows/tests.yml) could flag independent of whatever
// DetectBestDotRowTier() itself does -- which would leave a TSan finding on this cell
// ambiguous between "the seam's own counter is unsynchronized" and "the production
// magic-static init is unsynchronized" (the property actually under test). An atomic counter
// removes the seam itself as a possible source of the race, so a TSan finding on this cell is
// attributable to DetectBestDotRowTier()'s own construction, not to this instrument.
inline std::atomic<long long> g_dot_row_tier_probe_invocations{0};

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SUPPORT_MATMUL_DISPATCH_INSTRUMENT_H
