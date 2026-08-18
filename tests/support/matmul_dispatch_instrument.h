// T-2158 test-only seam for T-2149's dispatch-cache instrumentation
// (Claude/Vitruvius/t2149-avx-kernel-design-2026-08-18.md §10 dimensions 1 and 3;
// Claude/Curie/t2158-t2149-avx-red-suite-2026-08-18.md). Header-only, mirroring
// tests/support/bad_alloc_injection.h's own shape exactly (a single test-only translation
// unit, tests/test_main.cpp, includes every seam header directly -- no new build-system
// entry is needed for the header itself).
//
// Current state, corrected fold round 4 (D-SLM3572; code review T-2170, Finding 6 found
// this comment describing a world that no longer existed at the reviewed commit): the
// counter IS wired into production -- `src/matmul.cpp`'s `DetectBestDotRowTier()` (design
// §6.2, built at `src/matmul.cpp:317-347`) increments
// `superslm_test::g_dot_row_tier_probe_invocations` exactly once, from inside its own
// C++11 magic-static initializer body -- so the increment runs under the same
// happens-before guarantee the initializer itself carries, which is the property design
// §10 dimension 1's "selected exactly once per process" claim and dimension 3's
// first-call-race cell both need observed -- and ONLY when
// SUPERSLM_ENABLE_MATMUL_DISPATCH_INSTRUMENT is defined. CMakeLists.txt defines that macro
// (alongside SUPERSLM_T2149_AVX_TIERS_BUILT) for the `superslm_test_injection` library
// target AND the three SSE2/AVX2/AVX512 `*_forced` library targets, under the
// `SUPERSLM_T2149_AVX_TIERS_BUILT` option (ON by default) -- never for the production
// `superslm` library, `sslm_verify`, or any other consumer -- matching
// SUPERSLM_ENABLE_BAD_ALLOC_INJECTION's own isolation (src/bad_alloc_wrap.h's header
// comment): a release build never references tests/support/* and the seam compiles to
// nothing (Layer 1 stays independently embeddable, D-SLM13).
//
// This header, the two cells that include it (design §10 dimensions 1/3,
// `tests/test_main.cpp`), and the production increment site all compile into every
// binary in the option-ON matrix, x64 and non-x64 alike -- the option being ON is not by
// itself a non-x64 no-op. The declaration below and the two cells that read it are
// additionally scoped to `SUPERSLM_MATMUL_HAVE_SIMD_X64` (include/superslm/matmul.h),
// exactly the guard `DetectBestDotRowTier()` itself compiles under (fold round 4,
// D-SLM3568; code review T-2170, Finding 2): on a non-x64 target the counter, the
// increment site, and both test cells do not exist to assert anything -- there is no
// mechanism there for them to observe.
#ifndef SUPERSLM_TESTS_SUPPORT_MATMUL_DISPATCH_INSTRUMENT_H
#define SUPERSLM_TESTS_SUPPORT_MATMUL_DISPATCH_INSTRUMENT_H

#include "superslm/matmul.h"

#if SUPERSLM_MATMUL_HAVE_SIMD_X64

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

#endif  // SUPERSLM_MATMUL_HAVE_SIMD_X64

#endif  // SUPERSLM_TESTS_SUPPORT_MATMUL_DISPATCH_INSTRUMENT_H
