// Test-only fault-injection seam for the T-2246 speculative-decoding verify primitive
// (plan Claude/Plans/SuperSLM_SpecDecoding_SubPlan_2026-08-22.md SS3.6 CM-G3). Header-only,
// following tests/support/bad_alloc_injection.h's established shape: a single-shot inline
// thread_local kind-flag with Arm/Disarm helpers, so a failing cell cannot leak an armed seam
// into the next test.
//
// ONE INDEPENDENT SLOT, never a reuse of g_inject_throw: the shipped lesson is one independent
// slot per region (the post-load pin, D-SLM3466) -- arming one call path must not be
// consumable by another region's consultation point. No shipped slot covers mid-CPU-verify;
// this is that region's own slot.
//
// The production consultation point (src/sslm_abi.cpp, inside the verify primitive's staged
// walk after at least one KV landing) is compiled only under
// SUPERSLM_ENABLE_T2246_SPECDEC_FAULT_INJECTION -- the existing injection-gating convention.
// In non-test builds the seam is never consulted and costs nothing on the walk.
#ifndef SUPERSLM_TESTS_SUPPORT_SPECDEC_INJECTION_H
#define SUPERSLM_TESTS_SUPPORT_SPECDEC_INJECTION_H

namespace superslm_test {

enum class SpecDecFaultKind {
	kNone = 0,
	kNonOkMidVerify,  // a genuine non-Ok raised between per-position stages, after at least
	                  // one KV landing -- the granularity forward_sites.cpp's chunk-batched
	                  // commit loop names
};

inline thread_local SpecDecFaultKind g_inject_specdec_fault = SpecDecFaultKind::kNone;

// Consulted at the verify primitive's staged midpoint. Single-shot: the read disarms, so a
// second call in the same walk never re-fires. Returns true when a fault was armed (the
// caller turns that into its documented non-Ok status); false otherwise.
inline bool ConsultSpecDecFault() {
	const SpecDecFaultKind kind = g_inject_specdec_fault;
	g_inject_specdec_fault = SpecDecFaultKind::kNone;
	return kind != SpecDecFaultKind::kNone;
}

// Test-side convenience for arming before the call under test.
inline void ArmSpecDecFault(SpecDecFaultKind kind) { g_inject_specdec_fault = kind; }

// Belt-and-braces disarm after every cell, armed or not.
inline void DisarmSpecDecFault() { g_inject_specdec_fault = SpecDecFaultKind::kNone; }

}  // namespace superslm_test

#endif  // SUPERSLM_TESTS_SUPPORT_SPECDEC_INJECTION_H
