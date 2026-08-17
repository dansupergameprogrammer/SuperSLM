// t2139_gate_c_sentinel_negative.cpp -- Gate C's MUST-REJECT sibling for the SENTINEL IDENTITY
// mechanism specifically (S4, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md): proves
// the SSLM_STATUS_NEXT_FREE registry-top identity assertion (F1's own structural remedy) fires on
// a real desynchronization -- two shared ordinals claimed by two different names, the exact class
// (F1's own Probe C) a per-name check structurally cannot see, since two enumerators that share a
// numeric value but not a name never appear in the same per-name check call.
//
// Construction: the REAL library side (#include "superslm/sslm_abi.h", global scope, unmodified).
// The "suite side" is a namespaced transcription that DESYNCHRONIZES the two sentinels: it carries
// every real name at its real ordinal EXCEPT it inserts one extra enumerator
// (SSLM_PROBE_SENTINEL_NEG_EXTRA) before its own sentinel, so its own SSLM_STATUS_NEXT_FREE
// auto-values to 27 instead of the real header's 26 -- a one-sided append, structurally identical
// to S1's own commissioned P2/P5 probes. Every per-name check for the 26 real, shared names still
// passes (none of their own ordinals moved); ONLY the sentinel identity assertion catches this.
//
// THIS FILE'S OWN CI STEP MUST FAIL TO COMPILE -- if it ever starts compiling clean, the sentinel
// mechanism itself has regressed (a one-sided append is no longer caught), and the build fails
// loudly on the regression itself, matching the sibling negative file's own convention.
#include "superslm/sslm_abi.h"

namespace t2139_gate_c_sentinel_negative_suite_side {
typedef enum sslm_status {
	SSLM_OK = 0,
	SSLM_INVALID_ARGUMENT = 1,
	SSLM_BUFFER_TOO_SMALL = 2,
	SSLM_MISALIGNED_BUFFER = 3,
	SSLM_ARTIFACT_REJECTED = 4,
	SSLM_ADAPTER_MODEL_MISMATCH = 5,
	SSLM_RESTORE_MODEL_MISMATCH = 6,
	SSLM_RESTORE_KV_MISMATCH = 7,
	SSLM_MODEL_HAS_LIVE_SEQUENCES = 8,
	SSLM_POOL_HAS_LIVE_HANDLES = 9,
	SSLM_ADAPTER_HAS_LIVE_SEQUENCES = 10,
	SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED = 11,
	SSLM_SEQ_RESET_MIDTOKEN_REJECTED = 12,
	SSLM_PREFIX_FROZEN_REJECTED = 13,
	SSLM_KV_POOL_EXHAUSTED = 14,
	SSLM_TOKEN_ID_OUT_OF_RANGE = 15,
	SSLM_CONTEXT_CAP_EXCEEDED = 16,
	SSLM_TOKEN_ID_UNMAPPED = 17,
	SSLM_SCHEMA_NOT_FOUND = 18,
	SSLM_SCHEMA_BIND_REJECTED = 19,
	SSLM_SCHEMA_SPAN_UNBOUND = 20,
	SSLM_PREFIX_SCHEMA_MISMATCH = 21,
	SSLM_RESTORE_SCHEMA_MISMATCH = 22,
	SSLM_SCHEMA_SPAN_UNREACHABLE = 23,
	SSLM_SCHEMA_UNSATISFIABLE = 24,
	SSLM_ALLOCATION_FAILED = 25,
	// One-sided append: an extra enumerator this namespace's own suite side carries that the real
	// library header does not -- pushes THIS side's own auto-valued sentinel to 27 while the real
	// library header's own SSLM_STATUS_NEXT_FREE stays at 26. Every per-name check above still
	// passes (no shared name's ordinal moved); only the sentinel comparison below can see this.
	SSLM_PROBE_SENTINEL_NEG_EXTRA = 26,
	SSLM_STATUS_NEXT_FREE
} sslm_status;
}  // namespace t2139_gate_c_sentinel_negative_suite_side

// Per-name checks -- all 26 real, shared names -- must all PASS (this construction's own point is
// that per-name checks alone are blind to the desynchronization below).
#define T2139_GATE_C_SENTINEL_NEG_CHECK(NAME)                                                     \
	static_assert(                                                                                    \
	    static_cast<int>(t2139_gate_c_sentinel_negative_suite_side::NAME) == static_cast<int>(::NAME), \
	    #NAME " diverges (unexpected -- this construction's own per-name checks should all pass)")
#define T2139_GATE_C_SENTINEL_NEG_CHECK_GEN_(NAME) T2139_GATE_C_SENTINEL_NEG_CHECK(NAME);
SSLM_STATUS_ENUM_LIST(T2139_GATE_C_SENTINEL_NEG_CHECK_GEN_)
#undef T2139_GATE_C_SENTINEL_NEG_CHECK_GEN_
#undef T2139_GATE_C_SENTINEL_NEG_CHECK

// THE must-reject assertion: the one-sided append above desynchronizes the two sentinels -- this
// must fail.
static_assert(
    static_cast<int>(t2139_gate_c_sentinel_negative_suite_side::SSLM_STATUS_NEXT_FREE) ==
        static_cast<int>(::SSLM_STATUS_NEXT_FREE),
    "registry-top divergence (deliberate one-sided append, must-reject construction): the two "
    "SSLM_STATUS_NEXT_FREE sentinels no longer agree");

int main() { return 0; }
