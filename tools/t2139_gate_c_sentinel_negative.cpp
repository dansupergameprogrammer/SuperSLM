// t2139_gate_c_sentinel_negative.cpp -- Gate C's MUST-REJECT sibling for the SENTINEL IDENTITY
// mechanism specifically (S4, Claude/Poirot/3bcbe43-t2139-fourth-confirmation-review.md): proves
// the SSLM_STATUS_NEXT_FREE registry-top identity assertion (F1's own structural remedy) fires on
// a real desynchronization -- two shared ordinals claimed by two different names, the exact class
// (F1's own Probe C) a per-name check structurally cannot see, since two enumerators that share a
// numeric value but not a name never appear in the same per-name check call.
//
// Construction: the REAL library side (#include "superslm/sslm_abi.h", global scope, unmodified).
// The "suite side" is a namespaced, GENERATED copy that DESYNCHRONIZES the two sentinels: it
// carries every real name at its real ordinal (re-expanded from SSLM_STATUS_ENUM_LIST, see below)
// plus one extra enumerator (SSLM_PROBE_SENTINEL_NEG_EXTRA) inserted before its own sentinel, so
// its own SSLM_STATUS_NEXT_FREE auto-values one past the real header's -- a one-sided append,
// structurally identical to S1's own commissioned P2/P5 probes. Every per-name check for the real,
// shared names still passes (none of their own ordinals moved); ONLY the sentinel identity
// assertion catches this.
//
// THIS FILE'S OWN CI STEP MUST FAIL TO COMPILE -- if it ever starts compiling clean, the sentinel
// mechanism itself has regressed (a one-sided append is no longer caught), and the build fails
// loudly on the regression itself, matching the sibling negative file's own convention.
//
// GENERATED suite side (S3, Claude/Poirot/ce5aff2-t2139-fifth-confirmation-review.md, remedy (b)):
// a hand-transcribed 26-name copy went vacuous the first time a governed enumerator append landed
// on both real headers -- the append moved the REAL sentinel to 27 while this file's own
// hand-transcribed copy, unaware of the new name, still auto-valued its extra enumerator to 27 too,
// so the two sentinels re-aligned and the construction silently stopped demonstrating anything
// (executed: C2039 name-lookup only, the intended registry-top static_assert never reached).
// Re-expanding SSLM_STATUS_ENUM_LIST here instead removes the term: the suite side now mirrors
// whatever the real header's list currently contains, on every governed append, with nothing to
// hand-update -- and the ONE deliberate extra enumerator inserted after the generated names keeps
// this side's own auto-valued sentinel exactly one past the real header's, regardless of how many
// entries the list expands to.
#include "superslm/sslm_abi.h"

namespace t2139_gate_c_sentinel_negative_suite_side {
typedef enum sslm_status {
#define SSLM_GATE_C_SENTINEL_NEG_ENUM_VALUE_(name) name,
	SSLM_STATUS_ENUM_LIST(SSLM_GATE_C_SENTINEL_NEG_ENUM_VALUE_)
#undef SSLM_GATE_C_SENTINEL_NEG_ENUM_VALUE_
	// One-sided append: an extra enumerator this namespace's own suite side carries that the real
	// library header does not -- pushes THIS side's own auto-valued sentinel one past the real
	// library header's own SSLM_STATUS_NEXT_FREE, no matter how many entries the generated list
	// above expands to. Every per-name check below still passes (no shared name's ordinal moved);
	// only the sentinel comparison below can see this.
	SSLM_PROBE_SENTINEL_NEG_EXTRA,
	SSLM_STATUS_NEXT_FREE
} sslm_status;
}  // namespace t2139_gate_c_sentinel_negative_suite_side

// Per-name checks -- every real, shared name -- must all PASS (this construction's own point is
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
