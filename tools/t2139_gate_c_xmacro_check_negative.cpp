// t2139_gate_c_xmacro_check_negative.cpp -- Gate C's MUST-REJECT sibling for the X-MACRO-
// GENERATED per-name check mechanism specifically (S4, Claude/Poirot/
// 3bcbe43-t2139-fourth-confirmation-review.md): tools/t2139_gate_c_type_identity_check_negative.cpp
// (unchanged since 201bb94) proves only that static_assert itself works -- two hand-written toy
// enums, no #include "superslm/sslm_abi.h", no SSLM_STATUS_ENUM_LIST expansion at all. It exercises
// no part of Gate C's REAL mechanism (S4's own finding, verbatim). THIS file closes that gap for
// the X-macro-generation half specifically: it re-expands the REAL SSLM_STATUS_ENUM_LIST macro
// (the exact mechanism tools/t2139_gate_c_type_identity_check.cpp's own must-accept construction
// uses) against a deliberately-corrupted "suite side" -- one shared name given a wrong ordinal --
// so THIS file's own compile failure demonstrates the generated check itself catching a divergence,
// not merely that static_assert compiles.
//
// Construction: the REAL library side (#include "superslm/sslm_abi.h", global scope, unmodified --
// the SAME real header the must-accept construction uses). The "suite side" is a namespaced
// transcription with exactly ONE enumerator corrupted (SSLM_ARTIFACT_REJECTED, given ordinal 40
// instead of the real header's 4) -- everything else copied verbatim so this is a single, isolated
// divergence, not a wholesale toy enum. The SAME T2139_GATE_C_STATUS_CHECK-shaped macro the
// must-accept construction uses re-expands SSLM_STATUS_ENUM_LIST over BOTH sides; the generated
// check for SSLM_ARTIFACT_REJECTED is the one that must fail.
//
// THIS FILE'S OWN CI STEP MUST FAIL TO COMPILE -- if it ever starts compiling clean, the X-macro
// generation mechanism itself has regressed (a generated check silently stopped catching a real
// per-name divergence), and the build fails loudly on the regression itself, matching the sibling
// negative file's own convention.
#include "superslm/sslm_abi.h"

namespace t2139_gate_c_xmacro_negative_suite_side {
// Deliberately corrupted: every name from SSLM_STATUS_ENUM_LIST, verbatim, EXCEPT
// SSLM_ARTIFACT_REJECTED, which is given ordinal 40 instead of the real header's 4 -- a single,
// isolated per-name divergence the X-macro-generated check must catch.
typedef enum sslm_status {
	SSLM_OK = 0,
	SSLM_INVALID_ARGUMENT = 1,
	SSLM_BUFFER_TOO_SMALL = 2,
	SSLM_MISALIGNED_BUFFER = 3,
	SSLM_ARTIFACT_REJECTED = 40,  // CORRUPTED -- real header has this at 4
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
} sslm_status;
}  // namespace t2139_gate_c_xmacro_negative_suite_side

// The SAME generation shape tools/t2139_gate_c_type_identity_check.cpp's own must-accept
// construction uses -- re-expanding the REAL SSLM_STATUS_ENUM_LIST, one static_assert per name.
#define T2139_GATE_C_XMACRO_NEG_CHECK(NAME)                                                       \
	static_assert(                                                                                   \
	    static_cast<int>(t2139_gate_c_xmacro_negative_suite_side::NAME) == static_cast<int>(::NAME), \
	    #NAME " diverges (deliberate corruption, must-reject construction)")
#define T2139_GATE_C_XMACRO_NEG_CHECK_GEN_(NAME) T2139_GATE_C_XMACRO_NEG_CHECK(NAME);
SSLM_STATUS_ENUM_LIST(T2139_GATE_C_XMACRO_NEG_CHECK_GEN_)
#undef T2139_GATE_C_XMACRO_NEG_CHECK_GEN_
#undef T2139_GATE_C_XMACRO_NEG_CHECK

int main() { return 0; }
