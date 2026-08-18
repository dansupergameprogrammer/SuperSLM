// t2141_gate_c_t2138_suite_side_check_negative.cpp -- MUST-REJECT sibling for
// tools/t2141_gate_c_t2138_suite_side_check.cpp, same convention as tools/
// t2139_gate_c_xmacro_check_negative.cpp: the must-accept construction reads the REAL
// tests/t2138-abi-red-suite/sslm_abi.h directly, so it cannot itself be the vehicle for a
// deliberate corruption (that file is read-only per this construction's own invocation
// contract). This file proves the CHECK MECHANISM -- re-expanding the REAL library
// SSLM_STATUS_ENUM_LIST X-macro against a per-name generated check -- can actually fail, the
// same demonstration t2139_gate_c_xmacro_check_negative.cpp already gives for the sslm_g5.h-
// facing construction: a hand-written, deliberately corrupted "suite side" namespace, with
// exactly ONE enumerator given a wrong ordinal, re-checked by the SAME generated macro shape
// the must-accept construction uses.
//
// THIS FILE'S OWN CI STEP MUST FAIL TO COMPILE -- if it ever starts compiling clean, the
// per-name generation mechanism itself has regressed, matching every sibling negative's own
// convention.
#include "superslm/sslm_abi.h"

namespace t2141_gate_c_t2138_negative_suite_side {
// Deliberately corrupted: every name from SSLM_STATUS_ENUM_LIST, verbatim, EXCEPT
// SSLM_ARTIFACT_REJECTED, which is given ordinal 40 instead of the real header's 4 -- the same
// single, isolated divergence tools/t2139_gate_c_xmacro_check_negative.cpp already uses for its
// own sslm_g5.h-facing sibling.
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
}  // namespace t2141_gate_c_t2138_negative_suite_side

#define T2141_GATE_C_T2138_NEG_CHECK(NAME)                                                       \
	static_assert(                                                                                \
	    static_cast<int>(t2141_gate_c_t2138_negative_suite_side::NAME) == static_cast<int>(::NAME), \
	    #NAME " diverges (deliberate corruption, must-reject construction)")
#define T2141_GATE_C_T2138_NEG_CHECK_GEN_(NAME) T2141_GATE_C_T2138_NEG_CHECK(NAME);
SSLM_STATUS_ENUM_LIST(T2141_GATE_C_T2138_NEG_CHECK_GEN_)
#undef T2141_GATE_C_T2138_NEG_CHECK_GEN_
#undef T2141_GATE_C_T2138_NEG_CHECK

int main() { return 0; }
