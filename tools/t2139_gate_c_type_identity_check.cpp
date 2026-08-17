// t2139_gate_c_type_identity_check.cpp -- Gate C (design Sec9): type identity, closing the
// blind spot Gate A cannot see (Gate A reuses one side's types to avoid a redefinition error,
// so it can never itself detect a struct/enum BODY divergence under a shared name). MUST-ACCEPT
// construction: never linked, compile-only.
//
// Construction (design Sec9, literal, S8 fix round -- Claude/Poirot/
// 2c18dab-t2139-abi-build-review.md): for the LIBRARY side, this file now includes
// include/superslm/sslm_abi.h DIRECTLY, at global scope -- not a second hand transcription. The
// suite side alone stays a transcription (into its own namespace), because including BOTH real
// headers in one TU would collide on their shared extern "C" function names declared with
// different types ([dcl.link], the same collision Gate A's own macro-rename technique routes
// around) -- but nothing stops the LIBRARY side from being the real header, since only ONE side
// needs to avoid the collision for it to disappear. This removes HALF the drift class S8 named:
// a change to sslm_abi.h now perturbs this gate automatically, because it no longer merely
// re-states sslm_abi.h's own body a second time by hand.
//
// The OTHER half -- two headers assigning the SAME ordinal to DIFFERENT enumerator names, which
// a per-shared-NAME check structurally cannot see -- is closed by COMPLETE per-name coverage
// against a COMPLETE mirror, per the T-2133 enum-governance ruling (design Sec6, design commit
// 4f4eb23896): design Sec6 is the single-authority complete ordinal registry for sslm_status,
// and sslm_g5.h mirrors it VERBATIM (all 26 entries, 0..25, no gaps; next-free is a Sec6 fact).
// The earlier "base low / G5 appended above, disjoint ranges" arrangement is RETIRED by that
// ruling -- the suite header is no longer a base-plus-additions overlay but a full mirror, so
// the disjointness assertion this file previously carried (correctly red at the time, P2,
// Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.4) is replaced by the registry-top
// identity assertion below: with every library enumerator per-name checked against the complete
// mirror, a rename is caught by name lookup failing, a move is caught by value inequality, and
// a one-sided append is caught by the registry-top identity -- the different-name class has no
// remaining silent path.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "superslm/sslm_abi.h"

namespace t2139_gate_c_suite_side {
// Transcribed from tests/t2130-g5-red-suite/sslm_g5.h@52dc6cd, verbatim (the complete 26-entry
// Sec6 registry mirror -- every enumerator is transcribed, even though only the names sslm_abi.h
// also declares are checked below, so this namespace stays a faithful copy of the real header
// rather than a pre-filtered stand-in).
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
	// SSLM_TOKEN_ID_UNMAPPED reconciled in at 17 (curie/t2130-g5-red-suite@59e26ff, closing the
	// coordination this file's own top comment named) -- G5's own seven additions shifted 17-23
	// -> 18-24 to make room, base ordinal stability preserved for every enumerator above.
	SSLM_TOKEN_ID_UNMAPPED = 17,
	SSLM_SCHEMA_NOT_FOUND = 18,
	SSLM_SCHEMA_BIND_REJECTED = 19,
	SSLM_SCHEMA_SPAN_UNBOUND = 20,
	SSLM_PREFIX_SCHEMA_MISMATCH = 21,
	SSLM_RESTORE_SCHEMA_MISMATCH = 22,
	SSLM_SCHEMA_SPAN_UNREACHABLE = 23,
	SSLM_SCHEMA_UNSATISFIABLE = 24,
	// Mirrored in at 25 (curie/t2130-g5-red-suite@52dc6cd, executing the T-2133 Sec6 ruling:
	// sslm_g5.h mirrors the complete registry verbatim; next-free is 26, a Sec6 fact).
	SSLM_ALLOCATION_FAILED = 25
} sslm_status;

typedef enum sslm_span_kind {
	SSLM_SPAN_PROMPT = 0,
	SSLM_SPAN_SCHEMA_CONTENT = 1
} sslm_span_kind;

typedef struct sslm_decode_params {
	int32_t layer_budget;
} sslm_decode_params;

typedef struct sslm_stats_out {
	int64_t decode_step_ceiling;
	int64_t decode_step_actual;
	int64_t forced_token_count;
	int32_t kv_blocks_resident;
} sslm_stats_out;
}  // namespace t2139_gate_c_suite_side

// --- sslm_status: per-shared-enumerator-name value equality, against the REAL library header
// (::sslm_status, from #include "superslm/sslm_abi.h" above -- no second transcription). ---
#define T2139_GATE_C_STATUS_CHECK(NAME) \
	static_assert(static_cast<int>(t2139_gate_c_suite_side::NAME) == static_cast<int>(::NAME), \
	              #NAME " diverges")
T2139_GATE_C_STATUS_CHECK(SSLM_OK);
T2139_GATE_C_STATUS_CHECK(SSLM_INVALID_ARGUMENT);
T2139_GATE_C_STATUS_CHECK(SSLM_BUFFER_TOO_SMALL);
T2139_GATE_C_STATUS_CHECK(SSLM_MISALIGNED_BUFFER);
T2139_GATE_C_STATUS_CHECK(SSLM_ARTIFACT_REJECTED);
T2139_GATE_C_STATUS_CHECK(SSLM_ADAPTER_MODEL_MISMATCH);
T2139_GATE_C_STATUS_CHECK(SSLM_RESTORE_MODEL_MISMATCH);
T2139_GATE_C_STATUS_CHECK(SSLM_RESTORE_KV_MISMATCH);
T2139_GATE_C_STATUS_CHECK(SSLM_MODEL_HAS_LIVE_SEQUENCES);
T2139_GATE_C_STATUS_CHECK(SSLM_POOL_HAS_LIVE_HANDLES);
T2139_GATE_C_STATUS_CHECK(SSLM_ADAPTER_HAS_LIVE_SEQUENCES);
T2139_GATE_C_STATUS_CHECK(SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED);
T2139_GATE_C_STATUS_CHECK(SSLM_SEQ_RESET_MIDTOKEN_REJECTED);
T2139_GATE_C_STATUS_CHECK(SSLM_PREFIX_FROZEN_REJECTED);
T2139_GATE_C_STATUS_CHECK(SSLM_KV_POOL_EXHAUSTED);
T2139_GATE_C_STATUS_CHECK(SSLM_TOKEN_ID_OUT_OF_RANGE);
T2139_GATE_C_STATUS_CHECK(SSLM_CONTEXT_CAP_EXCEEDED);
// RECONCILED (curie/t2130-g5-red-suite@59e26ff, closing the coordination this file's own top
// comment named): sslm_g5.h now carries SSLM_TOKEN_ID_UNMAPPED at 17, matching sslm_abi.h.
// Checked per-name now, same as every other base enumerator above.
T2139_GATE_C_STATUS_CHECK(SSLM_TOKEN_ID_UNMAPPED);
// Mirrored (curie/t2130-g5-red-suite@52dc6cd, the T-2133 Sec6 ruling's execution): the mirror
// now carries SSLM_ALLOCATION_FAILED at 25, so it is a shared name and is checked per-name like
// every other. This closes P2's live half -- the ordinal the library appended is now declared
// identically on both sides.
T2139_GATE_C_STATUS_CHECK(SSLM_ALLOCATION_FAILED);
#undef T2139_GATE_C_STATUS_CHECK

// --- S8's own second half: no ordinal is claimed by both sides under DIFFERENT names. A
// per-shared-NAME check (above) is blind to exactly this class -- two enumerators that happen to
// share a numeric value but not a name never appear in the same T2139_GATE_C_STATUS_CHECK call.
//
// SHAPE UPDATED at the T-2133 enum-governance ruling's landing (design Sec6, design commit
// 4f4eb23896), exactly as the previous revision's own comment instructed ("take that ruling at
// landing if it changes this assertion's own shape" -- it does). The ruling retires the
// "base low / G5 appended above, disjoint ranges" arrangement the previous assertion here
// (SSLM_STATUS_BASE_MAX strictly below the first G5-only ordinal, correctly red as P2's own
// disclosure) was built on: design Sec6 is now the single-authority complete ordinal registry,
// and sslm_g5.h mirrors it verbatim -- one interleaved registry, not two ranges. Under that
// arrangement the different-name class is closed by construction plus one identity: every
// enumerator sslm_abi.h declares is per-name checked above against the complete mirror (a
// rename fails name lookup; a move fails value equality), and the registry-top identity below
// pins the mirror's own top to the library's declared maximum, so a one-sided append -- the
// exact mechanics by which P2 arose (N3 appended 25 to the library while the mirror ended at
// 24) -- fails to compile on whichever side lagged. ---
static_assert(static_cast<int>(t2139_gate_c_suite_side::SSLM_ALLOCATION_FAILED) ==
                  static_cast<int>(::SSLM_STATUS_BASE_MAX),
              "registry-top divergence: sslm_g5.h's mirror of the Sec6 registry does not top out "
              "at sslm_abi.h's own SSLM_STATUS_BASE_MAX -- one side has appended an enumerator "
              "the other has not taken (the exact one-sided-append mechanics of P2, "
              "Claude/Poirot/2c18dab-t2139-abi-build-review.md Sec7.4); reconcile both to design "
              "Sec6, the single-authority complete ordinal registry (T-2133 ruling, design "
              "commit 4f4eb23896)");

// --- sslm_span_kind: per-shared-enumerator-name value equality ---
static_assert(static_cast<int>(t2139_gate_c_suite_side::SSLM_SPAN_PROMPT) ==
                  static_cast<int>(::SSLM_SPAN_PROMPT),
              "SSLM_SPAN_PROMPT diverges");
static_assert(static_cast<int>(t2139_gate_c_suite_side::SSLM_SPAN_SCHEMA_CONTENT) ==
                  static_cast<int>(::SSLM_SPAN_SCHEMA_CONTENT),
              "SSLM_SPAN_SCHEMA_CONTENT diverges");

// --- sslm_decode_params: sizeof/alignof/offsetof per field ---
static_assert(sizeof(t2139_gate_c_suite_side::sslm_decode_params) ==
                  sizeof(::sslm_decode_params),
              "sslm_decode_params: sizeof diverges");
static_assert(alignof(t2139_gate_c_suite_side::sslm_decode_params) ==
                  alignof(::sslm_decode_params),
              "sslm_decode_params: alignof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_decode_params, layer_budget) ==
                  offsetof(::sslm_decode_params, layer_budget),
              "sslm_decode_params::layer_budget: offsetof diverges");

// --- sslm_stats_out: sizeof/alignof/offsetof per field ---
static_assert(sizeof(t2139_gate_c_suite_side::sslm_stats_out) == sizeof(::sslm_stats_out),
              "sslm_stats_out: sizeof diverges");
static_assert(alignof(t2139_gate_c_suite_side::sslm_stats_out) == alignof(::sslm_stats_out),
              "sslm_stats_out: alignof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, decode_step_ceiling) ==
                  offsetof(::sslm_stats_out, decode_step_ceiling),
              "sslm_stats_out::decode_step_ceiling: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, decode_step_actual) ==
                  offsetof(::sslm_stats_out, decode_step_actual),
              "sslm_stats_out::decode_step_actual: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, forced_token_count) ==
                  offsetof(::sslm_stats_out, forced_token_count),
              "sslm_stats_out::forced_token_count: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, kv_blocks_resident) ==
                  offsetof(::sslm_stats_out, kv_blocks_resident),
              "sslm_stats_out::kv_blocks_resident: offsetof diverges");

int main() { return 0; }
