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
// a per-shared-NAME check structurally cannot see -- is closed by one more assertion below: the
// library's own highest base ordinal must be STRICTLY BELOW the suite's first G5-only ordinal.
// That assertion is expected to FAIL to compile right now: sslm_abi.h's own SSLM_TOKEN_ID_UNMAPPED
// (added this same fix round, ordinal 17) and tests/t2130-g5-red-suite/sslm_g5.h's own
// SSLM_SCHEMA_NOT_FOUND (G5's first schema-scope enumerator, ALSO ordinal 17) now collide -- the
// exact drift S8 found already in the tree. Reconciling sslm_g5.h itself is suite ownership
// (design Sec6's own coordination note, sslm_abi.h's header comment, StandardsDocument Sec6.4) --
// not performed here. This gate is therefore CORRECTLY red until that reconciliation lands; build
// tooling driving this file records that as a known, disclosed, cross-suite-coordination block,
// never silences the assertion to force a green.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "superslm/sslm_abi.h"

namespace t2139_gate_c_suite_side {
// Transcribed from tests/t2130-g5-red-suite/sslm_g5.h@a7655dd, verbatim (this design's own
// 0-16 base plus G5's own 17-23 appended -- every enumerator is transcribed, even though only
// the 0-16 shared subset is checked below, so this namespace stays a faithful copy of the real
// header rather than a pre-filtered stand-in).
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
	SSLM_SCHEMA_NOT_FOUND = 17,
	SSLM_SCHEMA_BIND_REJECTED = 18,
	SSLM_SCHEMA_SPAN_UNBOUND = 19,
	SSLM_PREFIX_SCHEMA_MISMATCH = 20,
	SSLM_RESTORE_SCHEMA_MISMATCH = 21,
	SSLM_SCHEMA_SPAN_UNREACHABLE = 22,
	SSLM_SCHEMA_UNSATISFIABLE = 23
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
// SSLM_TOKEN_ID_UNMAPPED (library ordinal 17, this fix round) is deliberately NOT checked here
// per-name -- sslm_g5.h has no enumerator of that name yet (design's own coordination note,
// sslm_abi.h's header comment). The ordinal-disjointness assertion below is what stands in for
// it until the suite's own reconciliation lands.
#undef T2139_GATE_C_STATUS_CHECK

// --- S8's own second half: no ordinal is claimed by both sides under DIFFERENT names. A
// per-shared-NAME check (above) is blind to exactly this class -- two enumerators that happen to
// share a numeric value but not a name never appear in the same T2139_GATE_C_STATUS_CHECK call.
// Stated as "the library's highest base ordinal is strictly below the suite's first G5-only
// ordinal" (design Sec6's own reconciled-shape statement: this design's base occupies low
// ordinals, G5's own additions are appended above all of them, never interleaved). EXPECTED TO
// FAIL TO COMPILE right now (see this file's own top comment): sslm_abi.h's SSLM_TOKEN_ID_UNMAPPED
// = 17 and sslm_g5.h's own first G5 enumerator, SSLM_SCHEMA_NOT_FOUND, are ALSO = 17 today. That
// is not a defect in this assertion -- it is this assertion doing its job. ---
static_assert(static_cast<int>(::SSLM_TOKEN_ID_UNMAPPED) <
                  static_cast<int>(t2139_gate_c_suite_side::SSLM_SCHEMA_NOT_FOUND),
              "ordinal collision: sslm_abi.h's highest base enumerator is not strictly below "
              "sslm_g5.h's first G5-only enumerator -- sslm_g5.h needs SSLM_TOKEN_ID_UNMAPPED "
              "reconciled in at ordinal 17 with its own G5 additions shifted to 18-24 (design "
              "Sec6's own coordination note; suite ownership, not performed by this gate)");

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
