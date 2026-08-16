// t2139_gate_c_type_identity_check.cpp -- Gate C (design Sec9): type identity, closing the
// blind spot Gate A cannot see (Gate A reuses one side's types to avoid a redefinition error,
// so it can never itself detect a struct/enum BODY divergence under a shared name). MUST-ACCEPT
// construction: never linked, compile-only.
//
// Construction (design Sec9, literal): transcribe each shared type's body TWICE, once per side,
// each inside its own namespace -- never through #include, which would pull in the colliding
// function declarations Gate A already had to route around. For an enum, static_assert per
// SHARED enumerator name (never a whole-body comparison); for a struct meant to be
// byte-identical, static_assert on sizeof/alignof/offsetof per field.
//
// sslm_status -- RECONCILED, exclusion LIFTED (Claude/Brunel/t2139-abi-build-2026-08-16.md
// Sec4/Sec5's own executed finding, resolved). This design's own 17-enumerator taxonomy
// (Sec6) and tests/t2130-g5-red-suite/sslm_g5.h's own enum previously carried different
// ordinal values for three shared names -- found by executing this exact check and watching it
// fail (StandardsDocument Sec5.4). The G5 suite (curie/t2130-g5-red-suite@a7655dd) has since
// reconciled: sslm_g5.h's own enum now carries this design's 0-16 base verbatim (renumbered),
// with G5's own seven additions appended at 17-23. Every one of this design's own 17
// enumerators is checked below, per-shared-name, at both sides' real, current bodies.
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

namespace t2139_gate_c_library_side {
// Transcribed from include/superslm/sslm_abi.h (this design's own Sec8), verbatim.
typedef enum sslm_status {
	SSLM_OK = 0,
	SSLM_INVALID_ARGUMENT,
	SSLM_BUFFER_TOO_SMALL,
	SSLM_MISALIGNED_BUFFER,
	SSLM_ARTIFACT_REJECTED,
	SSLM_ADAPTER_MODEL_MISMATCH,
	SSLM_RESTORE_MODEL_MISMATCH,
	SSLM_RESTORE_KV_MISMATCH,
	SSLM_MODEL_HAS_LIVE_SEQUENCES,
	SSLM_POOL_HAS_LIVE_HANDLES,
	SSLM_ADAPTER_HAS_LIVE_SEQUENCES,
	SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED,
	SSLM_SEQ_RESET_MIDTOKEN_REJECTED,
	SSLM_PREFIX_FROZEN_REJECTED,
	SSLM_KV_POOL_EXHAUSTED,
	SSLM_TOKEN_ID_OUT_OF_RANGE,
	SSLM_CONTEXT_CAP_EXCEEDED
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
}  // namespace t2139_gate_c_library_side

// --- sslm_status: per-shared-enumerator-name value equality (exclusion lifted) ---
#define T2139_GATE_C_STATUS_CHECK(NAME)                                                        \
	static_assert(static_cast<int>(t2139_gate_c_suite_side::NAME) ==                             \
	                  static_cast<int>(t2139_gate_c_library_side::NAME),                          \
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
#undef T2139_GATE_C_STATUS_CHECK

// --- sslm_span_kind: per-shared-enumerator-name value equality ---
static_assert(static_cast<int>(t2139_gate_c_suite_side::SSLM_SPAN_PROMPT) ==
                  static_cast<int>(t2139_gate_c_library_side::SSLM_SPAN_PROMPT),
              "SSLM_SPAN_PROMPT diverges");
static_assert(static_cast<int>(t2139_gate_c_suite_side::SSLM_SPAN_SCHEMA_CONTENT) ==
                  static_cast<int>(t2139_gate_c_library_side::SSLM_SPAN_SCHEMA_CONTENT),
              "SSLM_SPAN_SCHEMA_CONTENT diverges");

// --- sslm_decode_params: sizeof/alignof/offsetof per field ---
static_assert(sizeof(t2139_gate_c_suite_side::sslm_decode_params) ==
                  sizeof(t2139_gate_c_library_side::sslm_decode_params),
              "sslm_decode_params: sizeof diverges");
static_assert(alignof(t2139_gate_c_suite_side::sslm_decode_params) ==
                  alignof(t2139_gate_c_library_side::sslm_decode_params),
              "sslm_decode_params: alignof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_decode_params, layer_budget) ==
                  offsetof(t2139_gate_c_library_side::sslm_decode_params, layer_budget),
              "sslm_decode_params::layer_budget: offsetof diverges");

// --- sslm_stats_out: sizeof/alignof/offsetof per field ---
static_assert(sizeof(t2139_gate_c_suite_side::sslm_stats_out) ==
                  sizeof(t2139_gate_c_library_side::sslm_stats_out),
              "sslm_stats_out: sizeof diverges");
static_assert(alignof(t2139_gate_c_suite_side::sslm_stats_out) ==
                  alignof(t2139_gate_c_library_side::sslm_stats_out),
              "sslm_stats_out: alignof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, decode_step_ceiling) ==
                  offsetof(t2139_gate_c_library_side::sslm_stats_out, decode_step_ceiling),
              "sslm_stats_out::decode_step_ceiling: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, decode_step_actual) ==
                  offsetof(t2139_gate_c_library_side::sslm_stats_out, decode_step_actual),
              "sslm_stats_out::decode_step_actual: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, forced_token_count) ==
                  offsetof(t2139_gate_c_library_side::sslm_stats_out, forced_token_count),
              "sslm_stats_out::forced_token_count: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, kv_blocks_resident) ==
                  offsetof(t2139_gate_c_library_side::sslm_stats_out, kv_blocks_resident),
              "sslm_stats_out::kv_blocks_resident: offsetof diverges");

int main() { return 0; }
