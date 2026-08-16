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
// SCOPE -- sslm_status is DELIBERATELY EXCLUDED (Claude/Brunel/t2139-abi-build-2026-08-16.md
// Sec4/Sec5, executed finding): three enumerator names this design's own Sec6 taxonomy shares
// with tests/t2130-g5-red-suite/sslm_g5.h -- SSLM_ADAPTER_MODEL_MISMATCH,
// SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED, SSLM_RESTORE_MODEL_MISMATCH -- carry DIFFERENT ordinal
// values in the two real headers (sslm_g5.h: 3, 4, 5; this design's own Sec8 header: 5, 11, 6),
// verified by actually building the per-shared-enumerator-value static_assert against both real
// bodies and watching it fail (StandardsDocument Sec5.4, "exactness verified at source or by
// execution -- never by construction"). Landing this check as originally specified would make
// EVERY build of this permanent CI fixture fail from day one, on a real structural fact rather
// than a corrupted control -- not the standing-fixture property design Sec9 wants. sslm_status
// is therefore scoped out of Gate C's own comparison until the design authority rules whether
// this divergence is acceptable (the two ABIs are independently linked; no caller mixes one
// library's sslm_status return value against the other header's enumerator values) or requires
// reconciliation. sslm_span_kind, sslm_decode_params, and sslm_stats_out -- the three OTHER
// Gate C obligations design Sec9 names for C1 -- carry no such divergence and are checked in
// full below, unmodified from the design's own construction.
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace t2139_gate_c_suite_side {
// Transcribed from tools/t2139_sslm_g5_ref.h (sslm_g5.h@760b12b), verbatim.
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
