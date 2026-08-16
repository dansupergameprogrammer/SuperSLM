// t2139_gate_c_type_identity_check_negative.cpp -- Gate C's MUST-REJECT sibling (design Sec9).
// A permanent, deliberately-corrupted mutation of sslm_stats_out: two fields swapped in
// declaration order, shifting every offsetof after the swap. This file's own CI step MUST FAIL
// TO COMPILE -- if it ever starts compiling clean, that is Gate C regressing, and the build
// fails loudly on the regression itself.
//
// Hand-maintained, not script-generated (same stated limitation as
// t2139_gate_a_header_parity_check_negative.cpp) -- a future change to sslm_stats_out's real
// field order needs a matching manual update here to stay a real corruption.
#include <cstddef>
#include <cstdint>

namespace t2139_gate_c_suite_side {
typedef struct sslm_stats_out {
	int64_t decode_step_ceiling;
	int64_t decode_step_actual;
	int64_t forced_token_count;
	int32_t kv_blocks_resident;
} sslm_stats_out;
}  // namespace t2139_gate_c_suite_side

namespace t2139_gate_c_library_side {
// Corrupted: decode_step_actual and forced_token_count swapped, shifting both fields' offsetof
// relative to the suite side above.
typedef struct sslm_stats_out {
	int64_t decode_step_ceiling;
	int64_t forced_token_count;
	int64_t decode_step_actual;
	int32_t kv_blocks_resident;
} sslm_stats_out;
}  // namespace t2139_gate_c_library_side

static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, decode_step_actual) ==
                  offsetof(t2139_gate_c_library_side::sslm_stats_out, decode_step_actual),
              "sslm_stats_out::decode_step_actual: offsetof diverges");
static_assert(offsetof(t2139_gate_c_suite_side::sslm_stats_out, forced_token_count) ==
                  offsetof(t2139_gate_c_library_side::sslm_stats_out, forced_token_count),
              "sslm_stats_out::forced_token_count: offsetof diverges");

int main() { return 0; }
