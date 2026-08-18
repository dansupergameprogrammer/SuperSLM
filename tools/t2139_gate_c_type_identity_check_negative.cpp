// t2139_gate_c_type_identity_check_negative.cpp -- Gate C's MUST-REJECT sibling (design Sec9).
// A permanent, deliberately-corrupted mutation of sslm_status: one enumerator's ordinal value
// changed under a shared name (SSLM_ARTIFACT_REJECTED, 4 vs 42) -- the sslm_status ordinal-
// perturbation case, now that Gate C's own sslm_status exclusion is lifted (Claude/Brunel/
// t2139-abi-build-2026-08-16.md Sec4/Sec5, resolved). This file's own CI step MUST FAIL TO
// COMPILE -- if it ever starts compiling clean, that is Gate C regressing on the exact
// divergence class its own commissioning already found once by execution, and the build fails
// loudly on the regression itself.
//
// Hand-maintained, not script-generated (same stated limitation as the other Gate A/C negative
// siblings) -- a future renumbering of either real sslm_status needs a matching manual update
// here to stay a real corruption.
#include <type_traits>

namespace t2139_gate_c_suite_side {
typedef enum sslm_status {
	SSLM_OK = 0,
	SSLM_ARTIFACT_REJECTED = 4
} sslm_status;
}  // namespace t2139_gate_c_suite_side

namespace t2139_gate_c_library_side {
// Corrupted: SSLM_ARTIFACT_REJECTED given a wildly different ordinal under the same name.
typedef enum sslm_status {
	SSLM_OK = 0,
	SSLM_ARTIFACT_REJECTED = 42
} sslm_status;
}  // namespace t2139_gate_c_library_side

static_assert(static_cast<int>(t2139_gate_c_suite_side::SSLM_ARTIFACT_REJECTED) ==
                  static_cast<int>(t2139_gate_c_library_side::SSLM_ARTIFACT_REJECTED),
              "SSLM_ARTIFACT_REJECTED diverges");

int main() { return 0; }
