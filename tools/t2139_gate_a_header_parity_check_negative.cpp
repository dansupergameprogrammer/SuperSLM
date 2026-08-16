// t2139_gate_a_header_parity_check_negative.cpp -- Gate A's MUST-REJECT sibling (design Sec9).
// A permanent, deliberately-corrupted single-verb mutation of
// include/superslm/sslm_abi_functions_g5_comparable.inc: sslm_model_map gains a spurious
// trailing `int corruption_marker` parameter it does not really have (the F1-class divergence
// the design's own Loki commissioning probe used, adapted to this arc's own shipped verb set).
// This file's own CI step MUST FAIL TO COMPILE -- if it ever starts compiling clean, that is
// Gate A regressing (a real signature divergence it should catch is no longer caught), and the
// build fails loudly on the regression itself per design Sec9's own standing-fixture
// requirement.
//
// This corruption is hand-maintained, not script-generated (design Sec9's own "ideally
// script-generated... so the corruption tracks the header rather than bit-rotting against it"
// is not built this round -- flagged here rather than silently claimed done): if
// sslm_abi_functions_g5_comparable.inc's own sslm_model_map declaration ever changes shape,
// this file's hand-copied, hand-corrupted declaration below needs a matching manual update to
// stay a real corruption of the current real declaration rather than a stale one.
#include <type_traits>

#include "t2130-g5-red-suite/sslm_g5.h"

// The corrupted declaration (a spurious extra parameter), standing in for
// sslm_abi_functions_g5_comparable.inc's own sslm_model_map -- NOT included from that file, so
// this corruption is real and independent of it.
extern "C" sslm_status t2139_gate_a_lib_sslm_model_map(const void* data, size_t size,
                                                        sslm_model* out, int corruption_marker);

static_assert(std::is_same_v<decltype(&sslm_model_map), decltype(&t2139_gate_a_lib_sslm_model_map)>,
              "sslm_model_map: library signature diverges from tests/t2130-g5-red-suite/sslm_g5.h");

int main() { return 0; }
