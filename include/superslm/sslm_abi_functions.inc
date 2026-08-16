/* sslm_abi_functions.inc -- the sole source of every sslm_* function DECLARATION this ABI
 * ships (design Sec8). include/superslm/sslm_abi.h #includes this file, inside its own
 * extern "C" block, after defining every type the declarations reference (sslm_model,
 * sslm_status, sslm_config, sslm_detok_state, etc.).
 *
 * Split into two pieces (Claude/Brunel/t2139-abi-build-2026-08-16.md, the Gate C sslm_status
 * finding's own sibling): sslm_abi_functions_g5_comparable.inc holds every verb whose signature
 * uses ONLY types tests/t2130-g5-red-suite/sslm_g5.h itself defines -- included here AND,
 * separately, from tools/t2139_gate_a_header_parity_check.cpp (macro-renamed) so Gate A checks
 * the declarations this library actually ships, not a hand-maintained copy of them. The two
 * workspace-sizing declarations below take the caller-memory call-shape struct, which has no
 * counterpart in sslm_g5.h at all (design Sec9 residual d) -- they stay directly in this file,
 * outside the g5-comparable subset, and are correspondingly outside Gate A's own comparison
 * surface until/unless that struct gains a second, independent definition to compare against.
 * The text-I/O declaration below takes the incremental-detokenize state struct, the same
 * residual's other type, for the identical reason.
 */
#include "superslm/sslm_abi_functions_g5_comparable.inc"

/* Takes the caller-memory call-shape struct (design Sec7.1) -- not part of the g5-comparable
 * subset (see above). */
size_t sslm_workspace_size(sslm_model model, const sslm_config* config);
sslm_status sslm_workspace_create(sslm_model model, const sslm_config* config,
                                   void* buf, size_t buf_size, sslm_workspace* out);

/* Takes the incremental-detokenize state struct (design Sec7.4) -- not part of the
 * g5-comparable subset (see above). */
sslm_status sslm_detokenize_stream(sslm_model model, sslm_detok_state* state, const int32_t* tokens,
                                    int32_t n, char* utf8, int32_t* out_n);
