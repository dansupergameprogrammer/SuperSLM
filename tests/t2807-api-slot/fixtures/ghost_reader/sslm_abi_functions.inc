/* T-2835 (Curie) -- the C verb reader's must-reject fixture (plan Sec3.7 item 3, T-2824 G6, T-2825 F-3).
 * Not an engine header: read only by tests/t2807-api-slot/test_c_abi_reader.ps1. The reader must count
 * exactly the two declarations in this fixture -- sslm_ghost_split (this file: the return type on the line
 * before the name) and sslm_ghost_macro (the other file: behind an unknown prefix macro) -- and must not
 * count the comment mention below. */
#include "superslm/sslm_abi_functions_g5_comparable.inc"

/* A comment that writes the call form sslm_ghost_comment( is a mention, never a declaration. */
sslm_status
sslm_ghost_split(sslm_model model, int32_t n);
