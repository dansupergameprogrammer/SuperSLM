/* X2 red population: the shipped direct declarations plus planned CPU query. */
#include "superslm/api.h"
#include "superslm/sslm_abi_functions_g5_comparable.inc"
SUPERSLM_API sslm_status sslm_decode_step_v2(sslm_model model, sslm_seq* seqs, int32_t n,
 const sslm_decode_params* params, sslm_workspace ws, int32_t* out_tokens);
SUPERSLM_API sslm_status sslm_decode_params_init(sslm_model model, int32_t mode,
 int32_t layer_budget, sslm_decode_params* out);
SUPERSLM_API size_t sslm_workspace_size(sslm_model model, const sslm_config* config);
SUPERSLM_API sslm_status sslm_workspace_create(sslm_model model, const sslm_config* config,
 void* buf, size_t buf_size, sslm_workspace* out);
SUPERSLM_API sslm_status sslm_detokenize_stream(sslm_model model, sslm_detok_state* state,
 const int32_t* tokens, int32_t n, char* utf8, int32_t* out_n);
SUPERSLM_API sslm_status sslm_seq_schema_bound(sslm_seq seq, int32_t* out_schema_bound);
