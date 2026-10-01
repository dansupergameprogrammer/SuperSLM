// Paged-KV plan (rev 16.1) step C1: the engine surface the C2 cells call (§3.2).
//
// Transcribed from §3.2 items 1-3. `KvPageView` is a type, so it is defined here only until the
// shipped forward_sites.h defines SUPERSLM_HAS_KV_PAGE_VIEW (the C2 builder adds the macro with the
// struct); the functions are re-declared, which C++ allows when the declarations agree, so a
// signature shipped differently stops the suite compiling. Red by link until C2.
//
// The overloads mirror the threaded flat overloads (forward_sites.h, decode-threading §3.4) with
// `(workspace, workspace_size)` replaced by `const KvPageView&`, every parameter explicit, so the
// flat overloads can become thin wrappers that build the one-page view (§3.2 item 2).

#ifndef SUPERSLM_TESTS_PKV_ENGINE_API_H
#define SUPERSLM_TESTS_PKV_ENGINE_API_H

#include "superslm/forward_sites.h"
#include "superslm/matmul.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace superslm {

#ifndef SUPERSLM_HAS_KV_PAGE_VIEW
// §3.2 item 1. Address of (l, half, h, pos, d) (§3.1):
//   pool_base + page_table[pos / page_positions] * page_bytes
//             + ((l * 2 + half) * H_kv + h) * page_positions * D + (pos % page_positions) * D + d.
struct KvPageView {
	uint8_t* pool_base;
	const uint32_t* page_table;
	uint32_t mapped_pages;
	int64_t page_positions;
	size_t page_bytes;
};
#endif

// §3.2 item 1: the view accessors. The flat accessors stay unchanged.
const int8_t* KeyRow(const KvPageView& view, uint32_t layer, size_t num_kv_heads, size_t head_dim,
                     size_t kv_head, int64_t position) noexcept;
const int8_t* ValueRow(const KvPageView& view, uint32_t layer, size_t num_kv_heads, size_t head_dim,
                       size_t kv_head, int64_t position) noexcept;
int8_t* MutableKeyRow(const KvPageView& view, uint32_t layer, size_t num_kv_heads, size_t head_dim,
                      size_t kv_head, int64_t position) noexcept;
int8_t* MutableValueRow(const KvPageView& view, uint32_t layer, size_t num_kv_heads, size_t head_dim,
                        size_t kv_head, int64_t position) noexcept;

// §3.2 item 2: the view overloads of the two layer loops.
SslmForwardStatus RunLayerLoop(SequenceLayerState& seq, const LayerWeights* layers, uint32_t num_hidden_layers,
                               uint32_t layer_budget, size_t hidden_size, size_t head_dim,
                               size_t num_key_value_heads, size_t intermediate_size, int64_t context_cap,
                               const SslmTensorManifest& rope_tables, const KvPageView& view,
                               OptionGKLandingMode option_g_k_landing_mode, std::string_view site_prefix,
                               size_t token_index, SslmTraceHookState* trace_hook_state, size_t q_width,
                               const GemmThreading& threading);
SslmForwardStatus RunLayerLoopChunkBatched(int8_t* hidden_codes_chunk, CarriedScale* hidden_scales,
                                           size_t chunk_tokens, const LayerWeights* layers,
                                           uint32_t num_hidden_layers, size_t hidden_size, size_t head_dim,
                                           size_t num_key_value_heads, size_t intermediate_size,
                                           int64_t context_cap, int64_t context_length_start,
                                           const SslmTensorManifest& rope_tables, const KvPageView& view,
                                           bool option_g_fused_k_landing, uint64_t* kv_saturation_count,
                                           std::string_view site_prefix, SslmTraceHookState* trace_hook_state,
                                           size_t q_width, uint64_t* out_kv_landing_saturation_count,
                                           uint64_t* out_k_channel_landing_saturation_count,
                                           uint64_t* out_rope_q_saturation_count,
                                           uint64_t* out_rope_k_saturation_count, const GemmThreading& threading);

// §3.2 item 3: GemmProbQ15Accumulate without the zeroing; adds into `out_ctx`. The same contract
// otherwise (probs has `width` elements, values `width * head_dim`, row-major).
void GemmProbQ15AccumulateInto(const int64_t* probs, const int8_t* values, size_t width, size_t head_dim,
                               int64_t* out_ctx);

}  // namespace superslm

#endif  // SUPERSLM_TESTS_PKV_ENGINE_API_H
