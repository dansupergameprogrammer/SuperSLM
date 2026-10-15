// Paged-KV plan (rev 16.2) step G1: the ONE provisional 'SLM6' byte layout the GPU red suite reads and
// edits (g1b_helpers.h slm6::, g1e_helpers.h Slm6Parse). Before unification the two groups pinned it
// separately; they agreed on the fixed fields and disagreed on the K/V section (see g1-unify-notes.md).
//
// Plan §4 item 10: "SLM5 fields, plus kv_mode, budget, origin and the canonical [0, L') K/V gather of
// §3.7". §3.7's K/V section: "kv_positions (u64) = L' = context_length + (layer_index > 0 ? 1 : 0), then
// the rows of positions [0, L') in canonical layout" ([layer][K|V][head][pos < L'][d]). On the GPU (G5
// review ruling) L' = context_length + (0 < layer_index < num_hidden_layers ? 1 : 0): the GPU's rest at
// layer_index == num_hidden_layers is a completed token, as the CPU's rest at 0 is. The plan fixes
// the fields, not the offsets: these offsets are the suite's reading, and G5's builder re-anchors them
// HERE, once, if the writer differs. Every cell that saves an SLM6 self-checks the layout against a state
// it built (g1b Slm6LayoutMatches, g1e Slm6SelfCheck), so a stale pin turns cells red, never green.
//
//   [0, 120)    SLM5's header (superslm_gpu.cpp GpuSeqBlobHeader), magic 'SLM6'
//   [120, 132)  SLM5's 12-byte tail (bound_schema_index, dfa_walk_state, ready)
//   132         kv_mode  u32 (0 whole_reserve, 1 budget; the writer rule makes every SLM6 budget)
//   136         budget   i32
//   140         origin   i64
//   148         the residual (hidden_codes_size bytes, header offset 40)
//   148 + R     kv_positions u64 = L'
//   156 + R     L' rows, canonical layout
#ifndef SSLM_PKVG_SLM6_LAYOUT_H
#define SSLM_PKVG_SLM6_LAYOUT_H

#include <cstddef>
#include <cstdint>

namespace pkvg_slm6 {
constexpr size_t kSlm5HeaderBytes = 120;
constexpr size_t kSlm5TailBytes = 12;
constexpr size_t kOffKvMode = kSlm5HeaderBytes + kSlm5TailBytes;  // 132
constexpr size_t kOffBudget = kOffKvMode + 4;                     // 136
constexpr size_t kOffOrigin = kOffBudget + 4;                     // 140
constexpr size_t kFixedBytes = kOffOrigin + 8;                    // 148
constexpr size_t kKvPositionsBytes = 8;
// The K/V section's offsets, given the residual's size (the header's hidden_codes_size).
constexpr size_t OffKvPositions(uint64_t residual_bytes) { return kFixedBytes + static_cast<size_t>(residual_bytes); }
constexpr size_t OffRows(uint64_t residual_bytes) { return OffKvPositions(residual_bytes) + kKvPositionsBytes; }
static_assert(kOffKvMode == 132 && kOffBudget == 136 && kOffOrigin == 140 && kFixedBytes == 148, "SLM6 pin");
}  // namespace pkvg_slm6

#endif  // SSLM_PKVG_SLM6_LAYOUT_H
