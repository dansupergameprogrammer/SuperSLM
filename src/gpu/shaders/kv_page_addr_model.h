// Paged-KV plan (rev 16.1) §3.1 and §4 item 1: the C++ transliteration of the shaders' K/V
// page-table addressing -- `KvInPageRowOffsetGpu`, `KvPageRowAddrGpu` and `KvRowAddrGpu` in
// site_common.hlsli, and the page-run loop bounds of context_accumulate_site.hlsl. Kept next to
// the shaders on purpose: a change to one side is made to the other in the same commit.
//
// Every line tagged `SSLM-KV-ADDR:<name>` holds formula text that must read the same, character for
// character after whitespace is collapsed, here and in the shader file the tag names
// (tests/gpu_kv_page_addr_tests.cpp checks it, so the two cannot drift apart silently). That is why
// this file spells HLSL's `uint` and `min` (both defined below) and takes the shader's root
// constants as parameters of the same names.
//
// Header-only and free of D3D12. The host offset builders and KeyRowGpu/ValueRowGpu
// (superslm_gpu.cpp) instantiate these templates with size_t, so the host reads exactly the bytes the
// shaders write. The cell instantiates them with uint32_t, which is HLSL's `uint` with its modulo
// 2^32 arithmetic, and with a checked 32-bit type that fails on any intermediate at or above 2^32.
//
// The address of (layer, half, kv_head, pos) through a page table (§3.1):
//   table[pos / B] * page_bytes + ((layer * 2 + half) * H_kv + kv_head) * B * D + (pos % B) * D
// with half 0 for K and 1 for V. For the one-page view (table {0}, B = context_cap) the first term is
// 0 and the rest is the flat layout term for term (§3.1's one-page identity), which is what makes a
// legacy dedicated sequence's output byte-identical through the table (§4 item 2).
#ifndef SSLM_GPU_KV_PAGE_ADDR_MODEL_H
#define SSLM_GPU_KV_PAGE_ADDR_MODEL_H

#include <cstdint>

namespace superslm_gpu {
namespace kv_page_addr_model {

using uint = uint32_t;

template <typename U>
constexpr U min(U a, U b) {
	return b < a ? b : a;
}

// site_common.hlsli KvInPageRowOffsetGpu: the byte offset of (layer, half, kv_head, pos_in_page)
// inside one page.
template <typename U>
constexpr U KvInPageRowOffsetGpu(U layer, U half, U num_kv_heads, U head_dim, U kv_head, U page_positions,
                                 U pos_in_page) {
	return ((layer * 2u + half) * num_kv_heads + kv_head) * page_positions * head_dim + pos_in_page * head_dim;  // SSLM-KV-ADDR:in-page
}

// site_common.hlsli KvPageRowAddrGpu: the row's byte address, given the physical page its logical
// page maps to.
template <typename U>
constexpr U KvPageRowAddrGpu(U phys_page, U page_bytes, U page_positions, U layer, U half, U num_kv_heads,
                             U head_dim, U kv_head, U pos) {
	return phys_page * page_bytes + KvInPageRowOffsetGpu(layer, half, num_kv_heads, head_dim, kv_head, page_positions, pos % page_positions);  // SSLM-KV-ADDR:row
}

// site_common.hlsli KvRowAddrGpu: one table read, `KvPageTable.Load((pos / g_page_positions) * 4u)`
// there, `page_table[pos / page_positions]` here.
template <typename U>
constexpr U KvRowAddrGpu(const uint32_t* page_table, U page_bytes, U page_positions, U layer, U half,
                         U num_kv_heads, U head_dim, U kv_head, U pos) {
	return KvPageRowAddrGpu(U(page_table[static_cast<uint64_t>(pos / page_positions)]), page_bytes,
	                        page_positions, layer, half, num_kv_heads, head_dim, kv_head, pos);
}

// context_accumulate_site.hlsl's page-run loop: rows [p0, p0 + rows) of one page per iteration, in
// ascending order, reading the table once per run. `f(p0, rows)` stands for the run's body.
template <typename U, typename F>
constexpr void ForEachKvPageRunGpu(U width_u, U page_positions, F&& f) {
	for (U p0 = 0u; p0 < width_u; p0 += page_positions)  // SSLM-KV-ADDR:page-run-loop
	{
		U rows = min(page_positions, width_u - p0);  // SSLM-KV-ADDR:page-run-rows
		f(p0, rows);
	}
}

}  // namespace kv_page_addr_model
}  // namespace superslm_gpu

#endif  // SSLM_GPU_KV_PAGE_ADDR_MODEL_H
