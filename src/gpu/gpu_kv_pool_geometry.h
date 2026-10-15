// Paged-KV plan (rev 16.1) §4 items 3, 6a and 7, step G3: the GPU page pool's host-only arithmetic.
//
// D3D12-free and header-only, so the Linux build can run it (tests/gpu_kv_pool_geometry_tests.cpp)
// and gpu_1p0.cpp / superslm_gpu.cpp use the same definitions:
//   - the page geometry (§3.1): B = kv_block_size when it divides the cap, else the cap; page_bytes =
//     L * 2 * H_kv * B * D at the GPU's int8 K/V element (the GPU tier check admits int8 K/V only);
//   - the size refusal (§4 item 3): page_count * page_bytes above 2^32, the reach of the shaders'
//     uint byte offsets, is SSLM_GPU_KV_POOL_TOO_LARGE;
//   - the reservation R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)) and the budget domain
//     [1, cap] (§3.4, the CPU module's formula);
//   - the zero-fill's staging size min(bytes, 64 MiB) and its CopyBufferRegion chunks, computed in
//     place so the recording window allocates nothing (§4 item 3's invariant);
//   - the coverage-guard twin over a chunk (§3.2 item 4, §4 item 6a): every row the call writes lies
//     in a mapped page.
#ifndef SUPERSLM_SRC_GPU_GPU_KV_POOL_GEOMETRY_H
#define SUPERSLM_SRC_GPU_GPU_KV_POOL_GEOMETRY_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace superslm_gpu {
namespace kv_pool {

// The shaders' byte offsets are uint (§4 item 1): a pool of more than 2^32 bytes is refused.
constexpr uint64_t kMaxPoolBytes = uint64_t{1} << 32;
// The zero-fill's staging buffer is at most this size (§4 item 3, rev 15).
constexpr uint64_t kZeroFillStagingMaxBytes = uint64_t{64} << 20;
// The GPU's K/V element width: the residency tier check admits int8 K/V only.
constexpr size_t kGpuKvElementBytes = 1;

struct GpuPageGeometry {
	bool ok = false;
	int64_t page_positions = 0;  // B
	size_t page_bytes = 0;
	uint32_t cap_pages = 0;      // ceil(cap / B), the table's entry count
};

// §3.1 on the GPU model's own geometry. Not ok for a zero dimension, a cap below 1 or above the u32
// range, or an overflowing product.
inline GpuPageGeometry ComputeGpuPageGeometry(uint32_t num_hidden_layers, uint32_t num_key_value_heads,
                                              uint32_t head_dim, int64_t context_cap,
                                              uint32_t kv_block_size) {
	GpuPageGeometry g;
	if (context_cap < 1 || context_cap > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) return g;
	if (num_hidden_layers == 0 || num_key_value_heads == 0 || head_dim == 0) return g;
	const uint64_t cap = static_cast<uint64_t>(context_cap);
	const uint64_t B = (kv_block_size >= 1 && kv_block_size <= cap && cap % kv_block_size == 0)
	                       ? static_cast<uint64_t>(kv_block_size)
	                       : cap;
	size_t v = static_cast<size_t>(num_hidden_layers);
	const size_t factors[] = {size_t{2}, static_cast<size_t>(num_key_value_heads), static_cast<size_t>(B),
	                          static_cast<size_t>(head_dim), kGpuKvElementBytes};
	for (size_t f : factors) {
		if (v > std::numeric_limits<size_t>::max() / f) return g;
		v *= f;
	}
	g.ok = true;
	g.page_positions = static_cast<int64_t>(B);
	g.page_bytes = v;
	g.cap_pages = static_cast<uint32_t>((cap + B - 1) / B);
	return g;
}

// §4 item 3: the size refusal. Computed in 128-bit-safe form: page_bytes above 2^32 refuses any
// nonzero count, and otherwise the u64 product of two values below 2^32 cannot overflow.
inline bool PoolBytesTooLarge(uint32_t page_count, size_t page_bytes) {
	if (static_cast<uint64_t>(page_bytes) > kMaxPoolBytes) return page_count != 0;
	return static_cast<uint64_t>(page_count) * static_cast<uint64_t>(page_bytes) > kMaxPoolBytes;
}

// The largest page count PoolBytesTooLarge admits (0 for a zero page size).
inline uint32_t LargestAdmissiblePageCount(size_t page_bytes) {
	if (page_bytes == 0) return 0;
	const uint64_t n = kMaxPoolBytes / static_cast<uint64_t>(page_bytes);
	return n > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
	                                                 : static_cast<uint32_t>(n);
}

// §3.4 / §4 item 7: a budget outside [1, cap] is refused.
inline bool BudgetInDomain(int32_t budget, int64_t context_cap) {
	return budget >= 1 && static_cast<int64_t>(budget) <= context_cap;
}

// §3.4: R(budget) = min(ceil(budget / B) + 1, ceil(cap / B)). 0 outside the domain.
inline uint32_t ReservationPages(int32_t budget, const GpuPageGeometry& g, int64_t context_cap) {
	if (!g.ok || !BudgetInDomain(budget, context_cap)) return 0;
	const int64_t pages = (static_cast<int64_t>(budget) + g.page_positions - 1) / g.page_positions + 1;
	return pages < static_cast<int64_t>(g.cap_pages) ? static_cast<uint32_t>(pages) : g.cap_pages;
}

// §3.4: limit = min(origin + budget, cap).
inline int64_t HolderLimit(int64_t origin, int32_t budget, int64_t context_cap) {
	const int64_t l = origin + static_cast<int64_t>(budget);
	return l < context_cap ? l : context_cap;
}

// The pages a holder must map to hold rows [0, positions): ceil(positions / B).
inline uint32_t PagesToCover(int64_t positions, int64_t page_positions) {
	if (positions <= 0 || page_positions <= 0) return 0;
	return static_cast<uint32_t>((positions + page_positions - 1) / page_positions);
}

// §4 item 3: the zero-fill's staging buffer, min(bytes, 64 MiB).
inline uint64_t ZeroFillStagingBytes(uint64_t bytes) {
	return bytes < kZeroFillStagingMaxBytes ? bytes : kZeroFillStagingMaxBytes;
}

// ceil(bytes / staging): the CopyBufferRegion count of the one submission.
inline uint64_t ZeroFillChunkCount(uint64_t bytes) {
	const uint64_t s = ZeroFillStagingBytes(bytes);
	return s == 0 ? 0 : (bytes + s - 1) / s;
}

// Chunk `i`'s destination offset and length, computed in place (no container, so the recording
// window that calls it allocates nothing and cannot throw).
inline void ZeroFillChunk(uint64_t bytes, uint64_t i, uint64_t* offset, uint64_t* len) {
	const uint64_t s = ZeroFillStagingBytes(bytes);
	*offset = i * s;
	const uint64_t rest = bytes - *offset;
	*len = rest < s ? rest : s;
}

// The coverage guard's GPU twin (§3.2 items 4-5, §4 item 6a), over the rows one call writes:
// [context_length, context_length + coverage_tokens), clamped to the cap. The single-token form
// (coverage_tokens == 1) is context_length / B < mapped_pages; the chunk form is
// context_length_start + chunk_tokens <= mapped_pages * B. The clamp keeps a legacy one-page view
// (mapped * B == cap) exactly at its pre-G3 outcome: a chunk crossing the cap is the cap guard's,
// never this one's. Callers have established 0 <= context_length < cap and page_positions >= 1.
// True means refused (KvPageUnmapped).
inline bool CoverageRefused(int64_t context_length, uint32_t coverage_tokens, int64_t context_cap,
                            int64_t page_positions, uint32_t mapped_pages) {
	const uint64_t start = static_cast<uint64_t>(context_length);
	uint64_t end = start + (coverage_tokens == 0 ? 1u : coverage_tokens);  // exclusive
	if (end > static_cast<uint64_t>(context_cap)) end = static_cast<uint64_t>(context_cap);
	if (end <= start) end = start + 1;
	return (end - 1) / static_cast<uint64_t>(page_positions) >= static_cast<uint64_t>(mapped_pages);
}

}  // namespace kv_pool
}  // namespace superslm_gpu

#endif  // SUPERSLM_SRC_GPU_GPU_KV_POOL_GEOMETRY_H
