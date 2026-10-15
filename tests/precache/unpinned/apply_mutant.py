"""The named mutants of the unpinned E2 cells (plan revs 1.13-1.17: 4.7, 8.13, 9.8, 7.12's pool-kind leg).

usage: apply_mutant.py <path to src/sslm_abi.cpp> <mutant>

Applies one mutant to the engine source in place, by exact single-occurrence replacement (it refuses when the
anchor text is missing or repeated). Build superslm_precache_unpinned, run the cell, then restore the file with
`git checkout -- src/sslm_abi.cpp`. Never commit a mutated tree. build/e2-ruleb-cells.md carries the receipts."""
import sys
path, name = sys.argv[1], sys.argv[2]
src = open(path).read()
M = {
 # 4.7: the restore's draw computed as floor(L/B)
 "draw-floor": [(
"""	const uint32_t pages = static_cast<uint32_t>((length + pool_ptr->page_positions - 1) /
	                                              pool_ptr->page_positions);""",
"""	const uint32_t pages = static_cast<uint32_t>(length / pool_ptr->page_positions);  // MUTANT draw-floor""")],
 # 4.7: PKV's MapPagesFor table bound computed as cap / B, truncating (premise mutant)
 "table-cap-div-b": [(
"""	if (needed > holder->table_entries) return SSLM_INVALID_ARGUMENT;""",
"""	if (needed > static_cast<uint64_t>(pool->model->view.config.context_cap) / static_cast<uint64_t>(B))
		return SSLM_INVALID_ARGUMENT;  // MUTANT table-cap-div-b""")],
 # 4.7 (variant): the same truncating bound applied by the restore alone, before its map
 "table-cap-div-b-restore": [(
"""	const sslm_status mapped = MapPagesFor(pool_ptr, holder, length);
	if (mapped != SSLM_OK) {
		ReleaseHolder(pool_ptr, holder);
		return mapped;
	}""",
"""	if (static_cast<uint64_t>((length + pool_ptr->page_positions - 1) / pool_ptr->page_positions) >
	    static_cast<uint64_t>(c.context_cap) / static_cast<uint64_t>(pool_ptr->page_positions)) {
		ReleaseHolder(pool_ptr, holder);
		return SSLM_INVALID_ARGUMENT;  // MUTANT table-cap-div-b-restore
	}
	const sslm_status mapped = MapPagesFor(pool_ptr, holder, length);
	if (mapped != SSLM_OK) {
		ReleaseHolder(pool_ptr, holder);
		return mapped;
	}""")],
 # 4.7: a scatter that ignores the run-slot rotation (slot r of every logical page)
 "scatter-unrotated": [(
"""			std::memcpy(HolderRow(pool, holder, r, p0, D), run_src + static_cast<size_t>(p0) * D,
			            static_cast<size_t>(n) * D);""",
"""			uint8_t* unrotated = superslm::kv_pages::PageBase(pool->pages,
			    superslm::kv_pages::HolderTableEntryForAddress(holder, static_cast<uint32_t>(p0 / B))) +
			    r * static_cast<size_t>(B) * D;  // MUTANT scatter-unrotated
			std::memcpy(unrotated, run_src + static_cast<size_t>(p0) * D, static_cast<size_t>(n) * D);""")],
 # 7.12 pool-kind leg (and 4.7): a writer that serializes rows in the holder's page order
 "writer-page-order": [(
"""	for (size_t run = 0; run < KvRuns(c); ++run) {
		uint8_t* out = dst + run * static_cast<size_t>(length) * D;
		for (int64_t position = 0; position < length; position += B) {
			const size_t bytes = static_cast<size_t>(std::min<int64_t>(B, length - position)) * D;
			std::memcpy(out + static_cast<size_t>(position) * D,
			            HolderRow(prefix->pool, prefix->kv, run, position, D), bytes);
		}
	}""",
"""	size_t at = 0;  // MUTANT writer-page-order: page by page, each page's runs in turn
	for (int64_t position = 0; position < length; position += B) {
		const size_t bytes = static_cast<size_t>(std::min<int64_t>(B, length - position)) * D;
		for (size_t run = 0; run < KvRuns(c); ++run) {
			std::memcpy(dst + at, HolderRow(prefix->pool, prefix->kv, run, position, D), bytes);
			at += bytes;
		}
	}""")],
 # 8.13: PKV's page-size check in adopt removed (A22)
 "adopt-no-page-size-check": [(
"""	if (seq->pool->page_positions != prefix->pool->page_positions) return SSLM_INVALID_ARGUMENT;""",
"""	// MUTANT adopt-no-page-size-check: A22 removed""")],
 # 9.8: the restore takes B from the model's geometry rather than from the pool
 "restore-b-from-model": [(
"""	const uint32_t pages = static_cast<uint32_t>((length + pool_ptr->page_positions - 1) /
	                                              pool_ptr->page_positions);""",
"""	const int64_t model_B = ComputePageGeometry(model).page_positions;  // MUTANT restore-b-from-model
	const uint32_t pages = static_cast<uint32_t>((length + model_B - 1) / model_B);""")],
}
for old, new in M[name]:
    n = src.count(old)
    if n != 1:
        sys.exit(f"{name}: expected one occurrence, found {n}")
    src = src.replace(old, new)
open(path, "w").write(src)
print("applied", name)
