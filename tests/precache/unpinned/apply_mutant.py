"""The named mutants of the unpinned cells (plan revs 1.13-1.17: 4.7, 8.13, 9.8, 7.12's pool-kind leg) and of the
E3 restore hint (4.7's and 9.8's nested legs, and the e1 suite's [E3] cells).

usage: apply_mutant.py <path to src/sslm_abi.cpp> <mutant>

Applies one mutant to the engine source in place, by exact single-occurrence replacement (it refuses when the
anchor text is missing or repeated). Build superslm_precache_unpinned, run the cell, then restore the file with
`git checkout -- src/sslm_abi.cpp`. Never commit a mutated tree. build/e2-ruleb-cells.md and
build/e3-nested-cloud-2026-10-17.md carry the receipts."""
import sys
path, name = sys.argv[1], sys.argv[2]
src = open(path).read()
M = {
 # 4.7: the restore's draw computed as floor(L/B)
 "draw-floor": [(
"""	const uint32_t pages = static_cast<uint32_t>((length + B - 1) / B);""",
"""	const uint32_t pages = static_cast<uint32_t>(length / B);  // MUTANT draw-floor""")],
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
"""	const uint32_t pages = static_cast<uint32_t>((length + B - 1) / B);""",
"""	const int64_t model_B = ComputePageGeometry(model).page_positions;  // MUTANT restore-b-from-model
	const uint32_t pages = static_cast<uint32_t>((length + model_B - 1) / model_B);""")],
 # ---- E3: the restore hint (plan §3.5) ----
 # E3 baseline: the hint ignored (s = 0 always), the E2 private restore
 "hint-ignored": [(
"""	if (parent_or_null && origin > 0 &&""",
"""	if (false && parent_or_null && origin > 0 &&  // MUTANT hint-ignored""")],
 # 9.8 nested leg (named): the hint's s from the model's rule B rather than the restore pool's B
 "hint-s-from-model": [(
"""		share = static_cast<uint32_t>(origin / B);""",
"""		share = static_cast<uint32_t>(origin / ComputePageGeometry(model).page_positions);  // MUTANT hint-s-from-model""")],
 # 9.8 nested leg (named, second form): (f)'s span from the model's rule B rather than the pool's
 "hint-span-from-model": [(
"""	const int64_t B = pool->page_positions;
	const int64_t span = (origin / B) * B;""",
"""	const int64_t B = ComputePageGeometry(model).page_positions;  // MUTANT hint-span-from-model
	const int64_t span = (origin / B) * B;""")],
 # 4.7 nested leg: the draw ignores the shared pages (ceil(L/B) drawn, not ceil(L/B) - s)
 "hint-draw-whole": [(
"""	const sslm_status draw = CreateHolder(pool_ptr, pages - share, &holder);""",
"""	const sslm_status draw = CreateHolder(pool_ptr, pages, &holder);  // MUTANT hint-draw-whole""")],
 # 4.7 nested leg: the parent's partial tail page shared too (s = ceil(origin/B))
 "hint-share-ceil": [(
"""		share = static_cast<uint32_t>(origin / B);""",
"""		share = static_cast<uint32_t>((origin + B - 1) / B);  // MUTANT hint-share-ceil""")],
 # 4.7 nested leg: the scatter starts at origin, not at s*B (the private partial page's rows [s*B, origin) unwritten)
 "hint-scatter-from-origin": [(
"""	ScatterRows(pool_ptr, holder, c, rows, length, static_cast<int64_t>(share) * B, length);""",
"""	ScatterRows(pool_ptr, holder, c, rows, length, share ? origin : 0, length);  // MUTANT hint-scatter-from-origin""")],
 # 7.5: (f) compares only layer 0
 "hint-f-layer0": [(
"""	for (size_t r = 0; r < runs; ++r) {  // (f)""",
"""	for (size_t r = 0; r < 2u * c.num_key_value_heads; ++r) {  // (f) MUTANT hint-f-layer0""")],
 # 5.11/E3: a refused share returns without releasing the holder (drawn pages leak)
 "hint-share-refusal-leaks": [(
"""		if (shared_st != superslm::kv_pages::PageStatus::kOk) {
			ReleaseHolder(pool_ptr, holder);
			return MapPageStatus(shared_st);""",
"""		if (shared_st != superslm::kv_pages::PageStatus::kOk) {
			return MapPageStatus(shared_st);  // MUTANT hint-share-refusal-leaks""")],
 # 5.2/E3: the post-map allocation failure returns only the reserve (HolderReturnReserve), not the holder: the
 # private pages stay mapped and the shared references stay taken (T-3027 F13's form, on the nested restore)
 "alloc-fail-reserve-only": [(
"""		FirePrefixRestoreFault(3, pool_ptr, holder);
		ReleaseHolder(pool_ptr, holder);""",
"""		FirePrefixRestoreFault(3, pool_ptr, holder);
		superslm::kv_pages::HolderReturnReserve(holder, nullptr);  // MUTANT alloc-fail-reserve-only""")],
 # 2.6 / 5.10 (T-3060 F8): a private restore that underdraws by the s a matching hint would have shared
 "hint-underdraw-private": [(
"""	const sslm_status draw = CreateHolder(pool_ptr, pages - share, &holder);""",
"""	const sslm_status draw = CreateHolder(pool_ptr, pages - static_cast<uint32_t>(origin / B), &holder);  // MUTANT hint-underdraw-private""")],
 # 5.2/E3 (T-3060 F5): the post-map allocation failure releases nothing (the holder, its private pages and its shared
 # references all leak)
 "alloc-fail-leak-holder": [(
"""		FirePrefixRestoreFault(3, pool_ptr, holder);
		ReleaseHolder(pool_ptr, holder);""",
"""		FirePrefixRestoreFault(3, pool_ptr, holder);  // MUTANT alloc-fail-leak-holder""")],
 # 5.11/E3 (T-3060 F5): the map refusal returns without releasing the holder (the shared references leak)
 "map-refusal-leaks": [(
"""	if (mapped != SSLM_OK) {
		ReleaseHolder(pool_ptr, holder);
		return mapped;
	}
#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
	if (g_prefix_restore_fault.position == 3) {""",
"""	if (mapped != SSLM_OK) {
		return mapped;  // MUTANT map-refusal-leaks
	}
#ifdef SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS
	if (g_prefix_restore_fault.position == 3) {""")],
}
for old, new in M[name]:
    n = src.count(old)
    if n != 1:
        sys.exit(f"{name}: expected one occurrence, found {n}")
    src = src.replace(old, new)
open(path, "w").write(src)
print("applied", name)
