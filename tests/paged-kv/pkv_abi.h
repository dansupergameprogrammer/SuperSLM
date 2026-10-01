/* Paged-KV plan (rev 16.1) step C1: the red suite's declared CPU surface (§3.6), and the test seams
 * the CPU cells read (§3.3, §3.6, §7).
 *
 * Transcribed from plan §3.6's verb table: same names, argument order and types, per verb. The
 * builder ships each declaration in include/superslm/sslm_abi_functions.inc (C3-C5); this file then
 * re-declares the same functions, which C++ allows when the declarations agree, so a signature the
 * builder ships differently stops the suite compiling rather than passing silently. The two stats
 * structs and the new status are types and a value, not functions: they are defined here only
 * until the shipped header defines SSLM_HAS_PAGED_KV_ABI (the builder adds that macro with them),
 * so the suite compiles before and after. The structs' field order and types are the test
 * author's reading of §3.6's field lists; the builder keeps them or updates this file and the cells
 * together, never the assertions.
 *
 * RED BY LINK. No source in the library defines any symbol below at C1. Each per-step executable
 * (tests/paged-kv/CMakeLists.txt) links only once its step's builder has defined every symbol its
 * cells call; until then tools/check_paged_kv_red.sh confirms the link fails on exactly the symbols
 * that step owes, which is the "red for the stated reason" of C1's gate.
 *
 * Which step defines which symbol (§9, §9.3's ledger):
 *   C4: the page-index peek, the table read, the pool-mutex counter, the table sentinel and the
 *       reserve-drain seam (the legacy holders' seams, §3.6 "the mutex counter lands at C4");
 *   C5: every verb of §3.6's table and the stats structs.
 */

#ifndef SUPERSLM_TESTS_PKV_ABI_H
#define SUPERSLM_TESTS_PKV_ABI_H

#include "superslm/sslm_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* §3.6: SSLM_KV_BUDGET_EXCEEDED, appended as ordinal 29. Cells compare against this constant;
 * once the shipped enum carries the name, pkv_common.h static_asserts the two agree. */
#define PKV_KV_BUDGET_EXCEEDED ((sslm_status)29)

#ifndef SSLM_HAS_PAGED_KV_ABI
/* §3.6 sslm_kv_pool_stats: struct_size-versioned. shared_pages is the number of distinct pages at
 * least one live holder maps as shared, by §3.3's provenance definition (never refcount >= 2). */
typedef struct sslm_kv_pool_stats_out {
	uint32_t struct_size;    /* caller sets sizeof(sslm_kv_pool_stats_out) */
	uint32_t page_count;
	uint32_t free_pages;
	uint32_t shared_pages;
	int64_t page_positions;  /* B */
} sslm_kv_pool_stats_out;

/* §3.6 sslm_seq_kv_stats: struct_size-versioned; takes the lifecycle lock. mode: 0 whole_reserve,
 * 1 budget (the SSB6 kv_mode values, §3.7). */
typedef struct sslm_seq_kv_stats_out {
	uint32_t struct_size;  /* caller sets sizeof(sslm_seq_kv_stats_out) */
	uint32_t mode;
	int64_t origin;
	int32_t budget;
	int32_t reserved0;     /* must read 0 */
	int64_t limit;
	int64_t context_length;
	uint32_t mapped_private_pages;
	uint32_t shared_pages;
	uint32_t reserve_pages;
	uint32_t materialized_pages;
} sslm_seq_kv_stats_out;
#endif

/* ---- §3.6's new verbs (C5) ------------------------------------------------------------------ */
SUPERSLM_API int64_t sslm_kv_page_positions(sslm_model model);
SUPERSLM_API size_t sslm_kv_page_size(sslm_model model);
SUPERSLM_API size_t sslm_kv_pages_for_budget(sslm_model model, int32_t budget);
SUPERSLM_API size_t sslm_kv_page_pool_overhead_size(sslm_model model, uint32_t page_count);
SUPERSLM_API sslm_status sslm_kv_page_pool_create(sslm_model model, void* buf, size_t buf_size, uint32_t page_count,
                                     sslm_kv_pool* out);
SUPERSLM_API sslm_status sslm_seq_create_budgeted(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_seq* out);
SUPERSLM_API sslm_status sslm_prefix_begin_budgeted(sslm_model model, sslm_kv_pool* pool, int32_t budget, sslm_prefix* out);
SUPERSLM_API sslm_status sslm_prefix_begin_from(sslm_prefix parent, int32_t budget, sslm_prefix* out);
SUPERSLM_API sslm_status sslm_kv_pool_stats(sslm_kv_pool pool, sslm_kv_pool_stats_out* out);
SUPERSLM_API sslm_status sslm_seq_kv_stats(sslm_seq seq, sslm_seq_kv_stats_out* out);
SUPERSLM_API sslm_status sslm_seq_restore_shared(sslm_model model, sslm_kv_pool* pool, const void* blob, size_t size,
                                    sslm_prefix prefix_or_null, sslm_seq* out, uint32_t* out_shared_pages);

/* ---- test-only seams (C4) --------------------------------------------------------------------
 * The precedent is sslm_g5_test_only_peek_kv_block_bytes (src/sslm_abi.cpp): extern "C", declared
 * in no public header, bounds-checked like any verb. Each is defined in every library build (the
 * peek's own convention); the sentinel's address-path trap is compiled only into the
 * superslm_test_injection library (SUPERSLM_ENABLE_BAD_ALLOC_INJECTION builds), so a production
 * address path carries no extra branch. */

/* G19's peek ported to pages (§3.6): copies n <= page_bytes raw bytes of pool page `page_index`
 * into out_buf, bypassing every holder. SSLM_INVALID_ARGUMENT for a null pool or buffer, an index
 * >= page_count, or n > page_bytes. */
sslm_status sslm_pkv_test_only_peek_page_bytes(sslm_kv_pool pool, uint32_t page_index, uint8_t* out_buf,
                                               size_t n);

/* A holder's table entry i (i < mapped), so a cell can find the page it peeks. Reading at or past
 * `mapped` is SSLM_INVALID_ARGUMENT, never a stale read. *out_mapped receives `mapped` when non-null. */
sslm_status sslm_pkv_test_only_seq_table_entry(sslm_seq seq, uint32_t i, uint32_t* out_page,
                                               uint32_t* out_mapped);
sslm_status sslm_pkv_test_only_prefix_table_entry(sslm_prefix prefix, uint32_t i, uint32_t* out_page,
                                                  uint32_t* out_mapped);

/* §3.6 hot-path rule, cell 7.10: the number of times `pool`'s pool mutex has been acquired since the
 * pool was created. */
uint64_t sslm_pkv_test_only_pool_mutex_acquisitions(sslm_kv_pool pool);

/* Cell 7.13: with the sentinel on, every page table created afterwards is filled with 0xFFFFFFFF,
 * every entry an unmap (reset, adopt's unmap, release) releases is rewritten to it, and the address
 * path traps (std::abort, with a message naming the seam) when it reads it. Process-wide. A cell
 * that expects no trap passes by finishing; the mutants of 7.13 abort the process. */
void sslm_pkv_test_only_set_table_sentinel(int enabled);

/* Cell 11.2: moves every page of `seq`'s private reserve back to the pool, leaving its table as it is,
 * so the next map below `limit` finds an empty reserve. */
sslm_status sslm_pkv_test_only_drain_reserve(sslm_seq seq);

#ifdef __cplusplus
}
#endif

#endif /* SUPERSLM_TESTS_PKV_ABI_H */
