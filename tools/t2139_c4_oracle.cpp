// t2139_c4_oracle.cpp -- C4's own gate (Claude/Vitruvius/t2133-layer1-c-abi-design-2026-08-16.md
// Sec9): "single-sequence greedy decode through this ABI reproduces RunGreedyDecodeLoop's own
// direct-call output bit-for-bit." Drives the SAME real prompt through two independent paths
// against the SAME real artifact:
//
//   (A) this ABI: sslm_model_map -> sslm_kv_pool_create(1) -> sslm_seq_create ->
//       sslm_prefill(whole prompt) -> sslm_decode_step (repeated, one token per call, full
//       layer_budget) -> N produced tokens, plus the K/V pages the sequence's own holder maps.
//   (B) the direct engine: tests/t2138-abi-red-suite/fixture_common.h's own CpuOracleModel +
//       RunGreedyOracle (a thin, already-reviewed wrapper around superslm::RunGreedyDecodeLoop),
//       reused verbatim rather than re-derived (StandardsDocument Sec6.6) -- the SAME construction
//       T-2112's own sibling GPU-ABI gate and this suite's own dim6/dim10 cells already use for
//       the identical "this ABI's output bit-equals the direct-engine call" oracle.
//
// Compares: the produced token sequence (must be identical, element for element) and the
// sequence's own K/V bytes against the oracle's own flat workspace.
//
// The paged read. The default pool is paged and its pages rotate (paged-KV plan 3.1), so the
// bytes are read page by page: the holder's table entry j names the physical page
// (sslm_pkv_test_only_seq_table_entry), the page's bytes come from
// sslm_pkv_test_only_peek_page_bytes, and the row of (layer, half, kv_head, pos) on logical page
// j = pos / B is at ((r + j) mod runs) * B * D + (pos mod B) * D in that page, with
// r = (layer * 2 + half) * H_kv + kv_head and runs = L * 2 * H_kv. The oracle's flat row is at
// r * cap * D + pos * D. This tool computes both offsets itself and never calls
// KvPageInPageOffset, KvPageRunSlot or the engine's flat accessors, so a slot defect in the
// engine's formula cannot cancel against the reference. B and the page size are the ABI's own
// (sslm_kv_page_positions, sslm_kv_page_size); the tool refuses unless the page size is
// L * 2 * H_kv * B * D. Every position of every mapped page is compared, and the oracle's
// workspace must be zero from the first unmapped position on, so the ABI needed no page it did
// not map.
//
// The zero-tail dependency. Positions of the last mapped page that the run did not write are
// compared as zero bytes against the oracle's zero workspace. That comparison holds only because
// the pool buffer is this tool's own zero-initialised allocation: pages are poisoned only on
// release and never filled at draw (src/sslm_abi.cpp:4763-4765), so an unwritten position of a
// drawn page keeps the allocation's zero bytes. A pool that filled pages at draw would fail this
// comparison for a reason that is not a rotation defect.
//
// The run length. The default run writes at least 2B + 1 positions where the cap allows, so
// logical pages 1 and 2 are read (page 0 alone cannot show a rotation error: every run's slot is
// its own index there). The tool reads B before the oracle runs, sets the default length from it,
// and exits non-zero when it reached fewer than min(3, ceil(cap / B)) pages, whatever length it
// was given. It prints B, the run count and the pages reached. A failure names the first
// divergent row: page, position, run, layer, half and kv head.
//
// Usage: t2139_c4_oracle.exe <path-to-real.sslm> [max_new_tokens] [layer_budget]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "sslm_abi.h"
#include "fixture_common.h"

// Declared locally -- test-only, not part of any public header (as
// tools/t2132_s4_leak_guard_mutation_pin.cpp declares them). The signatures match src/sslm_abi.cpp.
// The paged-geometry verbs (include/superslm/sslm_abi_functions.inc) are absent from the T-2138 suite
// header this tool includes, so they are declared here with the same signatures.
extern "C" int64_t sslm_kv_page_positions(sslm_model model);
extern "C" size_t sslm_kv_page_size(sslm_model model);
extern "C" sslm_status sslm_pkv_test_only_peek_page_bytes(sslm_kv_pool pool, uint32_t page_index,
                                                          uint8_t* out_buf, size_t n);
extern "C" sslm_status sslm_pkv_test_only_seq_table_entry(sslm_seq seq, uint32_t i, uint32_t* out_page,
                                                         uint32_t* out_mapped);

namespace {
bool ReadFile(const char* path, std::vector<uint8_t>* out) {
	std::ifstream f(path, std::ios::binary);
	if (!f) return false;
	f.seekg(0, std::ios::end);
	const std::streamoff size = f.tellg();
	if (size < 0) return false;
	f.seekg(0, std::ios::beg);
	out->resize(static_cast<size_t>(size));
	if (size > 0) f.read(reinterpret_cast<char*>(out->data()), size);
	return static_cast<bool>(f) || f.eof();
}

// The geometry the comparison is a function of: the oracle's own config and the ABI's own B.
struct PageGeometry {
	size_t layers = 0;
	size_t kv_heads = 0;
	size_t head_dim = 0;
	size_t cap = 0;
	size_t page_positions = 0;  // B
	size_t runs = 0;            // L * 2 * H_kv
	size_t page_bytes = 0;
};

struct RowId {
	size_t page = 0;
	size_t pos = 0;
	size_t run = 0;
};

// The first divergent row between the ABI's pages (`pages` holds `mapped_pages` pages in logical
// order, page_bytes each) and the oracle's flat workspace. Pages ascend, then positions, then
// runs, so the first row found is the first in that order. Returns true when a row diverges.
// Both offsets are stated here, one expression each, and no engine offset function is called.
bool FirstDivergentRow(const uint8_t* pages, size_t mapped_pages, const uint8_t* workspace,
                       const PageGeometry& g, RowId* row, size_t* rows_compared) {
	const size_t B = g.page_positions;
	const size_t D = g.head_dim;
	const size_t cap = g.cap;
	const size_t runs = g.runs;
	size_t compared = 0;
	for (size_t j = 0; j < mapped_pages; ++j) {
		const uint8_t* page = pages + j * g.page_bytes;
		const size_t first = j * B;
		const size_t last = first + B < cap ? first + B : cap;
		for (size_t pos = first; pos < last; ++pos) {
			for (size_t r = 0; r < runs; ++r) {
				const size_t slot = (r + j) % runs;
				const size_t abi_off = slot * B * D + (pos % B) * D;
				const size_t flat_off = r * cap * D + pos * D;
				if (std::memcmp(page + abi_off, workspace + flat_off, D) != 0) {
					row->page = j;
					row->pos = pos;
					row->run = r;
					if (rows_compared) *rows_compared = compared;
					return true;
				}
				++compared;
			}
		}
	}
	if (rows_compared) *rows_compared = compared;
	return false;
}

// The first non-zero byte of the oracle's workspace from the first unmapped position on. Returns
// true when one exists; `row` names it (page is the first unmapped logical page).
bool FirstNonzeroBeyond(const uint8_t* workspace, size_t mapped_pages, const PageGeometry& g,
                        RowId* row) {
	const size_t D = g.head_dim;
	for (size_t pos = mapped_pages * g.page_positions; pos < g.cap; ++pos) {
		for (size_t r = 0; r < g.runs; ++r) {
			const uint8_t* bytes = workspace + r * g.cap * D + pos * D;
			for (size_t k = 0; k < D; ++k) {
				if (bytes[k] != 0) {
					row->page = mapped_pages;
					row->pos = pos;
					row->run = r;
					return true;
				}
			}
		}
	}
	return false;
}

#ifdef T2139_C4_CAPTURE
// A harness that includes this source with main renamed defines T2139_C4_CAPTURE and reads the
// last comparison's inputs here: the mapped pages in logical order, the oracle workspace, the
// geometry and the mapped count. The constructions (plan 38.4, MR-b and MR-h) flip bytes of a copy
// of the workspace and call the two comparators above. Production builds do not define it.
struct C4Capture {
	std::vector<uint8_t> pages;
	std::vector<uint8_t> workspace;
	PageGeometry geometry;
	size_t mapped_pages = 0;
};
C4Capture g_c4_capture;
#endif
}  // namespace

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <path-to-real.sslm> [max_new_tokens] [layer_budget]\n",
		             argv[0]);
		return 1;
	}
	// layer_budget < num_hidden_layers exercises the mid-token PENDING (-1) resumption path --
	// sslm_decode_step called multiple times per token, each advancing layer_index by at most
	// layer_budget, until a call completes the token. Same oracle comparison either way: the
	// FINAL produced token sequence must still be bit-identical regardless of how many
	// bounded calls it took to get there (StandardsDocument Sec5.4 -- executed, not reasoned).
	const int32_t forced_layer_budget = argc >= 4 ? std::atoi(argv[3]) : 0;
	const bool tokens_given = argc >= 3;
	size_t max_new_tokens = tokens_given ? static_cast<size_t>(std::atoi(argv[2])) : 8;

	std::vector<uint8_t> bytes;
	if (!ReadFile(argv[1], &bytes)) {
		std::fprintf(stderr, "FAIL: could not read %s\n", argv[1]);
		return 1;
	}

	// --- (B) the direct-engine oracle, built once from the SAME bytes ---
	superslm::SslmModelView oracle_view;
	std::string oracle_err;
	if (superslm::SslmModel::Load(bytes.data(), bytes.size(), oracle_view, &oracle_err) !=
	    superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "FAIL: oracle SslmModel::Load: %s\n", oracle_err.c_str());
		return 1;
	}
	CpuOracleModel oracle;
	if (!LoadCpuOracleModel(oracle_view, &oracle, &oracle_err)) {
		std::fprintf(stderr, "FAIL: LoadCpuOracleModel: %s\n", oracle_err.c_str());
		return 1;
	}
	if (oracle.kv_precision != superslm::SslmKvPrecision::Int8) {
		std::printf("SKIP: artifact declares a non-Int8 kv_precision -- RunGreedyDecodeLoop "
		            "itself rejects Int16 (KvPrecisionUnsupported); no decodable oracle exists "
		            "to compare against.\n");
		return 0;
	}

	const int32_t prompt_tokens[] = {1, 2, 3, 4, 5};
	const size_t prompt_len = 5;

	// --- (A) this ABI, over the SAME artifact bytes and the SAME prompt ---
	// The model is mapped first so that B is known before the oracle runs and the default run
	// length can be set from it.
	sslm_model model = nullptr;
	sslm_status st = sslm_model_map(bytes.data(), bytes.size(), &model);
	if (st != SSLM_OK || !model) {
		std::fprintf(stderr, "FAIL: sslm_model_map returned %d\n", static_cast<int>(st));
		return 1;
	}

	PageGeometry geometry;
	geometry.layers = static_cast<size_t>(oracle.num_hidden_layers);
	geometry.kv_heads = static_cast<size_t>(oracle.num_kv_heads);
	geometry.head_dim = static_cast<size_t>(oracle.head_dim);
	geometry.cap = static_cast<size_t>(oracle.context_cap);
	geometry.runs = geometry.layers * 2 * geometry.kv_heads;
	const int64_t page_positions = sslm_kv_page_positions(model);
	const size_t abi_page_bytes = sslm_kv_page_size(model);
	if (page_positions < 1) {
		std::fprintf(stderr, "FAIL: page geometry: B=%lld page_size=%zu, expected L*2*H_kv*B*D=%zu\n",
		             static_cast<long long>(page_positions), abi_page_bytes, static_cast<size_t>(0));
		return 1;
	}
	geometry.page_positions = static_cast<size_t>(page_positions);
	geometry.page_bytes = abi_page_bytes;
	const size_t expected_page_bytes =
	    geometry.layers * 2 * geometry.kv_heads * geometry.page_positions * geometry.head_dim;
	if (abi_page_bytes != expected_page_bytes) {
		std::fprintf(stderr, "FAIL: page geometry: B=%lld page_size=%zu, expected L*2*H_kv*B*D=%zu\n",
		             static_cast<long long>(page_positions), abi_page_bytes, expected_page_bytes);
		return 1;
	}
	std::printf("KV geometry: B=%zu runs=%zu page_bytes=%zu\n", geometry.page_positions,
	            geometry.runs, geometry.page_bytes);

	// The default run writes at least 2B + 1 positions where the cap allows (positions written =
	// prompt_len + produced - 1), so logical pages 0, 1 and 2 hold rows.
	if (!tokens_given) {
		const size_t want_positions = 2 * geometry.page_positions + 1;
		const size_t positions = want_positions < geometry.cap ? want_positions : geometry.cap;
		const size_t base = positions + 1 > prompt_len ? positions + 1 - prompt_len : 1;
		max_new_tokens = base < 8 ? 8 : base;
	}

	std::vector<int32_t> oracle_tokens;
	size_t oracle_produced = 0;
	superslm::SslmDecodeStopReason oracle_stop{};
	const superslm::SslmForwardStatus ora_st =
	    RunGreedyOracle(oracle, prompt_tokens, prompt_len, max_new_tokens, &oracle_tokens,
	                     &oracle_produced, &oracle_stop);
	if (ora_st != superslm::SslmForwardStatus::Ok) {
		std::fprintf(stderr, "FAIL: RunGreedyOracle returned status %d\n", static_cast<int>(ora_st));
		return 1;
	}
	oracle_tokens.resize(oracle_produced);

	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	// S2 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create now checks
	// alignment -- over-allocate and round up, matching the workspace buffer's own established
	// pattern (tools/t2139_c2_smoke.cpp). `pool_raw` retained as the aligned void* base of the
	// buffer this tool allocated and passed to sslm_kv_pool_create. The vector value-initialises
	// it, so every byte starts at zero (the zero-tail dependency in the header comment).
	const size_t pool_buf_size = block_bytes * block_count + overhead;
	std::vector<uint8_t> pool_raw_storage(pool_buf_size + 63);
	void* pool_raw = pool_raw_storage.data();
	size_t pool_raw_space = pool_raw_storage.size();
	std::align(64, pool_buf_size, pool_raw, pool_raw_space);
	sslm_kv_pool pool = nullptr;
	st = sslm_kv_pool_create(model, pool_raw, pool_buf_size, block_count, &pool);
	if (st != SSLM_OK || !pool) {
		std::fprintf(stderr, "FAIL: sslm_kv_pool_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	sslm_seq seq = nullptr;
	st = sslm_seq_create(model, &pool, &seq);
	if (st != SSLM_OK || !seq) {
		std::fprintf(stderr, "FAIL: sslm_seq_create returned %d\n", static_cast<int>(st));
		return 1;
	}

	int32_t consumed = 0;
	st = sslm_prefill(model, seq, prompt_tokens, static_cast<int32_t>(prompt_len),
	                   static_cast<int32_t>(prompt_len), SSLM_SPAN_PROMPT, nullptr, &consumed);
	if (st != SSLM_OK || consumed != static_cast<int32_t>(prompt_len)) {
		std::fprintf(stderr, "FAIL: sslm_prefill returned %d, consumed %d\n",
		             static_cast<int>(st), consumed);
		return 1;
	}

	std::vector<int32_t> abi_tokens;
	sslm_decode_params params{};
	// T-2199 Phase D review addendum (D-SLM3797, Dan): struct_size is the FIRST
	// new field the library now validates -- an unrecognized size is a loud rejection.
	params.struct_size = sizeof(params);
	params.layer_budget = forced_layer_budget > 0 ? forced_layer_budget
	                                               : static_cast<int32_t>(oracle.num_hidden_layers);
	sslm_seq seqs[1] = {seq};
	int32_t calls_this_run = 0;
	for (size_t t = 0; t < max_new_tokens; ++t) {
		int32_t out_token = -1;
		int32_t guard = 0;
		while (out_token < 0) {
			st = sslm_decode_step(model, seqs, 1, &params, nullptr, &out_token);
			++calls_this_run;
			if (st != SSLM_OK) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (token %zu) returned %d\n", t,
				             static_cast<int>(st));
				return 1;
			}
			if (++guard > 10000) {
				std::fprintf(stderr, "FAIL: sslm_decode_step (token %zu) never completed -- "
				                      "possible resumption bug\n", t);
				return 1;
			}
		}
		abi_tokens.push_back(out_token);
	}
	if (forced_layer_budget > 0) {
		std::printf("bounded layer_budget=%d: %d sslm_decode_step calls to produce %zu tokens\n",
		            forced_layer_budget, calls_this_run, max_new_tokens);
	}

	// --- comparison ---
	bool ok = true;
	if (abi_tokens.size() != oracle_tokens.size()) {
		std::fprintf(stderr, "FAIL: token count diverges: ABI=%zu oracle=%zu\n",
		             abi_tokens.size(), oracle_tokens.size());
		ok = false;
	} else {
		for (size_t i = 0; i < abi_tokens.size(); ++i) {
			if (abi_tokens[i] != oracle_tokens[i]) {
				std::fprintf(stderr, "FAIL: token[%zu] diverges: ABI=%d oracle=%d\n", i,
				             abi_tokens[i], oracle_tokens[i]);
				ok = false;
			}
		}
	}
	if (ok) {
		std::printf("token sequence: BIT-IDENTICAL (%zu tokens): ", abi_tokens.size());
		for (int32_t t : abi_tokens) std::printf("%d ", t);
		std::printf("\n");
	}

	// KV pages: compare the pages this ABI's holder maps against the oracle's own flat workspace.
	{
		const size_t oracle_kv_bytes = static_cast<size_t>(oracle.num_hidden_layers) *
		                                static_cast<size_t>(oracle.context_cap) *
		                                oracle.num_kv_heads * oracle.head_dim * 2;
		if (oracle_kv_bytes != block_bytes) {
			std::fprintf(stderr,
			             "FAIL: KV byte-count formulas diverge: ABI sslm_kv_block_size=%zu "
			             "oracle=%zu\n",
			             block_bytes, oracle_kv_bytes);
			ok = false;
		} else {
			// Re-run the oracle's own RunGreedyOracle a second time, independently, to obtain
			// its own final workspace bytes for comparison (RunGreedyOracle's own workspace is
			// local to that call and was not returned above) -- a fresh run against the
			// IDENTICAL prompt/max_new_tokens is bit-reproducible by this project's own proven
			// determinism property (S-HARDEN family), so comparing against a second run's own
			// buffer is equivalent to comparing against the first.
			const size_t kv_bytes = static_cast<size_t>(oracle.num_hidden_layers) *
			                         static_cast<size_t>(oracle.context_cap) * oracle.num_kv_heads *
			                         oracle.head_dim * 2;
			std::vector<uint8_t> oracle_workspace(kv_bytes);
			std::vector<int8_t> oracle_hidden(oracle.hidden_size);
			superslm::SequenceLayerState oracle_seq{};
			oracle_seq.hidden_codes = oracle_hidden.data();
			std::vector<int32_t> oracle_tokens2(max_new_tokens, 0);
			std::vector<int32_t> oracle_logit_rows(max_new_tokens *
			                                        static_cast<size_t>(oracle.vocab_size));
			size_t produced2 = 0;
			superslm::SslmDecodeStopReason stop2{};
			const superslm::SslmForwardStatus st2 = superslm::RunGreedyDecodeLoop(
			    oracle_seq, oracle.layers.data(), oracle.num_hidden_layers, oracle.hidden_size,
			    oracle.head_dim, oracle.num_kv_heads, oracle.intermediate_size, oracle.context_cap,
			    *oracle.rope_tables, prompt_tokens, prompt_len, oracle.embed_weights,
			    oracle.embed_site_constant, oracle.final_norm_gain.data(),
			    oracle.final_norm_site_constant, oracle.head_weights, oracle.vocab_size, nullptr,
			    0, max_new_tokens, oracle_workspace.data(), oracle_workspace.size(),
			    oracle_tokens2.data(), oracle_logit_rows.data(), oracle_tokens2.size(), &produced2,
			    &stop2, oracle.kv_precision, oracle.option_g_fused_k_landing,
			    oracle.num_attention_heads);
			if (st2 != superslm::SslmForwardStatus::Ok) {
				std::fprintf(stderr, "FAIL: second oracle run returned status %d\n",
				             static_cast<int>(st2));
				ok = false;
			} else {
				uint32_t mapped = 0;
				uint32_t entry_page = 0;
				const sslm_status ts = sslm_pkv_test_only_seq_table_entry(seq, 0, &entry_page, &mapped);
				if (ts != SSLM_OK || mapped == 0) {
					std::fprintf(stderr, "FAIL: holder table entry 0: status %d, mapped %u\n",
					             static_cast<int>(ts), static_cast<unsigned>(mapped));
					ok = false;
				} else {
					// The mapped pages, copied out in logical order.
					std::vector<uint8_t> pages(static_cast<size_t>(mapped) * geometry.page_bytes);
					bool pages_ok = true;
					for (uint32_t j = 0; j < mapped; ++j) {
						uint32_t phys = 0;
						uint32_t unused = 0;
						if (sslm_pkv_test_only_seq_table_entry(seq, j, &phys, &unused) != SSLM_OK ||
						    sslm_pkv_test_only_peek_page_bytes(
						        pool, phys, pages.data() + static_cast<size_t>(j) * geometry.page_bytes,
						        geometry.page_bytes) != SSLM_OK) {
							std::fprintf(stderr, "FAIL: could not read logical page %u\n",
							             static_cast<unsigned>(j));
							pages_ok = false;
							break;
						}
					}
#ifdef T2139_C4_CAPTURE
					g_c4_capture.pages = pages;
					g_c4_capture.workspace = oracle_workspace;
					g_c4_capture.geometry = geometry;
					g_c4_capture.mapped_pages = mapped;
#endif
					size_t rows_compared = 0;
					if (pages_ok) {
						RowId bad;
						if (FirstDivergentRow(pages.data(), mapped, oracle_workspace.data(), geometry,
						                      &bad, &rows_compared)) {
							const size_t H = geometry.kv_heads;
							std::fprintf(stderr,
							             "FAIL: KV row diverges from the oracle's own workspace: "
							             "page %zu, pos %zu, run %zu (layer %zu, half %zu, kv_head %zu)\n",
							             bad.page, bad.pos, bad.run, bad.run / (2 * H), (bad.run / H) % 2,
							             bad.run % H);
							pages_ok = false;
						} else if (FirstNonzeroBeyond(oracle_workspace.data(), mapped, geometry, &bad)) {
							// The ABI needed no page it did not map: the oracle's rows are zero from the
							// first unmapped position on.
							std::fprintf(stderr,
							             "FAIL: the oracle wrote pos %zu (run %zu) beyond the %u mapped "
							             "pages\n",
							             bad.pos, bad.run, static_cast<unsigned>(mapped));
							pages_ok = false;
						}
					}
					// A run that reached fewer pages than the cap allows up to three cannot show a
					// rotation error, whatever length it was given.
					const size_t cap_pages = (geometry.cap + geometry.page_positions - 1) / geometry.page_positions;
					const size_t required_pages = cap_pages < 3 ? cap_pages : 3;
					if (static_cast<size_t>(mapped) < required_pages) {
						std::fprintf(stderr, "FAIL: pages reached %u, fewer than %zu\n",
						             static_cast<unsigned>(mapped), required_pages);
						pages_ok = false;
					}
					if (!pages_ok) {
						ok = false;
					} else {
						std::printf("KV pages: B=%zu runs=%zu page_bytes=%zu, %u pages reached, "
						            "%zu rows BIT-IDENTICAL\n",
						            geometry.page_positions, geometry.runs, geometry.page_bytes,
						            static_cast<unsigned>(mapped), rows_compared);
					}
				}
			}
		}
	}

	if (!ok) {
		std::fprintf(stderr, "t2139_c4_oracle: FAIL\n");
		return 1;
	}
	std::printf("t2139_c4_oracle: PASS\n");
	return 0;
}
