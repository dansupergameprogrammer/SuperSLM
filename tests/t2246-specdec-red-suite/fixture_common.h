// T-2246 (test design) -- shared CHECK harness, real-artifact loading, greedy-reference oracle,
// canonicalized save-blob comparison, and fixture builders for the speculative-decoding red
// suite. Reuses this repo's established conventions rather than inventing new ones:
//   - the CHECK/CHECK_MSG/SKIP_MSG counter triple, AlignedBuffer/SinglePool/MakePool,
//     SeqBlobBuffer, ParseFixtureArgs, and LoadRealModelView are
//     tests/t2138-abi-red-suite/fixture_common.h's own shapes, reused;
//   - the CpuOracleModel marshal + RunGreedyDecodeLoop reference oracle is that same file's
//     construction (itself tools/t2100_gpu_throughput.cpp's load path), extended here with an
//     explicit stop-id set so boundary cells drive the shipped loop through the identical
//     stop/cap tests the speculate walk must reproduce (forward_sites.cpp:2615-2628);
//   - the save-blob field offsets cited below are read from src/sslm_abi.cpp:2609-2716
//     (kSeqBlobFixedHeaderBytes = 120 and each WriteLE site in sslm_seq_save);
//   - digest helpers call superslm::ComputeTokenDigest / ComputeFinalLogitDigest directly
//     (include/superslm/decode_digest.h; contract D-SLM3795).
//
// Model/adapter/corpus paths are NEVER hardcoded: g_model_path/g_model_tok_path/g_corpus_path
// come from argv. A cell whose fixture is missing SKIPs (counted separately), never silently
// passes.
//
// RED STRUCTURE: every cell calls at least one symbol declared in
// sslm_specdec_red_contract.h -- none of which has a definition at pin f409bda -- so every
// cell file links RED BY LINK on recorded expected symbols until S-B/S-C/S-D/S-E land. The
// greedy-reference legs and fixture-search legs inside these files use only SHIPPED symbols
// and are exercised for real once the build lands.
#ifndef SSLM_T2246_FIXTURE_COMMON_H
#define SSLM_T2246_FIXTURE_COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/decode_digest.h"    // superslm::ComputeTokenDigest/ComputeFinalLogitDigest
#include "superslm/forward_sites.h"   // superslm::RunGreedyDecodeLoop -- the greedy reference
#include "superslm/layer_marshal.h"
#include "superslm/model.h"
#include "superslm/sha256.h"          // superslm::Sha256Hash -- corpus pin check (dim12)

// The REAL shipped CPU ABI surface (prefill/pool/save/restore/tokenize/decode_step_v2 ...).
#include "superslm/sslm_abi.h"

// The EXPECTED _v3/specdec surface this suite is authored against (see that file's header).
#include "sslm_specdec_red_contract.h"

// ---------------------------------------------------------------------------
// CHECK harness -- tests/t2138-abi-red-suite/fixture_common.h's own triple, verbatim shapes.

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	} while (0)

inline void PrintSummaryAndExit(int* out_ec) {
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	*out_ec = GFailures ? 1 : 0;
}

// Cell-local bail-out: records the failure and abandons THIS cell only (the calling Test*
// function returns), leaving later cells runnable -- dim5's per-cell isolation convention.
#define ASSERT_TRUE(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			return; \
		} \
	} while (0)

// ---------------------------------------------------------------------------
// Buffers, pools, blobs -- t2138's own shapes.

struct AlignedBuffer {
	explicit AlignedBuffer(size_t n)
	    : bytes_(n),
	      storage_(n > 0 ? ::operator new(n, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)) : nullptr) {}
	~AlignedBuffer() {
		if (storage_) ::operator delete(storage_, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer& operator=(const AlignedBuffer&) = delete;
	void* data() { return storage_; }
	const void* data() const { return storage_; }
	size_t size() const { return bytes_; }

  private:
	size_t bytes_;
	void* storage_;
};

struct SinglePool {
	std::unique_ptr<AlignedBuffer> buf;
	sslm_kv_pool pool = nullptr;
};

inline bool MakePool(sslm_model model, uint32_t block_count, SinglePool* out) {
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	out->buf = std::make_unique<AlignedBuffer>(block_count * block_bytes + overhead);
	return sslm_kv_pool_create(model, out->buf->data(), out->buf->size(), block_count,
	                            &out->pool) == SSLM_OK;
}

inline bool MakeSinglePool(sslm_model model, SinglePool* out) {
	return MakePool(model, 1, out);
}

struct SeqBlobBuffer {
	std::vector<uint8_t> bytes;
	size_t size = 0;
	explicit SeqBlobBuffer(sslm_model model) : bytes(sslm_seq_state_size(model)), size(bytes.size()) {}
};

// ---------------------------------------------------------------------------
// argv fixtures (t2138 convention): every flag optional; absent means SKIP.

static std::string g_model_path;      // real base artifact (the speculating model)
static std::string g_model_tok_path;  // artifact carrying a bound tokenizer, for corpus text
static std::string g_corpus_path;     // golden corpus JSONL (shopkeeper_corpus_v1.jsonl)

inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model=")) g_model_path = v;
		else if (const char* v = take("--modeltok=")) g_model_tok_path = v;
		else if (const char* v = take("--corpus=")) g_corpus_path = v;
	}
}

inline bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out_bytes) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) return false;
	} else {
		std::fclose(f);
	}
	return true;
}

inline bool LoadRealModelView(const std::string& path, superslm::SslmModelView* out_view,
                              std::vector<uint8_t>* out_bytes, std::string* out_err) {
	if (!ReadFileBytes(path, out_bytes)) {
		if (out_err) *out_err = "could not read " + path;
		return false;
	}
	const superslm::SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, out_err);
	return st == superslm::SslmModelStatus::Ok;
}

// ---------------------------------------------------------------------------
// Greedy-reference oracle (shipped symbols only).

struct CpuOracleModel {
	std::vector<superslm_marshal::LayerBacking> backings;
	std::vector<superslm::LayerWeights> layers;
	const int8_t* embed_weights = nullptr;
	superslm::CarriedScale embed_site_constant{};
	std::vector<int32_t> final_norm_gain;
	superslm::CarriedScale final_norm_site_constant{};
	const int8_t* head_weights = nullptr;
	uint32_t num_hidden_layers = 0;
	size_t hidden_size = 0, head_dim = 0, num_kv_heads = 0, intermediate_size = 0;
	int64_t context_cap = 0;
	int32_t vocab_size = 0;
	const superslm::SslmTensorManifest* rope_tables = nullptr;
	superslm::SslmKvPrecision kv_precision{};
	bool option_g_fused_k_landing = false;

	size_t kv_row_bytes() const {  // one cache position across all layers (kv_precision int8:
	                               // src/sslm_abi.cpp geometry, plan SS3.3's own 1 B/element)
		return static_cast<size_t>(num_hidden_layers) * num_kv_heads * head_dim * 2;
	}
};

inline bool LoadCpuOracleModel(const superslm::SslmModelView& view, CpuOracleModel* out,
                                std::string* err) {
	out->num_hidden_layers = view.config.num_hidden_layers;
	out->hidden_size = view.config.hidden_size;
	out->head_dim = view.config.head_dim;
	out->num_kv_heads = view.config.num_key_value_heads;
	out->intermediate_size = view.config.intermediate_size;
	out->context_cap = static_cast<int64_t>(view.config.context_cap);
	out->vocab_size = view.config.vocab_size;
	out->rope_tables = &view.rope_tables;
	out->kv_precision = view.config.kv_precision;
	out->option_g_fused_k_landing = view.option_g_fused_k_landing;
	out->backings.resize(out->num_hidden_layers);
	out->layers.resize(out->num_hidden_layers);
	for (uint32_t l = 0; l < out->num_hidden_layers; ++l) {
		if (!superslm_marshal::MarshalLayer(view, l, view.config.num_attention_heads,
		                                     view.config.num_key_value_heads, out->backings[l],
		                                     out->layers[l], err)) {
			return false;
		}
	}
	const superslm::SslmTensorView* embed_w = view.weights.Tensor("embed");
	const superslm::SslmTensorView* final_gain_w = view.weights.Tensor("final_norm.gain");
	if (!embed_w || !final_gain_w) {
		if (err) *err = "artifact has no embed or final_norm.gain tensor";
		return false;
	}
	out->final_norm_gain = superslm_marshal::WidenGainToInt32(*final_gain_w);
	out->embed_weights = reinterpret_cast<const int8_t*>(embed_w->data);
	bool ok = true;
	out->embed_site_constant =
	    superslm_marshal::ReadCarriedScale(view.composition_constants, "embed", &ok);
	out->final_norm_site_constant =
	    superslm_marshal::ReadCarriedScale(view.composition_constants, "final_norm", &ok);
	if (!ok) {
		if (err) *err = "artifact has no embed/final_norm site constant";
		return false;
	}
	if (view.config.tie_word_embeddings) {
		out->head_weights = out->embed_weights;
	} else {
		const superslm::SslmTensorView* lm_head_w = view.weights.Tensor("lm_head");
		if (!lm_head_w) {
			if (err) *err = "tie_word_embeddings=0 but no lm_head tensor present";
			return false;
		}
		out->head_weights = reinterpret_cast<const int8_t*>(lm_head_w->data);
	}
	return true;
}

struct GreedyRun {
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;  // produced * vocab_size, row-major
	size_t produced = 0;
	superslm::SslmDecodeStopReason reason = superslm::SslmDecodeStopReason::MaxTokensReached;
};

// One full greedy pass through the shipped entry point (tools/sslm_generate.cpp's own call),
// with the host stop set and cap exposed because the SS3.2 batch walk must reproduce exactly
// these tests per emitted position (append-before-stop-test, stop-then-cap).
inline bool RunGreedyReference(const CpuOracleModel& m, const int32_t* prompt_tokens,
                                size_t prompt_len, const std::vector<int32_t>& stop_ids,
                                size_t max_new_tokens, GreedyRun* out_run, std::string* err) {
	if (prompt_len < 1 || max_new_tokens < 1) {
		if (err) *err = "degenerate prompt/budget";
		return false;
	}
	const size_t kv_bytes = static_cast<size_t>(m.num_hidden_layers) *
	                        static_cast<size_t>(m.context_cap) * m.num_kv_heads * m.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(m.hidden_size);
	superslm::SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes.data();
	out_run->tokens.assign(max_new_tokens, 0);
	out_run->logit_rows.assign(max_new_tokens * static_cast<size_t>(m.vocab_size), 0);
	const superslm::SslmForwardStatus st = superslm::RunGreedyDecodeLoop(
	    seq, m.layers.data(), m.num_hidden_layers, m.hidden_size, m.head_dim, m.num_kv_heads,
	    m.intermediate_size, m.context_cap, *m.rope_tables, prompt_tokens, prompt_len,
	    m.embed_weights, m.embed_site_constant, m.final_norm_gain.data(),
	    m.final_norm_site_constant, m.head_weights, m.vocab_size,
	    stop_ids.empty() ? nullptr : stop_ids.data(), stop_ids.size(), max_new_tokens,
	    workspace.data(), workspace.size(), out_run->tokens.data(), out_run->logit_rows.data(),
	    out_run->tokens.size(), &out_run->produced, &out_run->reason, m.kv_precision,
	    m.option_g_fused_k_landing);
	if (st != superslm::SslmForwardStatus::Ok) {
		if (err) *err = "greedy reference rejected (" + std::to_string(static_cast<int>(st)) + ")";
		return false;
	}
	out_run->tokens.resize(out_run->produced);
	out_run->logit_rows.resize(out_run->produced * static_cast<size_t>(m.vocab_size));
	return true;
}

// Digest pair over a run's emissions, per the pinned mapping (plan SS1; D-SLM3795): token
// digest always, logit-row digest over the emitted rows.
inline void DigestRun(const GreedyRun& run, size_t vocab_size, uint8_t out_token[32],
                      uint8_t out_rows[32]) {
	superslm::ComputeTokenDigest(run.tokens.data(), run.tokens.size(), out_token);
	superslm::ComputeFinalLogitDigest(run.logit_rows.data(), run.produced, vocab_size, out_rows);
}

inline bool DigestEqual(const uint8_t* a, const uint8_t* b) {
	return std::memcmp(a, b, 32) == 0;
}

// Digest pair over the FIRST `prefix` emissions of a reference run -- what a truncated
// speculate call must cover exactly.
inline void DigestRunPrefix(const GreedyRun& run, size_t prefix, size_t vocab_size,
                            uint8_t out_token[32], uint8_t out_rows[32]) {
	superslm::ComputeTokenDigest(run.tokens.data(), prefix, out_token);
	superslm::ComputeFinalLogitDigest(run.logit_rows.data(), prefix, vocab_size, out_rows);
}

// Drives repeated _v3 speculate calls until `total_budget` emissions or a stop reason,
// accumulating tokens and rows so one digest pair covers the whole drive (the flagship
// cells' oracle input). Returns false when a call rejects (the cell then asserts on the
// rejection itself rather than looping).
struct SpecDrive {
	std::vector<int32_t> tokens;
	std::vector<int32_t> logit_rows;
	int32_t last_stop = SSLM_SPECULATE_STOP_MAX_TOKENS;
};

inline bool DriveSpeculate(sslm_model model, sslm_seq seq, const sslm_speculate_params& base,
                           sslm_workspace ws, size_t vocab_size, size_t total_budget,
                           SpecDrive* out, std::string* err) {
	sslm_speculate_params p = base;
	while (out->tokens.size() < total_budget) {
		p.max_new_tokens =
		    static_cast<int32_t>(total_budget - out->tokens.size());
		std::vector<int32_t> tok(static_cast<size_t>(p.max_new_tokens) + 1, 0);
		std::vector<int32_t> rows((static_cast<size_t>(p.max_new_tokens) + 1) * vocab_size, 0);
		int32_t produced = 0;
		int32_t stop = SSLM_SPECULATE_STOP_MAX_TOKENS;
		const sslm_status st =
		    sslm_speculate_step_v3(model, seq, &p, ws, tok.data(),
		                            static_cast<int32_t>(tok.size()), rows.data(),
		                            static_cast<int32_t>(rows.size()), &produced, &stop);
		if (st != SSLM_OK) {
			if (err) *err = "speculate rejected (" + std::to_string(static_cast<int>(st)) + ")";
			return false;
		}
		if (produced < 0 || static_cast<size_t>(produced) > tok.size()) {
			if (err) *err = "produced count out of range";
			return false;
		}
		out->tokens.insert(out->tokens.end(), tok.begin(), tok.begin() + produced);
		out->logit_rows.insert(out->logit_rows.end(), rows.begin(),
		                       rows.begin() + static_cast<size_t>(produced) *
		                                          static_cast<ptrdiff_t>(vocab_size));
		out->last_stop = stop;
		if (stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED) break;
		if (produced == 0 && stop == SSLM_SPECULATE_STOP_MAX_TOKENS &&
		    p.max_new_tokens > 0) {
			if (err) *err = "zero emissions with positive budget and no stop";
			return false;
		}
	}
	return true;
}

// Convenience: params_init then CHECK-visible failure. Returns false instead of asserting so
// callers decide whether a rejection is the cell's subject.
inline bool MakeSpecParams(sslm_model model, int32_t k_max, int32_t max_new_tokens,
                           const std::vector<int32_t>& stop_ids, sslm_speculate_params* out) {
	if (sslm_speculate_params_init(model, out) != SSLM_OK) return false;
	out->max_draft_tokens = k_max;
	out->max_new_tokens = max_new_tokens;
	if (!stop_ids.empty()) {
		out->stop_ids = stop_ids.data();
		out->stop_count = static_cast<int32_t>(stop_ids.size());
	} else {
		out->stop_ids = nullptr;
		out->stop_count = 0;
	}
	return true;
}

// Drives an ABI-level twin through `count` single-token v2 emissions at full layer budget --
// the never-speculated / pure-v2 witness for the rollback-exactness and interleave families
// (each completed call appends retention per SS3.0's "Decode step (any mode)" row, so both
// sides' sequence state advances through the same emitted prefix).
inline bool DriveV2Emissions(sslm_model model, sslm_seq* seq, uint32_t num_hidden_layers,
                             sslm_workspace ws, size_t count) {
	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = static_cast<int32_t>(num_hidden_layers);
	for (size_t i = 0; i < count; ++i) {
		int32_t t = 0;
		if (sslm_decode_step_v2(model, seq, 1, &dp, ws, &t) != SSLM_OK) return false;
		if (t < 0) return false;  // -1 pending => the full-budget call did not complete a token
	}
	return true;
}

// ---------------------------------------------------------------------------
// Save-blob layout helpers. Field offsets cited from src/sslm_abi.cpp's sslm_seq_save:
// fixed header = 120 bytes (:2609-2615), then variable residual (hidden_size bytes iff
// mid-token), anti-LM history (4 B x history_count), kv_block_count(4), then the whole KV
// block image (block_size bytes, :2713-2716).

struct BlobLayout {
	static constexpr size_t kContextLength = 60;    // LE64 (:2674)
	static constexpr size_t kLayerIndex = 68;       // LE32 (:2676)
	static constexpr size_t kCurrentToken = 72;     // LE32 (:2687)
	static constexpr size_t kSaturationCount = 92;  // LE64 (:2693)
	static constexpr size_t kHistoryCount = 112;    // LE64 (:2702)
	static constexpr size_t kFixedHeaderBytes = 120;
};

inline uint64_t BlobReadLE64(const uint8_t* p) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
	return v;
}

inline uint32_t BlobReadLE32(const uint8_t* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
	       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

inline int64_t BlobContextLength(const std::vector<uint8_t>& b) {
	return static_cast<int64_t>(BlobReadLE64(b.data() + BlobLayout::kContextLength));
}

inline int64_t BlobSaturationCount(const std::vector<uint8_t>& b) {
	return static_cast<int64_t>(BlobReadLE64(b.data() + BlobLayout::kSaturationCount));
}

// Byte offset where the KV block image begins inside a saved blob.
inline size_t BlobKvRegionStart(const std::vector<uint8_t>& b, size_t hidden_size) {
	const bool mid_token = BlobReadLE32(b.data() + BlobLayout::kLayerIndex) != 0;
	const size_t residual_len = mid_token ? hidden_size : 0;
	const uint64_t hist = BlobReadLE64(b.data() + BlobLayout::kHistoryCount);
	return BlobLayout::kFixedHeaderBytes + residual_len + static_cast<size_t>(hist) * 4 + 4;
}

// The plan SS3.3 canonicalization as code: sequence-state equality asserted over the save-blob
// layout with raw KV-block bytes BEYOND the live rows excluded (path-dependent by design,
// sslm_abi.cpp:1864-1869). Header fields, residual, and anti-LM regions compare byte-exact;
// live KV rows below context_length compare byte-exact inside the block image. Live-byte span
// assumes cache positions occupy ascending addresses within each layer segment -- the order
// landings are written (row index = position); both layer-major and flat layouts satisfy it.
inline bool CanonicalizedBlobsEqual(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                                    size_t hidden_size, size_t block_size, int64_t context_cap,
                                    std::string* why_not) {
	if (a.size() != b.size()) {
		if (why_not) *why_not = "blob sizes differ";
		return false;
	}
	const size_t kv_start = BlobKvRegionStart(a, hidden_size);
	if (std::memcmp(a.data(), b.data(), kv_start) != 0) {
		if (why_not) *why_not = "header/residual/anti-LM region differs";
		return false;
	}
	const int64_t ctx_a = BlobContextLength(a);
	const int64_t ctx_b = BlobContextLength(b);
	if (ctx_a != ctx_b) {
		if (why_not) *why_not = "context_length differs";
		return false;
	}
	const size_t per_position = block_size / static_cast<size_t>(context_cap);
	size_t live_bytes = per_position * static_cast<size_t>(ctx_a);
	if (live_bytes > block_size) live_bytes = block_size;  // paranoia; ctx <= cap by validation
	if (std::memcmp(a.data() + kv_start, b.data() + kv_start, live_bytes) != 0) {
		if (why_not) *why_not = "live KV rows below context_length differ";
		return false;
	}
	return true;
}

// Cheap near-cap occupancy without real compute (dim5 C11's grounded tamper route): save a
// small real sequence, patch ONLY its context_length field to `new_ctx`, restore into a fresh
// handle. Restore reads both counters from the blob verbatim (sslm_abi.cpp:2909-2910).
inline bool RestoreWithTamperedContextLength(sslm_model model, sslm_kv_pool* pool,
                                             const std::vector<uint8_t>& src_blob,
                                             int64_t new_ctx, sslm_seq* out_seq) {
	std::vector<uint8_t> patched = src_blob;
	const uint64_t v = static_cast<uint64_t>(new_ctx);
	for (int i = 0; i < 8; ++i) {
		patched[BlobLayout::kContextLength + i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
	}
	return sslm_seq_restore(model, pool, patched.data(), patched.size(), out_seq) == SSLM_OK;
}

// ---------------------------------------------------------------------------
// Corpus + tokenizer fixtures.

// Extracts the first max_count "utterance" string values from a JSONL corpus (naive scan --
// the corpus is machine-generated JSONL with stable field order; shopkeeper_corpus_v1.jsonl).
inline bool LoadCorpusUtterances(const std::string& path, size_t max_count,
                                 std::vector<std::string>* out) {
	std::vector<uint8_t> bytes;
	if (!ReadFileBytes(path, &bytes) || bytes.empty()) return false;
	const char* p = reinterpret_cast<const char*>(bytes.data());
	const size_t n = bytes.size();
	const std::string key = "\"utterance\"";
	size_t i = 0;
	while (out->size() < max_count && i + key.size() < n) {
		const size_t at = std::string(p + i, n - i).find(key);
		if (at == std::string::npos) break;
		i += at + key.size();
		size_t j = i;
		while (j < n && p[j] != ':') ++j;
		while (j < n && (p[j] == ':' || p[j] == ' ' || p[j] == '\t')) ++j;
		if (j >= n || p[j] != '"') continue;
		++j;
		std::string value;
		while (j < n && p[j] != '"') {
			if (p[j] == '\\' && j + 1 < n) {
				value.push_back(p[++j]);
			} else {
				value.push_back(p[j]);
			}
			++j;
		}
		i = j;
		if (!value.empty()) out->push_back(value);
	}
	return !out->empty();
}

// sslm_tokenize two-call sizing (sslm_abi_functions.inc:133; same query/fill shape the schema
// name verb documents).
inline bool TokenizeUtf8(sslm_model tok_model, const std::string& utf8,
                         std::vector<int32_t>* out) {
	int32_t need = 0;
	sslm_status st = sslm_tokenize(tok_model, utf8.c_str(), nullptr, &need);
	if (st != SSLM_BUFFER_TOO_SMALL && !(st == SSLM_OK && need == 0)) return false;
	out->assign(static_cast<size_t>(need), 0);
	int32_t got = need;
	st = sslm_tokenize(tok_model, utf8.c_str(), out->empty() ? nullptr : out->data(), &got);
	return st == SSLM_OK && got == need;
}

// A prompt whose token ids NEVER repeat (strictly increasing arithmetic ids): drives the
// drafter's no-match arm for the empty-draft fallback cell and the broken-drafter negative
// control without any injection seam. Ids stay inside [0, vocab).
inline std::vector<int32_t> NoRepeatPrompt(int32_t vocab_size, size_t len) {
	std::vector<int32_t> ids;
	ids.reserve(len);
	int32_t id = 11;
	for (size_t i = 0; i < len && id < vocab_size; ++i, id += 97) ids.push_back(id);
	return ids;
}

// Full-K acceptance fixture search (plan SS5 S-A full-K cell's own stated construction route:
// "forcing target agreement needs prompt iteration"). Scans short windows of a tokenized
// corpus stream for a context whose LONGEST suffix match (per the documented selection rule,
// plan SS3.1) carries a continuation of length K that the target itself reproduces -- i.e.
// pure greedy from that context emits those exact K ids and then a bonus. Returns the prompt
// (the context window) and leaves the expected acceptance to be re-derived by driving the
// shipped greedy loop inside the cell. Uses ONLY the shipped CPU oracle; runs today.
inline bool FindFullKAcceptancePrompt(const CpuOracleModel& m, const std::vector<int32_t>& stream,
                                      int32_t k_max, size_t max_attempts,
                                      std::vector<int32_t>* out_prompt, std::string* err) {
	constexpr size_t kWindow = 40;
	if (stream.size() < kWindow + static_cast<size_t>(k_max) + 1) {
		if (err) *err = "corpus stream too short for a full-K window";
		return false;
	}
	GreedyRun run;
	size_t attempts = 0;
	for (size_t s = 0; s + kWindow < stream.size() && attempts < max_attempts; ++s, ++attempts) {
		const int32_t* ctx = stream.data() + s;
		if (!RunGreedyReference(m, ctx, kWindow, {}, static_cast<size_t>(k_max) + 1, &run, err)) {
			continue;
		}
		if (run.produced < static_cast<size_t>(k_max) + 1) continue;
		// Longest suffix match of the context within itself (fixture SELECTION uses the
		// documented rule; the ASSERTION stays _v3-vs-greedy equivalence).
		size_t best_len = 0, best_pos = 0;  // best_pos: start of the EARLIEST longest match
		for (size_t len = 1; len + 1 <= kWindow; ++len) {
			for (size_t q = 0; q + len <= kWindow - len; ++q) {  // earlier occurrence only
				if (std::memcmp(ctx + q, ctx + kWindow - len, len * sizeof(int32_t)) == 0) {
					if (len > best_len) {
						best_len = len;
						best_pos = q;
					} else if (len == best_len) {
						break;  // earliest occurrence already recorded at this length
					}
				}
			}
		}
		if (best_len == 0) continue;
		const size_t cont_start = best_pos + best_len;
		const size_t cont_avail = kWindow - cont_start;
		if (cont_avail < static_cast<size_t>(k_max)) continue;
		bool agree = true;
		for (int32_t i = 0; i < k_max; ++i) {
			if (ctx[cont_start + static_cast<size_t>(i)] !=
			    run.tokens[static_cast<size_t>(i)]) {
				agree = false;
				break;
			}
		}
		if (agree) {
			out_prompt->assign(ctx, ctx + kWindow);
			return true;
		}
	}
	if (err) *err = "no full-K acceptance prompt found in the searched corpus windows";
	return false;
}

#endif  // SSLM_T2246_FIXTURE_COMMON_H
