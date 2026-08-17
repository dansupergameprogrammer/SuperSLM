// T-2138 (Curie) -- shared CHECK harness and real-artifact loading helpers for the Layer-1
// CPU-side sslm_* consumer C ABI red suite. Reuses this repo's own established conventions
// rather than inventing new ones: the CHECK/CHECK_MSG/SKIP_MSG counter triple and the argv
// fixture convention are tests/t2112-gpu-1p0-red-suite/fixture_common.h's own shapes
// (tests/t2130-g5-red-suite/fixture_common.h's own copy of the identical convention, itself
// tests/t2018-slora-serial/t2018_offline_red.cpp's convention, itself tests/test_main.cpp's);
// the real-artifact load path (SslmModel::Load over a file read into memory, and the CPU
// forward-pass oracle built from it) is tools/t2100_gpu_throughput.cpp's / this suite's own two
// sibling suites' convention, reused verbatim.
//
// Model/adapter paths are NEVER hardcoded (StandardsDocument.md Sec5.2, cold-read
// self-contained): every product cell that needs a real artifact takes it from
// g_model_path/g_adapter_path, populated from argv by each suite binary's own main(). A cell
// whose product half needs an artifact argv did not supply reports SKIP (counted separately
// from PASS/FAIL/RED so a suite run without artifacts on the invoking machine is still
// legible), never a silent pass.
//
// RED-BY-LINK STRUCTURE (matching tests/t2130-g5-red-suite's own convention exactly, since this
// ABI is equally unbuilt, D-SLM3450): every Test* function below and in every dimN_*_red.cpp
// file is a REAL function performing a real call sequence against sslm_abi.h's declared
// surface -- product cells check g_model_path/g_adapter_path and SKIP their own real-artifact
// half when absent, exactly as they will once the build lands and a driver actually invokes
// them with live handles. At the RED phase, no cell is actually CALLED: each file's own main()
// only takes the address of every Test* function (forcing it to be emitted, which is what
// forces the linker to see -- and fail to resolve -- every sslm_* symbol the function body
// references). This is what makes the suite fail at LINK (LNK2019), never at compile and never
// by silently passing nothing.
#ifndef SSLM_T2138_FIXTURE_COMMON_H
#define SSLM_T2138_FIXTURE_COMMON_H

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "superslm/artifact.h"
#include "superslm/forward_sites.h"   // superslm::RunGreedyDecodeLoop -- the C4 dim6/dim10 oracle
#include "superslm/layer_marshal.h"   // superslm_marshal::LayerBacking/MarshalLayer/ReadCarriedScale
#include "superslm/model.h"

// The declared CPU ABI surface this whole suite is written against (Claude/Vitruvius/
// t2133-layer1-c-abi-design-2026-08-16.md Sec8). Promoted copy, this directory -- see
// sslm_abi.h's own header comment.
#include "sslm_abi.h"

// T-2139 fix round S2/S3 (Claude/Poirot/2c18dab-t2139-abi-build-review.md): sslm_kv_pool_create
// now checks its caller-supplied buffer's alignment the same way sslm_workspace_create already
// did (SSLM_ABI_ALIGNMENT_BYTES, this suite's own sslm_abi.h, reconciled this revision) -- a
// plain std::vector<uint8_t>::data() carries no alignment guarantee past whatever the allocator
// happens to return. Every fixture in this suite that constructs a workspace or pool buffer now
// uses this instead -- same data()/size() interface, so the affected call sites are a type swap,
// not a rewrite.
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

// T-2139 fix round S6/nullptr-pool sweep: sslm_seq_create/sslm_prefix_begin/sslm_seq_restore
// all require a REAL, bound sslm_kv_pool since design commit fab235c1c6 (the buffer-mapping
// ruling) -- a literal `nullptr` pool argument is a suite-authored convenience that predates
// that ruling and is itself rejected (SSLM_INVALID_ARGUMENT: no pool pointer at all, not "no
// pool needed"). Every cell that previously hardcoded `nullptr` for its own pool argument now
// builds one of these instead. Owns both the pool handle and its backing storage so the caller
// need only keep this object alive for the pool's own lifetime -- block_count=1, the
// single-sequence case design Sec7.2's own sizing recipe names.
struct SinglePool {
	std::unique_ptr<AlignedBuffer> buf;
	sslm_kv_pool pool = nullptr;
};

// Leaves a REAL sequence genuinely mid-token (layer_index != 0, current_token pending) --
// house precedent tools/t2139_c6_smoke.cpp's own comment: "the FIRST decode_step call after a
// completed prefill is always the free ready_for_logits step -- no RunLayerLoop work, so a real
// token completes in one call regardless of layer_budget. A SECOND, bounded call is what
// genuinely starts a new token and leaves it mid-token." Every cell in this suite that needs a
// mid-token precondition (SSLM_ADAPTER_SWAP_MIDTOKEN_REJECTED, SSLM_SEQ_RESET_MIDTOKEN_REJECTED,
// a mid-token save/restore) calls this rather than re-deriving the two-call shape per file.
// Returns true iff the sequence is confirmed pending (out_token < 0) after the second call.
inline bool EnterMidToken(sslm_model model, sslm_seq seq) {
	sslm_decode_params params{};
	params.layer_budget = 1;
	sslm_seq batch[1] = {seq};
	int32_t out_token = 0;
	if (sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) != SSLM_OK) return false;
	out_token = 0;
	if (sslm_decode_step(model, batch, 1, &params, nullptr, &out_token) != SSLM_OK) return false;
	return out_token < 0;
}

// A real save/restore blob buffer, sized from sslm_seq_state_size(model) -- NOT a fixed 64 KiB
// stand-in. The blob carries the sequence's own full KV block content (design Sec7.3:
// "kv_blocks: kv_block_count * sslm_kv_block_size(model) bytes"), which is the same
// hundreds-of-MB-per-sequence figure sslm_kv_block_size itself reports (design Sec7.2, revised
// buffer model) -- a fixed-size stand-in silently BUFFER_TOO_SMALLs against any real artifact.
struct SeqBlobBuffer {
	std::vector<uint8_t> bytes;
	size_t size = 0;
	explicit SeqBlobBuffer(sslm_model model) : bytes(sslm_seq_state_size(model)), size(bytes.size()) {}
	void ResetSize() { size = bytes.size(); }
};

inline bool MakeSinglePool(sslm_model model, SinglePool* out) {
	const uint32_t block_count = 1;
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	out->buf = std::make_unique<AlignedBuffer>(block_count * block_bytes + overhead);
	return sslm_kv_pool_create(model, out->buf->data(), out->buf->size(), block_count,
	                            &out->pool) == SSLM_OK;
}

// The N-block generalization of SinglePool/MakeSinglePool -- for cells that need more than one
// block concurrently live (e.g. a save/restore cell whose original handle is still held when the
// restored one is created).
inline bool MakePool(sslm_model model, uint32_t block_count, SinglePool* out) {
	const size_t block_bytes = sslm_kv_block_size(model);
	const size_t overhead = sslm_kv_pool_overhead_size(model, block_count);
	out->buf = std::make_unique<AlignedBuffer>(block_count * block_bytes + overhead);
	return sslm_kv_pool_create(model, out->buf->data(), out->buf->size(), block_count,
	                            &out->pool) == SSLM_OK;
}

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

// argv-populated real-artifact paths (StandardsDocument.md Sec5.4 real-workload rule). Empty
// means "not supplied on this invocation" -- the cell SKIPs its product half rather than
// silently passing or fabricating a result.
static std::string g_model_path;          // real base artifact (e.g. qwen2.5-1.5b-instruct.sslm)
static std::string g_model_variant_path;  // a second real artifact, same tier, different content
                                           // hash (design Sec7.3/Sec10 dim2's "same-shape,
                                           // different-content" cell -- an adapter-merged
                                           // derivative of g_model_path, e.g.
                                           // qwen2.5-1.5b-shopkeeper-lora-v2-merged-t2102.sslm)
static std::string g_adapter_path;        // real LoRA adapter artifact (shopkeeper-v2), for C6/dim8/dim10
static std::string g_foreign_model_path;  // a real artifact of a DIFFERENT shape (e.g. the
                                           // 0.5B tier) -- dim2's foreign-blob/shape-mismatch cell
static std::string g_model_tok_path;      // a SEPARATE real artifact carrying a bound tokenizer
                                           // (house convention: build.bat's own T2139_MODEL vs
                                           // T2139_MODEL_TOK split) -- g_model_path is the base
                                           // the real adapter was compiled against and often has
                                           // NO tokenizer bound; text-I/O cells (sslm_tokenize/
                                           // sslm_detokenize_stream) need this one instead. Empty
                                           // means "not supplied" -- a cell needing it SKIPs.

// Reads a whole file into memory, no parsing -- the shape sslm_model_map/sslm_adapter_map take
// directly (const void* data, size_t size), distinct from LoadRealModel below which ALSO
// invokes the already-shipped internal SslmModel::Load parse for cells that need a CPU-oracle
// SslmModelView, not merely raw bytes to feed the (unbuilt) sslm_model_map.
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

// Reuses tools/t2100_gpu_throughput.cpp's own load path verbatim (T-2100, this repo) -- the
// internal parse, for cells that need a real superslm::SslmModelView to build the C4/dim6/dim10
// CPU-forward-pass oracle from (RunGreedyDecodeLoop, EmbedEntry), never a substitute for calling
// this suite's OWN sslm_model_map (that call is the subject under test in C2's own cells, and
// remains RED BY LINK independent of this helper).
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

// Parses this suite's own argv convention: <binary> [--model=PATH] [--modelvariant=PATH]
// [--adapter=PATH] [--foreignmodel=PATH] [--modeltok=PATH]. Every flag optional; a cell whose
// fixture is missing SKIPs rather than asserting against nothing.
inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model=")) g_model_path = v;
		else if (const char* v = take("--modelvariant=")) g_model_variant_path = v;
		else if (const char* v = take("--adapter=")) g_adapter_path = v;
		else if (const char* v = take("--foreignmodel=")) g_foreign_model_path = v;
		else if (const char* v = take("--modeltok=")) g_model_tok_path = v;
	}
}

// A KNOWN-VALID sslm_config (design Sec7.1, revised buffer model, commit fab235c1c6) for any
// cell that needs a real, accepted workspace-sizing call -- since Sec7.1's own domain makes an
// all-zero/null config HOSTILE input (sslm_workspace_size returns 0, sslm_workspace_create
// rejects SSLM_INVALID_ARGUMENT), a cell that expects SSLM_OK from sslm_workspace_create can no
// longer pass a null config as a convenience default the way a reserved-only GPU-ABI-style
// config would allow. max_batch/max_chunk_budget are sized generously (16/256) for any cell
// that composes a handful of sequences or a few hundred prompt tokens in one call;
// max_layer_budget is clamped to the real model's own num_hidden_layers, the one field whose
// valid domain depends on the artifact.
inline sslm_config ValidWorkspaceConfig(int32_t num_hidden_layers) {
	sslm_config c{};
	c.max_batch = 16;
	c.max_chunk_budget = 256;
	c.max_layer_budget = num_hidden_layers;
	c.reserved = 0;
	return c;
}

// The C4/dim6/dim10 CPU-forward-pass oracle -- the SAME construction
// tests/t2112-gpu-1p0-red-suite/fixture_common.h's own CpuOracleModel/StepCpu establishes,
// reused verbatim (StandardsDocument.md Sec6.6: one real implementation, not a second drifting
// copy) because this ABI's own dim6 obligation (design Sec10 dim6) is the identical
// "this ABI's output bit-equals the direct-engine call" consistency oracle, one layer up from
// where T-2112 already proved the same construction out. A full greedy decode loop (not the
// per-layer RunLayerLoop alone) is what design Sec9 C4's own gate names
// ("RunGreedyDecodeLoop's own direct-call output bit-for-bit"), so this oracle drives that
// entry point directly.
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
};

// Marshals a real SslmModelView into the CPU-side, already-materialized LayerWeights[] array
// RunGreedyDecodeLoop consumes directly -- byte-for-byte the same marshal sequence
// tools/sslm_generate.cpp's own driver runs (embed/final_norm/head resolution, including the
// tie_word_embeddings branch tools/sslm_generate.cpp's own comment names: "a fact about THIS
// artifact, not a fact this driver may assume").
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

// Runs a full greedy-decode prompt->N-tokens pass through the direct internal engine call
// (superslm::RunGreedyDecodeLoop) -- the C4/dim6/dim10 consistency oracle this ABI's own
// sslm_prefill+sslm_decode_step sequence is checked against, byte-for-byte the same call
// tools/sslm_generate.cpp's own driver makes.
inline superslm::SslmForwardStatus RunGreedyOracle(
    const CpuOracleModel& m, const int32_t* prompt_tokens, size_t prompt_len,
    size_t max_new_tokens, std::vector<int32_t>* out_tokens, size_t* out_tokens_produced,
    superslm::SslmDecodeStopReason* out_stop_reason) {
	const size_t kv_bytes = static_cast<size_t>(m.num_hidden_layers) *
	                         static_cast<size_t>(m.context_cap) * m.num_kv_heads * m.head_dim * 2;
	std::vector<uint8_t> workspace(kv_bytes);
	std::vector<int8_t> hidden_codes(m.hidden_size);
	superslm::SequenceLayerState seq{};
	seq.hidden_codes = hidden_codes.data();
	out_tokens->assign(max_new_tokens, 0);
	std::vector<int32_t> out_logit_rows(max_new_tokens * static_cast<size_t>(m.vocab_size));
	return superslm::RunGreedyDecodeLoop(
	    seq, m.layers.data(), m.num_hidden_layers, m.hidden_size, m.head_dim, m.num_kv_heads,
	    m.intermediate_size, m.context_cap, *m.rope_tables, prompt_tokens, prompt_len,
	    m.embed_weights, m.embed_site_constant, m.final_norm_gain.data(),
	    m.final_norm_site_constant, m.head_weights, m.vocab_size, /*stop_ids=*/nullptr,
	    /*stop_count=*/0, max_new_tokens, workspace.data(), workspace.size(), out_tokens->data(),
	    out_logit_rows.data(), out_tokens->size(), out_tokens_produced, out_stop_reason,
	    m.kv_precision, m.option_g_fused_k_landing);
}

#endif  // SSLM_T2138_FIXTURE_COMMON_H
