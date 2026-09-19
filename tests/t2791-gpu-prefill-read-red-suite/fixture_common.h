// T-2791 (Curie) -- shared harness for the SuperSLM 1.6.0 GPU embedding-read red suite.
// Realizes the Coverage Model of the GPU-path plan's Stage 1 (the prefill snapshot and
// `sslm_gpu_seq_read_prefill_final_hidden`, plan Sec3.1/Sec3.4) and its transition census
// (plan Sec2.6). The test-design record, per cell with its oracle and its red reading, is
// Claude/Curie/t2791-gpu-read-red-2026-09-18.md in the records tree.
//
// Conventions reused rather than invented: the CHECK/CHECK_MSG/SKIP_MSG trio and argv-supplied
// artifact paths are tests/t2112-gpu-1p0-red-suite/fixture_common.h's; the batch-built,
// documented-local shape (no GPU CI runner, D-SLM3432) is tests/t2178-gpu-batched-prefill-red-
// suite's; the independent SchemaMasks parse is tests/t2130-g5-red-suite/fixture_common.h's.
//
// HOW THIS SUITE IS RED BEFORE THE BUILD. The verb, the width query and the two statuses do not
// exist at v1.5.0. The verb and the width query are declared below with exactly the signatures
// plan Sec3.1 gives, so every cell COMPILES against v1.5.0 and fails to LINK (LNK2019 on those
// two symbols). The two statuses are referenced through the ordinals plan Sec3.1 fixes ("appended
// LAST"): cell_status_ordinals.cpp pins the enumerator names to those ordinals and is the one
// cell that is red by COMPILE. Once include/superslm/gpu_1p0.h declares the verb, the
// redeclarations below must match it exactly: a signature that drifts from Sec3.1 turns every
// cell into a compile error, which is the intended pin.
#ifndef SSLM_T2791_FIXTURE_COMMON_H
#define SSLM_T2791_FIXTURE_COMMON_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"  // superslm::CarriedScale
#include "superslm/forward_sites.h"         // RmsNormSite, LogitsSite, EmbedEntry, RunLayerLoop
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_1p0_g5_bridge.h"
#include "superslm/gpu_port.h"              // superslm_gpu::DispatchesPerLayer
#include "superslm/layer_marshal.h"         // WidenGainToInt32, ReadCarriedScale, MarshalLayer
#include "superslm/model.h"
#include "superslm/schema_masks.h"
#include "superslm/sha256.h"

using enum SslmGpuStatus;

// ---------------------------------------------------------------------------------------------
// The surface under test, exactly as plan Sec3.1 specifies it.
// ---------------------------------------------------------------------------------------------
SslmGpuStatus sslm_gpu_model_hidden_size(const SslmGpuModelHandle* model,
                                         uint32_t* out_hidden_size) noexcept;
SslmGpuStatus sslm_gpu_seq_read_prefill_final_hidden(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                                                     int8_t* out_codes, size_t out_capacity,
                                                     size_t* out_required, int64_t* out_scale_m,
                                                     int64_t* out_scale_e) noexcept;

// At v1.5.0 the last SslmGpuStatus enumerator is SSLM_GPU_ALLOCATION_FAILED == 16
// (cell_status_ordinals.cpp pins all seventeen). Plan Sec3.1 appends SSLM_OUTPUT_BUFFER_TOO_SMALL
// then SSLM_PREFILL_HIDDEN_UNAVAILABLE, so they are 17 and 18.
constexpr SslmGpuStatus kOutputBufferTooSmall = static_cast<SslmGpuStatus>(17u);
constexpr SslmGpuStatus kPrefillHiddenUnavailable = static_cast<SslmGpuStatus>(18u);

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------
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

inline int FinishSuite(const char* name) {
	std::printf("%s: checks=%d failures=%d skips=%d -> %s\n", name, GChecks, GFailures, GSkips,
	            GFailures == 0 ? "PASS" : "FAIL");
	return GFailures == 0 ? 0 : 1;
}

// argv convention, every flag optional; a cell whose artifact is not supplied SKIPs:
//   --qwen3=PATH       the 1.5.0 Qwen3-Embedding-0.6B artifact (plan Sec2.6's executed cell)
//   --synthetic=PATH   the synthetic fused-K fixture (plan Sec3.4 row 4): SuperEmbedder's
//                      u1_pair_model.sslm, SHA-256
//                      a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70, 729,680 bytes,
//                      written by tests/fixtures/regenerate_tokenizer_pair_fixture.py at SuperEmbedder
//                      701ef1a or later against v1.5.0 (hidden 32, 8 layers, 4 heads and 2 K/V heads at
//                      head_dim 128, QK-norm fused-K, vocab 384, cap 64, tied embeddings)
//   --a2fn=PATH        fixture A2-fn (plan Sec3.4 row 5): the --synthetic fixture with its final_norm
//                      composition constant patched, built by make_a2fn_fixture.py
//   --g5fixture=PATH   a schema-bearing artifact that loads at 1.5.0 (plan Sec3.4 row 5)
//   --commission=PATH  T-2780's 256-row token file (plan Sec3.4 row 10)
static std::string g_qwen3_path;
static std::string g_synthetic_path;
static std::string g_a2fn_path;
static std::string g_g5_path;
static std::string g_commission_path;
static std::string g_schema_name = "shopkeeper_intent_extraction";

inline void ParseFixtureArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--qwen3=")) g_qwen3_path = v;
		else if (const char* v = take("--synthetic=")) g_synthetic_path = v;
		else if (const char* v = take("--a2fn=")) g_a2fn_path = v;
		else if (const char* v = take("--g5fixture=")) g_g5_path = v;
		else if (const char* v = take("--commission=")) g_commission_path = v;
		else if (const char* v = take("--schema=")) g_schema_name = v;
	}
}

inline const char* StatusName(SslmGpuStatus s) {
	switch (static_cast<uint32_t>(s)) {
		case 0: return "SSLM_OK";
		case 1: return "SSLM_DISPATCH_BUDGET_TOO_SMALL";
		case 2: return "SSLM_BUSY";
		case 3: return "SSLM_CONTEXT_HAS_LIVE_HANDLES";
		case 4: return "SSLM_MODEL_HAS_LIVE_SEQUENCES";
		case 5: return "SSLM_ADAPTER_MODEL_MISMATCH";
		case 6: return "SSLM_ADAPTER_BASE_HASH_MISMATCH";
		case 7: return "SSLM_SEQUENCE_KV_BUFFER_MISMATCH";
		case 8: return "SSLM_DEVICE_LOST";
		case 9: return "SSLM_BATCH_BUDGET_EXHAUSTED";
		case 10: return "SSLM_TOKEN_ID_OUT_OF_RANGE";
		case 11: return "SSLM_SEQUENCE_REJECTED";
		case 12: return "SSLM_RESTORE_MODEL_MISMATCH";
		case 13: return "SSLM_MODEL_HAS_LIVE_ADAPTERS";
		case 14: return "SSLM_ADAPTER_HAS_BOUND_SEQUENCES";
		case 15: return "SSLM_GPU_SHADER_BINARY_STALE";
		case 16: return "SSLM_GPU_ALLOCATION_FAILED";
		case 17: return "SSLM_OUTPUT_BUFFER_TOO_SMALL";
		case 18: return "SSLM_PREFILL_HIDDEN_UNAVAILABLE";
		default: return "UNKNOWN_STATUS";
	}
}

inline bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, static_cast<size_t>(sz), f) : 0;
	std::fclose(f);
	return n == out->size();
}

// ---------------------------------------------------------------------------------------------
// One mapped artifact, plus the host-side inputs of the ORACLE. The oracle for "the frame of a
// completed prefill" is final_norm applied to that prefill's completed last-position residual:
// RmsNormSite with the artifact's own final_norm.gain (widened) and "final_norm" composition
// constant, read here from an SslmModelView this harness parsed itself, over the residual the
// bench accessor exposes IMMEDIATELY after the prefill returned SSLM_OK -- the one moment the
// live residual is, by plan Sec3.1's own definition, the snapshot's content. None of those
// inputs is produced by the snapshot or the read under test. (Row 6's cell closes the remaining
// link: that residual equals the CPU forward's.)
// ---------------------------------------------------------------------------------------------
struct GpuModelFixture {
	std::string path;
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	size_t hidden = 0;
	uint32_t layers = 0;
	int32_t vocab = 0;
	int64_t model_cap = 0;
	bool has_qk_norm = false;
	uint32_t one_layer_budget = 0;
	std::vector<int32_t> final_gain;
	superslm::CarriedScale final_const{};
	bool ok = false;

	bool Open(const std::string& p, SslmGpuContext* shared_ctx) {
		path = p;
		std::string err;
		if (!ReadFileBytes(p, &bytes)) {
			std::printf("fixture: cannot read %s\n", p.c_str());
			return false;
		}
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) !=
		    superslm::SslmModelStatus::Ok) {
			std::printf("fixture: load %s failed: %s\n", p.c_str(), err.c_str());
			return false;
		}
		hidden = view.config.hidden_size;
		layers = view.config.num_hidden_layers;
		vocab = static_cast<int32_t>(view.config.vocab_size);
		model_cap = static_cast<int64_t>(view.config.context_cap);
		for (uint32_t l = 0; l < layers; ++l) {
			if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain") != nullptr) {
				has_qk_norm = true;
			}
		}
		one_layer_budget = superslm_gpu::DispatchesPerLayer(has_qk_norm);
		const superslm::SslmTensorView* g = view.weights.Tensor("final_norm.gain");
		if (!g) {
			std::printf("fixture: %s has no final_norm.gain\n", p.c_str());
			return false;
		}
		final_gain = superslm_marshal::WidenGainToInt32(*g);
		bool cok = true;
		final_const =
		    superslm_marshal::ReadCarriedScale(view.composition_constants, "final_norm", &cok);
		if (!cok) {
			std::printf("fixture: %s has no final_norm constant\n", p.c_str());
			return false;
		}
		ctx = shared_ctx;
		if (sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != SSLM_OK || !model) {
			std::printf("fixture: sslm_gpu_model_map(%s) failed\n", p.c_str());
			return false;
		}
		ok = true;
		std::printf("fixture: %s hidden=%zu layers=%u vocab=%d cap=%lld qk_norm=%d schemas=%d\n",
		            p.c_str(), hidden, layers, vocab, static_cast<long long>(model_cap),
		            has_qk_norm ? 1 : 0, SslmGpuModelHasSchemasForG5Bridge(model) ? 1 : 0);
		return true;
	}

	void Close() {
		if (model) sslm_gpu_model_unmap(ctx, model);
		model = nullptr;
	}

	// Three prompt tokens and a two-token continuation, valid for this vocabulary. On the Qwen3
	// artifact they are the census's own (plan Sec2.6: P = {1000, 2000, 151643}, Q = {42, 43}).
	std::vector<int32_t> TokensP() const {
		if (vocab > 151643) return {1000, 2000, 151643};
		return {vocab / 7, vocab / 3, vocab - 1};
	}
	std::vector<int32_t> TokensQ() const {
		if (vocab > 151643) return {42, 43};
		return {vocab / 5, vocab / 2};
	}
	int32_t SomeToken() const { return vocab > 151643 ? 42 : vocab / 5; }
	int32_t OtherToken() const { return vocab > 151643 ? 43 : vocab / 2; }
	// `n` distinct-enough valid tokens starting at `base`.
	std::vector<int32_t> Run(size_t n, int32_t base) const {
		std::vector<int32_t> t;
		for (size_t i = 0; i < n; ++i) t.push_back(static_cast<int32_t>((base + 37 * i) % vocab));
		return t;
	}
};

// A read outcome, with every out-parameter captured so a cell can assert what was, and was not,
// written. Buffers are pre-filled with sentinels.
constexpr int8_t kCodeSentinel = 0x5A;
constexpr int64_t kScaleMSentinel = INT64_C(0x1111111111111111);
constexpr int64_t kScaleESentinel = INT64_C(0x2222222222222222);
constexpr size_t kRequiredSentinel = static_cast<size_t>(0x3333333333333333ull);

struct Frame {
	SslmGpuStatus status = SSLM_DEVICE_LOST;
	std::vector<int8_t> codes;  // capacity-sized, sentinel-filled before the call
	int64_t m = kScaleMSentinel, e = kScaleESentinel;
	size_t required = kRequiredSentinel;

	bool IsOk() const { return status == SSLM_OK; }
	// The first `n` codes plus the scale equal `o`'s (a frame compared byte for byte).
	bool SameFrame(const Frame& o, size_t n) const {
		if (status != SSLM_OK || o.status != SSLM_OK) return false;
		if (codes.size() < n || o.codes.size() < n) return false;
		return std::equal(codes.begin(), codes.begin() + n, o.codes.begin()) && m == o.m && e == o.e;
	}
	bool OutputsUntouched() const {
		for (int8_t c : codes) {
			if (c != kCodeSentinel) return false;
		}
		return m == kScaleMSentinel && e == kScaleESentinel;
	}
};

inline size_t DiffCodes(const Frame& a, const Frame& b, size_t n) {
	size_t d = 0;
	for (size_t i = 0; i < n && i < a.codes.size() && i < b.codes.size(); ++i) d += a.codes[i] != b.codes[i];
	return d;
}

inline Frame ReadVerb(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, size_t capacity) {
	Frame f;
	f.codes.assign(capacity, kCodeSentinel);
	f.status = sslm_gpu_seq_read_prefill_final_hidden(ctx, seq, capacity ? f.codes.data() : nullptr,
	                                                  capacity, &f.required, &f.m, &f.e);
	return f;
}

inline Frame ReadVerb(const GpuModelFixture& fx, SslmGpuSequenceHandle* seq) {
	return ReadVerb(fx.ctx, seq, fx.hidden);
}

// The oracle frame: final_norm of the sequence's live residual, computed here. Call it only
// where the live residual is authoritative (immediately after a prefill returned SSLM_OK).
inline Frame OracleFromLive(const GpuModelFixture& fx, SslmGpuSequenceHandle* seq) {
	Frame f;
	f.codes.assign(fx.hidden, 0);
	const int8_t* live = SslmGpuSeqHandleHiddenCodesForBench(seq);
	const superslm::CarriedScale live_scale = *SslmGpuSeqHandleHiddenScaleForBench(seq);
	superslm::CarriedScale out{};
	const superslm::SslmForwardStatus st =
	    superslm::RmsNormSite(live, fx.final_gain.data(), fx.hidden, live_scale, fx.final_const,
	                          f.codes.data(), &out, "final_norm");
	f.status = st == superslm::SslmForwardStatus::Ok ? SSLM_OK : SSLM_SEQUENCE_REJECTED;
	f.m = out.m;
	f.e = out.e;
	f.required = fx.hidden;
	return f;
}

inline SslmGpuStatus Prefill(const GpuModelFixture& fx, SslmGpuSequenceHandle* seq,
                             const std::vector<int32_t>& t, uint32_t budget = 0) {
	return SslmGpuSeqPrefillPromptForG5Bridge(fx.ctx, seq, t.data(), static_cast<int32_t>(t.size()),
	                                          budget ? budget : fx.one_layer_budget);
}

inline SslmGpuStatus Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq) {
	int32_t ready = 0;
	SslmGpuStatus out_status = SSLM_OK;
	SslmGpuStatus st = SSLM_OK;
	while (!ready) {
		st = sslm_gpu_ready(ctx, seq, /*block=*/1, &ready, &out_status);
		if (st != SSLM_OK) return st;
	}
	return out_status;
}

inline uint32_t LayerIndex(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleLayerIndexForBench(s); }
inline int64_t ContextLength(SslmGpuSequenceHandle* s) { return *SslmGpuSeqHandleContextLengthForBench(s); }

// Every host-visible piece of a sequence's state the bench bridge exposes -- used to show a
// read writes nothing on the sequence.
struct SeqState {
	std::vector<int8_t> codes;
	int64_t m = 0, e = 0;
	uint32_t layer = 0;
	uint64_t kv_sat = 0;
	int64_t ctxlen = 0;
	uint32_t walk = 0;
	bool operator==(const SeqState& o) const {
		return codes == o.codes && m == o.m && e == o.e && layer == o.layer && kv_sat == o.kv_sat &&
		       ctxlen == o.ctxlen && walk == o.walk;
	}
};
inline SeqState CaptureState(SslmGpuSequenceHandle* s) {
	SeqState st;
	const size_t h = SslmGpuSeqHandleHiddenSizeForBench(s);
	const int8_t* c = SslmGpuSeqHandleHiddenCodesForBench(s);
	st.codes.assign(c, c + h);
	st.m = SslmGpuSeqHandleHiddenScaleForBench(s)->m;
	st.e = SslmGpuSeqHandleHiddenScaleForBench(s)->e;
	st.layer = LayerIndex(s);
	st.kv_sat = *SslmGpuSeqHandleKvSaturationForBench(s);
	st.ctxlen = ContextLength(s);
	st.walk = SslmGpuSeqWalkStateForG5Bridge(s);
	return st;
}

// SHA-256 of an arbitrary byte span, hex.
inline std::string Sha256Hex(const void* data, size_t n) {
	uint8_t d[32];
	superslm::Sha256Hash(static_cast<const uint8_t*>(data), n, d);
	return superslm::ToHex(d);
}

// ---------------------------------------------------------------------------------------------
// Independent SchemaMasks walk (tests/t2130-g5-red-suite/fixture_common.h's technique): a second
// parse of the artifact bytes, so DFA chains are built without the GPU model handle's own table.
// ---------------------------------------------------------------------------------------------
struct IndependentSchema {
	superslm::SslmModelView view{};
	superslm::SchemaMasksTable table;
	const superslm::SchemaEntry* entry = nullptr;

	bool Build(const std::vector<uint8_t>& bytes, const std::string& name) {
		std::string err;
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) !=
		    superslm::SslmModelStatus::Ok) {
			return false;
		}
		const superslm::SslmSectionView* s = view.Section(superslm::SslmSectionType::SchemaMasks);
		if (!s) return false;
		if (!superslm::SchemaMasksTable::Parse(s->data, s->byte_size, view.config.vocab_size, table,
		                                       &err)) {
			return false;
		}
		entry = table.ByName(name);
		return entry != nullptr;
	}
	static uint32_t Le32(const uint8_t* p) {
		return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
		       (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
	}
	// The first legal token at `state`, or -1.
	int32_t FirstLegal(uint32_t state) const {
		const uint32_t b = Le32(entry->state_offsets_le + static_cast<size_t>(state) * 4);
		const uint32_t en = Le32(entry->state_offsets_le + static_cast<size_t>(state + 1) * 4);
		return b < en ? static_cast<int32_t>(Le32(entry->transitions_le + static_cast<size_t>(b) * 8)) : -1;
	}
	bool Next(uint32_t state, int32_t tok, uint32_t* next) const {
		return table.Transition(*entry, state, static_cast<uint32_t>(tok), next);
	}
	// The lowest valid token id that is NOT legal at `state`, or -1.
	int32_t FirstIllegal(uint32_t state) const {
		for (int32_t t = 0; t < static_cast<int32_t>(view.config.vocab_size); ++t) {
			if (!table.Transition(*entry, state, static_cast<uint32_t>(t), nullptr)) return t;
		}
		return -1;
	}
};

#endif  // SSLM_T2791_FIXTURE_COMMON_H
