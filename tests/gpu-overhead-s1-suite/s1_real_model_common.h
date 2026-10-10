// GPU per-token overhead plan rev 3.1, slice S1: the shared harness for the real-artifact S1
// cells (cell_s1_real_model.cpp) and for the v1.11.0 SLM5 blob writer (write_v1p11_blob.cpp).
//
// Everything in this header uses only the 1.0 GPU API and bench accessors as they stand at v1.11.0
// -- no timing-mode seam -- so write_v1p11_blob.cpp builds against the v1.11.0 tree unchanged and
// the blob it writes is the "v1.9.0-saved blob" of cell 9.1, re-anchored to v1.11.0 (the common
// brief's re-anchor: v1.11.0's recorder is byte-for-byte v1.9.0's).
//
// The CHECK/SKIP convention is tests/t2178-gpu-batched-prefill-red-suite/fixture_common.h's: a
// cell whose artifact argv did not supply SKIPs by name rather than passing.
#ifndef SSLM_GPU_OVERHEAD_S1_COMMON_H
#define SSLM_GPU_OVERHEAD_S1_COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/checked_chain_funnel.h"
#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/sha256.h"

using enum SslmGpuStatus;

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

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

namespace gpu_overhead_s1_suite {

inline bool ReadWholeFile(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const bool ok = sz <= 0 || std::fread(out->data(), 1, out->size(), f) == out->size();
	std::fclose(f);
	return ok;
}

inline bool WriteWholeFile(const std::string& path, const std::vector<uint8_t>& bytes) {
	std::FILE* f = std::fopen(path.c_str(), "wb");
	if (!f) return false;
	const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
	std::fclose(f);
	return ok;
}

inline bool LoadRealModel(const std::string& path, superslm::SslmModelView* view,
                          std::vector<uint8_t>* bytes, std::string* err) {
	if (!ReadWholeFile(path, bytes)) {
		if (err) *err = "cannot read " + path;
		return false;
	}
	return superslm::SslmModel::Load(bytes->data(), bytes->size(), *view, err) ==
	       superslm::SslmModelStatus::Ok;
}

inline bool ModelHasQkNorm(const superslm::SslmModelView& view) {
	// The 1.0 map derives the same flag from MarshalLayer's "layer<l>.q_norm.gain" /
	// "layer<l>.k_norm.gain" tensors (any layer); a Qwen3-shaped model carries them on every layer.
	for (uint32_t l = 0; l < view.config.num_hidden_layers; ++l) {
		const std::string prefix = "layer" + std::to_string(l) + ".";
		if (view.weights.Tensor(prefix + "q_norm.gain") != nullptr ||
		    view.weights.Tensor(prefix + "k_norm.gain") != nullptr) {
			return true;
		}
	}
	return false;
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

// One decode call at `budget` dispatches, drained.
inline SslmGpuStatus StepBlocking(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, uint32_t budget) {
	const SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq, nullptr, budget);
	if (st != SSLM_OK) return st;
	return Drain(ctx, seq);
}

inline bool SaveBlob(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, std::vector<uint8_t>* blob) {
	size_t size = 0;
	if (sslm_gpu_seq_save(ctx, seq, nullptr, &size) != SSLM_OK || size == 0) return false;
	blob->assign(size, 0);
	return sslm_gpu_seq_save(ctx, seq, blob->data(), &size) == SSLM_OK && size == blob->size();
}

inline std::string Sha256Hex(const std::vector<uint8_t>& bytes) {
	uint8_t digest[32];
	superslm::Sha256Hash(bytes.data(), bytes.size(), digest);
	static const char* kHex = "0123456789abcdef";
	std::string hex;
	for (uint8_t b : digest) {
		hex += kHex[b >> 4];
		hex += kHex[b & 15];
	}
	return hex;
}

// The fixed SLM5 procedure cells 8.2 and 9.1 share (and write_v1p11_blob.cpp runs at v1.11.0):
// prefill three prompt tokens, embed a fourth, advance it four layers in one call (a mid-token
// slice boundary), and save. `unit` is the model's dispatches per layer.
inline constexpr int32_t kProcedurePrompt[4] = {872, 13, 576, 1879};
inline constexpr int64_t kProcedureContextCap = 64;
inline constexpr uint32_t kProcedureSliceLayers = 4;

inline bool RunSlm5ProcedureToMidToken(SslmGpuContext* ctx, SslmGpuModelHandle* model,
                                       uint32_t num_hidden_layers, uint32_t unit,
                                       SslmGpuSequenceHandle** out_seq, std::vector<uint8_t>* blob,
                                       std::string* err) {
	*out_seq = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_seq_create(ctx, model, kProcedureContextCap, &seq) != SSLM_OK || !seq) {
		if (err) *err = "sslm_gpu_seq_create failed";
		return false;
	}
	SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, kProcedurePrompt, 3,
	                                                      unit * num_hidden_layers);
	if (st == SSLM_OK) st = sslm_gpu_seq_embed_token(ctx, seq, kProcedurePrompt[3]);
	if (st == SSLM_OK) st = StepBlocking(ctx, seq, unit * kProcedureSliceLayers);
	if (st != SSLM_OK) {
		if (err) *err = "procedure call returned status " + std::to_string(static_cast<int>(st));
		sslm_gpu_seq_release(ctx, seq);
		return false;
	}
	if (*SslmGpuSeqHandleLayerIndexForBench(seq) != kProcedureSliceLayers) {
		if (err) *err = "procedure did not stop at layer 4";
		sslm_gpu_seq_release(ctx, seq);
		return false;
	}
	if (!SaveBlob(ctx, seq, blob)) {
		if (err) *err = "sslm_gpu_seq_save failed";
		sslm_gpu_seq_release(ctx, seq);
		return false;
	}
	*out_seq = seq;
	return true;
}

// From a mid-token sequence: finish the token's remaining layers, take its token, then decode
// `more` greedy tokens. Returns every token and the final blob.
inline bool ContinueAndDecode(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq,
                              uint32_t num_hidden_layers, uint32_t unit, int more,
                              std::vector<int32_t>* tokens, std::vector<uint8_t>* final_blob,
                              std::string* err) {
	const uint32_t at = *SslmGpuSeqHandleLayerIndexForBench(seq);
	SslmGpuStatus st = StepBlocking(ctx, seq, unit * (num_hidden_layers - at));
	int32_t token = -1;
	if (st == SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &token);
	if (st != SSLM_OK) {
		if (err) *err = "finishing the restored token returned " + std::to_string(static_cast<int>(st));
		return false;
	}
	tokens->push_back(token);
	for (int i = 0; i < more; ++i) {
		int32_t next = -1;
		st = SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, token, unit * num_hidden_layers, &next);
		if (st != SSLM_OK) {
			if (err) *err = "decode step " + std::to_string(i) + " returned " + std::to_string(static_cast<int>(st));
			return false;
		}
		tokens->push_back(next);
		token = next;
	}
	if (!SaveBlob(ctx, seq, final_blob)) {
		if (err) *err = "final sslm_gpu_seq_save failed";
		return false;
	}
	return true;
}

}  // namespace gpu_overhead_s1_suite

#endif  // SSLM_GPU_OVERHEAD_S1_COMMON_H
