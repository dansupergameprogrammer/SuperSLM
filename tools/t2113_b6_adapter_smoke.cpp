// T-2113 (B6): the bench proof for design Sec10 B6's own residency/guard gate --
// `sslm_gpu_adapter_map`/`sslm_gpu_adapter_unmap`, adapter residency, base-hash validation, and
// the AdapterModelMismatch guard on `sslm_decode_step_gpu`. Same convention as
// t2113_b1_context_smoke.cpp/t2113_b2_model_smoke.cpp/t2113_b3_sequence_smoke.cpp -- a dedicated
// build-seat bench tool, not part of Curie's T-2112 red suite (blocked on D-SLM3367, the missing
// production token-embed entry point, Claude/Brunel/t2113-1p0-core-build-2026-08-15.md Sec8.3).
//
// NOT proven here (named, not silently claimed): the GPU dispatch-recording path does not yet
// read a bound adapter's own resident buffers (Sec9 of the build log) -- so this tool proves
// residency/guards/lifecycle, never a numerical divergence from a bound adapter. That gate
// (design Sec10 B6's own "runtime-vs-baked, runtime-vs-base divergence... reproduced on the GPU
// path") is NOT reachable until the delta-application dispatches exist.
//
// Usage: t2113_b6_adapter_smoke <model1p5b.sslm> <model0p5b.sslm> <adapter.sslm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

using namespace superslm;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
	do {                                                                         \
		++g_checks;                                                                \
		if (!(cond)) {                                                             \
			++g_failures;                                                            \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);        \
		}                                                                           \
	} while (0)

static bool LoadModel(const std::string& path, SslmModelView* out_view,
                       std::vector<uint8_t>* out_bytes, std::string* err) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) { *err = "could not open " + path; return false; }
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) { *err = "short read on " + path; return false; }
	} else {
		std::fclose(f);
	}
	const SslmModelStatus st = SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, err);
	return st == SslmModelStatus::Ok;
}

int main(int argc, char** argv) {
	if (argc < 4) {
		std::fprintf(stderr, "usage: t2113_b6_adapter_smoke <model1p5b.sslm> <model0p5b.sslm> <adapter.sslm>\n");
		return 2;
	}
	const std::string model1p5b_path = argv[1];
	const std::string model0p5b_path = argv[2];
	const std::string adapter_path = argv[3];

	std::string err;
	std::vector<uint8_t> bytes_1p5b, bytes_0p5b, bytes_adapter;
	SslmModelView view_1p5b{}, view_0p5b{}, view_adapter{};
	CHECK(LoadModel(model1p5b_path, &view_1p5b, &bytes_1p5b, &err), ("load 1.5B: " + err).c_str());
	CHECK(LoadModel(model0p5b_path, &view_0p5b, &bytes_0p5b, &err), ("load 0.5B: " + err).c_str());
	CHECK(LoadModel(adapter_path, &view_adapter, &bytes_adapter, &err), ("load adapter: " + err).c_str());

	GpuContextConfig cctx{};
	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(cctx, &ctx) == SSLM_OK, "context_create failed");
	if (!ctx) return 1;

	GpuResidencyConfig rcfg{};
	SslmGpuModelHandle* model_1p5b = nullptr;
	SslmGpuModelHandle* model_0p5b = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &view_1p5b, rcfg, &model_1p5b) == SSLM_OK, "model_map 1.5B failed");
	CHECK(sslm_gpu_model_map(ctx, &view_0p5b, rcfg, &model_0p5b) == SSLM_OK, "model_map 0.5B failed");

	// 1. Base-hash mismatch: the shopkeeper adapter is trained against the 1.5B base, never the
	// 0.5B -- mapping it against model_0p5b must be rejected via AdapterBaseHashMismatch (design
	// Sec9), *out_adapter left null, no residency uploaded.
	{
		SslmGpuAdapterHandle* bad = nullptr;
		const SslmGpuStatus st = sslm_gpu_adapter_map(ctx, model_0p5b, &view_adapter, &bad);
		CHECK(st == SSLM_ADAPTER_BASE_HASH_MISMATCH,
		      "adapter_map against the WRONG base did not return SSLM_ADAPTER_BASE_HASH_MISMATCH");
		CHECK(bad == nullptr, "adapter_map left *out_adapter non-null on a rejected map");
	}

	// 2. Real map against the correct (1.5B) base succeeds.
	SslmGpuAdapterHandle* adapter = nullptr;
	CHECK(sslm_gpu_adapter_map(ctx, model_1p5b, &view_adapter, &adapter) == SSLM_OK,
	      "adapter_map against the correct base did not return SSLM_OK");
	CHECK(adapter != nullptr, "adapter_map left *out_adapter null on SSLM_OK");

	// 3. AdapterModelMismatch: a sequence created against model_0p5b, decoded with the
	// 1.5B-mapped adapter bound, must be rejected via the per-call guard (design Sec5.2/Sec9,
	// pointer-equality against the model handle each side carries) -- never silently accepted.
	{
		SslmGpuSequenceHandle* seq_0p5b = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model_0p5b, 64, &seq_0p5b) == SSLM_OK,
		      "seq_create (0.5B) failed");
		if (seq_0p5b) {
			const SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq_0p5b, adapter, 24u);
			CHECK(st == SSLM_ADAPTER_MODEL_MISMATCH,
			      "decode with a cross-model adapter binding did not return SSLM_ADAPTER_MODEL_MISMATCH");
			CHECK(sslm_gpu_seq_release(ctx, seq_0p5b) == SSLM_OK, "seq_release (0.5B) failed");
		}
	}

	// 4. NULL-adapter is unaffected: a sequence against the SAME (1.5B) model the adapter is
	// bound to, decoded with adapter_or_null=nullptr, is accepted (base-only path, unperturbed
	// by this section's own additions -- confirmed by the guard NOT firing).
	{
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, model_1p5b, 64, &seq) == SSLM_OK, "seq_create (1.5B) failed");
		if (seq) {
			const SslmGpuStatus st = sslm_decode_step_gpu(ctx, seq, nullptr, 672u);
			CHECK(st != SSLM_ADAPTER_MODEL_MISMATCH, "NULL adapter incorrectly triggered the guard");
			CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK, "seq_release (1.5B) failed");
		}
	}

	// 5. Adapter unmap is Ok, unmap(nullptr) is a documented no-op, and unmapping does not
	// disturb the context's own ContextHasLiveHandles accounting for the still-live models.
	CHECK(sslm_gpu_adapter_unmap(ctx, adapter) == SSLM_OK, "adapter_unmap failed");
	CHECK(sslm_gpu_adapter_unmap(ctx, nullptr) == SSLM_OK, "adapter_unmap(nullptr) is not a no-op");

	CHECK(sslm_gpu_model_unmap(ctx, model_1p5b) == SSLM_OK, "model_unmap 1.5B failed");
	CHECK(sslm_gpu_model_unmap(ctx, model_0p5b) == SSLM_OK, "model_unmap 0.5B failed");
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK, "context_destroy failed (live handles leaked?)");

	std::printf("t2113_b6_adapter_smoke: checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures ? 1 : 0;
}
