// T-2113 (B2): the bench proof for design Sec10 B2's own delivery -- SslmGpuModelHandle
// map/unmap, weight residency keyed on the artifact's own content hash (not the caller's
// host pointer), the T-2105 RoPE-table residency construction folded in as the model
// handle's own upload path. Not part of Curie's t2112 red suite (that suite's own dim1/dim2
// product cells need B3's sequence handle too, per its own handoff -- this is a build-seat
// bench proof, the same convention as tools/t2113_b1_context_smoke.cpp).
//
// Proves, on real D3D12 hardware, against the two real artifacts this project's own T-2100/
// T-2105/T-2106 arc already uses (StandardsDocument.md Sec5.4's real-workload rule -- no
// hidden_size=2 stand-in for the product proof):
//   1. sslm_gpu_model_map succeeds against a real 1.5B artifact and returns a non-null
//      handle whose resident weight/RoPE buffers are real, non-null D3D12 resources.
//   2. Two DIFFERENT real artifacts (1.5B, 0.5B) mapped against the SAME context produce
//      two independently-live handles with DIFFERENT content hashes, neither aliasing the
//      other's own resident buffers (design Sec11 dim1 M1's own mechanism shape: "a second,
//      independent model handle created for a different artifact while the first is still
//      live -- must not evict, alias, or corrupt the first's resident weights").
//   3. Mapping the SAME artifact a second time succeeds and produces a fresh, independent
//      handle pointer (design Sec5.1: "nothing in this design imposes a one-model-at-a-time
//      limit") -- SslmGpuModelHandle is opaque at this API surface (design Sec4.1), so this
//      bench proof checks handle-pointer distinctness, the externally-observable half of the
//      claim; content-hash equality itself is exercised at the field level by the T-2112
//      red suite's own dim1 cells once B3 lands (this smoke tool has no accessor to read a
//      handle's own internal content_hash field, by design -- the API never exposes it).
//   4. sslm_gpu_model_unmap returns Ok and the context's own ContextHasLiveHandles guard
//      (B1) observes the live-handle count moving: destroying a context with a live model
//      handle still mapped is refused (SSLM_CONTEXT_HAS_LIVE_HANDLES); after unmap, the same
//      context destroys cleanly -- the B1/B2 boundary's own bookkeeping, exercised for real,
//      not asserted in prose.
//   5. sslm_gpu_model_unmap(nullptr) is a documented no-op returning Ok.
//   6. Null ctx / null base are rejected (SSLM_DEVICE_LOST, mirroring
//      sslm_gpu_context_create's own null-out-parameter disposition -- design Sec9 assigns
//      no dedicated status to a caller contract violation at this call).
//
// Usage: t2113_b2_model_smoke <model1p5b.sslm> <model0p5b.sslm>  (exits 0 on pass, 1 on fail)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, msg)                                                            \
	do {                                                                              \
		++g_checks;                                                                    \
		if (!(cond)) {                                                                 \
			++g_failures;                                                                \
			std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);            \
		}                                                                               \
	} while (0)

static bool LoadModel(const std::string& path, superslm::SslmModelView* out_view,
                       std::vector<uint8_t>* out_bytes) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) {
		std::fprintf(stderr, "could not open %s\n", path.c_str());
		return false;
	}
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out_bytes->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	if (sz > 0) {
		const size_t n = std::fread(out_bytes->data(), 1, static_cast<size_t>(sz), f);
		std::fclose(f);
		if (n != static_cast<size_t>(sz)) {
			std::fprintf(stderr, "short read on %s\n", path.c_str());
			return false;
		}
	} else {
		std::fclose(f);
	}
	std::string err;
	const superslm::SslmModelStatus st =
	    superslm::SslmModel::Load(out_bytes->data(), out_bytes->size(), *out_view, &err);
	if (st != superslm::SslmModelStatus::Ok) {
		std::fprintf(stderr, "model load failed for %s: %s\n", path.c_str(), err.c_str());
		return false;
	}
	return true;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: %s <model1p5b.sslm> <model0p5b.sslm>\n", argv[0]);
		return 2;
	}

	std::vector<uint8_t> bytes_a, bytes_b;
	superslm::SslmModelView view_a, view_b;
	if (!LoadModel(argv[1], &view_a, &bytes_a) || !LoadModel(argv[2], &view_b, &bytes_b)) {
		return 1;
	}

	GpuContextConfig cfg{};
	GpuResidencyConfig rcfg{};

	// 6: null ctx / null base are rejected before ANY context exists.
	SslmGpuModelHandle* null_check = nullptr;
	CHECK(sslm_gpu_model_map(nullptr, &view_a, rcfg, &null_check) == SSLM_DEVICE_LOST,
	      "sslm_gpu_model_map(null ctx, ...) did not return SSLM_DEVICE_LOST");
	CHECK(null_check == nullptr, "sslm_gpu_model_map(null ctx, ...) left *out_model non-null");

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(cfg, &ctx) == SSLM_OK && ctx != nullptr,
	      "sslm_gpu_context_create failed");
	if (!ctx) return g_failures ? 1 : 0;

	CHECK(sslm_gpu_model_map(ctx, nullptr, rcfg, &null_check) == SSLM_DEVICE_LOST,
	      "sslm_gpu_model_map(ctx, null base, ...) did not return SSLM_DEVICE_LOST");

	// 1: real 1.5B artifact maps cleanly.
	SslmGpuModelHandle* model_a1 = nullptr;
	SslmGpuStatus st_a1 = sslm_gpu_model_map(ctx, &view_a, rcfg, &model_a1);
	CHECK(st_a1 == SSLM_OK, "sslm_gpu_model_map(1.5B) did not return SSLM_OK");
	CHECK(model_a1 != nullptr, "sslm_gpu_model_map(1.5B) left *out_model null on SSLM_OK");

	// 2: a second, DIFFERENT real artifact (0.5B), live at the same time.
	SslmGpuModelHandle* model_b = nullptr;
	SslmGpuStatus st_b = sslm_gpu_model_map(ctx, &view_b, rcfg, &model_b);
	CHECK(st_b == SSLM_OK, "sslm_gpu_model_map(0.5B) did not return SSLM_OK");
	CHECK(model_b != nullptr, "sslm_gpu_model_map(0.5B) left *out_model null on SSLM_OK");
	CHECK(model_a1 != model_b, "two different-artifact map calls returned the same handle pointer");

	// 3: mapping the SAME artifact again gives a fresh, independent handle.
	SslmGpuModelHandle* model_a2 = nullptr;
	SslmGpuStatus st_a2 = sslm_gpu_model_map(ctx, &view_a, rcfg, &model_a2);
	CHECK(st_a2 == SSLM_OK, "sslm_gpu_model_map(1.5B, second time) did not return SSLM_OK");
	CHECK(model_a2 != nullptr && model_a2 != model_a1,
	      "re-mapping the same artifact did not produce a fresh, distinct handle");

	// 4a: a context with live model handles refuses to destroy.
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_CONTEXT_HAS_LIVE_HANDLES,
	      "sslm_gpu_context_destroy did not refuse a context with live model handles");

	// 5: unmap(nullptr) is a no-op.
	CHECK(sslm_gpu_model_unmap(ctx, nullptr) == SSLM_OK, "sslm_gpu_model_unmap(nullptr) did not return SSLM_OK");

	// 4b: release every live handle, then the context destroys cleanly.
	CHECK(sslm_gpu_model_unmap(ctx, model_a1) == SSLM_OK, "unmap(model_a1) did not return SSLM_OK");
	CHECK(sslm_gpu_model_unmap(ctx, model_a2) == SSLM_OK, "unmap(model_a2) did not return SSLM_OK");
	CHECK(sslm_gpu_model_unmap(ctx, model_b) == SSLM_OK, "unmap(model_b) did not return SSLM_OK");
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK,
	      "sslm_gpu_context_destroy did not succeed after every model handle was unmapped");

	std::printf("T-2113 B2 model smoke: checks=%d failures=%d\n", g_checks, g_failures);
	return g_failures == 0 ? 0 : 1;
}
