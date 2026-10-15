// Paged-KV plan (rev 16.2) §7 cell 9.8 [G5], leg 2: the OLD reader. Group g1e, step G1.
//
// Built against the v1.12.0 tree (the GPU slice's base; the dedicated restore of the tree before G5),
// never against the paged build: it uses only that tree's shipped surface (sslm_gpu_context_create,
// sslm_gpu_model_map, sslm_gpu_seq_restore, sslm_gpu_seq_release, SslmGpuSeqDecodeStepForG5Bridge,
// SslmModel::Load) and only status names v1.12.0 has, so it compiles there unchanged. See
// g1e_cell_9_8.cpp (leg 1) for the design; it is the CPU 9.3 box leg's (SUPERSLM_PKV_93_OUT).
//
//   1. The control: v1.12.0 restores the paged build's dedicated 'SLM5' (the writer rule keeps it
//      SLM5) with SSLM_OK, and its next 8 greedy tokens equal the ones leg 1 recorded. A failure here
//      is the harness's (fixture, files, driver), not the cell's verdict.
//   2. The verdict: v1.12.0 refuses the paged build's 'SLM6' with SSLM_SEQUENCE_KV_BUFFER_MISMATCH
//      (its one status for an unrecognized magic, PeekGpuSeqBlobWorkspaceSize) and leaves the
//      out-handle null. The G5 reader returns SSLM_GPU_HOLDER_MODE_MISMATCH here (A11), so this leg
//      also fails when it is built against the wrong tree.
//
// RED NOW: no input (leg 1 writes it only once G5's SLM6 writer exists).
// GREEN: both checks pass on both GPUs.
//
// Run (box): g1e_old_reader_9_8.exe --model=DIR\pkv_def.sslm --in=%SUPERSLM_PKV_GPU_98_OUT%
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_port.h"  // superslm_gpu::kQkNormDispatchesPerLayer
#include "superslm/model.h"

namespace {

int g_checks = 0, g_failures = 0;
#define OR_CHECK(cond, ...)                                     \
	do {                                                        \
		++g_checks;                                             \
		if (!(cond)) {                                          \
			++g_failures;                                       \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__);                           \
			std::printf("\n");                                  \
		}                                                       \
	} while (0)

bool ReadAll(const std::string& p, std::vector<uint8_t>* out) {
	std::ifstream f(p, std::ios::binary);
	if (!f) return false;
	out->assign(std::istreambuf_iterator<char>(f), {});
	return true;
}

unsigned St(SslmGpuStatus s) { return static_cast<unsigned>(s); }

}  // namespace

int main(int argc, char** argv) {
	std::string model_path, in_dir;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--in=", 0) == 0) in_dir = a.substr(5);
	}
	OR_CHECK(!model_path.empty() && !in_dir.empty(), "usage: g1e_old_reader_9_8 --model=PATH --in=DIR");
	std::vector<uint8_t> model_bytes, slm5, slm6, expect;
	OR_CHECK(ReadAll(model_path, &model_bytes) && !model_bytes.empty(), "cannot read %s", model_path.c_str());
	OR_CHECK(ReadAll(in_dir + "/g1e_98_dedicated.slm5", &slm5) && slm5.size() > 4, "no g1e_98_dedicated.slm5 in %s (run leg 1)",
	         in_dir.c_str());
	OR_CHECK(ReadAll(in_dir + "/g1e_98_pooled.slm6", &slm6) && slm6.size() > 4, "no g1e_98_pooled.slm6 in %s (run leg 1)",
	         in_dir.c_str());
	OR_CHECK(ReadAll(in_dir + "/g1e_98_expect.txt", &expect) && !expect.empty(), "no g1e_98_expect.txt in %s", in_dir.c_str());
	if (g_failures) {
		std::printf("g1e_old_reader_9_8: checks=%d failures=%d -> FAIL\n", g_checks, g_failures);
		return 1;
	}
	int last = -1;
	std::vector<int32_t> want;
	{
		const std::string t(expect.begin(), expect.end());
		const size_t lp = t.find("last="), np = t.find("next=");
		OR_CHECK(lp != std::string::npos && np != std::string::npos, "malformed g1e_98_expect.txt");
		if (lp != std::string::npos) last = std::atoi(t.c_str() + lp + 5);
		for (size_t p = np + 5; np != std::string::npos && p < t.size() && t[p] != '\n';) {
			want.push_back(std::atoi(t.c_str() + p));
			const size_t c = t.find(',', p);
			if (c == std::string::npos) break;
			p = c + 1;
		}
	}

	superslm::SslmModelView view{};
	std::string err;
	OR_CHECK(superslm::SslmModel::Load(model_bytes.data(), model_bytes.size(), view, &err) == superslm::SslmModelStatus::Ok,
	         "SslmModel::Load: %s", err.c_str());
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	OR_CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SslmGpuStatus::SSLM_OK && ctx, "context create");
	if (ctx) OR_CHECK(sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) == SslmGpuStatus::SSLM_OK && model, "model map");
	if (model) {
		const uint32_t budget = superslm_gpu::kQkNormDispatchesPerLayer * view.config.num_hidden_layers;
		// 1. the control
		SslmGpuSequenceHandle* r = nullptr;
		const SslmGpuStatus s5 = sslm_gpu_seq_restore(ctx, model, slm5.data(), slm5.size(), &r);
		OR_CHECK(s5 == SslmGpuStatus::SSLM_OK && r, "control: v1.12.0 did not restore the paged build's SLM5 (status %u)", St(s5));
		if (r) {
			std::vector<int32_t> got;
			int32_t tok = last;
			for (size_t i = 0; i < want.size(); ++i) {
				int32_t out = -1;
				if (SslmGpuSeqDecodeStepForG5Bridge(ctx, r, tok, budget, &out) != SslmGpuStatus::SSLM_OK) break;
				got.push_back(out);
				tok = out;
			}
			OR_CHECK(got == want && want.size() == 8, "control: v1.12.0's continuation of the SLM5 differs from the paged build's");
			sslm_gpu_seq_release(ctx, r);
		}
		// 2. the verdict
		SslmGpuSequenceHandle* r6 = reinterpret_cast<SslmGpuSequenceHandle*>(uintptr_t{0x1});
		const SslmGpuStatus s6 = sslm_gpu_seq_restore(ctx, model, slm6.data(), slm6.size(), &r6);
		OR_CHECK(std::memcmp(slm6.data(), "SLM6", 4) == 0, "the new data is not an 'SLM6' blob");
		// kills: an old reader that accepts SLM6 (or a new magic it happens to read)
		OR_CHECK(s6 == SslmGpuStatus::SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		         "v1.12.0 sslm_gpu_seq_restore of SLM6 returned %u, not SSLM_SEQUENCE_KV_BUFFER_MISMATCH (7): either it "
		         "accepted the blob, or this is not the v1.12.0 binary (the G5 reader returns 27)",
		         St(s6));
		OR_CHECK(r6 == nullptr, "v1.12.0 left the out-handle non-null after refusing SLM6");
		if (r6 && r6 != reinterpret_cast<SslmGpuSequenceHandle*>(uintptr_t{0x1})) sslm_gpu_seq_release(ctx, r6);
		sslm_gpu_model_unmap(ctx, model);
	}
	if (ctx) sslm_gpu_context_destroy(ctx);
	std::printf("g1e_old_reader_9_8: checks=%d failures=%d -> %s\n", g_checks, g_failures, g_failures ? "FAIL" : "PASS");
	return g_failures ? 1 : 0;
}
