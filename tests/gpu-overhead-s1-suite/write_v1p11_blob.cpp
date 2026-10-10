// GPU per-token overhead plan rev 3.1, cell 9.1: writes the "v1.9.0-saved blob" -- re-anchored to
// v1.11.0 by the common brief -- that cell_s1_real_model.cpp restores and continues in bracket mode.
// Built at v1.11.0 (copy this file and s1_real_model_common.h into a v1.11.0 tree's
// tests/gpu-overhead-s1-suite/ and build it with build_and_run.bat's `writer` arm): it uses only the
// 1.0 API as it stands there, and runs the same fixed procedure the cell runs.
//
// Usage: write_v1p11_blob.exe --model0p5b=PATH --out=PATH
#include <string>
#include <vector>

#include "s1_real_model_common.h"

using namespace gpu_overhead_s1_suite;

int main(int argc, char** argv) {
	std::string model_path, out_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model0p5b=", 0) == 0) model_path = a.substr(12);
		else if (a.rfind("--out=", 0) == 0) out_path = a.substr(6);
	}
	if (model_path.empty() || out_path.empty()) {
		std::printf("usage: write_v1p11_blob --model0p5b=PATH --out=PATH\n");
		return 2;
	}
	superslm::SslmModelView view{};
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModel(model_path, &view, &bytes, &err)) {
		std::printf("FAILED: %s\n", err.c_str());
		return 1;
	}
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK ||
	    sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != SSLM_OK) {
		std::printf("FAILED: context or model map\n");
		return 1;
	}
	const uint32_t unit = superslm_gpu::DispatchesPerLayer(ModelHasQkNorm(view));
	SslmGpuSequenceHandle* seq = nullptr;
	std::vector<uint8_t> blob;
	if (!RunSlm5ProcedureToMidToken(ctx, model, view.config.num_hidden_layers, unit, &seq, &blob, &err)) {
		std::printf("FAILED: %s\n", err.c_str());
		return 1;
	}
	sslm_gpu_seq_release(ctx, seq);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
	if (!WriteWholeFile(out_path, blob)) {
		std::printf("FAILED: cannot write %s\n", out_path.c_str());
		return 1;
	}
	std::printf("wrote %zu bytes, sha256 %s, to %s\n", blob.size(), Sha256Hex(blob).c_str(),
	            out_path.c_str());
	return 0;
}
