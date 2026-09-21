// T-2900 (Curie) -- the restore half of the legacy 'SLM4' backward-compatibility cell (see
// cell_gpu_slm4_dump.cpp's own header). Linked against T-2895's own v5-patched GPU objects; reads
// the 'SLM4' blob the dump process wrote and restores it with the v5 restore code. A legacy blob
// carries no schema tail, so the restored copy must come back unbound (the unused walk sentinel),
// exactly as documented today -- proven by execution across a real magic-version boundary rather
// than by reading the `is_v5` branch alone.
//
// ADOPTED, PER BRIEF, NOT RE-DERIVED, from `Claude/Vitruvius/t2895-probe/restore_slm4_blob.cpp`.
//
// Run: cell_gpu_slm4_restore.exe --model=PATH --in=BLOBPATH
#include "fixture_common.h"
#include <fstream>

int main(int argc, char** argv) {
	std::string model_path, in_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--in=", 0) == 0) in_path = a.substr(5);
	}
	if (model_path.empty() || in_path.empty()) {
		std::printf("SKIP cell_gpu_slm4_restore -- needs --model=PATH --in=BLOBPATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	CHECK(sslm_gpu_context_create(GpuContextConfig{}, &ctx) == SSLM_OK);
	GpuModelFixture fx;
	CHECK_MSG(fx.Open(model_path, ctx), "GPU open failed");
	if (GFailures > 0) {
		std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
		return 1;
	}

	std::ifstream in_f(in_path, std::ios::binary);
	std::vector<uint8_t> blob((std::istreambuf_iterator<char>(in_f)), std::istreambuf_iterator<char>());
	CHECK_MSG(blob.size() >= 4 && blob[0] == 'S' && blob[1] == 'L' && blob[2] == 'M' && blob[3] == '4',
	          "expected to read back an 'SLM4' blob (bytes=%zu)", blob.size());
	std::printf("read blob_bytes=%zu magic=%c%c%c%c\n", blob.size(), blob.size() >= 1 ? blob[0] : '?',
	            blob.size() >= 2 ? blob[1] : '?', blob.size() >= 3 ? blob[2] : '?',
	            blob.size() >= 4 ? blob[3] : '?');

	SslmGpuSequenceHandle* r = nullptr;
	const SslmGpuStatus rs = sslm_gpu_seq_restore(ctx, fx.model, blob.data(), blob.size(), &r);
	CHECK_MSG(rs == SSLM_OK && r, "restore returned %s", StatusName(rs));
	if (r) {
		const uint32_t walk = SslmGpuSeqWalkStateForG5Bridge(r);
		CHECK_MSG(walk == 0xFFFFFFFFu,
		          "a legacy 'SLM4' blob (no schema tail) must restore UNBOUND (walk=0xFFFFFFFF); got walk=%u", walk);
		std::printf("restored (legacy SLM4 blob, via v5 restore): walk=%u ctx=%lld layer=%u\n", walk,
		            static_cast<long long>(ContextLength(r)), LayerIndex(r));
		sslm_gpu_seq_release(ctx, r);
	}

	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
}
