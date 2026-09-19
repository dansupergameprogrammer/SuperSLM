// T-2791 (Curie) -- plan Sec3.4 row 10 (functional achievement), the product claim at production
// size: "The real 1.5.0 artifact over T-2780's 256-row commission tokens, through context, map,
// reset, prefill and read, reproduces the CPU driver's 51616b69...0f4e output byte for byte,
// 270,336 bytes."
//
// FEATURE ORACLE: SHA-256 51616b69a8fce7d811a8630cf81b9cca05645a4d75e0a4a9cb14ad1dbdf30f4e over
// 270,336 bytes is T-2780's CPU driver output (EmbedEntry + RunLayerLoop per token, then
// RmsNormSite "final_norm"; Claude/Brunel/t2776-baseline-instruments-2026-09-17.md, "256-row CPU
// driver/probe commission"), reproduced there by an independent per-text probe. It takes no input
// from the GPU path. Input: T-2780's tokens.txt (SHA-256 01c9b047...9f82), one comma-separated
// token row per line, 256 rows, 54,587 tokens; supplied as --commission=PATH and hash-checked.
// Each row's frame is serialized exactly as that driver's WriteFinalHiddenFrame does: LE64 magic
// 0x54474D5331373032, LE64 count, LE64 m, LE64 e, then `count` int8 codes -- 1,056 bytes per row.
//
// Sequence sizing: one sequence per row at cap max(64, next power of two >= row length); plan
// Sec2.2 measured bytes invariant to the cap. Cost: about 11.4 GPU minutes on an RTX 2080 SUPER
// (plan Sec3.4 row 10), so this cell is run at release (plan Sec3.5 step 5), not per edit.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_functional_commission.exe --qwen3=PATH --commission=PATH [--rows=N --reference=CPU_DRIVER_BIN]
// (a --rows=N run is graded only against the CPU driver's output prefix, never reported as the
// row-10 result).
#include <fstream>

#include "fixture_common.h"

namespace {
constexpr const char* kTokensSha = "01c9b0471695086cd708e02ad9826d70d8745fc3296610b2c2420f6df9091f82";
constexpr const char* kCpuDriverSha = "51616b69a8fce7d811a8630cf81b9cca05645a4d75e0a4a9cb14ad1dbdf30f4e";
constexpr size_t kExpectedBytes = 270336;

void PutLe64(std::vector<uint8_t>* out, uint64_t v) {
	for (int i = 0; i < 8; ++i) out->push_back(static_cast<uint8_t>(v >> (8 * i)));
}
}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	size_t max_rows = SIZE_MAX;
	std::string ref_path;
	for (int i = 1; i < argc; ++i) {
		if (std::strncmp(argv[i], "--rows=", 7) == 0) max_rows = static_cast<size_t>(std::strtoull(argv[i] + 7, nullptr, 10));
		if (std::strncmp(argv[i], "--reference=", 12) == 0) ref_path = argv[i] + 12;
	}
	if (g_qwen3_path.empty() || g_commission_path.empty()) {
		SKIP_MSG("row 10 needs --qwen3=PATH and --commission=PATH");
		return FinishSuite("cell_functional_commission");
	}
	std::vector<uint8_t> tok_bytes;
	CHECK_MSG(ReadFileBytes(g_commission_path, &tok_bytes), "cannot read %s", g_commission_path.c_str());
	const std::string tok_sha = Sha256Hex(tok_bytes.data(), tok_bytes.size());
	CHECK_MSG(tok_sha == kTokensSha, "SETUP: the commission token file is not T-2780's (sha256 %s)", tok_sha.c_str());

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	GpuModelFixture fx;
	if (!fx.Open(g_qwen3_path, ctx)) {
		CHECK_MSG(false, "could not open %s", g_qwen3_path.c_str());
		return FinishSuite("cell_functional_commission");
	}
	std::vector<uint8_t> out;
	std::ifstream in(g_commission_path);
	std::string line;
	size_t rows = 0, tokens_total = 0, failed = 0;
	while (rows < max_rows && std::getline(in, line)) {
		if (!line.empty() && line.back() == '\r') line.pop_back();
		if (line.empty()) continue;
		std::vector<int32_t> t;
		for (size_t pos = 0; pos < line.size();) {
			const size_t end = line.find(',', pos);
			t.push_back(static_cast<int32_t>(std::stol(line.substr(pos, end == std::string::npos ? end : end - pos))));
			if (end == std::string::npos) break;
			pos = end + 1;
		}
		int64_t cap = 64;
		while (cap < static_cast<int64_t>(t.size())) cap *= 2;
		SslmGpuSequenceHandle* s = nullptr;
		Frame f;
		if (sslm_gpu_seq_create(ctx, fx.model, cap, &s) == SSLM_OK && s && sslm_gpu_seq_reset(ctx, s) == SSLM_OK &&
		    Prefill(fx, s, t) == SSLM_OK) {
			f = ReadVerb(fx, s);
		}
		if (s) sslm_gpu_seq_release(ctx, s);
		if (!f.IsOk()) {
			++failed;
			std::printf("row %zu: read %s\n", rows, StatusName(f.status));
			f.codes.assign(fx.hidden, 0);
		}
		PutLe64(&out, UINT64_C(0x54474D5331373032));
		PutLe64(&out, static_cast<uint64_t>(fx.hidden));
		PutLe64(&out, static_cast<uint64_t>(f.m));
		PutLe64(&out, static_cast<uint64_t>(f.e));
		out.insert(out.end(), reinterpret_cast<const uint8_t*>(f.codes.data()),
		           reinterpret_cast<const uint8_t*>(f.codes.data()) + fx.hidden);
		++rows;
		tokens_total += t.size();
		if (rows % 32 == 0) std::printf("  %zu rows, %zu tokens\n", rows, tokens_total);
	}
	const std::string sha = Sha256Hex(out.data(), out.size());
	std::printf("row 10: rows=%zu tokens=%zu bytes=%zu sha256=%s\n", rows, tokens_total, out.size(), sha.c_str());
	CHECK_MSG(failed == 0, "%zu rows failed to read", failed);
	if (max_rows == SIZE_MAX) {
		CHECK_MSG(out.size() == kExpectedBytes && sha == kCpuDriverSha,
		          "row 10: the GPU read's %zu bytes (sha256 %s) do not reproduce the CPU driver's %zu bytes (%s)", out.size(),
		          sha.c_str(), kExpectedBytes, kCpuDriverSha);
	} else if (!ref_path.empty()) {
		// A partial run is graded against the same-length prefix of the CPU driver's own output
		// file (T-2780's t2780-v15-driver-cpu-256.bin, whose whole-file digest is kCpuDriverSha).
		std::vector<uint8_t> ref;
		CHECK_MSG(ReadFileBytes(ref_path, &ref) && Sha256Hex(ref.data(), ref.size()) == kCpuDriverSha,
		          "SETUP: --reference is not the CPU driver's 51616b69... output");
		CHECK_MSG(ref.size() >= out.size() && std::equal(out.begin(), out.end(), ref.begin()),
		          "row 10 (first %zu rows): the GPU read's frames differ from the CPU driver's", rows);
	} else {
		SKIP_MSG("--rows=%zu without --reference: partial run, not graded", max_rows);
	}
	fx.Close();
	sslm_gpu_context_destroy(ctx);
	return FinishSuite("cell_functional_commission");
}
