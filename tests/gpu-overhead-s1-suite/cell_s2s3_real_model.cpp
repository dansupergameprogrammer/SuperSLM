// GPU per-token overhead plan rev 3.1, slices S2 and S3: the §7 cells that need a real artifact.
// The synthetic-fixture S2/S3 cells are in tests/test_main.cpp (RunGpuOverheadS2S3Cells); 7.3's
// killing value (k = 12 through sslm_decode_step_gpu, every configuration) is cell_s1_real_model's
// Cell7_3. Every cell here runs every bracket fusion configuration that can ship
// (tests/support/gpu_fusion_configs.h) against per-site mode on the same build:
//
//   4.1/4.2/6.1  bit identity, tokens and SLM5 blobs, on each model supplied (A-EX Qwen2.5-0.5B,
//        Qwen2.5-1.5B, one QK-norm model), on every path: the one-call decode path; composed
//        slices of 1, 4, 5 and all layers (sslm_decode_step_gpu at s x unit); the chunk bridge at
//        1, 4, 8 and 256 tokens; batch decode with 2 and 4 sequences.
//   1.1  the fused pipelines are built once and reused across sequences and models (0.5B -> 1.5B
//        -> 0.5B, and the QK model when supplied); needs SUPERSLM_ENABLE_GPU_RECORD_LOG
//        (build_and_run.bat's s2s3 arm defines it).
//   8.2  save (SLM5) mid-token between two slices, restore, continue: the blobs and tokens in
//        bracket mode equal per-site mode's, every configuration.
//
// Usage: cell_s2s3_real_model.exe --model0p5b=PATH [--model1p5b=PATH] [--modelqk=PATH]
#include <algorithm>
#include <string>
#include <vector>

#include "s1_real_model_common.h"
#include "support/gpu_fusion_configs.h"

using namespace gpu_overhead_s1_suite;
using namespace superslm_gpu_fusion_configs;
using superslm_gpu::GpuTimingMode;

namespace {

std::string g_model_0p5b_path, g_model_1p5b_path, g_model_qk_path;

struct ScopedMode {
	GpuTimingMode previous;
	explicit ScopedMode(GpuTimingMode mode) : previous(superslm_gpu::GetGpuTimingMode()) {
		superslm_gpu::SetGpuTimingMode(mode);
	}
	~ScopedMode() { superslm_gpu::SetGpuTimingMode(previous); }
	ScopedMode(const ScopedMode&) = delete;
	ScopedMode& operator=(const ScopedMode&) = delete;
};

struct MappedModel {
	superslm::SslmModelView view{};
	std::vector<uint8_t> bytes;
	SslmGpuModelHandle* handle = nullptr;
	uint32_t layers = 0;
	uint32_t unit = 0;
	bool qk = false;
};

bool MapModel(SslmGpuContext* ctx, const std::string& path, MappedModel* out) {
	std::string err;
	if (!LoadRealModel(path, &out->view, &out->bytes, &err)) {
		CHECK_MSG(false, "cannot load %s: %s", path.c_str(), err.c_str());
		return false;
	}
	const SslmGpuStatus st = sslm_gpu_model_map(ctx, &out->view, GpuResidencyConfig{}, &out->handle);
	CHECK_MSG(st == SSLM_OK && out->handle != nullptr, "sslm_gpu_model_map(%s) returned %d", path.c_str(),
	          static_cast<int>(st));
	out->layers = out->view.config.num_hidden_layers;
	out->qk = ModelHasQkNorm(out->view);
	out->unit = superslm_gpu::DispatchesPerLayer(out->qk);
	return st == SSLM_OK && out->handle != nullptr;
}

// What one scenario leaves behind: every token it produced and every blob it saved.
struct Outcome {
	bool ok = false;
	std::string error;
	std::vector<int32_t> tokens;
	std::vector<std::vector<uint8_t>> blobs;
	bool operator==(const Outcome& o) const { return ok == o.ok && tokens == o.tokens && blobs == o.blobs; }
};

int32_t PromptToken(int i) { return (i * 37 + 11) % 1000 + 100; }

constexpr int64_t kCap = 320;
constexpr int kDecodeSteps = 6;

// The one-call path: a 3-token prompt, then kDecodeSteps decode steps at the full-depth budget.
Outcome OneCallDecode(SslmGpuContext* ctx, const MappedModel& m) {
	Outcome o;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_seq_create(ctx, m.handle, kCap, &seq) != SSLM_OK || !seq) {
		o.error = "seq_create";
		return o;
	}
	const int32_t prompt[3] = {PromptToken(0), PromptToken(1), PromptToken(2)};
	SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt, 3, m.unit * m.layers);
	int32_t token = PromptToken(3);
	for (int i = 0; i < kDecodeSteps && st == SSLM_OK; ++i) {
		int32_t next = -1;
		st = SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, token, m.unit * m.layers, &next);
		o.tokens.push_back(next);
		token = next;
	}
	std::vector<uint8_t> blob;
	o.ok = st == SSLM_OK && SaveBlob(ctx, seq, &blob);
	if (!o.ok) o.error = "status " + std::to_string(static_cast<int>(st));
	o.blobs.push_back(blob);
	sslm_gpu_seq_release(ctx, seq);
	return o;
}

// Composed slices: each token embedded, then driven `slice` layers per sslm_decode_step_gpu call
// (budget slice x unit) to full depth, then finished. The blob is saved after every call, so a
// slice boundary inside a token is compared too.
Outcome ComposedSlices(SslmGpuContext* ctx, const MappedModel& m, uint32_t slice) {
	Outcome o;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_seq_create(ctx, m.handle, kCap, &seq) != SSLM_OK || !seq) {
		o.error = "seq_create";
		return o;
	}
	const int32_t prompt[3] = {PromptToken(0), PromptToken(1), PromptToken(2)};
	SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt, 3, m.unit * m.layers);
	int32_t token = PromptToken(3);
	for (int i = 0; i < 3 && st == SSLM_OK; ++i) {
		st = sslm_gpu_seq_embed_token(ctx, seq, token);
		while (st == SSLM_OK && *SslmGpuSeqHandleLayerIndexForBench(seq) < m.layers) {
			st = StepBlocking(ctx, seq, slice * m.unit);
			std::vector<uint8_t> blob;
			if (st == SSLM_OK && !SaveBlob(ctx, seq, &blob)) st = SSLM_DEVICE_LOST;
			o.blobs.push_back(blob);
		}
		int32_t next = -1;
		if (st == SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &next);
		o.tokens.push_back(next);
		token = next;
	}
	o.ok = st == SSLM_OK;
	if (!o.ok) o.error = "status " + std::to_string(static_cast<int>(st));
	sslm_gpu_seq_release(ctx, seq);
	return o;
}

// The chunk bridge: an n-token prompt in one prefill call (sub-chunk lists of at most
// kT2169TdrSafeMaxChunkTokens), then one decode step.
Outcome Chunk(SslmGpuContext* ctx, const MappedModel& m, int n) {
	Outcome o;
	SslmGpuSequenceHandle* seq = nullptr;
	if (sslm_gpu_seq_create(ctx, m.handle, kCap, &seq) != SSLM_OK || !seq) {
		o.error = "seq_create";
		return o;
	}
	std::vector<int32_t> prompt(static_cast<size_t>(n));
	for (int i = 0; i < n; ++i) prompt[static_cast<size_t>(i)] = PromptToken(i);
	SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, prompt.data(), n, m.unit * m.layers);
	std::vector<uint8_t> blob;
	if (st == SSLM_OK && !SaveBlob(ctx, seq, &blob)) st = SSLM_DEVICE_LOST;
	o.blobs.push_back(blob);
	int32_t next = -1;
	if (st == SSLM_OK) st = SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, PromptToken(n), m.unit * m.layers, &next);
	o.tokens.push_back(next);
	o.ok = st == SSLM_OK;
	if (!o.ok) o.error = "status " + std::to_string(static_cast<int>(st));
	sslm_gpu_seq_release(ctx, seq);
	return o;
}

// Batch decode: n sequences with distinct prompts, three batch steps at an unlimited budget, each
// sequence finished after each step.
Outcome Batch(SslmGpuContext* ctx, const MappedModel& m, uint32_t n) {
	Outcome o;
	std::vector<SslmGpuSequenceHandle*> seqs(n, nullptr);
	SslmGpuStatus st = SSLM_OK;
	std::vector<int32_t> token(n);
	for (uint32_t i = 0; i < n && st == SSLM_OK; ++i) {
		st = sslm_gpu_seq_create(ctx, m.handle, kCap, &seqs[i]);
		const int32_t prompt[2] = {PromptToken(static_cast<int>(10 * i)), PromptToken(static_cast<int>(10 * i + 1))};
		if (st == SSLM_OK) st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, seqs[i], prompt, 2, m.unit * m.layers);
		token[i] = PromptToken(static_cast<int>(10 * i + 2));
	}
	for (int step = 0; step < 3 && st == SSLM_OK; ++step) {
		for (uint32_t i = 0; i < n && st == SSLM_OK; ++i) st = sslm_gpu_seq_embed_token(ctx, seqs[i], token[i]);
		std::vector<SslmGpuStatus> sts(n, SSLM_DEVICE_LOST);
		if (st == SSLM_OK) st = sslm_decode_step_batch_gpu(ctx, seqs.data(), nullptr, n, 0xFFFFFFFFu, sts.data());
		for (uint32_t i = 0; i < n && st == SSLM_OK; ++i) {
			st = sts[i];
			int32_t next = -1;
			if (st == SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(ctx, seqs[i], &next);
			o.tokens.push_back(next);
			token[i] = next;
		}
	}
	for (uint32_t i = 0; i < n; ++i) {
		std::vector<uint8_t> blob;
		if (st == SSLM_OK && seqs[i] && !SaveBlob(ctx, seqs[i], &blob)) st = SSLM_DEVICE_LOST;
		o.blobs.push_back(blob);
		if (seqs[i]) sslm_gpu_seq_release(ctx, seqs[i]);
	}
	o.ok = st == SSLM_OK;
	if (!o.ok) o.error = "status " + std::to_string(static_cast<int>(st));
	return o;
}

// One scenario: per-site mode's outcome, then bracket mode's under every configuration, equal.
template <typename Scenario>
void CompareEveryConfiguration(const char* model_label, const std::string& scenario_label, Scenario scenario) {
	Outcome per_site;
	{
		ScopedMode mode(GpuTimingMode::PerSite);
		per_site = scenario();
	}
	CHECK_MSG(per_site.ok, "4.1/4.2 %s %s (per-site): %s", model_label, scenario_label.c_str(), per_site.error.c_str());
	if (!per_site.ok) return;
	for (const ConfigRow& row : ShippableConfigs()) {
		ScopedFusionConfig config(row.config);
		CHECK_MSG(config.accepted, "4.1/4.2: configuration %s refused", row.name);
		ScopedMode mode(GpuTimingMode::Bracket);
		const Outcome bracket = scenario();
		CHECK_MSG(bracket == per_site,
		          "6.1/4.1/4.2 %s %s, bracket %s %s: tokens and blobs must equal per-site mode's (%s; %zu vs "
		          "%zu tokens, %zu vs %zu blobs)",
		          model_label, scenario_label.c_str(), row.name, ConfigName(row.config).c_str(),
		          bracket.ok ? "ok" : bracket.error.c_str(), bracket.tokens.size(), per_site.tokens.size(),
		          bracket.blobs.size(), per_site.blobs.size());
	}
}

void BitIdentityOnEveryPath(SslmGpuContext* ctx, const MappedModel& m, const char* label) {
	CompareEveryConfiguration(label, "one-call decode", [&] { return OneCallDecode(ctx, m); });
	for (uint32_t slice : {1u, 4u, 5u, m.layers}) {
		CompareEveryConfiguration(label, "slices of " + std::to_string(slice),
		                          [&] { return ComposedSlices(ctx, m, slice); });
	}
	for (int n : {1, 4, 8, 256}) {
		CompareEveryConfiguration(label, "chunk of " + std::to_string(n), [&] { return Chunk(ctx, m, n); });
	}
	for (uint32_t n : {2u, 4u}) {
		CompareEveryConfiguration(label, "batch of " + std::to_string(n), [&] { return Batch(ctx, m, n); });
	}
}

// 8.2: the SLM5 procedure (s1_real_model_common.h), restored and continued, every configuration.
void Cell8_2(SslmGpuContext* ctx, const MappedModel& m, const char* label) {
	auto run = [&](Outcome* o) {
		SslmGpuSequenceHandle* seq = nullptr;
		std::vector<uint8_t> mid;
		std::string err;
		if (!RunSlm5ProcedureToMidToken(ctx, m.handle, m.layers, m.unit, &seq, &mid, &err)) {
			o->error = err;
			return;
		}
		sslm_gpu_seq_release(ctx, seq);
		SslmGpuSequenceHandle* restored = nullptr;
		if (sslm_gpu_seq_restore(ctx, m.handle, mid.data(), mid.size(), &restored) != SSLM_OK || !restored) {
			o->error = "restore";
			return;
		}
		std::vector<uint8_t> final_blob;
		o->ok = ContinueAndDecode(ctx, restored, m.layers, m.unit, 4, &o->tokens, &final_blob, &err);
		if (!o->ok) o->error = err;
		o->blobs = {mid, final_blob};
		sslm_gpu_seq_release(ctx, restored);
	};
	CompareEveryConfiguration(label, "8.2 save mid-token / restore / continue", [&] {
		Outcome o;
		run(&o);
		return o;
	});
}

void RunModel(SslmGpuContext* ctx, const std::string& path, const char* label, bool full) {
	MappedModel m;
	if (!MapModel(ctx, path, &m)) return;
	std::printf("%s: %u layers, unit %u%s\n", label, m.layers, m.unit, m.qk ? " (QK norm)" : "");
	if (full) {
		BitIdentityOnEveryPath(ctx, m, label);
		Cell8_2(ctx, m, label);
	} else {
		CompareEveryConfiguration(label, "one-call decode", [&] { return OneCallDecode(ctx, m); });
	}
	sslm_gpu_model_unmap(ctx, m.handle);
}

}  // namespace

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		auto take = [&](const char* flag) -> const char* {
			const size_t n = std::strlen(flag);
			return a.compare(0, n, flag) == 0 ? a.c_str() + n : nullptr;
		};
		if (const char* v = take("--model0p5b=")) g_model_0p5b_path = v;
		else if (const char* v2 = take("--model1p5b=")) g_model_1p5b_path = v2;
		else if (const char* v3 = take("--modelqk=")) g_model_qk_path = v3;
	}
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	CHECK_MSG(superslm_gpu::GetGpuBracketFusionConfig() == superslm_gpu::DefaultGpuBracketFusionConfig(),
	          "the process must start in the compiled-in default configuration");
	if (g_model_0p5b_path.empty()) {
		SKIP_MSG("4.1/4.2/6.1, 8.2 and 1.1 on the 0.5B: --model0p5b not supplied");
	} else {
		RunModel(ctx, g_model_0p5b_path, "0.5B", /*full=*/true);
	}
	if (g_model_1p5b_path.empty()) {
		SKIP_MSG("4.1 on the 1.5B: --model1p5b not supplied");
	} else {
		RunModel(ctx, g_model_1p5b_path, "1.5B", /*full=*/true);
		// 1.1: back to the 0.5B after the 1.5B.
		if (!g_model_0p5b_path.empty()) RunModel(ctx, g_model_0p5b_path, "0.5B again", /*full=*/false);
	}
	if (g_model_qk_path.empty()) {
		SKIP_MSG("4.1 on a QK-norm model: --modelqk not supplied");
	} else {
		RunModel(ctx, g_model_qk_path, "QK-norm", /*full=*/true);
		if (!g_model_0p5b_path.empty()) RunModel(ctx, g_model_0p5b_path, "0.5B after QK", /*full=*/false);
	}
#if defined(SUPERSLM_ENABLE_GPU_RECORD_LOG)
	// 1.1: every fused pipeline any configuration used was built exactly once across every model,
	// sequence and configuration above.
	for (const char* name : {"fused_o_residual_norm_site", "fused_down_residual_commit_site", "gate_up_gemm_site",
	                         "up_proj_gemm_wide_b_site", "fused_gate_up_site"}) {
		const uint32_t built = superslm_gpu::ComposedPipelineBuildCount(name);
		CHECK_MSG(g_model_0p5b_path.empty() || built == 1, "1.1: %s was built %u times across models, want once",
		          name, built);
	}
#else
	CHECK_MSG(false, "1.1: build with /DSUPERSLM_ENABLE_GPU_RECORD_LOG (build_and_run.bat s2s3)");
#endif
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}
