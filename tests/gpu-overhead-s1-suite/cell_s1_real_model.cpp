// GPU per-token overhead plan rev 3.1, slice S1: the §7 cells that need a real artifact. The
// synthetic-fixture S1 cells are in tests/test_main.cpp (RunGpuOverheadS1Cells); this binary takes
// the A-EX Qwen2.5-0.5B artifact (24 layers, the legacy 24-dispatch unit) and, optionally, a
// QK-norm model (Qwen3-0.6B, 28 layers, unit 25) and the v1.11.0 SLM5 blob write_v1p11_blob.exe
// wrote. A cell whose artifact was not supplied SKIPs by name.
//
//   7.2  gpu_busy_ms keeps its meaning: > 0 and <= submit_wait_ms in both modes, and bracket >=
//        0.5 x per-site on the same call shape, on calls that follow another call, on the
//        single-token path and on the chunk path (one 4-token list). Kills a recorder that writes
//        both bracket queries before the body. Margin unverified until B0.3 runs (plan §7 7.2).
//   7.3  at dispatch_budget = k x DispatchesPerLayer, k in {1, 4, 12, 24}, both modes advance
//        exactly k layers through sslm_decode_step_gpu under every shippable fusion configuration,
//        and record k x the configuration's per-layer count (k = 12 is S2/S3's killing value).
//   8.2  save (SLM5) mid-token between two slices, restore, continue: the blob, every token and the
//        final blob in bracket mode equal per-site mode's.
//   8.3  the device-head finish (SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE) after a bracket-mode slice:
//        the token equals per-site mode's (and the host head's). The logits row itself is not on
//        the public surface; the token and the final blob are what this cell compares.
//   9.1  the mid-token blob saved in either mode is byte-equal, and equal to the one v1.11.0 saved
//        on the same procedure; that v1.11.0 blob restores and continues identically in bracket
//        mode.
//   6.3  repeatability: the 0.5B run of --steps tokens (default 200), three times in bracket mode,
//        gives one digest.
//
// Usage: cell_s1_real_model.exe --model0p5b=PATH [--modelqk=PATH] [--v1p11-blob=PATH] [--steps=N]
#include <algorithm>
#include <string>
#include <vector>

#include "s1_real_model_common.h"
#include "support/gpu_fusion_configs.h"

using namespace gpu_overhead_s1_suite;
using superslm_gpu::GpuTimingMode;

namespace {

std::string g_model_0p5b_path, g_model_qk_path, g_v1p11_blob_path;
int g_steps = 200;

const char* ModeName(GpuTimingMode mode) {
	return mode == GpuTimingMode::PerSite ? "per-site" : "bracket";
}

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
	uint32_t unit = 0;  // dispatches per layer: the budget unit
};

bool MapModel(SslmGpuContext* ctx, const std::string& path, uint32_t residency_flags,
              MappedModel* out) {
	std::string err;
	if (!LoadRealModel(path, &out->view, &out->bytes, &err)) {
		CHECK_MSG(false, "cannot load %s: %s", path.c_str(), err.c_str());
		return false;
	}
	GpuResidencyConfig cfg{};
	cfg.flags = residency_flags;
	const SslmGpuStatus st = sslm_gpu_model_map(ctx, &out->view, cfg, &out->handle);
	CHECK_MSG(st == SSLM_OK && out->handle != nullptr, "sslm_gpu_model_map(%s) returned %d",
	          path.c_str(), static_cast<int>(st));
	out->layers = out->view.config.num_hidden_layers;
	out->unit = superslm_gpu::DispatchesPerLayer(ModelHasQkNorm(out->view));
	return st == SSLM_OK && out->handle != nullptr;
}

double Median(std::vector<double> v) {
	std::sort(v.begin(), v.end());
	return v.empty() ? 0.0 : v[v.size() / 2];
}

// 7.2 on one path. `run_one` performs one call of the measured shape on `seq` and returns its
// status; the first call is a warm-up and is not measured, so every measured call follows another.
template <typename RunOne>
void CheckGpuBusyBound(const char* path_label, RunOne run_one) {
	std::vector<double> busy[2];
	const GpuTimingMode modes[2] = {GpuTimingMode::PerSite, GpuTimingMode::Bracket};
	{
		ScopedMode warm(GpuTimingMode::PerSite);
		CHECK_MSG(run_one() == SSLM_OK, "7.2 %s: warm-up call failed", path_label);
	}
	for (int round = 0; round < 5; ++round) {
		for (int m = 0; m < 2; ++m) {
			ScopedMode scoped(modes[m]);
			const SslmGpuStatus st = run_one();
			const superslm_gpu::GpuCallTiming t = superslm_gpu::LastCallTiming();
			CHECK_MSG(st == SSLM_OK, "7.2 %s (%s) round %d: status %d", path_label, ModeName(modes[m]),
			          round, static_cast<int>(st));
			CHECK_MSG(t.gpu_busy_ms > 0.0 && t.gpu_busy_ms <= t.submit_wait_ms,
			          "7.2 %s (%s) round %d: gpu_busy_ms=%.4f submit_wait_ms=%.4f, want 0 < busy <= wait",
			          path_label, ModeName(modes[m]), round, t.gpu_busy_ms, t.submit_wait_ms);
			busy[m].push_back(t.gpu_busy_ms);
		}
	}
	const double per_site = Median(busy[0]), bracket = Median(busy[1]);
	std::printf("7.2 %s: median gpu_busy_ms per-site %.4f, bracket %.4f (ratio %.3f)\n", path_label,
	            per_site, bracket, per_site > 0.0 ? bracket / per_site : 0.0);
	CHECK_MSG(bracket >= 0.5 * per_site,
	          "7.2 %s: median bracket gpu_busy_ms %.4f < 0.5 x per-site %.4f -- a recorder that writes "
	          "both bracket queries before the body reads near zero here",
	          path_label, bracket, per_site);
}

void Cell7_2(SslmGpuContext* ctx, const MappedModel& m) {
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK_MSG(sslm_gpu_seq_create(ctx, m.handle, 64, &seq) == SSLM_OK && seq, "7.2: seq_create");
	if (!seq) return;
	int32_t token = kProcedurePrompt[0];
	CheckGpuBusyBound("single-token", [&]() {
		SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, token);
		if (st == SSLM_OK) st = StepBlocking(ctx, seq, m.unit * m.layers);
		token = (token + 1) % 1000 + 100;
		return st;
	});
	sslm_gpu_seq_release(ctx, seq);

	CheckGpuBusyBound("chunk (one 4-token list)", [&]() {
		// A fresh sequence per call; four tokens <= kT2169TdrSafeMaxChunkTokens is one command list,
		// so LastCallTiming is that list's.
		SslmGpuSequenceHandle* chunk_seq = nullptr;
		SslmGpuStatus st = sslm_gpu_seq_create(ctx, m.handle, 64, &chunk_seq);
		if (st != SSLM_OK || !chunk_seq) return st != SSLM_OK ? st : SSLM_DEVICE_LOST;
		st = SslmGpuSeqPrefillPromptForG5Bridge(ctx, chunk_seq, kProcedurePrompt, 4, m.unit * m.layers);
		sslm_gpu_seq_release(ctx, chunk_seq);
		return st;
	});
}

// 7.3: the budget unit stays DispatchesPerLayer while bracket mode records the fusion
// configuration's per-layer count. At S1 the mutant "unit = recorded count" is equivalent (both are
// 24/25), so k in {1, 4, 24} is a regression; from S2 on the killing value is k = 12 in every
// shippable configuration (plan §7 7.3, rev 3.1): the mutant's floor(12 x unit / recorded) is 13-16
// layers, never 12. This is the cell that can see it -- the budget goes through
// sslm_decode_step_gpu, whose unit is the 1.0 API's own choice (PlanDispatchBudgetGpu takes its
// unit from the caller and ignores the mode).
void Cell7_3(SslmGpuContext* ctx, const MappedModel& m, const char* label) {
	using namespace superslm_gpu_fusion_configs;
	for (const ConfigRow& row : ShippableConfigs()) {
		ScopedFusionConfig config(row.config);
		CHECK_MSG(config.accepted, "7.3 %s: configuration %s refused", label, row.name);
		for (GpuTimingMode mode : {GpuTimingMode::Bracket, GpuTimingMode::PerSite}) {
			ScopedMode scoped(mode);
			for (uint32_t k : {1u, 4u, 12u, 24u}) {
				if (k > m.layers) continue;
				SslmGpuSequenceHandle* seq = nullptr;
				CHECK_MSG(sslm_gpu_seq_create(ctx, m.handle, 64, &seq) == SSLM_OK && seq, "7.3: seq_create");
				if (!seq) continue;
				SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, kProcedurePrompt[0]);
				if (st == SSLM_OK) st = StepBlocking(ctx, seq, k * m.unit);
				const uint32_t layer = *SslmGpuSeqHandleLayerIndexForBench(seq);
				const uint32_t recorded = superslm_gpu::LastCallRecordedDispatchCount();
				CHECK_MSG(st == SSLM_OK && layer == k,
				          "7.3 %s %s (%s) k=%u: dispatch_budget %u -> status %d, layer_index %u; want exactly "
				          "%u layers",
				          label, row.name, ModeName(mode), k, k * m.unit, static_cast<int>(st), layer, k);
				const uint32_t want =
				    k * (mode == GpuTimingMode::Bracket ? row.PerLayer(m.unit == superslm_gpu::kQkNormDispatchesPerLayer)
				                                        : m.unit);
				CHECK_MSG(recorded == want,
				          "7.3 %s %s %s (%s) k=%u: recorded %u dispatches, want %u (the unit stays %u per layer)",
				          label, row.name, ConfigName(row.config).c_str(), ModeName(mode), k, recorded, want, m.unit);
				sslm_gpu_seq_release(ctx, seq);
			}
		}
	}
}

struct ProcedureRun {
	bool ok = false;
	std::vector<uint8_t> mid_blob, final_blob;
	std::vector<int32_t> tokens;
};

// The SLM5 procedure, then restore its mid-token blob into a fresh handle and continue there.
ProcedureRun RunProcedureWithRestore(SslmGpuContext* ctx, const MappedModel& m, const char* label) {
	ProcedureRun r;
	SslmGpuSequenceHandle* seq = nullptr;
	std::string err;
	if (!RunSlm5ProcedureToMidToken(ctx, m.handle, m.layers, m.unit, &seq, &r.mid_blob, &err)) {
		CHECK_MSG(false, "%s: procedure failed: %s", label, err.c_str());
		return r;
	}
	sslm_gpu_seq_release(ctx, seq);
	SslmGpuSequenceHandle* restored = nullptr;
	const SslmGpuStatus st =
	    sslm_gpu_seq_restore(ctx, m.handle, r.mid_blob.data(), r.mid_blob.size(), &restored);
	CHECK_MSG(st == SSLM_OK && restored, "%s: sslm_gpu_seq_restore returned %d", label, static_cast<int>(st));
	if (!restored) return r;
	r.ok = ContinueAndDecode(ctx, restored, m.layers, m.unit, /*more=*/4, &r.tokens, &r.final_blob, &err);
	CHECK_MSG(r.ok, "%s: continuing the restored sequence failed: %s", label, err.c_str());
	sslm_gpu_seq_release(ctx, restored);
	return r;
}

void Cell8_2And9_1(SslmGpuContext* ctx, const MappedModel& m, std::vector<int32_t>* host_tokens) {
	ProcedureRun runs[2];
	const GpuTimingMode modes[2] = {GpuTimingMode::PerSite, GpuTimingMode::Bracket};
	for (int i = 0; i < 2; ++i) {
		ScopedMode scoped(modes[i]);
		runs[i] = RunProcedureWithRestore(ctx, m, ModeName(modes[i]));
	}
	if (!runs[0].ok || !runs[1].ok) return;
	CHECK_MSG(runs[0].mid_blob == runs[1].mid_blob,
	          "9.1: the mid-token SLM5 blob saved in bracket mode (%zu bytes) must be byte-equal to "
	          "per-site mode's (%zu bytes)",
	          runs[1].mid_blob.size(), runs[0].mid_blob.size());
	CHECK_MSG(runs[0].tokens == runs[1].tokens,
	          "8.2: tokens after save mid-token / restore / continue differ between bracket and "
	          "per-site mode");
	CHECK_MSG(runs[0].final_blob == runs[1].final_blob,
	          "8.2: the final blob after save / restore / continue differs between bracket and "
	          "per-site mode");
	*host_tokens = runs[0].tokens;

	// 9.1, the v1.11.0 half.
	if (g_v1p11_blob_path.empty()) {
		CHECK_MSG(false,
		          "9.1: v1.11.0 blob not filed -- pass --v1p11-blob=PATH (write_v1p11_blob.exe, built "
		          "at v1.11.0, run on this same artifact); this cell stays red until it is");
		return;
	}
	std::vector<uint8_t> old_blob;
	if (!ReadWholeFile(g_v1p11_blob_path, &old_blob)) {
		CHECK_MSG(false, "9.1: cannot read %s", g_v1p11_blob_path.c_str());
		return;
	}
	CHECK_MSG(old_blob == runs[1].mid_blob,
	          "9.1: the v1.11.0-saved blob (%zu bytes, sha256 %s) must equal the one this build saves "
	          "in bracket mode on the same procedure (%zu bytes, sha256 %s)",
	          old_blob.size(), Sha256Hex(old_blob).c_str(), runs[1].mid_blob.size(),
	          Sha256Hex(runs[1].mid_blob).c_str());
	ScopedMode bracket(GpuTimingMode::Bracket);
	SslmGpuSequenceHandle* restored = nullptr;
	const SslmGpuStatus st = sslm_gpu_seq_restore(ctx, m.handle, old_blob.data(), old_blob.size(), &restored);
	CHECK_MSG(st == SSLM_OK && restored, "9.1: restoring the v1.11.0 blob returned %d", static_cast<int>(st));
	if (!restored) return;
	std::vector<int32_t> tokens;
	std::vector<uint8_t> final_blob;
	std::string err;
	const bool ok = ContinueAndDecode(ctx, restored, m.layers, m.unit, 4, &tokens, &final_blob, &err);
	CHECK_MSG(ok, "9.1: continuing the v1.11.0 blob failed: %s", err.c_str());
	CHECK_MSG(ok && tokens == runs[0].tokens && final_blob == runs[0].final_blob,
	          "9.1: the v1.11.0 blob, restored and continued in bracket mode, must give per-site "
	          "mode's tokens and final blob");
	sslm_gpu_seq_release(ctx, restored);
}

void Cell8_3(SslmGpuContext* ctx, const std::vector<int32_t>& host_tokens) {
	MappedModel m;
	if (!MapModel(ctx, g_model_0p5b_path, SSLM_GPU_RESIDENCY_HEAD_ON_DEVICE, &m)) return;
	ProcedureRun runs[2];
	const GpuTimingMode modes[2] = {GpuTimingMode::PerSite, GpuTimingMode::Bracket};
	for (int i = 0; i < 2; ++i) {
		ScopedMode scoped(modes[i]);
		runs[i] = RunProcedureWithRestore(ctx, m, ModeName(modes[i]));
	}
	if (runs[0].ok && runs[1].ok) {
		CHECK_MSG(runs[0].tokens == runs[1].tokens && runs[0].final_blob == runs[1].final_blob,
		          "8.3: the device-head finish after bracket-mode slices must give per-site mode's "
		          "tokens and final blob");
		CHECK_MSG(host_tokens.empty() || runs[1].tokens == host_tokens,
		          "8.3: the device-head tokens must equal the host head's (gpu_1p0.h: the token is "
		          "identical on every path)");
	}
	sslm_gpu_model_unmap(ctx, m.handle);
}

void Cell6_3(SslmGpuContext* ctx, const MappedModel& m) {
	ScopedMode bracket(GpuTimingMode::Bracket);
	std::string digests[3];
	for (int run = 0; run < 3; ++run) {
		SslmGpuSequenceHandle* seq = nullptr;
		const int64_t cap = static_cast<int64_t>(g_steps) + 8;
		CHECK_MSG(sslm_gpu_seq_create(ctx, m.handle, cap, &seq) == SSLM_OK && seq, "6.3: seq_create");
		if (!seq) return;
		std::vector<uint8_t> record;
		int32_t token = kProcedurePrompt[0];
		bool ok = true;
		for (int i = 0; i < g_steps && ok; ++i) {
			int32_t next = -1;
			ok = SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, token, m.unit * m.layers, &next) == SSLM_OK;
			const uint8_t* p = reinterpret_cast<const uint8_t*>(&next);
			record.insert(record.end(), p, p + sizeof(next));
			token = next;
		}
		CHECK_MSG(ok, "6.3 run %d: a decode step failed", run);
		std::vector<uint8_t> blob;
		CHECK_MSG(SaveBlob(ctx, seq, &blob), "6.3 run %d: final save failed", run);
		record.insert(record.end(), blob.begin(), blob.end());
		digests[run] = Sha256Hex(record);
		std::printf("6.3 run %d: %d tokens, digest %s\n", run, g_steps, digests[run].c_str());
		sslm_gpu_seq_release(ctx, seq);
	}
	CHECK_MSG(digests[0] == digests[1] && digests[1] == digests[2],
	          "6.3: three bracket-mode runs of %d tokens must give one digest", g_steps);
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
		else if (const char* v2 = take("--modelqk=")) g_model_qk_path = v2;
		else if (const char* v3 = take("--v1p11-blob=")) g_v1p11_blob_path = v3;
		else if (const char* v4 = take("--steps=")) g_steps = std::max(1, std::atoi(v4));
	}
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	if (g_model_0p5b_path.empty()) {
		SKIP_MSG("7.2, 7.3, 8.2, 8.3, 9.1, 6.3: --model0p5b not supplied");
	} else {
		MappedModel m;
		if (MapModel(ctx, g_model_0p5b_path, 0, &m)) {
			CHECK_MSG(m.unit == superslm_gpu::kLegacyDispatchesPerLayer && m.layers == 24,
			          "the 0.5B artifact must be the 24-layer legacy model (unit 24); got %u layers, "
			          "unit %u",
			          m.layers, m.unit);
			Cell7_2(ctx, m);
			Cell7_3(ctx, m, "0.5B");
			std::vector<int32_t> host_tokens;
			Cell8_2And9_1(ctx, m, &host_tokens);
			Cell6_3(ctx, m);
			sslm_gpu_model_unmap(ctx, m.handle);
			Cell8_3(ctx, host_tokens);
		}
	}
	if (g_model_qk_path.empty()) {
		SKIP_MSG("7.3 at unit 25: --modelqk not supplied");
	} else {
		MappedModel qk;
		if (MapModel(ctx, g_model_qk_path, 0, &qk)) {
			CHECK_MSG(qk.unit == superslm_gpu::kQkNormDispatchesPerLayer,
			          "--modelqk must carry QK norm (unit 25); got unit %u", qk.unit);
			Cell7_3(ctx, qk, "QK-norm");
			sslm_gpu_model_unmap(ctx, qk.handle);
		}
	}
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures == 0 ? 0 : 1;
}
