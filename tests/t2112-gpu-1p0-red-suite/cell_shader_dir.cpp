// T-2948: process-wide shader override against a poisoned executable-side shader set.
// Run each arm in a fresh process. The real 1.5B model and real compiled shader set are required.
#include "fixture_common.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <utility>

#include "../../src/gpu/d3d12_harness.h"
#include <shlwapi.h>

namespace fs = std::filesystem;

static std::string g_arm;
static std::string g_shader_dir;

static bool Require(SslmGpuStatus got, SslmGpuStatus want, const char* step) {
	CHECK_MSG(got == want, "%s: got status %u, expected %u", step,
	          static_cast<unsigned>(got), static_cast<unsigned>(want));
	return got == want;
}

static void InvalidRows() {
	const fs::path selected(g_shader_dir);
	const fs::path empty_dir = selected.parent_path() / "t2808" / "empty";
	const std::string missing = (selected / "t2948-no-such-directory").string();
	const std::string file = g_model_1p5b_path;
	const std::string empty = empty_dir.string();
	const char invalid_utf8[] = "\xC3\x28";
	const std::string rooted = selected.string().substr(2);
	const std::string drive_relative = std::string(1, selected.string()[0]) + ":shaders";
	const char* values[] = {"", "shaders", missing.c_str(), file.c_str(),
	                        empty.c_str(), invalid_utf8, "\\shaders", "C:shaders",
	                        rooted.c_str(), drive_relative.c_str()};
	const char* names[] = {"empty", "relative", "missing", "file", "no-cso", "invalid-utf8",
	                       "rooted-literal", "drive-relative-literal", "rooted-existing",
	                       "drive-relative-existing"};
	const fs::path original_cwd = fs::current_path();
	fs::current_path(selected.parent_path());
	for (size_t i = 0; i < 10; ++i) {
		std::wstring normalized;
		const auto pre = superslm_gpu::harness::CheckShaderDirOverride(values[i], &normalized);
		CHECK_MSG(pre == superslm_gpu::harness::ShaderDirCheck::Invalid,
		          "%s: pre-device validator accepted a path dependent on current directory",
		          names[i]);
		if (pre != superslm_gpu::harness::ShaderDirCheck::Invalid) continue;
		GpuContextConfig cfg{};
		cfg.shader_dir = values[i];
		SslmGpuContext* rejected = reinterpret_cast<SslmGpuContext*>(uintptr_t{1});
		const SslmGpuStatus st = sslm_gpu_context_create(cfg, &rejected);
		CHECK_MSG(st == SSLM_GPU_SHADER_DIR_INVALID,
		          "%s: got %u, expected INVALID", names[i], static_cast<unsigned>(st));
		CHECK_MSG(rejected == nullptr, "%s: rejected create retained a context", names[i]);
		if (rejected && rejected != reinterpret_cast<SslmGpuContext*>(uintptr_t{1})) {
			sslm_gpu_context_destroy(rejected);
		}
	}
	fs::current_path(original_cwd);
}

static void AbsolutePathControls() {
	const std::string backslash = g_shader_dir;
	std::string slash = backslash;
	std::replace(slash.begin(), slash.end(), '\\', '/');
	const std::string unc = "\\\\localhost\\" + std::string(1, backslash[0]) + "$" +
	                        backslash.substr(2);
	for (const auto& row : {std::pair{"drive-backslash", backslash},
	                        std::pair{"drive-forward-slash", slash}}) {
		std::wstring normalized;
		const auto st = superslm_gpu::harness::CheckShaderDirOverride(row.second.c_str(),
		                                                            &normalized);
		CHECK_MSG(st == superslm_gpu::harness::ShaderDirCheck::Ok,
		          "%s: existing absolute shader directory rejected (%u)", row.first,
		          static_cast<unsigned>(st));
	}
	std::error_code unc_ec;
	if (fs::is_directory(unc, unc_ec)) {
		std::wstring normalized;
		const auto st = superslm_gpu::harness::CheckShaderDirOverride(unc.c_str(), &normalized);
		CHECK_MSG(st == superslm_gpu::harness::ShaderDirCheck::Ok,
		          "UNC: existing absolute shader directory rejected (%u)", static_cast<unsigned>(st));
	} else {
		std::printf("UNC control: local administrative share unavailable; syntax-only pin applies\n");
	}
	// These exact spellings are syntactically absolute. Their absent directories can still
	// make the production validator return INVALID at the later existence check.
	CHECK(PathIsRelativeW(L"C:\\shaders") == FALSE);
	CHECK(PathIsRelativeW(L"D:/x") == FALSE);
	CHECK(PathIsRelativeW(L"\\\\server\\share") == FALSE);
}

static void OverrideAndConflict() {
	GpuContextConfig cfg{};
	cfg.shader_dir = g_shader_dir.c_str();
	SslmGpuContext* ctx = nullptr;
	if (!Require(sslm_gpu_context_create(cfg, &ctx), SSLM_OK, "override create") || !ctx) return;

	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		CHECK_MSG(false, "real model load: %s", err.c_str());
		sslm_gpu_context_destroy(ctx);
		return;
	}
	CpuOracleModel cpu{};
	if (!LoadCpuOracleModel(view, &cpu, &err)) {
		CHECK_MSG(false, "CPU oracle load: %s", err.c_str());
		sslm_gpu_context_destroy(ctx);
		return;
	}
	SslmGpuModelHandle* model = nullptr;
	if (!Require(sslm_gpu_model_map(ctx, &view, {}, &model), SSLM_OK, "model map") || !model) {
		sslm_gpu_context_destroy(ctx);
		return;
	}
	SslmGpuSequenceHandle* seq = nullptr;
	if (!Require(sslm_gpu_seq_create(ctx, model, 128, &seq), SSLM_OK, "sequence create") || !seq) {
		sslm_gpu_model_unmap(ctx, model);
		sslm_gpu_context_destroy(ctx);
		return;
	}
	CpuOracleRunner oracle{};
	oracle.Init(cpu);
	const int32_t prompt[8] = {5, 6, 7, 8, 9, 10, 11, 12};
	bool ok = true;
	for (int32_t token : prompt) {
		if (StepCpu(oracle.seq, token, cpu, oracle.ws.data(), oracle.ws.size()) !=
		    superslm::SslmForwardStatus::Ok) ok = false;
	}
	CHECK_MSG(ok, "CPU oracle must complete the eight-token prompt");
	const SslmGpuStatus prefill = SslmGpuSeqPrefillPromptForG5Bridge(
	    ctx, seq, prompt, 8, FullTokenBudget(cpu.num_hidden_layers));
	CHECK_MSG(prefill == SSLM_OK, "override prefill got %u", static_cast<unsigned>(prefill));
	if (ok && prefill == SSLM_OK) {
		SeqSnapshot snap{};
		CHECK(CaptureSnapshot(seq, &snap));
		CHECK_MSG(oracle.seq.context_length == snap.context_length &&
		          oracle.seq.kv_saturation_count == snap.kv_saturation_count &&
		          oracle.codes == snap.hidden_codes,
		          "eight-token prefill diverged from the independent CPU forward oracle");
		int32_t previous = prompt[7];
		for (int step = 0; step < 64; ++step) {
			int32_t next = -1;
			const SslmGpuStatus st = SslmGpuSeqDecodeStepForG5Bridge(
			    ctx, seq, previous, FullTokenBudget(cpu.num_hidden_layers), &next);
			CHECK_MSG(st == SSLM_OK && next >= 0, "decode step %d: status %u token %d",
			          step, static_cast<unsigned>(st), next);
			if (st != SSLM_OK || next < 0) break;
			if (step > 0) {
				CHECK(StepCpu(oracle.seq, previous, cpu, oracle.ws.data(), oracle.ws.size()) ==
				      superslm::SslmForwardStatus::Ok);
			}
			CHECK(CaptureSnapshot(seq, &snap));
			CHECK_MSG(oracle.seq.context_length == snap.context_length &&
			          oracle.seq.kv_saturation_count == snap.kv_saturation_count &&
			          oracle.codes == snap.hidden_codes,
			          "decode step %d hidden state diverged from CPU", step);
			previous = next;
		}
	}

	// Validation and process-wide inheritance occur after the successful override in this process.
	InvalidRows();
	GpuContextConfig inherited{};
	SslmGpuContext* null_ctx = nullptr;
	CHECK(sslm_gpu_context_create(inherited, &null_ctx) == SSLM_OK && null_ctx);
	if (null_ctx) {
		SslmGpuModelHandle* null_model = nullptr;
		CHECK(sslm_gpu_model_map(null_ctx, &view, {}, &null_model) == SSLM_OK && null_model);
		if (null_model) {
			SslmGpuSequenceHandle* null_seq = nullptr;
			CHECK(sslm_gpu_seq_create(null_ctx, null_model, 128, &null_seq) == SSLM_OK && null_seq);
			if (null_seq) {
				CHECK(SslmGpuSeqPrefillPromptForG5Bridge(null_ctx, null_seq, prompt, 1,
				      FullTokenBudget(cpu.num_hidden_layers)) == SSLM_OK);
				CHECK(sslm_gpu_seq_release(null_ctx, null_seq) == SSLM_OK);
			}
			CHECK(sslm_gpu_model_unmap(null_ctx, null_model) == SSLM_OK);
		}
		CHECK(sslm_gpu_context_destroy(null_ctx) == SSLM_OK);
	}
	std::string trailing = g_shader_dir + "\\";
	std::string slash = g_shader_dir;
	std::replace(slash.begin(), slash.end(), '\\', '/');
	std::string upper = g_shader_dir;
	std::transform(upper.begin(), upper.end(), upper.begin(),
	               [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
	for (const std::string& spelling : {trailing, slash, upper}) {
		GpuContextConfig same{};
		same.shader_dir = spelling.c_str();
		SslmGpuContext* equal_ctx = nullptr;
		CHECK_MSG(sslm_gpu_context_create(same, &equal_ctx) == SSLM_OK && equal_ctx,
		          "equivalent spelling rejected: %s", spelling.c_str());
		if (equal_ctx) CHECK(sslm_gpu_context_destroy(equal_ctx) == SSLM_OK);
	}
	const fs::path poison = fs::path(g_shader_dir).parent_path() / "t2808" / "shaders";
	const std::string poison_string = poison.string();
	GpuContextConfig conflict{};
	conflict.shader_dir = poison_string.c_str();
	SslmGpuContext* rejected = reinterpret_cast<SslmGpuContext*>(uintptr_t{1});
	CHECK(sslm_gpu_context_create(conflict, &rejected) == SSLM_GPU_SHADER_DIR_CONFLICT);
	CHECK(rejected == nullptr);
	if (rejected && rejected != reinterpret_cast<SslmGpuContext*>(uintptr_t{1}))
		sslm_gpu_context_destroy(rejected);
	CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
}

static void PoisonedDefault() {
	SslmGpuContext* ctx = nullptr;
	if (!Require(sslm_gpu_context_create({}, &ctx), SSLM_OK, "default create") || !ctx) return;
	std::vector<uint8_t> bytes;
	SslmModelView view{};
	std::string err;
	if (!LoadRealModel(g_model_1p5b_path, &view, &bytes, &err)) {
		CHECK_MSG(false, "real model load: %s", err.c_str());
		sslm_gpu_context_destroy(ctx);
		return;
	}
	SslmGpuModelHandle* model = nullptr;
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_model_map(ctx, &view, {}, &model) == SSLM_OK && model);
	if (model) CHECK(sslm_gpu_seq_create(ctx, model, 128, &seq) == SSLM_OK && seq);
	if (seq) {
		const int32_t prompt[] = {5};
		const SslmGpuStatus st = SslmGpuSeqPrefillPromptForG5Bridge(
		    ctx, seq, prompt, 1, FullTokenBudget(view.config.num_hidden_layers));
		CHECK_MSG(st != SSLM_OK, "poisoned executable-side .cso unexpectedly ran");
		CHECK(sslm_gpu_seq_release(ctx, seq) == SSLM_OK);
	}
	if (model) CHECK(sslm_gpu_model_unmap(ctx, model) == SSLM_OK);
	CHECK(sslm_gpu_context_destroy(ctx) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	for (int i = 1; i < argc; ++i) {
		const std::string arg = argv[i];
		if (arg.rfind("--arm=", 0) == 0) g_arm = arg.substr(6);
		if (arg.rfind("--shader-dir=", 0) == 0) g_shader_dir = arg.substr(13);
	}
	if ((g_arm != "invalid-only" && (g_model_1p5b_path.empty() ||
	     !fs::is_regular_file(g_model_1p5b_path))) ||
	    (g_arm != "poisoned-default" && g_arm != "override" && g_arm != "invalid-only") ||
	    (g_arm != "poisoned-default" && !fs::is_directory(g_shader_dir))) {
		std::printf("USAGE: --model1p5b=<real model> --arm=poisoned-default|override "
		            "--shader-dir=<absolute compiled .cso directory>\n");
		return 2;
	}
	if (g_arm == "invalid-only") { InvalidRows(); AbsolutePathControls(); }
	else if (g_arm == "poisoned-default") PoisonedDefault();
	else OverrideAndConflict();
	std::printf("arm=%s checks=%d failures=%d skips=%d\n", g_arm.c_str(), GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
