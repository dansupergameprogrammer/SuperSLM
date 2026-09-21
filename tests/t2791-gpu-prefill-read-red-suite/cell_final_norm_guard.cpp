// T-2795 (Curie) -- plan Sec3.4 row 5, the final-norm guard refusal (plan Sec3.1 step 5), on
// fixture A2-fn; and its row 11 guard-vitality mutant (g).
//
// FIXTURE. A2-fn is the synthetic fused-K fixture (--synthetic, the U1 pair model) with its
// CompositionConstants entry "final_norm" set to mantissa -2,147,483,647, its exponent unchanged,
// and its integrity SHA-256 recomputed. make_a2fn_fixture.py in this directory builds it; this
// cell pins both the source's and the output's SHA-256.
//
// CELL, per prompt (six prompts of 3, 1, 8, 16, 40 and 63 tokens, sequence cap 64):
//   G1 A2-fn loads and maps; create, reset and SslmGpuSeqPrefillPromptForG5Bridge return SSLM_OK;
//   G2 the read returns SSLM_SEQUENCE_REJECTED, leaves the sentinel-filled out_codes, out_scale_m and
//      out_scale_e unchanged, and sets *out_required == hidden_size.
// MUST-ACCEPT NEIGHBOUR: the same prompts on the unpatched --synthetic fixture read the O-LIVE frame.
// SETUP PRECONDITION, per A2-fn prompt: this cell's own CPU RmsNormSite, over the residual the bench
// accessor exposes right after the prefill, with A2-fn's parsed final_norm gain and constant, must
// return CarriedScaleMantissaOutOfDomain. The refusal depends on the residual (plan Sec3.4 row 5),
// so a prompt that misses the precondition fails as SETUP and is never counted as a refusal cell.
//
// GUARD VITALITY: mutant (g) (the verb ignores RmsNormSite's status, returns SSLM_OK and copies its
// verb-local storage out) turns G2 red by status.
//
// ORACLE: O-LIVE (fixture_common.h) for the must-accept frame; O-CONTRACT (plan Sec3.1 steps 4-5)
// for the refusal. Neither takes input from the snapshot or the read.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_final_norm_guard.exe --synthetic=PATH --a2fn=PATH
#include "fixture_common.h"

namespace {

constexpr const char* kA2Sha256 = "a231d9ed9dd3944a253201aa9be418fb3b7f7ad258f79e57cf39297dc8954f70";
constexpr const char* kA2fnSha256 = "73a1ec9e1846be16af7f8dc13511207d4129a9e825cc5271f783f6bc1ff4f940";
constexpr int64_t kA2FinalNormM = 1090717716;
constexpr int64_t kA2FinalNormE = -60;
constexpr int64_t kA2fnFinalNormM = -2147483647;

bool FileSha256(const std::string& path, std::string* out) {
	std::vector<uint8_t> b;
	if (!ReadFileBytes(path, &b)) return false;
	*out = Sha256Hex(b.data(), b.size());
	return true;
}

std::vector<std::vector<int32_t>> Prompts(const GpuModelFixture& fx) {
	return {fx.TokensP(), fx.Run(1, 5), fx.Run(8, 11), fx.Run(16, 200), fx.Run(40, 3), fx.Run(63, 17)};
}

// Plan Sec3.4 row 5's setup precondition: the CPU site refuses on this residual.
superslm::SslmForwardStatus CpuFinalNorm(const GpuModelFixture& fx, SslmGpuSequenceHandle* seq) {
	std::vector<int8_t> out(fx.hidden, 0);
	superslm::CarriedScale scale{};
	return superslm::RmsNormSite(SslmGpuSeqHandleHiddenCodesForBench(seq), fx.final_gain.data(), fx.hidden,
	                             superslm::CarriedScale{}, fx.final_const, out.data(), &scale, "final_norm");
}

void MustAccept(SslmGpuContext* ctx) {
	GpuModelFixture fx;
	const bool opened = fx.Open(g_synthetic_path, ctx);
	CHECK_MSG(opened, "A2 must-accept: the unpatched fixture did not load and map");
	if (!opened) return;
	CHECK_MSG(fx.final_const.m == kA2FinalNormM && fx.final_const.e == kA2FinalNormE,
	          "SETUP: A2's final_norm constant is (%lld, %lld), want (%lld, %lld)",
	          static_cast<long long>(fx.final_const.m), static_cast<long long>(fx.final_const.e),
	          static_cast<long long>(kA2FinalNormM), static_cast<long long>(kA2FinalNormE));
	const auto prompts = Prompts(fx);
	for (size_t i = 0; i < prompts.size(); ++i) {
		SslmGpuSequenceHandle* s = nullptr;
		CHECK_MSG(sslm_gpu_seq_create(ctx, fx.model, 64, &s) == SSLM_OK && s, "A2 prompt%zu: create", i);
		if (!s) continue;
		CHECK_MSG(sslm_gpu_seq_reset(ctx, s) == SSLM_OK, "A2 prompt%zu: reset", i);
		const SslmGpuStatus ps = Prefill(fx, s, prompts[i]);
		CHECK_MSG(ps == SSLM_OK, "A2 prompt%zu (n=%zu): prefill returned %s", i, prompts[i].size(), StatusName(ps));
		if (ps == SSLM_OK) {
			const Frame want = OracleFromLive(fx, s);
			CHECK_MSG(want.IsOk(), "SETUP: A2 prompt%zu: the O-LIVE oracle refused on the unpatched fixture", i);
			const Frame got = ReadVerb(fx, s);
			CHECK_MSG(got.SameFrame(want, fx.hidden),
			          "A2 prompt%zu (n=%zu) must-accept: read returned %s, differing from O-LIVE in %zu of %zu codes",
			          i, prompts[i].size(), StatusName(got.status), DiffCodes(got, want, fx.hidden), fx.hidden);
			std::printf("    A2 prompt%zu n=%zu read=%s frame==O-LIVE=%d\n", i, prompts[i].size(),
			            StatusName(got.status), got.SameFrame(want, fx.hidden) ? 1 : 0);
		}
		sslm_gpu_seq_release(ctx, s);
	}
	fx.Close();
}

void GuardRefusal(SslmGpuContext* ctx) {
	GpuModelFixture fx;
	const bool opened = fx.Open(g_a2fn_path, ctx);
	CHECK_MSG(opened, "G1: A2-fn did not load and map (plan Sec3.4 row 5: if a later loader rejects a negative "
	                  "final_norm constant, this cell becomes a load-rejection cell)");
	if (!opened) return;
	CHECK_MSG(fx.final_const.m == kA2fnFinalNormM && fx.final_const.e == kA2FinalNormE,
	          "SETUP: A2-fn's parsed final_norm constant is (%lld, %lld), want (%lld, %lld)",
	          static_cast<long long>(fx.final_const.m), static_cast<long long>(fx.final_const.e),
	          static_cast<long long>(kA2fnFinalNormM), static_cast<long long>(kA2FinalNormE));
	const auto prompts = Prompts(fx);
	for (size_t i = 0; i < prompts.size(); ++i) {
		SslmGpuSequenceHandle* s = nullptr;
		CHECK_MSG(sslm_gpu_seq_create(ctx, fx.model, 64, &s) == SSLM_OK && s, "G1 prompt%zu: create", i);
		if (!s) continue;
		CHECK_MSG(sslm_gpu_seq_reset(ctx, s) == SSLM_OK, "G1 prompt%zu: reset", i);
		const SslmGpuStatus ps = Prefill(fx, s, prompts[i]);
		CHECK_MSG(ps == SSLM_OK, "G1 prompt%zu (n=%zu): prefill on A2-fn returned %s, want SSLM_OK", i,
		          prompts[i].size(), StatusName(ps));
		if (ps != SSLM_OK) {
			sslm_gpu_seq_release(ctx, s);
			continue;
		}
		const superslm::SslmForwardStatus pre = CpuFinalNorm(fx, s);
		const bool precondition = pre == superslm::SslmForwardStatus::CarriedScaleMantissaOutOfDomain;
		CHECK_MSG(precondition,
		          "SETUP: A2-fn prompt%zu (n=%zu): the CPU RmsNormSite did not return CarriedScaleMantissaOutOfDomain "
		          "(status %d); this residual cannot force the guard refusal",
		          i, prompts[i].size(), static_cast<int>(pre));
		if (precondition) {
			const Frame got = ReadVerb(fx, s);
			CHECK_MSG(got.status == SSLM_SEQUENCE_REJECTED,
			          "G2 prompt%zu (n=%zu): read returned %s, want SSLM_SEQUENCE_REJECTED (the step-5 guard refusal)",
			          i, prompts[i].size(), StatusName(got.status));
			CHECK_MSG(got.OutputsUntouched(), "G2 prompt%zu: a refused read wrote out_codes, out_scale_m or out_scale_e", i);
			CHECK_MSG(got.required == fx.hidden, "G2 prompt%zu: *out_required is %zu, want %zu", i, got.required,
			          fx.hidden);
			std::printf("    A2-fn prompt%zu n=%zu prefill=%s cpu_precondition=refuses read=%s untouched=%d required=%zu\n",
			            i, prompts[i].size(), StatusName(ps), StatusName(got.status), got.OutputsUntouched() ? 1 : 0,
			            got.required);
		}
		sslm_gpu_seq_release(ctx, s);
	}
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	RunAsLegDriverIfRequested(argc, argv, "cell_final_norm_guard");
	ParseFixtureArgs(argc, argv);
	if (g_synthetic_path.empty() || g_a2fn_path.empty()) {
		SKIP_MSG("the final-norm guard cell needs --synthetic=PATH and --a2fn=PATH");
		return FinishSuite("cell_final_norm_guard");
	}
	std::string h;
	const bool a2_ok = FileSha256(g_synthetic_path, &h) && h == kA2Sha256;
	CHECK_MSG(a2_ok, "SETUP: --synthetic is not the U1 pair model A2-fn is built from (sha256 %s, want %s)", h.c_str(),
	          kA2Sha256);
	const bool fn_ok = FileSha256(g_a2fn_path, &h) && h == kA2fnSha256;
	CHECK_MSG(fn_ok, "SETUP: --a2fn is not make_a2fn_fixture.py's output (sha256 %s, want %s)", h.c_str(), kA2fnSha256);
	if (!a2_ok || !fn_ok) return FinishSuite("cell_final_norm_guard");

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	MustAccept(ctx);
	GuardRefusal(ctx);
	sslm_gpu_context_destroy(ctx);
	return FinishSuite("cell_final_norm_guard");
}
