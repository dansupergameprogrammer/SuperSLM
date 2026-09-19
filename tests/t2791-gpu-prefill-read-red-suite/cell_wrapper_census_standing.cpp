// T-2791 (Curie) -- plan Sec3.4 row 7, "The snapshot changes no existing behaviour": "Every
// Sec2.6 member's next decode-wrapper status equals the 1.5.0 status recorded in
// te269-census-output.txt, reproduced on the 1.6.0 build."
//
// A STANDING cell, not a red one: it asserts preserved 1.5.0 behaviour, so it is green at
// v1.5.0 by construction and must stay green after the build (the same disposition as
// tests/t2178-gpu-batched-prefill-red-suite/cell_exit_census.cpp). What it pins is that adding
// the snapshot's writers to create, reset and the two prefill twins moves no decode observable.
//
// ORACLE: the "next_wrapper" column of the planner's executed census at v1.5.0
// (Claude/Vitruvius/te269-probe/te269-census-output.txt, Qwen3-Embedding-0.6B), identical on the
// re-strike's two Qwen2.5-1.5B runs (Claude/Loki/te270-census-qwen25-1p5b-*.txt). Both are 1.5.0
// readings taken before any 1.6.0 code existed. The M01, M02, M05, M07, M14 SEQUENCE_REJECTED
// rows are the shipped decode-shortcut defect plan Sec3.3 leaves unrepaired in 1.6.0 (P-9) --
// pinned here as they are, so a 1.6.0 build that silently changes them fails this cell and the
// change has to be made on purpose.
//
// It calls no read verb, so it links and runs at v1.5.0.
//
// Run: cell_wrapper_census_standing.exe [--qwen3=PATH] [--synthetic=PATH] [--g5fixture=PATH]
#include "fixture_common.h"

namespace {

void RunOn(const std::string& path, const char* flag, SslmGpuContext* ctx) {
	if (path.empty()) {
		SKIP_MSG("wrapper census: %s not supplied", flag);
		return;
	}
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "could not open %s", path.c_str());
		return;
	}
	const char* tag = path.c_str();
	SslmGpuSequenceHandle* seq = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, std::min<int64_t>(512, fx.model_cap), &seq) == SSLM_OK && seq);
	if (!seq) { fx.Close(); return; }
	const std::vector<int32_t> P = fx.TokensP(), Q = fx.TokensQ();
	auto base = [&]() {
		CHECK(sslm_gpu_seq_reset(ctx, seq) == SSLM_OK);
		CHECK(Prefill(fx, seq, P) == SSLM_OK);
	};
	auto wrapper = [&](SslmGpuSequenceHandle* s) {
		int32_t t = -1;
		return SslmGpuSeqDecodeStepForG5Bridge(ctx, s, 43 % fx.vocab, fx.one_layer_budget, &t);
	};
	auto expect = [&](const char* member, SslmGpuSequenceHandle* s, SslmGpuStatus want) {
		const SslmGpuStatus got = wrapper(s);
		CHECK_MSG(got == want, "[%s] %s: next decode-wrapper status %s, recorded at 1.5.0 as %s", tag, member,
		          StatusName(got), StatusName(want));
	};
	const int32_t tok = fx.SomeToken();

	base(); CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK); expect("M01", seq, SSLM_SEQUENCE_REJECTED);
	base(); { int32_t t = -1; CHECK(SslmGpuSeqFinishTokenForG5Bridge(ctx, seq, &t) == SSLM_OK); } expect("M02", seq, SSLM_SEQUENCE_REJECTED);
	base(); { int32_t t = -1; CHECK(SslmGpuSeqDecodeStepForG5Bridge(ctx, seq, 43 % fx.vocab, fx.one_layer_budget, &t) == SSLM_OK); } expect("M03", seq, SSLM_OK);
	base(); CHECK(sslm_gpu_seq_reset(ctx, seq) == SSLM_OK); expect("M04", seq, SSLM_OK);
	base(); CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK);
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, fx.one_layer_budget) == SSLM_OK); CHECK(Drain(ctx, seq) == SSLM_OK);
	expect("M05", seq, SSLM_SEQUENCE_REJECTED);
	base(); CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK);
	{
		SslmGpuStatus st = SSLM_OK;
		while (LayerIndex(seq) < fx.layers && st == SSLM_OK) {
			st = sslm_decode_step_gpu(ctx, seq, nullptr, fx.one_layer_budget * fx.layers);
			if (st == SSLM_OK) st = Drain(ctx, seq);
		}
	}
	expect("M06", seq, SSLM_OK);
	base(); CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK);
	{
		SslmGpuSequenceHandle* seqs[1] = {seq};
		SslmGpuStatus outs[1] = {SSLM_OK};
		CHECK(sslm_decode_step_batch_gpu(ctx, seqs, nullptr, 1, fx.one_layer_budget, outs) == SSLM_OK);
		Drain(ctx, seq);
	}
	expect("M07", seq, SSLM_SEQUENCE_REJECTED);
	base(); CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, fx.one_layer_budget) == SSLM_DISPATCH_BUDGET_TOO_SMALL); expect("M08", seq, SSLM_OK);
	base(); CHECK(Prefill(fx, seq, Q) == SSLM_OK); expect("M09", seq, SSLM_OK);
	base(); CHECK(Prefill(fx, seq, {tok, 43 % fx.vocab, -1}) == SSLM_TOKEN_ID_OUT_OF_RANGE); expect("M10", seq, SSLM_OK);
	base(); CHECK(Prefill(fx, seq, {-1}) == SSLM_TOKEN_ID_OUT_OF_RANGE); expect("M11", seq, SSLM_OK);
	base(); CHECK(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, P.data(), 0, fx.one_layer_budget) == SSLM_OK); expect("M12", seq, SSLM_OK);
	base(); CHECK(SslmGpuSeqPrefillPromptForG5Bridge(ctx, seq, Q.data(), 2, 0) == SSLM_SEQUENCE_KV_BUFFER_MISMATCH); expect("M13", seq, SSLM_OK);
	base(); CHECK(sslm_gpu_seq_embed_token(ctx, seq, tok) == SSLM_OK);
	CHECK(sslm_decode_step_gpu(ctx, seq, nullptr, fx.one_layer_budget) == SSLM_OK);
	CHECK(Prefill(fx, seq, Q) == SSLM_BUSY); CHECK(Drain(ctx, seq) == SSLM_OK);
	expect("M14c", seq, SSLM_SEQUENCE_REJECTED);
	base();
	{
		size_t need = 0;
		sslm_gpu_seq_save(ctx, seq, nullptr, &need);
		std::vector<uint8_t> blob(need);
		size_t got = need;
		CHECK(sslm_gpu_seq_save(ctx, seq, blob.data(), &got) == SSLM_OK);
		SslmGpuSequenceHandle* r = nullptr;
		CHECK(sslm_gpu_seq_restore(ctx, fx.model, blob.data(), got, &r) == SSLM_OK && r);
		if (r) { expect("M15b", r, SSLM_OK); sslm_gpu_seq_release(ctx, r); }
	}
	base();
	{
		CHECK(sslm_gpu_seq_bind_adapter(ctx, seq, nullptr) == SSLM_OK);
		CHECK(SslmGpuSeqSetSchemaForG5Bridge(ctx, seq, -1) == SSLM_OK);
		int32_t r = 0; SslmGpuStatus rs = SSLM_OK;
		CHECK(sslm_gpu_ready(ctx, seq, 0, &r, &rs) == SSLM_OK);
	}
	expect("M16", seq, SSLM_OK);
	{
		const int64_t cap = std::min<int64_t>(64, fx.model_cap);
		SslmGpuSequenceHandle* s = nullptr;
		CHECK(sslm_gpu_seq_create(ctx, fx.model, cap, &s) == SSLM_OK && s);
		if (s) {
			CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK && Prefill(fx, s, fx.Run(static_cast<size_t>(cap), 1000 % fx.vocab)) == SSLM_OK);
			CHECK(Prefill(fx, s, {tok}) == SSLM_DEVICE_LOST);
			expect("M20", s, SSLM_SEQUENCE_REJECTED);
			CHECK(sslm_gpu_seq_reset(ctx, s) == SSLM_OK && Prefill(fx, s, fx.Run(static_cast<size_t>(cap - 4), 1000 % fx.vocab)) == SSLM_OK);
			CHECK(Prefill(fx, s, fx.Run(8, 3000 % fx.vocab)) == SSLM_DEVICE_LOST);
			expect("M21", s, SSLM_SEQUENCE_REJECTED);
			sslm_gpu_seq_release(ctx, s);
		}
	}
	sslm_gpu_seq_release(ctx, seq);
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	ParseFixtureArgs(argc, argv);
	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK || !ctx) {
		std::printf("FATAL: sslm_gpu_context_create failed\n");
		return 2;
	}
	RunOn(g_qwen3_path, "--qwen3", ctx);
	RunOn(g_synthetic_path, "--synthetic", ctx);
	RunOn(g_g5_path, "--g5fixture", ctx);
	sslm_gpu_context_destroy(ctx);
	return FinishSuite("cell_wrapper_census_standing");
}
