// T-2791 (Curie) -- plan Sec3.4 row 3 (concurrency), beyond the M14 Submitted-window cells in
// cell_census_lifetime.cpp:
//   K1 "Reads on two sequences from two threads give the same bytes as serial reads": two
//      sequences prefilled with different prompts, each read 200 times on its own thread,
//      concurrently; every concurrent read equals that sequence's serial read and its oracle frame.
//   K2 SSLM_BUSY is per sequence (plan Sec3.1 check 2 tests "Sequence submitted" -- this handle's
//      state): while sequence B is Submitted, sequence A (idle, with a snapshot) reads its frame,
//      and B's own read returns SSLM_BUSY.
//   K3 A read writes nothing on the sequence (Sec3.1 "Non-mutating"): every host-visible field
//      the bench bridge exposes is identical before and after 50 reads.
// The read is host-side and submits nothing (Sec3.1), so K1 runs without the external submit
// serialization the header requires of submitting calls; the prefills before it are serial.
//
// ORACLE: final_norm of each sequence's live residual right after its own prefill
// (fixture_common.h, OracleFromLive), and the bench-bridge state capture for K3.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_concurrency.exe [--qwen3=PATH] [--synthetic=PATH] [--g5fixture=PATH]
#include <atomic>
#include <thread>

#include "fixture_common.h"

namespace {

void RunOn(const std::string& path, const char* flag, SslmGpuContext* ctx) {
	if (path.empty()) {
		SKIP_MSG("concurrency: %s not supplied", flag);
		return;
	}
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "could not open %s", path.c_str());
		return;
	}
	const char* tag = path.c_str();
	SslmGpuSequenceHandle* a = nullptr;
	SslmGpuSequenceHandle* b = nullptr;
	CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &a) == SSLM_OK && a);
	CHECK(sslm_gpu_seq_create(ctx, fx.model, 64, &b) == SSLM_OK && b);
	if (!a || !b) { fx.Close(); return; }
	CHECK(sslm_gpu_seq_reset(ctx, a) == SSLM_OK && Prefill(fx, a, fx.TokensP()) == SSLM_OK);
	const Frame FA = OracleFromLive(fx, a);
	CHECK(sslm_gpu_seq_reset(ctx, b) == SSLM_OK && Prefill(fx, b, fx.TokensQ()) == SSLM_OK);
	const Frame FB = OracleFromLive(fx, b);
	CHECK_MSG(!FA.SameFrame(FB, fx.hidden), "[%s] SETUP: the two sequences' frames must differ", tag);

	// K1
	const Frame SA = ReadVerb(fx, a);
	const Frame SB = ReadVerb(fx, b);
	CHECK_MSG(SA.status == SSLM_OK && SA.SameFrame(FA, fx.hidden), "[%s] K1 serial read of A (%s)", tag,
	          StatusName(SA.status));
	CHECK_MSG(SB.status == SSLM_OK && SB.SameFrame(FB, fx.hidden), "[%s] K1 serial read of B (%s)", tag,
	          StatusName(SB.status));
	std::atomic<int> bad_a{0}, bad_b{0};
	auto reader = [&](SslmGpuSequenceHandle* s, const Frame& want, std::atomic<int>* bad) {
		for (int i = 0; i < 200; ++i) {
			const Frame f = ReadVerb(fx, s);
			if (!(f.status == SSLM_OK && f.SameFrame(want, fx.hidden))) bad->fetch_add(1);
		}
	};
	std::thread ta(reader, a, std::cref(SA), &bad_a);
	std::thread tb(reader, b, std::cref(SB), &bad_b);
	ta.join();
	tb.join();
	CHECK_MSG(bad_a.load() == 0 && bad_b.load() == 0,
	          "[%s] K1 concurrent reads on two sequences: %d of 200 (A) and %d of 200 (B) differ from the serial read",
	          tag, bad_a.load(), bad_b.load());

	// K2
	CHECK(sslm_gpu_seq_embed_token(ctx, b, fx.SomeToken()) == SSLM_OK);
	CHECK(sslm_decode_step_gpu(ctx, b, nullptr, fx.one_layer_budget) == SSLM_OK);
	const Frame ra = ReadVerb(fx, a);
	CHECK_MSG(ra.status == SSLM_OK && ra.SameFrame(FA, fx.hidden),
	          "[%s] K2 an idle sequence reads its frame while ANOTHER sequence is Submitted (%s)", tag,
	          StatusName(ra.status));
	const Frame rb = ReadVerb(fx, b);
	CHECK_MSG(rb.status == SSLM_BUSY, "[%s] K2 the Submitted sequence's own read returns SSLM_BUSY (%s)", tag,
	          StatusName(rb.status));
	CHECK(Drain(ctx, b) == SSLM_OK);

	// K3
	const SeqState before = CaptureState(a);
	for (int i = 0; i < 50; ++i) (void)ReadVerb(fx, a);
	CHECK_MSG(CaptureState(a) == before, "[%s] K3 50 reads changed the sequence's host-visible state", tag);

	sslm_gpu_seq_release(ctx, a);
	sslm_gpu_seq_release(ctx, b);
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
	return FinishSuite("cell_concurrency");
}
