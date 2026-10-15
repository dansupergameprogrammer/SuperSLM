// Paged-KV plan (rev 16.2) §7 cell 3.6 [G3], group g1c.
//
//   "GPU: two paged sequences on one pool, with interleaved submissions each drained. The debug
//    layer is clean, no resume barrier is missing, and outputs byte-equal separate runs."
//
// CONSTRUCTION. Two budgeted sequences A and B on ONE pool (§4 item 3), each prompt-prefilled
// (10 tokens) and then decoded 24 tokens, so both cross the page boundaries at 16 and 32 (B = 16):
//   phase 1, token-interleaved: A one token (embed, sslm_decode_step_gpu at full depth,
//            sslm_gpu_ready, finish), then B one token, alternating, 12 tokens each;
//   phase 2, layer-interleaved: A one layer (sslm_decode_step_gpu at one layer's budget,
//            sslm_gpu_ready), then B one layer, alternating to full depth, then each finishes;
//            12 tokens each.
// Every submission is drained before the other sequence submits, which is what the pool latch
// (§4 item 4: `&pool->kv_needs_resume_barrier`, one per pool) is for: each drain's readback leaves the
// shared pool buffer in COPY_SOURCE, and the other sequence's next dispatch must find the latch set.
//
// OPERANDS (§7 grading rule):
//   - tokens: A's and B's 1 + 24 tokens equal SEPARATE RUNS, each sequence alone on its own fresh pool
//     with the same calls (the unsaved twin). A missing resume barrier makes the next dispatch read
//     the pool buffer in the wrong state, which is the defect class the tokens see;
//   - the debug-layer message count (WARNING or worse), read from the device's info queue after the
//     interleaved run: 0. The run must be started with SSLM_GPU_ENABLE_DEBUG_LAYER=1, and the cell
//     FAILS when the layer is not on (a count from no queue is vacuous). T-2195 found the layer blind
//     to a desynced latch on the RTX 2080 SUPER (driver 560.94), so on that device the latch rests on
//     the tokens; the count is still graded.
// Diagnostic only: the latch each handle passes (SslmGpuSeqHandleKvResumeFlagForBench), which from
// G3 is the pool's for a pooled handle (§4.4 *Three states*, last paragraph): printed, not graded.
//
// RED NOW (63e7b0e): BY LINK -- sslm_gpu_kv_pool_create, sslm_gpu_kv_pool_destroy and
// sslm_gpu_seq_create_budgeted have no definition (G3 delivers them, §9.3).
// GREEN AT G3 when both GPUs give equal tokens and a clean debug layer.
#include "g1c_helpers.h"

namespace {
using namespace g1c;

constexpr size_t kPrefill = 10;
constexpr size_t kPhase1 = 12;
constexpr size_t kPhase2 = 12;

struct Track {
	SslmGpuSequenceHandle* seq = nullptr;
	std::vector<int64_t> out;  // statuses and tokens, in call order
	int32_t last = -1;
	bool clean = true;  // every status SSLM_OK and every token >= 0
	void Status(SslmGpuStatus st) {
		out.push_back(static_cast<int64_t>(st));
		clean = clean && st == SSLM_OK;
	}
	void Token(int32_t tok) {
		out.push_back(tok);
		clean = clean && tok >= 0;
		last = tok;
	}
};

// The ready token after the prefill.
void FirstToken(Stack& s, Track& t) {
	int32_t tok = -1;
	const SslmGpuStatus st = SslmGpuSeqFinishTokenForG5Bridge(s.ctx, t.seq, &tok);
	t.Status(st);
	t.Token(tok);
}
// One token at full depth in one submission.
void TokenStep(Stack& s, Track& t) {
	SslmGpuStatus st = sslm_gpu_seq_embed_token(s.ctx, t.seq, t.last >= 0 ? t.last : 1);
	if (st == SSLM_OK) st = sslm_decode_step_gpu(s.ctx, t.seq, nullptr, s.full_depth);
	if (st == SSLM_OK) st = Ready(s.ctx, t.seq);
	int32_t tok = -1;
	if (st == SSLM_OK) st = SslmGpuSeqFinishTokenForG5Bridge(s.ctx, t.seq, &tok);
	t.Status(st);
	t.Token(tok);
}
void Embed(Stack& s, Track& t) {
	t.Status(sslm_gpu_seq_embed_token(s.ctx, t.seq, t.last >= 0 ? t.last : 1));
}
// One layer in one submission, drained.
void LayerStep(Stack& s, Track& t) {
	SslmGpuStatus st = sslm_decode_step_gpu(s.ctx, t.seq, nullptr, s.one_layer);
	if (st == SSLM_OK) st = Ready(s.ctx, t.seq);
	t.Status(st);
}
void Finish(Stack& s, Track& t) {
	int32_t tok = -1;
	const SslmGpuStatus st = SslmGpuSeqFinishTokenForG5Bridge(s.ctx, t.seq, &tok);
	t.Status(st);
	t.Token(tok);
}

bool Prefill(Stack& s, Track& t, int32_t base) {
	const std::vector<int32_t> p = Toks(s.art, kPrefill, base);
	const SslmGpuStatus st =
	    SslmGpuSeqPrefillPromptForG5Bridge(s.ctx, t.seq, p.data(), static_cast<int32_t>(p.size()), s.one_layer);
	G1C_CHECK(st == SSLM_OK, "SETUP: prompt prefill -> %s", St(st));
	return st == SSLM_OK;
}

// A run: `tracks` share pool 0 when interleaved; otherwise each runs alone (called once per track).
void Drive(Stack& s, std::vector<Track*> tracks) {
	for (Track* t : tracks) FirstToken(s, *t);
	for (size_t i = 0; i < kPhase1; ++i) {
		for (Track* t : tracks) TokenStep(s, *t);
	}
	for (size_t i = 0; i < kPhase2; ++i) {
		for (Track* t : tracks) Embed(s, *t);
		for (uint32_t l = 0; l < s.art.Layers(); ++l) {
			for (Track* t : tracks) LayerStep(s, *t);
		}
		for (Track* t : tracks) Finish(s, *t);
	}
}

void Cell36() {
	Stack* s = GetStack();
	if (!s) return;
	bool layer_on = false;
	(void)DebugLayerReportable(&layer_on);  // clear what earlier work queued
	G1C_CHECK(layer_on, "the debug layer is not on: run with SSLM_GPU_ENABLE_DEBUG_LAYER=1 (the count is an "
	                    "operand of this cell, and a count from no info queue is vacuous)");

	// Separate runs: each sequence alone on its own fresh pool.
	Track a_sep, b_sep;
	{
		World w(*s);
		w.Pool(64);
		a_sep.seq = w.Pooled(0, 64);
		if (!w.ok || !Prefill(*s, a_sep, 3)) return;
		Drive(*s, {&a_sep});
	}
	{
		World w(*s);
		w.Pool(64);
		b_sep.seq = w.Pooled(0, 64);
		if (!w.ok || !Prefill(*s, b_sep, 41)) return;
		Drive(*s, {&b_sep});
	}
	(void)DebugLayerReportable(&layer_on);

	// The interleaved run: both on one pool.
	Track a, b;
	{
		World w(*s);
		w.Pool(64);
		a.seq = w.Pooled(0, 64);
		b.seq = w.Pooled(0, 64);
		if (!w.ok || !Prefill(*s, a, 3) || !Prefill(*s, b, 41)) return;
		std::printf("  diagnostic: resume latch A=%p B=%p (G3: the pool's, the same for both)\n",
		            static_cast<void*>(SslmGpuSeqHandleKvResumeFlagForBench(a.seq)),
		            static_cast<void*>(SslmGpuSeqHandleKvResumeFlagForBench(b.seq)));
		Drive(*s, {&a, &b});
	}
	const size_t reportable = DebugLayerReportable(&layer_on);
	G1C_CHECK(layer_on && reportable == 0, "the interleaved run left %zu WARNING-or-worse debug-layer messages "
	                                       "(layer on: %d)", reportable, layer_on ? 1 : 0);
	G1C_CHECK(a.out == a_sep.out, "A's statuses and tokens interleaved differ from A alone (%zu vs %zu values)",
	          a.out.size(), a_sep.out.size());
	G1C_CHECK(b.out == b_sep.out, "B's statuses and tokens interleaved differ from B alone (%zu vs %zu values)",
	          b.out.size(), b_sep.out.size());
	// The separate runs themselves must have succeeded, or equality is between two failures.
	G1C_CHECK(a_sep.clean && b_sep.clean && !a_sep.out.empty() && !b_sep.out.empty(),
	          "SETUP: a separate run did not complete cleanly (A %d, B %d)", a_sep.clean ? 1 : 0, b_sep.clean ? 1 : 0);
}
}  // namespace

G1C_CELL("3.6/G3", "G3", Cell36);
