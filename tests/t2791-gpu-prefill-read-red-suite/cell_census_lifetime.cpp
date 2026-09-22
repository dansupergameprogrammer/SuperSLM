// T-2791 (Curie) -- plan Sec3.4 row 1 (lifetime and reuse), the prompt-twin half of row 5
// (M10, M11, M20, M21), the persistence half of row 9 (M15), and row 8's two-call continuation
// (M09) and generation (M17): one named cell per member of plan Sec2.6's transition census.
//
// Each cell starts from reset -> SslmGpuSeqPrefillPromptForG5Bridge(P) (frame F_P), applies its
// transition, and asserts what sslm_gpu_seq_read_prefill_final_hidden returns:
//   - "F_P": SSLM_OK and the whole frame (every code and the scale) equal to F_P;
//   - "F_PQ": SSLM_OK and the whole frame equal to the continuation's frame;
//   - "refuse": SSLM_PREFILL_HIDDEN_UNAVAILABLE and no frame output written;
//   - "busy": SSLM_BUSY and no frame output written.
// "No frame output written" on refuse and busy includes *out_required, which a Frame seeds with a
// sentinel (plan Sec3.1 check 4 writes it only once checks 1-3 pass; Sec3.4 row 7, T-2806 N4): the
// fresh-sequence and M04 cells pin check 3, M14a pins check 2.
//
// ORACLE (independent of the snapshot and the read): F_P is final_norm applied, by this file, to
// the live residual the bench accessor exposes immediately after the base prefill returned
// SSLM_OK (fixture_common.h, OracleFromLive). F_PQ is the same over a reset sequence prefilled
// with P ++ Q in one call. Expected outcomes per member are plan Sec2.6's "Snapshot read"
// column, executed by the planner (Claude/Vitruvius/te269-probe/te269-census-output.txt) and
// re-executed on two Qwen2.5-1.5B artifacts by the re-strike (Claude/Loki/te270-census-*.txt).
//
// VITALITY. Where a member is one of plan Sec3.4 row 11's mutant-(a) killers (M01, M05, M06, M07,
// M14c), the cell also asserts the member is discriminating ON THIS ARTIFACT: final_norm of the
// live residual after the transition differs from F_P. Without that, a read of the live residual
// (mutant (a)) could pass by coincidence and the cell would pin nothing.
//
// Red at v1.5.0 by LINK: LNK2019 on sslm_gpu_seq_read_prefill_final_hidden.
//
// Run: cell_census_lifetime.exe [--qwen3=PATH] [--synthetic=PATH] [--g5fixture=PATH]
// Each supplied artifact gets the whole census.
#include "fixture_common.h"

namespace {

struct Census {
	GpuModelFixture& fx;
	SslmGpuSequenceHandle* seq = nullptr;
	Frame FP, FPQ;
	std::vector<int32_t> P, Q;
	const char* tag;

	explicit Census(GpuModelFixture& f) : fx(f), tag(f.path.c_str()) {}

	bool Setup() {
		P = fx.TokensP();
		Q = fx.TokensQ();
		const int64_t cap = std::min<int64_t>(512, fx.model_cap);
		if (sslm_gpu_seq_create(fx.ctx, fx.model, cap, &seq) != SSLM_OK || !seq) return false;
		// F_PQ: P ++ Q in one call from reset.
		std::vector<int32_t> PQ = P;
		PQ.insert(PQ.end(), Q.begin(), Q.end());
		if (sslm_gpu_seq_reset(fx.ctx, seq) != SSLM_OK || Prefill(fx, seq, PQ) != SSLM_OK) return false;
		FPQ = OracleFromLive(fx, seq);
		return Base() && FPQ.IsOk() && !FPQ.SameFrame(FP, fx.hidden);
	}

	// reset -> prefill(P); refreshes F_P from the live residual at this authoritative moment.
	bool Base() {
		if (sslm_gpu_seq_reset(fx.ctx, seq) != SSLM_OK) return false;
		if (Prefill(fx, seq, P) != SSLM_OK) return false;
		FP = OracleFromLive(fx, seq);
		return FP.IsOk();
	}

	void ExpectFrame(const char* member, const Frame& got, const Frame& want) {
		CHECK_MSG(got.status == SSLM_OK, "[%s] %s: read returned %s, want SSLM_OK with the frame", tag,
		          member, StatusName(got.status));
		CHECK_MSG(got.SameFrame(want, fx.hidden),
		          "[%s] %s: read frame differs from the expected frame in %zu of %zu codes (m %lld vs %lld, "
		          "e %lld vs %lld)",
		          tag, member, DiffCodes(got, want, fx.hidden), fx.hidden, static_cast<long long>(got.m),
		          static_cast<long long>(want.m), static_cast<long long>(got.e), static_cast<long long>(want.e));
		CHECK_MSG(got.required == fx.hidden, "[%s] %s: *out_required=%zu, want hidden_size %zu", tag, member,
		          got.required, fx.hidden);
	}
	void ExpectRefuse(const char* member, const Frame& got) {
		CHECK_MSG(got.status == kPrefillHiddenUnavailable,
		          "[%s] %s: read returned %s, want SSLM_PREFILL_HIDDEN_UNAVAILABLE", tag, member,
		          StatusName(got.status));
		CHECK_MSG(got.OutputsUntouched(), "[%s] %s: a refused read wrote its codes or scale", tag, member);
		// Plan Sec3.4 row 7 (T-2806 N4, T-2814): check 3 leaves *out_required unwritten.
		CHECK_MSG(got.required == kRequiredSentinel, "[%s] %s: a check-3 refusal wrote *out_required (%zu)", tag,
		          member, got.required);
	}
	void ExpectBusy(const char* member, const Frame& got) {
		CHECK_MSG(got.status == SSLM_BUSY, "[%s] %s: read returned %s, want SSLM_BUSY", tag, member,
		          StatusName(got.status));
		CHECK_MSG(got.OutputsUntouched(), "[%s] %s: a BUSY read wrote its codes or scale", tag, member);
		// Plan Sec3.4 row 7 (T-2806 N4, T-2814): check 2 leaves *out_required unwritten.
		CHECK_MSG(got.required == kRequiredSentinel, "[%s] %s: a check-2 refusal wrote *out_required (%zu)", tag,
		          member, got.required);
	}
	// The member moved the live residual away from F_P (mutant (a) is killable here).
	void ExpectDiscriminating(const char* member) {
		const Frame live = OracleFromLive(fx, seq);
		CHECK_MSG(live.IsOk() && !live.SameFrame(FP, fx.hidden),
		          "[%s] %s: VITALITY -- final_norm of the live residual after the transition equals F_P, so a "
		          "read of the live residual (plan Sec3.4 row 11 mutant (a)) would pass this cell",
		          tag, member);
	}
	void Setup_(bool ok, const char* member, const char* what) {
		CHECK_MSG(ok, "[%s] %s: SETUP -- %s; the member was not reached", tag, member, what);
	}

	// --- one cell per plan Sec2.6 member -------------------------------------------------------

	void Fresh() {  // plan Sec3.1 check 3: a fresh sequence holds no snapshot.
		SslmGpuSequenceHandle* s = nullptr;
		Setup_(sslm_gpu_seq_create(fx.ctx, fx.model, 64, &s) == SSLM_OK && s, "fresh", "create");
		if (!s) return;
		ExpectRefuse("fresh sequence", ReadVerb(fx, s));
		sslm_gpu_seq_release(fx.ctx, s);
	}

	void AfterPrefill() {  // the must-accept every refusal cell is the neighbour of.
		Setup_(Base(), "after-prefill", "base prefill");
		ExpectFrame("after prefill (must-accept)", ReadVerb(fx, seq), FP);
	}

	void M01() {
		Setup_(Base(), "M01", "base prefill");
		Setup_(sslm_gpu_seq_embed_token(fx.ctx, seq, fx.SomeToken()) == SSLM_OK, "M01", "embed_token");
		ExpectDiscriminating("M01 embed_token");
		ExpectFrame("M01 embed_token", ReadVerb(fx, seq), FP);
	}

	void M02() {
		Setup_(Base(), "M02", "base prefill");
		int32_t t = -1;
		Setup_(SslmGpuSeqFinishTokenForG5Bridge(fx.ctx, seq, &t) == SSLM_OK, "M02", "FinishToken");
		ExpectFrame("M02 FinishToken", ReadVerb(fx, seq), FP);
	}

	void M03() {
		Setup_(Base(), "M03", "base prefill");
		int32_t t = -1;
		Setup_(SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, seq, fx.OtherToken(), fx.one_layer_budget, &t) == SSLM_OK,
		       "M03", "decode wrapper");
		ExpectFrame("M03 decode wrapper (ready branch)", ReadVerb(fx, seq), FP);
	}

	void M04() {
		Setup_(Base(), "M04", "base prefill");
		Setup_(sslm_gpu_seq_reset(fx.ctx, seq) == SSLM_OK, "M04", "reset");
		ExpectRefuse("M04 seq_reset", ReadVerb(fx, seq));
	}

	void M05() {
		Setup_(Base(), "M05", "base prefill");
		Setup_(sslm_gpu_seq_embed_token(fx.ctx, seq, fx.SomeToken()) == SSLM_OK, "M05", "embed");
		Setup_(sslm_decode_step_gpu(fx.ctx, seq, nullptr, fx.one_layer_budget) == SSLM_OK, "M05", "decode 1 layer");
		Setup_(Drain(fx.ctx, seq) == SSLM_OK, "M05", "drain");
		Setup_(LayerIndex(seq) > 0 && LayerIndex(seq) < fx.layers, "M05", "partial depth");
		ExpectDiscriminating("M05 embed + one hand-driven layer");
		ExpectFrame("M05 embed + one hand-driven layer", ReadVerb(fx, seq), FP);
	}

	void M06() {
		Setup_(Base(), "M06", "base prefill");
		Setup_(sslm_gpu_seq_embed_token(fx.ctx, seq, fx.SomeToken()) == SSLM_OK, "M06", "embed");
		SslmGpuStatus st = SSLM_OK;
		while (LayerIndex(seq) < fx.layers && st == SSLM_OK) {
			st = sslm_decode_step_gpu(fx.ctx, seq, nullptr, fx.one_layer_budget * fx.layers);
			if (st == SSLM_OK) st = Drain(fx.ctx, seq);
		}
		Setup_(st == SSLM_OK && LayerIndex(seq) == fx.layers, "M06", "hand drive to full depth");
		ExpectDiscriminating("M06 embed + hand-driven full depth");
		ExpectFrame("M06 embed + hand-driven full depth", ReadVerb(fx, seq), FP);
	}

	void M07() {
		Setup_(Base(), "M07", "base prefill");
		Setup_(sslm_gpu_seq_embed_token(fx.ctx, seq, fx.SomeToken()) == SSLM_OK, "M07", "embed");
		SslmGpuSequenceHandle* seqs[1] = {seq};
		SslmGpuStatus outs[1] = {SSLM_DEVICE_LOST};
		Setup_(sslm_decode_step_batch_gpu(fx.ctx, seqs, nullptr, 1, fx.one_layer_budget, outs) == SSLM_OK &&
		           outs[0] == SSLM_OK,
		       "M07", "batched decode 1 layer");
		// The batched call drains through sslm_gpu_ready itself (plan Sec2.6 M07); drain defensively.
		Drain(fx.ctx, seq);
		ExpectDiscriminating("M07 embed + one batched layer");
		ExpectFrame("M07 embed + one batched layer", ReadVerb(fx, seq), FP);
	}

	void M08() {
		Setup_(Base(), "M08", "base prefill");
		const SslmGpuStatus st = sslm_decode_step_gpu(fx.ctx, seq, nullptr, fx.one_layer_budget);
		Setup_(st == SSLM_DISPATCH_BUDGET_TOO_SMALL, "M08", "decode at full depth returns DISPATCH_BUDGET_TOO_SMALL");
		ExpectFrame("M08 decode at full depth", ReadVerb(fx, seq), FP);
	}

	void M09() {  // row 1 "a continuation's frame"; row 8's two-call composition.
		Setup_(Base(), "M09", "base prefill");
		Setup_(Prefill(fx, seq, Q) == SSLM_OK, "M09", "continuation prefill");
		const Frame live = OracleFromLive(fx, seq);
		CHECK_MSG(live.SameFrame(FPQ, fx.hidden),
		          "[%s] M09: SETUP -- the continuation's live residual differs from the one-call P++Q frame", tag);
		ExpectFrame("M09 prefill continuation", ReadVerb(fx, seq), FPQ);
	}

	void M10() {
		Setup_(Base(), "M10", "base prefill");
		const std::vector<int32_t> t = {fx.SomeToken(), fx.OtherToken(), -1};
		Setup_(Prefill(fx, seq, t) == SSLM_TOKEN_ID_OUT_OF_RANGE, "M10", "kEmbed exit after 2 admitted");
		Setup_(ContextLength(seq) == static_cast<int64_t>(P.size()) + 2, "M10", "two tokens committed");
		ExpectRefuse("M10 prefill kEmbed, 2 admitted", ReadVerb(fx, seq));
	}

	void M11() {
		Setup_(Base(), "M11", "base prefill");
		Setup_(Prefill(fx, seq, {-1}) == SSLM_TOKEN_ID_OUT_OF_RANGE, "M11", "kEmbed exit, 0 admitted");
		ExpectRefuse("M11 prefill kEmbed, 0 admitted", ReadVerb(fx, seq));
	}

	void M12() {
		Setup_(Base(), "M12", "base prefill");
		Setup_(SslmGpuSeqPrefillPromptForG5Bridge(fx.ctx, seq, P.data(), 0, fx.one_layer_budget) == SSLM_OK, "M12",
		       "count == 0 returns SSLM_OK");
		ExpectFrame("M12 prefill count == 0", ReadVerb(fx, seq), FP);
	}

	void M13() {
		Setup_(Base(), "M13", "base prefill");
		Setup_(SslmGpuSeqPrefillPromptForG5Bridge(fx.ctx, seq, Q.data(), static_cast<int32_t>(Q.size()), 0) ==
		           SSLM_SEQUENCE_KV_BUFFER_MISMATCH,
		       "M13", "malformed prefill (budget 0)");
		ExpectFrame("M13 malformed prefill", ReadVerb(fx, seq), FP);
	}

	void M14() {  // row 3's Submitted window, reached through the existing async path.
		Setup_(Base(), "M14", "base prefill");
		Setup_(sslm_gpu_seq_embed_token(fx.ctx, seq, fx.SomeToken()) == SSLM_OK, "M14", "embed");
		Setup_(sslm_decode_step_gpu(fx.ctx, seq, nullptr, fx.one_layer_budget) == SSLM_OK, "M14", "submit");
		ExpectBusy("M14a while Submitted", ReadVerb(fx, seq));
		CHECK_MSG(Prefill(fx, seq, Q) == SSLM_BUSY, "[%s] M14b: a prefill against the Submitted window must return "
		          "SSLM_BUSY (plan Sec3.4 row 3)", tag);
		ExpectBusy("M14b after a BUSY prefill", ReadVerb(fx, seq));
		Setup_(Drain(fx.ctx, seq) == SSLM_OK, "M14", "drain");
		ExpectDiscriminating("M14c after the drain");
		ExpectFrame("M14c after the drain", ReadVerb(fx, seq), FP);
	}

	void M15() {
		Setup_(Base(), "M15", "base prefill");
		size_t need = 0;
		sslm_gpu_seq_save(fx.ctx, seq, nullptr, &need);
		std::vector<uint8_t> blob(need);
		size_t got = need;
		Setup_(need > 4 && sslm_gpu_seq_save(fx.ctx, seq, blob.data(), &got) == SSLM_OK, "M15", "save");
		ExpectFrame("M15a save (source sequence)", ReadVerb(fx, seq), FP);
		// Row 9, stated not changed BY THIS PLAN: the read verb adds no blob writer of its own.
		// T-2905 landed a concurrent, unrelated format bump (T-2895/D-SLM7572, closing TE-362): every
		// fresh save now writes 'SLM5' (the v4 header plus a twelve-byte bound_schema_index/
		// dfa_walk_state/ready_for_logits tail); an old 'SLM4' blob still restores unchanged. This
		// cell saves fresh, so 'SLM5' is now the correct magic -- corrected here (T-2906), not a
		// row-11 mutant target.
		CHECK_MSG(got >= 4 && std::memcmp(blob.data(), "SLM5", 4) == 0,
		          "[%s] M15: a fresh save blob no longer starts with the v5 magic 'SLM5'", tag);
		SslmGpuSequenceHandle* restored = nullptr;
		Setup_(sslm_gpu_seq_restore(fx.ctx, fx.model, blob.data(), got, &restored) == SSLM_OK && restored, "M15",
		       "restore");
		if (restored) {
			ExpectRefuse("M15b restored handle", ReadVerb(fx, restored));
			sslm_gpu_seq_release(fx.ctx, restored);
		}
	}

	void M16() {
		Setup_(Base(), "M16", "base prefill");
		// T-2934 reconciliation: binding (including unbinding) is fresh-or-reset only.
		// Recreate the same F_P snapshot after exercising the two fresh-state no-ops.
		Setup_(sslm_gpu_seq_reset(fx.ctx, seq) == SSLM_OK, "M16", "reset before bind operations");
		Setup_(sslm_gpu_seq_bind_adapter(fx.ctx, seq, nullptr) == SSLM_OK, "M16", "bind(nullptr)");
		Setup_(SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, seq, -1) == SSLM_OK, "M16", "set-schema(-1)");
		Setup_(Prefill(fx, seq, P) == SSLM_OK, "M16", "recreate base prefill");
		int32_t r = 0;
		SslmGpuStatus rs = SSLM_OK;
		Setup_(sslm_gpu_ready(fx.ctx, seq, 0, &r, &rs) == SSLM_OK, "M16", "idle ready");
		(void)SslmGpuSeqWalkStateForG5Bridge(seq);
		ExpectFrame("M16 bind / set-schema / idle ready / walk-state", ReadVerb(fx, seq), FP);
	}

	void M17() {  // a generation after the prompt: the read still returns the prompt's frame.
		Setup_(Base(), "M17", "base prefill");
		int32_t t = -1;
		SslmGpuStatus st = SSLM_OK;
		for (int i = 0; i < 3 && st == SSLM_OK; ++i) {
			st = SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, seq, t < 0 ? fx.OtherToken() : t, fx.one_layer_budget, &t);
		}
		Setup_(st == SSLM_OK, "M17", "three decode-wrapper steps");
		ExpectFrame("M17 three-step generation", ReadVerb(fx, seq), FP);
	}

	// Row 1: read twice gives identical bytes; releasing a DIFFERENT sequence leaves it unchanged.
	void ReadTwiceAndReleaseOther() {
		Setup_(Base(), "read-twice", "base prefill");
		const Frame a = ReadVerb(fx, seq);
		const Frame b = ReadVerb(fx, seq);
		ExpectFrame("read twice (first)", a, FP);
		ExpectFrame("read twice (second)", b, FP);
		SslmGpuSequenceHandle* other = nullptr;
		Setup_(sslm_gpu_seq_create(fx.ctx, fx.model, 64, &other) == SSLM_OK && other, "release-other", "create");
		if (other) {
			Prefill(fx, other, Q);
			Setup_(sslm_gpu_seq_release(fx.ctx, other) == SSLM_OK, "release-other", "release");
		}
		ExpectFrame("after releasing a different sequence", ReadVerb(fx, seq), FP);
	}

	// Position-cap members on a cap-64 sequence (plan Sec3.4 row 4: caps 64 and 512).
	void M20M21() {
		const int64_t cap = std::min<int64_t>(64, fx.model_cap);
		SslmGpuSequenceHandle* s = nullptr;
		Setup_(sslm_gpu_seq_create(fx.ctx, fx.model, cap, &s) == SSLM_OK && s, "M20", "create cap-64 sequence");
		if (!s) return;
		const std::vector<int32_t> fill_full = fx.Run(static_cast<size_t>(cap), 1000 % fx.vocab);
		const std::vector<int32_t> fill_less4 = fx.Run(static_cast<size_t>(cap - 4), 1000 % fx.vocab);
		const std::vector<int32_t> next8 = fx.Run(8, 3000 % fx.vocab);

		Setup_(sslm_gpu_seq_reset(fx.ctx, s) == SSLM_OK && Prefill(fx, s, fill_full) == SSLM_OK, "M20", "fill to cap");
		const Frame F_full = OracleFromLive(fx, s);
		ExpectFrame("cap-64 sequence filled to its cap (must-accept)", ReadVerb(fx, s), F_full);
		Setup_(Prefill(fx, s, {fx.SomeToken()}) == SSLM_DEVICE_LOST, "M20", "kPositionCap exit, 0 admitted");
		ExpectRefuse("M20 kPositionCap, 0 admitted", ReadVerb(fx, s));

		Setup_(sslm_gpu_seq_reset(fx.ctx, s) == SSLM_OK && Prefill(fx, s, fill_less4) == SSLM_OK, "M21", "fill to cap-4");
		Setup_(Prefill(fx, s, next8) == SSLM_DEVICE_LOST, "M21", "kPositionCap exit, 4 of 8 admitted");
		Setup_(ContextLength(s) == cap, "M21", "4 tokens committed");
		ExpectRefuse("M21 kPositionCap, 4 of 8 admitted", ReadVerb(fx, s));
		sslm_gpu_seq_release(fx.ctx, s);
	}

	void RunAll() {
		std::printf("--- census on %s (F_P, F_PQ from the live residual) ---\n", tag);
		Fresh();
		AfterPrefill();
		M01(); M02(); M03(); M04(); M05(); M06(); M07(); M08(); M09(); M10(); M11(); M12(); M13();
		M14(); M15(); M16(); M17();
		ReadTwiceAndReleaseOther();
		M20M21();
		std::printf("    F_P m=%lld e=%lld; F_PQ m=%lld e=%lld\n", static_cast<long long>(FP.m),
		            static_cast<long long>(FP.e), static_cast<long long>(FPQ.m), static_cast<long long>(FPQ.e));
	}
};

void RunOn(const std::string& path, const char* flag, SslmGpuContext* ctx) {
	if (path.empty()) {
		SKIP_MSG("census: %s not supplied", flag);
		return;
	}
	GpuModelFixture fx;
	if (!fx.Open(path, ctx)) {
		CHECK_MSG(false, "census: could not open %s", path.c_str());
		return;
	}
	Census c(fx);
	if (!c.Setup()) {
		CHECK_MSG(false, "census: setup on %s failed (reference prefills)", path.c_str());
	} else {
		c.RunAll();
	}
	if (c.seq) sslm_gpu_seq_release(ctx, c.seq);
	fx.Close();
}

}  // namespace

int main(int argc, char** argv) {
	RunAsLegDriverIfRequested(argc, argv, "cell_census_lifetime");
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
	return FinishSuite("cell_census_lifetime");
}
