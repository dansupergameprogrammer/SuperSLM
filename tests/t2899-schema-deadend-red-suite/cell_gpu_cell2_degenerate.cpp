// T-2909 (Curie) -- plan Sec3.10.2 Cell 2, for real: the degenerate-row dead end at a
// reachable non-accepting, non-S_c state, through the named seam
// `ArmGpuFinishDegenerateLogitRowInjection()` (`gpu_1p0.cpp:2400-2428` at this tip, compiled
// under `SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION`). TE-365 C2: the prior form of this file
// armed the seam and printed `checks=0 failures=0 skips=0` without ever calling Finish --
// the seam had never been fired by a cell. Closed below.
//
// FIXTURE. `--stringschema=PATH`, `make_t2909_string_schema_fixture.py`'s own hermetic
// artifact: a real schema (T-2853's own free-text string leaf, TE-365 C1's own fixed
// vocabularies) compiled by the production compiler over a real, index-identical
// vocabulary. The C39 synthetic (`g5_minimal_one_field`) cannot serve this cell -- it
// compiles to 2 states and 1 transition, so its only non-start state IS the accepting
// state Cell 1 already dead-ends at, with no interior state distinct from it (that
// generator's own docstring; also this file's prior header comment, T-2900). The
// meaningful vocabulary ids this cell drives (`kTok*` below) are reproduced from that
// generator's own module constants, not re-derived independently -- the CPU twin
// (`cell_cpu_deadend_retry_reset.cpp`'s `CpuCell2DegenerateRowTwin`) reads the identical
// fixture and the identical ids.
//
// CONSTRUCTION. `ReachSe` binds schema index 0, prefills a prompt (three filler ids the
// schema's own DFA never transitions on), then drives THREE real, admitted schema-content
// transitions in one `SslmGpuSeqPrefillSchemaContentForG5Bridge` call: `kTokOpen` (state 0 ->
// the pre-value literal state, the literal prefix up to the key's own colon), `kTokOpenQuote`
// (-> S_c, T-2910's own depth-0-only opening -- the literal object skeleton and the value's
// opening quote no longer share one vocabulary piece, T-2915) and `kTokBackslash` (S_c ->
// S_e) -- all three ordinary, correct transitions the compiler's own DFA admits; none is the
// degenerate-row construction. The resulting sequence rests at S_e with `layer_index == 0`/
// `ready_for_logits == true` (the schema-content prefill's own documented postcondition,
// identical to Cell 1's route R), so the next decode call reaches Finish directly over the
// unchanged residual.
//
// ASSERTIONS (plan Sec3.10.2 Cell 2 (i)/(ii)). With the seam armed, Finish presents a
// synthetic all-`INT32_MIN` row at S_e: `-2` at `SSLM_OK` (never a produced token), and
// `dfa_walk_state` PINNED at S_e -- the seam is single-shot and is fully consumed by THIS
// call, never left armed for a later, unrelated cell (the exact side effect TE-365 named on
// the CPU twin). Must-accept neighbour: the IDENTICAL construction, un-armed -- the real,
// non-degenerate row lets the masked argmax select one of S_e's own three admitted escapes
// (`kTokBackslash`/`kTokEscapeN`/`kTokClose`, each admitted from S_e per the generator's own
// SETUP self-check) and the walk leaves S_e (every one of S_e's admitted tokens returns to
// S_c, since an escape sequence can never itself close the string).
//
// GUARD VITALITY (plan Sec3.10.3 row 11: "a single-point mutant that reverts the checked
// return to the unconditional `next_state` write must turn both Cell 1 and Cell 2 red").
// `build_red_suite_gpu.bat` links this cell against `gpu_asbuilt` and the generated
// `gpu_mut_cr` object. Under MUT_CHECKEDRETURN, a
// degenerate row's own lowest-index tie-break (token 0, `kTokOpen`) is written to
// `*out_token` unconditionally instead of `-2`, and `dfa_walk_state` advances off S_e
// instead of staying pinned. MUT_LATE restores late binding after prompt prefill and must
// turn the D-SLM7625 assertion red.
//
// Run: cell_gpu_cell2_degenerate.exe --stringschema=PATH
#include "fixture_common.h"

#if defined(SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION)
extern "C" void ArmGpuFinishDegenerateLogitRowInjection();

namespace {

// Token ids from `make_t2909_string_schema_fixture.py` (kept in lockstep with that
// generator's own module constants TOK_OPEN/TOK_OPEN_QUOTE/TOK_CONTENT/TOK_BACKSLASH/
// TOK_ESCAPE_N/TOK_CLOSE/TOK_BRACE). T-2915 (porting the plan's final Sec3.9 design): T-2910's
// boundary discipline refuses a token that opens the string's content sub-automaton past its
// own first byte, so the generator's own literal object skeleton and the value's opening
// quote no longer share one vocabulary piece -- reaching S_e now takes three fed tokens
// (open, open-quote, backslash) rather than two.
constexpr int32_t kTokOpen = 0;
constexpr int32_t kTokOpenQuote = 1;
constexpr int32_t kTokBackslash = 3;
constexpr int32_t kTokEscapeN = 4;
constexpr int32_t kTokClose = 5;
constexpr int32_t kFillerBase = 7;  // "<unused-N>" pieces: harmless prompt filler.
constexpr int32_t kCallerToken = 8;  // the decode call's own caller-supplied token; ignored
                                     // by the ready branch (schema-content prefill already
                                     // left ready_for_logits armed), matching Cell 1's route R.

// Drives a fresh sequence to S_e (the escape state, prefix `{"Prompt_Result":"\`) via three
// real, admitted schema-content transitions from a fresh bind.
SslmGpuSequenceHandle* ReachSe(const GpuModelFixture& fx) {
	SslmGpuSequenceHandle* seq = nullptr;
	sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &seq);
	CHECK_MSG(seq != nullptr, "sslm_gpu_seq_create failed");
	if (!seq) return nullptr;
	const std::vector<int32_t> P = {kFillerBase, kFillerBase + 1, kFillerBase + 2};
	CHECK_MSG(SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, seq, 0) == SSLM_OK, "bind schema 0");
	CHECK_MSG(Prefill(fx, seq, P) == SSLM_OK, "prompt prefill after bind");
	const std::vector<int32_t> content = {kTokOpen, kTokOpenQuote, kTokBackslash};
	int32_t consumed = -1;
	CHECK_MSG(SslmGpuSeqPrefillSchemaContentForG5Bridge(fx.ctx, seq, content.data(),
	                                                    static_cast<int32_t>(content.size()),
	                                                    fx.one_layer_budget,
	                                                    &consumed) == SSLM_OK &&
	              consumed == static_cast<int32_t>(content.size()),
	          "schema-content prefill: open+open-quote+backslash to reach S_e (consumed=%d)",
	          consumed);
	return seq;
}

}  // namespace
#endif  // SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION

int main(int argc, char** argv) {
#if !defined(SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION)
	(void)argc;
	(void)argv;
	std::printf("SKIP cell_gpu_cell2_degenerate -- built without "
	            "SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION\n");
	std::printf("checks=0 failures=0 skips=1\n");
	return 0;
#else
	std::string model_path;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--stringschema=", 0) == 0) model_path = a.substr(15);
	}
	if (model_path.empty()) {
		std::printf("SKIP cell_gpu_cell2_degenerate -- needs --stringschema=PATH\n");
		std::printf("checks=0 failures=0 skips=1\n");
		return 0;
	}

	SslmGpuContext* ctx = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != SSLM_OK) {
		std::printf("FAIL cell_gpu_cell2_degenerate -- sslm_gpu_context_create failed\n");
		return 1;
	}
	GpuModelFixture fx;
	if (!fx.Open(model_path, ctx)) {
		std::printf("FAIL cell_gpu_cell2_degenerate -- fixture open failed\n");
		return 1;
	}
	// D-SLM7625: a prompt prefill is generation. A late bind must be rejected
	// without changing the sequence; ReachSe uses the supported bind-first order.
	{
		SslmGpuSequenceHandle* seq = nullptr;
		CHECK(sslm_gpu_seq_create(fx.ctx, fx.model, fx.model_cap, &seq) == SSLM_OK && seq);
		if (seq) {
			const std::vector<int32_t> prompt = {kFillerBase, kFillerBase + 1, kFillerBase + 2};
			CHECK(Prefill(fx, seq, prompt) == SSLM_OK);
			const uint32_t walk = SslmGpuSeqWalkStateForG5Bridge(seq);
			const int64_t context = ContextLength(seq);
			size_t bytes = 0;
			(void)sslm_gpu_seq_save(fx.ctx, seq, nullptr, &bytes);
			CHECK(bytes > 0);
			std::vector<uint8_t> before(bytes);
			CHECK(sslm_gpu_seq_save(fx.ctx, seq, before.data(), &bytes) == SSLM_OK);
			CHECK(SslmGpuSeqSetSchemaForG5Bridge(fx.ctx, seq, 0) == SSLM_SEQUENCE_REJECTED);
			CHECK(SslmGpuSeqWalkStateForG5Bridge(seq) == walk && ContextLength(seq) == context);
			std::vector<uint8_t> after(bytes);
			CHECK(sslm_gpu_seq_save(fx.ctx, seq, after.data(), &bytes) == SSLM_OK);
			CHECK(after == before);
			sslm_gpu_seq_release(fx.ctx, seq);
		}
	}

	// Cell 2(i)/(ii): the degenerate row at S_e.
	{
		SslmGpuSequenceHandle* seq = ReachSe(fx);
		if (seq) {
			const uint32_t s_e = SslmGpuSeqWalkStateForG5Bridge(seq);
			ArmGpuFinishDegenerateLogitRowInjection();
			int32_t out = 12345;
			const SslmGpuStatus st =
			    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, seq, kCallerToken, fx.one_layer_budget, &out);
			CHECK_MSG(st == SSLM_OK && out == -2,
			          "degenerate row at S_e: st=%s out=%d, want -2/SSLM_OK", StatusName(st), out);
			CHECK_MSG(SslmGpuSeqWalkStateForG5Bridge(seq) == s_e,
			          "degenerate row at S_e: dfa_walk_state moved %u -> %u", s_e,
			          SslmGpuSeqWalkStateForG5Bridge(seq));
			sslm_gpu_seq_release(fx.ctx, seq);
		}
	}

	// Must-accept neighbour (plan Sec3.10.2): the identical construction, un-armed -- the
	// real finish bridge selects one of S_e's own admitted escapes and advances to S_c.
	{
		SslmGpuSequenceHandle* seq = ReachSe(fx);
		if (seq) {
			const uint32_t s_e = SslmGpuSeqWalkStateForG5Bridge(seq);
			int32_t out = 12345;
			const SslmGpuStatus st =
			    SslmGpuSeqDecodeStepForG5Bridge(fx.ctx, seq, kCallerToken, fx.one_layer_budget, &out);
			CHECK_MSG(st == SSLM_OK && out >= 0,
			          "must-accept neighbour: st=%s out=%d, want a real, non-negative token",
			          StatusName(st), out);
			CHECK_MSG(out == kTokBackslash || out == kTokEscapeN || out == kTokClose,
			          "must-accept neighbour: produced token %d is not one of S_e's own admitted "
			          "escapes {%d,%d,%d}",
			          out, kTokBackslash, kTokEscapeN, kTokClose);
			const uint32_t moved = SslmGpuSeqWalkStateForG5Bridge(seq);
			CHECK_MSG(moved != s_e, "must-accept neighbour: dfa_walk_state did not leave S_e (%u)", s_e);
			sslm_gpu_seq_release(fx.ctx, seq);
		}
	}

	fx.Close();
	sslm_gpu_context_destroy(ctx);
	std::printf("checks=%d failures=%d skips=0\n", GChecks, GFailures);
	return GFailures == 0 ? 0 : 1;
#endif  // SUPERSLM_GPU_G5_FINISH_ROW_FAULT_INJECTION
}
