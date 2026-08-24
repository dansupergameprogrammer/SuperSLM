// T-2246 (test design) -- Coverage Model row 12, Performance (audit G-5, plan r6 SS5 S-F cell
// 2a + the broken-drafter negative control). Exactly ONE asserting end-state cell carries a
// threshold: the per-emitted-token cost ratio on the hash-pinned shopkeeper corpus. The
// threshold and minimum effect size are RECORDED AT COMMISSIONING -- before the first real
// reading -- by Dan's own act; until then the asserting cell REFUSES (fails loudly as
// NOT-COMMISSIONED) rather than passing silently. M2 proves that cell can fail: the same
// procedure over a never-matching drafter must NOT satisfy the ratio assertion while still
// satisfying every equivalence cell. M3 is the budget-free report scaffold for S-F cells 1-2
// (outside the catalog boundary per the audit's own note; prints, asserts nothing): it
// reports the per-call ACCEPTANCE HISTOGRAM over a corpus-driven speculate drive -- emissions
// per call bucketed 1..K+1 -- plus the mean. No threshold, no verdict; commissioning-time
// readings only.
//
// Corpus integrity: SHA-256 pinned at commissioning time from this tree
// (tools/reference_pipeline/data/shopkeeper_corpus_v1.jsonl,
//  640a5770da9e093446e2aece8b3b7e2869476963e54039af1e897ec750e4d9ae).
//
// EXECUTION STATUS (fold round 1, 2026-08-22): S-E has LANDED -- sslm_speculate_step_v3 /
// sslm_speculate_params_init exist in production under the recorded names; every cell
// executes against the real mechanism. The not-commissioned refusal stays structural.
#include "fixture_common.h"

using namespace superslm;
#include <chrono>

namespace {

// Commissioning gate: SET THESE BEFORE FIRST READING (Dan, at commissioning; Dan may tighten
// to an acceptance-length floor instead -- plan SS5 S-F(2a)). kThresholdUnsentinel marks them
// unset: any read of an unset gate fails the cell loudly.
constexpr double kThresholdUnsentinel = -1.0;
double g_max_cost_ratio = kThresholdUnsentinel;        // assert: measured_ratio < this
double g_min_effect_size = kThresholdUnsentinel;       // smallest ratio improvement acted on

bool ThresholdsCommissioned() { return g_max_cost_ratio > 0.0 && g_min_effect_size >= 0.0; }

bool CorpusHashMatches(const std::string& path) {
	std::vector<uint8_t> bytes;
	if (!ReadFileBytes(path, &bytes)) return false;
	// superslm::Sha256Hash over the raw bytes (include/superslm/sha256.h; src/sha256.cpp
	// ships with the engine sources this suite links), against the commissioning-time pin.
	uint8_t digest[32];
	superslm::Sha256Hash(bytes.data(), bytes.size(), digest);
	static const uint8_t kPin[32] = {
	    0x64, 0x0a, 0x57, 0x70, 0xda, 0x9e, 0x09, 0x34, 0x46, 0xe2, 0xae, 0xce, 0x8b, 0x3b,
	    0x7e, 0x28, 0x69, 0x47, 0x69, 0x63, 0xe5, 0x40, 0x39, 0xaf, 0x1e, 0x89, 0x7e, 0xc7,
	    0x50, 0xe4, 0xd9, 0xae};
	return std::memcmp(digest, kPin, 32) == 0;
}

struct CostSample {
	double spec_per_emitted_ns = 0.0;
	double baseline_per_emitted_ns = 0.0;
	size_t emissions = 0;
};

// Times N speculate-driven emissions against the SAME count of baseline single-token v2
// emissions from twin sequences prefill'd identically (same corpus/hardware, plan SS5 S-F(1)).
bool MeasureCostRatio(sslm_model model, sslm_workspace ws, const CpuOracleModel& oracle,
                      const std::vector<int32_t>& prompt, size_t emissions_to_time,
                      CostSample* out) {
	GreedyRun want;
	std::string err;
	if (!RunGreedyReference(oracle, prompt.data(), prompt.size(), {}, emissions_to_time + 1,
	                        &want, &err)) {
		return false;
	}
	SinglePool sp;
	if (!MakePool(model, 2, &sp)) {
		CHECK(false);
		return false;
	}
	sslm_seq spec = nullptr, base = nullptr;
	if (sslm_seq_create(model, &sp.pool, &spec) != SSLM_OK ||
	    sslm_seq_create(model, &sp.pool, &base) != SSLM_OK) {
		CHECK(false);
		if (spec) sslm_seq_release(spec);
		if (base) sslm_seq_release(base);
		return false;
	}
	int32_t consumed = 0;
	if (sslm_prefill(model, spec, prompt.data(), static_cast<int32_t>(prompt.size()), 64,
	                 SSLM_SPAN_PROMPT, nullptr, &consumed) != SSLM_OK ||
	    sslm_prefill(model, base, prompt.data(), static_cast<int32_t>(prompt.size()), 64,
	                 SSLM_SPAN_PROMPT, nullptr, &consumed) != SSLM_OK) {
		CHECK(false);
		sslm_seq_release(spec);
		sslm_seq_release(base);
		return false;
	}

	sslm_speculate_params params{};
	if (!MakeSpecParams(model, 6, static_cast<int32_t>(emissions_to_time), {}, &params)) {
		CHECK(false);
		sslm_seq_release(spec);
		sslm_seq_release(base);
		return false;
	}
	sslm_decode_params dp{};
	dp.struct_size = sizeof(dp);
	dp.layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);

	using Clock = std::chrono::steady_clock;
	const auto t0 = Clock::now();
	std::vector<int32_t> tok(static_cast<size_t>(params.max_new_tokens) + 1, 0);
	std::vector<int32_t> rows((static_cast<size_t>(params.max_new_tokens) + 1) *
	                              static_cast<size_t>(oracle.vocab_size), 0);
	int32_t produced = 0, stop = -1;
	size_t spec_emitted = 0;
	bool drive_ok = true;
	while (spec_emitted < emissions_to_time) {
		params.max_new_tokens = static_cast<int32_t>(emissions_to_time - spec_emitted);
		if (sslm_speculate_step_v3(model, spec, &params, ws, tok.data(),
		                           static_cast<int32_t>(tok.size()), rows.data(),
		                           static_cast<int32_t>(rows.size()), &produced,
		                           &stop) != SSLM_OK) {
			drive_ok = false;
			break;
		}
		spec_emitted += static_cast<size_t>(produced);
		if (stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED || produced == 0) break;
	}
	const auto t1 = Clock::now();
	if (drive_ok) {
		for (size_t i = 0; i < emissions_to_time; ++i) {
			int32_t t = -1;
			if (sslm_decode_step_v2(model, &base, 1, &dp, ws, &t) != SSLM_OK) {
				drive_ok = false;
				break;
			}
		}
	}
	const auto t2 = Clock::now();
	sslm_seq_release(spec);
	sslm_seq_release(base);
	if (!drive_ok) return false;

	const auto ns = [](std::chrono::steady_clock::duration d) {
		return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
	};
	out->emissions = spec_emitted < emissions_to_time ? spec_emitted : emissions_to_time;
	out->spec_per_emitted_ns = ns(t1 - t0) / static_cast<double>(out->emissions);
	out->baseline_per_emitted_ns = ns(t2 - t1) / static_cast<double>(out->emissions);
	return out->emissions > 0 && out->spec_per_emitted_ns > 0.0 &&
	       out->baseline_per_emitted_ns > 0.0;
}

// M1 -- THE achievement assertion (S-F 2a): measured per-emitted-token cost ratio on the
// hash-pinned corpus below the commissioned threshold BY AT LEAST the commissioned minimum
// effect size -- the smallest improvement the assertion will act on. Refuses loudly when
// uncommissioned; requires the tokenized corpus (the claim is corpus-scoped).
void TestM1_AchievementCostRatioBelowCommissionedThreshold(sslm_model model, sslm_workspace ws,
                                                           const CpuOracleModel& oracle,
                                                           const std::vector<int32_t>& corpus_prompt) {
	CHECK_MSG(ThresholdsCommissioned(),
	          "NOT-COMMISSIONED: record max-cost-ratio and min-effect-size before first "
	          "reading (plan SS5 S-F(2a)); refusing to assert against nothing");
	if (!ThresholdsCommissioned()) return;

	CostSample sample;
	ASSERT_TRUE(MeasureCostRatio(model, ws, oracle, corpus_prompt,
	                             /*emissions_to_time=*/16, &sample));
	const double ratio = sample.spec_per_emitted_ns / sample.baseline_per_emitted_ns;
	std::printf("REPORT m1: ratio=%.6f (threshold=%.6f, min_effect=%.6f)\n", ratio,
	            g_max_cost_ratio, g_min_effect_size);
	CHECK_MSG(ratio <= g_max_cost_ratio - g_min_effect_size,
	          "achievement claim: per-emitted-token ratio %.6f must sit below threshold "
	          "%.6f by at least the minimum effect size %.6f",
	          ratio, g_max_cost_ratio, g_min_effect_size);
}

// M1b -- S-F(1)'s acceptance>=1 arm: the SAME commissioned measurement over a
// repetition-stream fixture (the full-K accepting-window prompt, where the drafter provably
// proposes and the target reproduces the continuation). This is the workload shape n-gram
// speculation exists for; the novel-chat drive M1 measures is its control.
void TestM1b_RepetitionStreamCostRatioBelowCommissionedThreshold(
    sslm_model model, sslm_workspace ws, const CpuOracleModel& oracle,
    const std::vector<int32_t>& corpus_stream) {
	CHECK_MSG(ThresholdsCommissioned(),
	          "NOT-COMMISSIONED: record max-cost-ratio and min-effect-size before first "
	          "reading (plan SS5 S-F(2a)); refusing to assert against nothing");
	if (!ThresholdsCommissioned()) return;
	std::string err;
	std::vector<int32_t> prompt;
	if (!FindFullKAcceptancePrompt(oracle, corpus_stream, /*kK=*/4, /*max_len=*/160, &prompt,
	                               &err)) {
		SKIP_MSG("repetition fixture not found on this artifact/corpus: %s", err.c_str());
		return;
	}
	CostSample sample;
	ASSERT_TRUE(MeasureCostRatio(model, ws, oracle, prompt,
	                             /*emissions_to_time=*/32, &sample));
	const double ratio = sample.spec_per_emitted_ns / sample.baseline_per_emitted_ns;
	std::printf("REPORT m1b: ratio=%.6f (threshold=%.6f, min_effect=%.6f, prompt=%zu)\n",
	            ratio, g_max_cost_ratio, g_min_effect_size, prompt.size());
	CHECK_MSG(ratio <= g_max_cost_ratio - g_min_effect_size,
	          "achievement claim (repetition stream): per-emitted-token ratio %.6f must sit "
	          "below threshold %.6f by at least the minimum effect size %.6f",
	          ratio, g_max_cost_ratio, g_min_effect_size);
}
// M2 -- Broken-drafter negative control: the same measurement procedure over a
// never-matching history must FAIL M1's predicate (ratio >= threshold -- verification
// overhead makes it slower, never faster), while every equivalence cell still holds on this
// fixture (dim07 F1 drives it). Proves M1 can fail.
void TestM2_BrokenDrafterNegativeControl(sslm_model model, sslm_workspace ws,
                                         const CpuOracleModel& oracle) {
	CHECK_MSG(ThresholdsCommissioned(), "control shares M1's commissioning gate");
	if (!ThresholdsCommissioned()) return;

	CostSample sample;
	ASSERT_TRUE(MeasureCostRatio(model, ws, oracle, NoRepeatPrompt(oracle.vocab_size, 12),
	                             /*emissions_to_time=*/16, &sample));
	const double broken_ratio = sample.spec_per_emitted_ns / sample.baseline_per_emitted_ns;
	std::printf("REPORT m2: broken-drafter ratio=%.6f\n", broken_ratio);
	CHECK_MSG(broken_ratio >= g_max_cost_ratio,
	          "the control MUST fail the achievement predicate -- a never-matching drafter "
	          "cannot be cheaper per emitted token");
}

// M3 -- Acceptance-histogram report scaffold (S-F cells 1-2, outside the catalog boundary;
// RS-G3): drives speculate over the corpus prompt and PRINTS the per-call emissions
// histogram -- bucket b counts calls that emitted exactly b tokens (b in [1, K+1]) -- plus
// the emission-weighted mean. Report-only: no threshold, no CHECK on the distribution.
void TestM3_AcceptanceHistogramReport(sslm_model model, sslm_workspace ws,
                                      const CpuOracleModel& oracle,
                                      const std::vector<int32_t>& corpus_prompt) {
	constexpr int32_t kK = 6;
	SinglePool sp;
	ASSERT_TRUE(MakeSinglePool(model, &sp));
	sslm_seq seq = nullptr;
	ASSERT_TRUE(sslm_seq_create(model, &sp.pool, &seq) == SSLM_OK);
	int32_t consumed = 0;
	ASSERT_TRUE(sslm_prefill(model, seq, corpus_prompt.data(),
	                         static_cast<int32_t>(corpus_prompt.size()), 64,
	                         SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	sslm_speculate_params params{};
	ASSERT_TRUE(MakeSpecParams(model, kK, 16, {}, &params));
	std::vector<int32_t> hist(static_cast<size_t>(kK) + 2, 0);  // index = emissions per call
	size_t calls = 0, total_emitted = 0;
	std::string err;
	while (total_emitted < 32 && calls < 64) {
		params.max_new_tokens =
		    static_cast<int32_t>(std::min<size_t>(kK + 1, 32 - total_emitted));
		std::vector<int32_t> tok(static_cast<size_t>(params.max_new_tokens), 0);
		std::vector<int32_t> rows(static_cast<size_t>(params.max_new_tokens) *
		                              static_cast<size_t>(oracle.vocab_size), 0);
		int32_t produced = -1, stop = -1;
		if (sslm_speculate_step_v3(model, seq, &params, ws, tok.data(),
		                           static_cast<int32_t>(tok.size()), rows.data(),
		                           static_cast<int32_t>(rows.size()), &produced,
		                           &stop) != SSLM_OK) {
			break;
		}
		if (produced <= 0) break;
		++calls;
		total_emitted += static_cast<size_t>(produced);
		++hist[static_cast<size_t>(produced)];
		if (stop == SSLM_SPECULATE_STOP_TOKEN_MATCHED) break;
	}
	if (calls > 0) {
		std::printf("REPORT m3: acceptance histogram over %zu speculate calls (%zu "
		            "emissions):\n",
		            calls, total_emitted);
		for (int32_t b = 1; b <= kK + 1; ++b) {
			if (hist[static_cast<size_t>(b)] == 0) continue;
			std::printf("REPORT m3:   %d emitted: %zu call(s)\n", b,
			            static_cast<size_t>(hist[static_cast<size_t>(b)]));
		}
		std::printf("REPORT m3:   mean emissions/call = %.4f\n",
		            static_cast<double>(total_emitted) / static_cast<double>(calls));
	} else {
		SKIP_MSG("M3 produced no speculate calls to report");
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

}  // namespace

int main(int argc, char** argv) {	// Fold round 2 commissioning input (D-SLM4094): the two gate values arrive as argv so
	// the recorded decision precedes any reading the assertion acts on. Unset keeps the
	// loud NOT-COMMISSIONED refusal.
	for (int ci = 1; ci < argc; ++ci) {
		const std::string ca = argv[ci];
		if (ca.rfind("--commission-max-cost-ratio=", 0) == 0)
			g_max_cost_ratio = std::atof(ca.substr(ca.find("=") + 1).c_str());
		else if (ca.rfind("--commission-min-effect-size=", 0) == 0)
			g_min_effect_size = std::atof(ca.substr(ca.find("=") + 1).c_str());
	}
	ParseFixtureArgs(argc, argv);
	if (g_model_path.empty()) {
		SKIP_MSG("--model=PATH not supplied -- dim12 M1-M2 not run");
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	if (!CorpusHashMatches(g_corpus_path)) {
		// The corpus pin is checked even when other fixtures are absent: a drifted corpus
		// silently re-bases every S-F number.
		SKIP_MSG("--corpus=PATH missing or hash mismatch against the commissioning pin");
	}
	SslmModelView view;
	std::vector<uint8_t> bytes;
	std::string err;
	if (!LoadRealModelView(g_model_path, &view, &bytes, &err)) {
		SKIP_MSG("could not load real artifact: %s", err.c_str());
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	CpuOracleModel oracle;
	if (!LoadCpuOracleModel(view, &oracle, &err)) {
		SKIP_MSG("could not marshal CPU oracle: %s", err.c_str());
		int ec;
		PrintSummaryAndExit(&ec);
		return ec;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	if (model) {
		sslm_config cfg{};
		cfg.max_batch = 16;
		cfg.max_chunk_budget = 256;
		cfg.max_layer_budget = static_cast<int32_t>(oracle.num_hidden_layers);
		const size_t ws_size = sslm_workspace_size(model, &cfg);
		AlignedBuffer ws_store(ws_size ? ws_size : 1);
		sslm_workspace ws = nullptr;
		if (sslm_workspace_create(model, &cfg, ws_store.data(), ws_store.size(), &ws) ==
		    SSLM_OK) {
			// M1 is corpus-scoped: tokenize the first corpus utterance with a tokenizer-bound
			// artifact. M2's control needs no corpus.
			std::vector<int32_t> corpus_prompt;
			if (!g_model_tok_path.empty()) {
				sslm_model tok_model = nullptr;
				std::vector<uint8_t> tok_bytes;
				if (ReadFileBytes(g_model_tok_path, &tok_bytes)) {
					CHECK(sslm_model_map(tok_bytes.data(), tok_bytes.size(),
					                     &tok_model) == SSLM_OK);
				}
				if (tok_model) {
					std::vector<std::string> utterances;
					if (LoadCorpusUtterances(g_corpus_path, 1, &utterances)) {
						TokenizeUtf8(tok_model, utterances.front(), &corpus_prompt);
					}
					CHECK(sslm_model_unmap(tok_model) == SSLM_OK);
				}
			}
			if (corpus_prompt.size() >= 8) {
				TestM1_AchievementCostRatioBelowCommissionedThreshold(model, ws, oracle,
				                                                      corpus_prompt);
				TestM3_AcceptanceHistogramReport(model, ws, oracle, corpus_prompt);
			} else {
				SKIP_MSG("M1/M3 need a tokenizer-bound artifact (--modeltok=PATH) to drive "
				         "the pinned corpus");
			}
			TestM2_BrokenDrafterNegativeControl(model, ws, oracle);
		if (!g_corpus_path.empty() && !g_model_tok_path.empty()) {
			std::vector<int32_t> rep_stream;
			sslm_model tok_model2 = nullptr;
			std::vector<uint8_t> tok_bytes2;
			if (ReadFileBytes(g_model_tok_path, &tok_bytes2)) {
				CHECK(sslm_model_map(tok_bytes2.data(), tok_bytes2.size(), &tok_model2) ==
				      SSLM_OK);
			}
			if (tok_model2) {
				std::vector<std::string> ut2;
				if (LoadCorpusUtterances(g_corpus_path, 24, &ut2)) {
					for (const auto& u : ut2) {
						std::vector<int32_t> ids;
						if (TokenizeUtf8(tok_model2, u, &ids) && ids.size() >= 8)
							rep_stream.insert(rep_stream.end(), ids.begin(), ids.end());
					}
				}
				CHECK(sslm_model_unmap(tok_model2) == SSLM_OK);
			}
			if (!rep_stream.empty())
				TestM1b_RepetitionStreamCostRatioBelowCommissionedThreshold(
				    model, ws, oracle, rep_stream);
		}
			CHECK(sslm_workspace_destroy(ws) == SSLM_OK);
		} else {
			SKIP_MSG("workspace create failed -- dim12 needs a workspace");
		}
		CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
	int ec;
	PrintSummaryAndExit(&ec);
	return ec;
}



