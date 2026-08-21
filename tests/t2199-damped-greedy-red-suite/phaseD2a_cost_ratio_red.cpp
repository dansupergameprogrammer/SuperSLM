// T-2199 (Curie) -- Phase D2a red suite: the re-sited cost-ratio cell (plan Sec8 D2a, Sec9
// dim6/dim7's own speed-headroom claim, ratio half). Measures TopKRenormalizeQ15's own
// per-token cost (already real, Phase C, this repo's own sslm_damped_greedy.h) against this
// engine's own REAL total per-token forward cost, taken through the wired sslm_decode_step
// entry point.
//
// MIGRATED 2026-08-20 (T-2199 Phase D review fix S5; conductor's follow-on commission, item 2):
// calls the real production sslm_decode_step/sslm_decode_params directly instead of the
// suite-compatibility shim (sslm_decode_step_damped_greedy/sslm_decode_params_damped_greedy) --
// see phaseD2_wiring_red.cpp's own header comment for the full migration rationale.
//
// Coverage Model cell realized here (plan Sec9 dim6/dim7, re-sited 2026-08-20):
//   Classification: GATE, authorized by D-SLM3719 (a cost-order-of-growth/affordability bound
//   is engine correctness, not a calibration reading). Budget: TopKRenormalizeQ15's own
//   measured cost must not exceed 5% of the total per-token forward cost (mean, both target
//   checkpoints) -- FLAGGED, per the plan's own text, as this plan's own proposal, not
//   independently derived from a product-side latency requirement; Charpy/Dan's to confirm or
//   tighten before this cell is read as a shipped gate. This cell authors the MEASUREMENT
//   HARNESS red now; the 5% figure is asserted as written, with the flag restated in the
//   failure message so a future reader is not left to rediscover it from prose alone.
//
// Needs a real base artifact (--model=PATH) for the total-forward-cost denominator -- SKIPs
// honestly when not supplied, matching this suite's own established discipline.
#include "sslm_phaseD_stub.h"
#include "sslm_phaseD_fixture.h"
#include "sslm_damped_greedy.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static int GChecks = 0;
static int GFailures = 0;
static int GSkips = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
			std::fflush(stdout); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
			std::fflush(stdout); \
		} \
	} while (0)

#define SKIP_MSG(...) \
	do { \
		++GSkips; \
		std::printf("SKIP %s:%d -- ", __FILE__, __LINE__); \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
		std::fflush(stdout); \
	} while (0)

using namespace superslm;
using namespace superslm_test_phaseD;

static std::string g_model_path;
static void ParseArgs(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		const char* kFlag = "--model=";
		if (a.compare(0, std::strlen(kFlag), kFlag) == 0) g_model_path = a.c_str() + std::strlen(kFlag);
	}
}
static bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	if (path.empty()) return false;
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? (size_t)sz : 0);
	if (sz > 0) {
		const size_t n = std::fread(out->data(), 1, (size_t)sz, f);
		std::fclose(f);
		if (n != (size_t)sz) return false;
	} else {
		std::fclose(f);
	}
	return true;
}
struct AlignedBuffer {
	explicit AlignedBuffer(size_t n)
	    : bytes_(n), storage_(n > 0 ? ::operator new(n, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)) : nullptr) {}
	~AlignedBuffer() {
		if (storage_) ::operator delete(storage_, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer& operator=(const AlignedBuffer&) = delete;
	void* data() { return storage_; }
	size_t size() const { return bytes_; }

  private:
	size_t bytes_;
	void* storage_;
};

// --- TestD2a_TopKRenormalizeQ15CostRatio_WithinFiveGercentOfRealForwardCost -----------------
static void TestD2a_TopKRenormalizeQ15CostRatio_WithinFivePercentOfRealForwardCost() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- cost-ratio cell not run "
		         "(needs a real per-token forward cost as denominator, plan Sec8 D2a's own "
		         "re-siting reason)");
		return;
	}
	std::vector<uint8_t> bytes;
	if (!ReadFileBytes(g_model_path, &bytes)) {
		SKIP_MSG("could not read %s", g_model_path.c_str());
		return;
	}
	SslmModelView view;
	std::string err;
	if (SslmModel::Load(bytes.data(), bytes.size(), view, &err) != SslmModelStatus::Ok) {
		SKIP_MSG("could not parse real artifact: %s", err.c_str());
		return;
	}
	sslm_model model = nullptr;
	CHECK(sslm_model_map(bytes.data(), bytes.size(), &model) == SSLM_OK);
	AlignedBuffer pool_buf(sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 1));
	sslm_kv_pool pool = nullptr;
	CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), 1, &pool) == SSLM_OK);
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(model, &pool, &seq) == SSLM_OK);
	const int32_t prompt[4] = {0, 1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefill(model, seq, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed) == SSLM_OK);

	// --- Denominator: this engine's own REAL total per-token forward cost, timed through the
	// wired damped-greedy entry point itself (the first point in this build where this
	// denominator exists at all, per the plan's own D2a re-siting reason) -- mean over several
	// real steps, matching the plan's own "mean, both target checkpoints" framing (single
	// checkpoint per invocation; a runner sweeps both by re-invoking with each --model=).
	constexpr int kSteps = 12;
	sslm_decode_params params{};
	params.struct_size = sizeof(params);  // D-SLM3797: caller-set, library-validated
	params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
	params.mode = SSLM_DECODE_MODE_DAMPED_GREEDY;
	params.alpha_q15 = int32_t{1} << 14;
	params.anti_lm_max_order = 2;
	params.top_k = 6;
	// FIXED 2026-08-20 (conductor's dispute-resolution commission, dispute 3): derives the real
	// (q_ln2, q_b, q_c) triple instead of leaving q_b/q_c at their zero-init default, which made
	// every call refuse (SSLM_ARTIFACT_REJECTED) before the timed/concurrent work ever ran.
	CHECK(t2199phaseD::DeriveDefaultScaleConstants(&params.q_ln2, &params.q_b, &params.q_c));
	sslm_seq batch[1] = {seq};
	double total_forward_ns = 0.0;
	for (int i = 0; i < kSteps; ++i) {
		int32_t tok = -1;
		const auto t0 = std::chrono::steady_clock::now();
		CHECK(sslm_decode_step(model, batch, 1, &params, nullptr, &tok) == SSLM_OK);
		const auto t1 = std::chrono::steady_clock::now();
		total_forward_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
	}
	const double mean_forward_ns = total_forward_ns / kSteps;

	// --- Numerator: TopKRenormalizeQ15's own per-token cost, isolated -- already real (Phase
	// C, this repo's own sslm_damped_greedy.h), timed directly, not through the wired call
	// (matching Phase C2a's own already-pinned "measured in isolation" framing; D2a's own job
	// is only the RATIO's denominator, not re-measuring the numerator a second, different way).
	std::vector<int32_t> row(static_cast<size_t>(view.config.vocab_size));
	for (int32_t t = 0; t < view.config.vocab_size; ++t) row[t] = (t * 2654435761u) % 4000 - 2000;
	std::vector<int32_t> indices(6);
	std::vector<uint8_t> mask((view.config.vocab_size + 7) / 8, 0xFF);
	FsdTopK(row.data(), mask.data(), view.config.vocab_size, 6, indices.data());
	constexpr int kRenormIters = 2000;
	int64_t out_q15[6];
	const auto rt0 = std::chrono::steady_clock::now();
	for (int i = 0; i < kRenormIters; ++i) {
		TopKRenormalizeQ15(row.data(), indices.data(), 6, 493, 0, 0, out_q15);
	}
	const auto rt1 = std::chrono::steady_clock::now();
	const double mean_renorm_ns =
	    std::chrono::duration<double, std::nano>(rt1 - rt0).count() / kRenormIters;

	CHECK_MSG(mean_forward_ns > 0.0, "measured mean forward cost must be positive (%f ns) -- a "
	                                  "zero or negative reading means the timer/harness is broken",
	          mean_forward_ns);
	const double ratio = mean_forward_ns > 0.0 ? (mean_renorm_ns / mean_forward_ns) : 1e300;
	std::printf("D2a: mean_forward_ns=%.1f mean_renorm_ns=%.1f ratio=%.4f%% (budget: <=5%%, "
	            "FLAGGED not independently derived -- plan Sec8 C2a/D2a)\n",
	            mean_forward_ns, mean_renorm_ns, ratio * 100.0);
	CHECK_MSG(ratio <= 0.05,
	          "TopKRenormalizeQ15's own per-token cost (%.1f ns) must be <=5%% of this engine's "
	          "real total per-token forward cost (%.1f ns), got %.4f%% -- budget FLAGGED "
	          "(plan Sec8 C2a/D2a: 'this plan's own proposal, not independently derived from a "
	          "product-side latency requirement -- Charpy/Dan should confirm or tighten it')",
	          mean_renorm_ns, mean_forward_ns, ratio * 100.0);

	CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseArgs(argc, argv);
	TestD2a_TopKRenormalizeQ15CostRatio_WithinFivePercentOfRealForwardCost();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
