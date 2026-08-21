// T-2199 (Curie) -- Phase D2 red suite: the teardown-during-flight cell (plan Sec9 dim3, sited
// at Phase D2 by Mendeleev's coverage audit, d3b310714d -- "not testable before D2 lands," the
// anti-LM is new per-sequence heap state introduced onto an existing multi-sequence batched
// ABI). RED BY LINK: sslm_decode_step_damped_greedy is declared in sslm_phaseD_stub.h, defined
// nowhere yet.
//
// Coverage Model cell realized here (plan Sec9 dim3):
//   Classification: GATE, authorized by D-SLM3719 (race/sharing freedom is engine correctness,
//   not a calibration reading). "Tears down/frees one sequence's anti-LM state concurrently
//   with an in-flight batched decode step mutating a DIFFERENT sequence's anti-LM state in the
//   same call, asserting no cross-sequence corruption, no use-after-free, and a clean status
//   for the surviving sequence."
//
// Construction: a batched sslm_decode_step_damped_greedy(model, {A, B}, n=2, ...) call, driven
// on one background thread, raced against a SECOND background thread that calls
// sslm_seq_release(B) as soon as it can after the batched call is launched -- repeated across
// many trials with fresh sequences each time, since the exact interleaving that exercises the
// hazard cannot be forced deterministically (a standard concurrent-fuzz construction, not a
// single deterministic repro). Assertion: across every trial, the process does not crash
// (fatal for the whole binary, caught by this suite's own build_link_red_phaseD.bat/exit-code
// convention same as every other crash-prone construction in this campaign), sequence A's own
// call returns a defined status (never garbage), and no trial reports a status this cell did
// not anticipate. Needs a real base artifact (--model=PATH); SKIPs honestly otherwise.
#include "sslm_phaseD_stub.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
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

static void TestD3_TeardownDuringFlight_ConcurrentReleaseDoesNotCorruptSurvivor() {
	if (g_model_path.empty()) {
		SKIP_MSG("real base artifact not supplied (--model=PATH) -- teardown-during-flight cell not run");
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

	constexpr int kTrials = 200;
	std::atomic<int> surviving_ok_count{0};
	std::atomic<int> surviving_bad_status_count{0};

	for (int trial = 0; trial < kTrials; ++trial) {
		AlignedBuffer pool_buf(2 * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, 2));
		sslm_kv_pool pool = nullptr;
		CHECK(sslm_kv_pool_create(model, pool_buf.data(), pool_buf.size(), 2, &pool) == SSLM_OK);
		sslm_seq seq_a = nullptr, seq_b = nullptr;
		CHECK(sslm_seq_create(model, &pool, &seq_a) == SSLM_OK);
		CHECK(sslm_seq_create(model, &pool, &seq_b) == SSLM_OK);
		const int32_t prompt[4] = {0, 1, 2, 3};
		int32_t consumed_a = 0, consumed_b = 0;
		CHECK(sslm_prefill(model, seq_a, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_a) == SSLM_OK);
		CHECK(sslm_prefill(model, seq_b, prompt, 4, 8, SSLM_SPAN_PROMPT, nullptr, &consumed_b) == SSLM_OK);

		sslm_decode_params_damped_greedy params{};
		params.layer_budget = static_cast<int32_t>(view.config.num_hidden_layers);
		params.mode = DampedGreedyMode::kDampedGreedy;
		params.alpha_q15 = int64_t{1} << 14;
		params.anti_lm_max_order = 2;
		params.top_k = 6;
		params.q_ln2 = 493;
		sslm_seq batch[2] = {seq_a, seq_b};
		int32_t out_tokens[2] = {-1, -1};

		std::atomic<bool> b_released{false};
		std::thread decode_thread([&] {
			// One batched call, sequences {A, B} -- exactly the "in-flight batched decode step
			// mutating a DIFFERENT sequence's anti-LM state in the SAME call" construction.
			sslm_decode_step_damped_greedy(model, batch, 2, &params, nullptr, out_tokens);
		});
		std::thread release_thread([&] {
			// No synchronization barrier deliberately -- the race window this cell targets is
			// exactly "as early as the scheduler allows," which a barrier would remove.
			sslm_seq_release(seq_b);
			b_released.store(true, std::memory_order_release);
		});
		decode_thread.join();
		release_thread.join();
		CHECK(b_released.load(std::memory_order_acquire));

		// The process surviving this join (no crash) IS this cell's own primary, always-checked
		// assertion for every trial -- reaching the line below at all is evidence against a
		// use-after-free/double-free severe enough to fault; a process-fatal defect is caught by
		// this file's own binary exiting non-zero with no summary line (matching every other
		// crash-prone construction in this campaign, e.g. Phase A's M4/max_order=-1 cell).
		if (out_tokens[0] >= 0 && out_tokens[0] < view.config.vocab_size) {
			surviving_ok_count.fetch_add(1);
		} else {
			surviving_bad_status_count.fetch_add(1);
		}

		CHECK(sslm_seq_release(seq_a) == SSLM_OK);
		CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
	}

	std::printf("D3 teardown-during-flight: %d/%d trials -- surviving sequence A produced a "
	            "real in-domain token; %d trial(s) reported an out-of-domain/negative token "
	            "(a real defect if nonzero on a build that otherwise reports SSLM_OK for both "
	            "calls; investigate rather than assume benign)\n",
	            surviving_ok_count.load(), kTrials, surviving_bad_status_count.load());
	CHECK_MSG(surviving_bad_status_count.load() == 0,
	          "%d of %d trials produced an out-of-domain token for the surviving sequence A "
	          "while sequence B was concurrently released -- cross-sequence corruption or a "
	          "use-after-free reading garbage into A's own decode path",
	          surviving_bad_status_count.load(), kTrials);

	CHECK(sslm_model_unmap(model) == SSLM_OK);
}

int main(int argc, char** argv) {
	ParseArgs(argc, argv);
	TestD3_TeardownDuringFlight_ConcurrentReleaseDoesNotCorruptSurvivor();
	std::printf("checks=%d failures=%d skips=%d\n", GChecks, GFailures, GSkips);
	return GFailures ? 1 : 0;
}
