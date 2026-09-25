// 1.8.x saturation census: the four per-site counts `SequenceLayerState::kv_saturation_count`
// sums must travel with that total through the two CPU ABI paths that move it without going
// through RunLayerLoop. `sslm_seq_adopt_prefix` copied the total but left the adopting sequence's
// four per-site counts at whatever they held before; `sslm_prefill`/`sslm_prefix_prefill`
// (PrefillWholeTokens -> RunLayerLoopChunkBatched) grew the total and never passed the four
// per-site out-parameters, so a prefilled sequence's census no longer summed to its total.
// Diagnostic only: neither path changes a token.
//
// These cells live in their own translation unit because they call the C ABI, and
// tests/test_main.cpp cannot include superslm/sslm_abi.h on Windows: its `using enum
// SslmGpuStatus;` brings the GPU enumerators (SSLM_OK, SSLM_ADAPTER_MODEL_MISMATCH, ...) into
// the global scope, where the C ABI's unscoped sslm_status enumerators of the same names
// collide with them. test_main.cpp calls RunSlm18xSaturationCensusCells and adds this unit's
// check and failure counts to its own totals.

#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "superslm/forward_sites.h"
#include "superslm/layer_marshal.h"
#include "superslm/model.h"
#include "superslm/sslm_abi.h"
#include "sslm_tokenizer_fixtures.h"

using superslm_test::ResolveFixturePath;

static int GChecks = 0;
static int GFailures = 0;

#define CHECK(cond) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
		} \
	} while (0)

#define CHECK_MSG(cond, ...) \
	do { \
		++GChecks; \
		if (!(cond)) { \
			++GFailures; \
			std::printf("FAIL %s:%d: %s — ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

extern "C" superslm::SequenceLayerState* SslmSeqLiveStateForTest(sslm_seq);
extern "C" superslm::SequenceLayerState* SslmPrefixLiveStateForTest(sslm_prefix);

namespace {

struct SatCensusAbiFixture {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	void* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	int32_t vocab_size = 0;

	bool Open(uint32_t block_count) {
		const std::string path = ResolveFixturePath("t2572_arm_c_non_qknorm_fixture.sslm");
		CHECK_MSG(!path.empty(), "fixture t2572_arm_c_non_qknorm_fixture.sslm not found");
		if (path.empty()) return false;
		CHECK_MSG(superslm_marshal::ReadFile(path.c_str(), bytes), "failed to read %s",
		          path.c_str());
		if (bytes.empty()) return false;
		superslm::SslmModelView view;
		std::string err;
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) !=
		    superslm::SslmModelStatus::Ok) {
			CHECK_MSG(false, "SslmModel::Load(fixture) failed: %s", err.c_str());
			return false;
		}
		vocab_size = static_cast<int32_t>(view.config.vocab_size);
		const sslm_status ms = sslm_model_map(bytes.data(), bytes.size(), &model);
		CHECK_MSG(ms == SSLM_OK, "sslm_model_map(fixture) == %d, want SSLM_OK", static_cast<int>(ms));
		if (ms != SSLM_OK) return false;
		const size_t size = block_count * sslm_kv_block_size(model) +
		                    sslm_kv_pool_overhead_size(model, block_count);
		pool_buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		const sslm_status ps = sslm_kv_pool_create(model, pool_buf, size, block_count, &pool);
		CHECK_MSG(ps == SSLM_OK, "sslm_kv_pool_create == %d, want SSLM_OK", static_cast<int>(ps));
		return ps == SSLM_OK;
	}
	~SatCensusAbiFixture() {
		if (pool) CHECK(sslm_kv_pool_destroy(pool) == SSLM_OK);
		if (pool_buf) ::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		if (model) CHECK(sslm_model_unmap(model) == SSLM_OK);
	}
};

uint64_t SatCensusSiteSum(const superslm::SequenceLayerState& s) {
	return s.kv_landing_saturation_count + s.k_channel_landing_saturation_count +
	       s.rope_q_saturation_count + s.rope_k_saturation_count;
}

}  // namespace

// Adoption replaces the sequence's origin with the prefix's, so the whole census moves with it:
// the adopting sequence's four per-site counts must equal the prefix's afterwards, not survive
// from the sequence's own prior history. Seeded through the test-only accessors (the dim1 M2
// reset cell's pattern) so the cell does not depend on this fixture saturating every site.
static void TestSlm18x_AdoptPrefixCarriesPerSiteSaturationCensus() {
	SatCensusAbiFixture f;
	if (!f.Open(/*block_count=*/2)) return;
	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(f.model, &f.pool, &prefix) == SSLM_OK);
	if (!prefix) return;
	const int32_t tokens[3] = {1, 2, 3};
	int32_t consumed = 0;
	CHECK(sslm_prefix_prefill(f.model, prefix, tokens, 3, 3, SSLM_SPAN_PROMPT, nullptr,
	                          &consumed) == SSLM_OK);
	CHECK(consumed == 3);
	superslm::SequenceLayerState* ps = SslmPrefixLiveStateForTest(prefix);
	CHECK(ps != nullptr);
	if (!ps) {
		CHECK(sslm_prefix_release(prefix) == SSLM_OK);
		return;
	}
	ps->kv_landing_saturation_count = 5;
	ps->k_channel_landing_saturation_count = 7;
	ps->rope_q_saturation_count = 11;
	ps->rope_k_saturation_count = 23;
	ps->kv_saturation_count = 46;
	CHECK(sslm_prefix_freeze(prefix) == SSLM_OK);

	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(f.model, &f.pool, &seq) == SSLM_OK);
	superslm::SequenceLayerState* ss = SslmSeqLiveStateForTest(seq);
	CHECK(ss != nullptr);
	if (ss) {
		// The sequence's own prior history, which adoption must discard.
		ss->kv_landing_saturation_count = 1000;
		ss->k_channel_landing_saturation_count = 2000;
		ss->rope_q_saturation_count = 3000;
		ss->rope_k_saturation_count = 4000;
		ss->kv_saturation_count = 10000;
		CHECK(sslm_seq_adopt_prefix(seq, prefix) == SSLM_OK);
		CHECK_MSG(ss->kv_saturation_count == 46, "adopted total == %llu, want 46",
		          static_cast<unsigned long long>(ss->kv_saturation_count));
		CHECK_MSG(ss->kv_landing_saturation_count == 5, "adopted kv_landing == %llu, want 5",
		          static_cast<unsigned long long>(ss->kv_landing_saturation_count));
		CHECK_MSG(ss->k_channel_landing_saturation_count == 7,
		          "adopted k_channel_landing == %llu, want 7",
		          static_cast<unsigned long long>(ss->k_channel_landing_saturation_count));
		CHECK_MSG(ss->rope_q_saturation_count == 11, "adopted rope_q == %llu, want 11",
		          static_cast<unsigned long long>(ss->rope_q_saturation_count));
		CHECK_MSG(ss->rope_k_saturation_count == 23, "adopted rope_k == %llu, want 23",
		          static_cast<unsigned long long>(ss->rope_k_saturation_count));
	}
	if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
	CHECK(sslm_prefix_release(prefix) == SSLM_OK);
}

// Prefill must keep the census summing to its total, as RunLayerLoop's decode path already does.
// The prompt is chosen so this fixture really saturates (the total is asserted non-zero first, so
// the cell cannot pass vacuously on a prompt that never clamps).
static void TestSlm18x_PrefillFillsPerSiteSaturationCensus() {
	SatCensusAbiFixture f;
	if (!f.Open(/*block_count=*/2)) return;
	std::vector<int32_t> prompt;
	for (int32_t t = 0; t < f.vocab_size && prompt.size() < 12; ++t) prompt.push_back(t);
	const int32_t n = static_cast<int32_t>(prompt.size());

	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(f.model, &f.pool, &seq) == SSLM_OK);
	if (!seq) return;
	int32_t consumed = 0;
	CHECK(sslm_prefill(f.model, seq, prompt.data(), n, n, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	CHECK(consumed == n);
	const superslm::SequenceLayerState* ss = SslmSeqLiveStateForTest(seq);
	CHECK(ss != nullptr);
	if (ss) {
		CHECK_MSG(ss->kv_saturation_count > 0,
		          "sslm_prefill saturation total == 0 on this prompt; the cell needs a prompt "
		          "that clamps");
		CHECK_MSG(SatCensusSiteSum(*ss) == ss->kv_saturation_count,
		          "sslm_prefill per-site census sums to %llu, total is %llu",
		          static_cast<unsigned long long>(SatCensusSiteSum(*ss)),
		          static_cast<unsigned long long>(ss->kv_saturation_count));
	}

	sslm_prefix prefix = nullptr;
	CHECK(sslm_prefix_begin(f.model, &f.pool, &prefix) == SSLM_OK);
	consumed = 0;
	CHECK(sslm_prefix_prefill(f.model, prefix, prompt.data(), n, n, SSLM_SPAN_PROMPT, nullptr,
	                          &consumed) == SSLM_OK);
	const superslm::SequenceLayerState* ps = SslmPrefixLiveStateForTest(prefix);
	CHECK(ps != nullptr);
	if (ps && ss) {
		CHECK_MSG(ps->kv_saturation_count == ss->kv_saturation_count,
		          "prefix and sequence prefill of the same prompt disagree on the total");
		CHECK_MSG(SatCensusSiteSum(*ps) == ps->kv_saturation_count,
		          "sslm_prefix_prefill per-site census sums to %llu, total is %llu",
		          static_cast<unsigned long long>(SatCensusSiteSum(*ps)),
		          static_cast<unsigned long long>(ps->kv_saturation_count));
	}
	if (seq) CHECK(sslm_seq_release(seq) == SSLM_OK);
	if (prefix) CHECK(sslm_prefix_release(prefix) == SSLM_OK);
}

void RunSlm18xSaturationCensusCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestSlm18x_AdoptPrefixCarriesPerSiteSaturationCensus();
	TestSlm18x_PrefillFillsPerSiteSaturationCensus();
	checks += GChecks;
	failures += GFailures;
}
