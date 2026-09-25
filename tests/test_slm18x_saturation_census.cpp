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
#include <cstring>
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

// --- 1.9.0: the per-site census through sslm_seq_save/sslm_seq_restore (TE-441). ---------------
// 'SSB5' is the 'SSB4' layout plus the four per-site counts, LE64 at 124, 132, 140, 148
// (kv_landing, k_channel_landing, rope_q, rope_k); its fixed header is 156 bytes. A legacy 'SSB4'
// blob restores its saved total with per-site counts of 0, because it never recorded them. The
// real-model cells for the same claims are tests/te441-sat-restore-red-suite/; these run in
// superslm_tests so the CI build, and its branch-coverage run, execute the save and restore paths.

namespace {

uint64_t SatCensusReadLE64(const std::vector<uint8_t>& b, size_t off) {
	uint64_t v = 0;
	for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(b[off + i]) << (8 * i);
	return v;
}

struct SatCensusSnapshot {
	uint64_t total, kv, kch, rq, rk;
};

SatCensusSnapshot SatCensusRead(const superslm::SequenceLayerState& s) {
	return {s.kv_saturation_count, s.kv_landing_saturation_count, s.k_channel_landing_saturation_count,
	        s.rope_q_saturation_count, s.rope_k_saturation_count};
}

bool SatCensusSave(sslm_seq seq, std::vector<uint8_t>* out) {
	size_t need = 0;
	if (sslm_seq_save(seq, nullptr, &need) != SSLM_BUFFER_TOO_SMALL || need == 0) return false;
	out->assign(need, 0);
	size_t wrote = out->size();
	if (sslm_seq_save(seq, out->data(), &wrote) != SSLM_OK) return false;
	out->resize(wrote);
	return true;
}

// The next greedy token, one layer per call (a call that finishes a token returns it, >= 0).
int32_t SatCensusNextToken(sslm_model model, sslm_seq seq) {
	sslm_decode_params p{};
	p.layer_budget = 1;
	sslm_seq b[1] = {seq};
	int32_t tok = -1;
	for (int guard = 0; tok < 0 && guard < 4096; ++guard) {
		if (sslm_decode_step(model, b, 1, &p, nullptr, &tok) != SSLM_OK) return -999;
	}
	return tok;
}

// A sequence prefilled on the committed fixture with the prompt the prefill cell above uses,
// which saturates on this fixture (each caller asserts the total is non-zero).
sslm_seq SatCensusPrefilled(SatCensusAbiFixture& f) {
	std::vector<int32_t> prompt;
	for (int32_t t = 0; t < f.vocab_size && prompt.size() < 12; ++t) prompt.push_back(t);
	const int32_t n = static_cast<int32_t>(prompt.size());
	sslm_seq seq = nullptr;
	CHECK(sslm_seq_create(f.model, &f.pool, &seq) == SSLM_OK);
	if (!seq) return nullptr;
	int32_t consumed = 0;
	CHECK(sslm_prefill(f.model, seq, prompt.data(), n, n, SSLM_SPAN_PROMPT, nullptr, &consumed) ==
	      SSLM_OK);
	CHECK(consumed == n);
	return seq;
}

}  // namespace

// Save writes 'SSB5' with the live per-site counts at their offsets, and restore carries them.
// Round 0 uses the counts a real prefill produced; round 1 seeds four distinct values (one above
// 2^32) through the test accessor, so a swapped or truncated field cannot pass.
static void TestSlm19x_SaveRestoreCarriesPerSiteSaturationCensus() {
	SatCensusAbiFixture f;
	if (!f.Open(/*block_count=*/3)) return;
	sslm_seq seq = SatCensusPrefilled(f);
	if (!seq) return;
	superslm::SequenceLayerState* ss = SslmSeqLiveStateForTest(seq);
	CHECK(ss != nullptr);
	if (!ss) {
		CHECK(sslm_seq_release(seq) == SSLM_OK);
		return;
	}
	CHECK_MSG(ss->kv_saturation_count > 0,
	          "the fixture prompt no longer saturates; the cell needs a prompt that clamps");

	for (int round = 0; round < 2; ++round) {
		if (round == 1) {
			ss->kv_landing_saturation_count = 3;
			ss->k_channel_landing_saturation_count = 5;
			ss->rope_q_saturation_count = (uint64_t{1} << 32) + 7;
			ss->rope_k_saturation_count = 11;
			ss->kv_saturation_count = 3 + 5 + (uint64_t{1} << 32) + 7 + 11;
		}
		const SatCensusSnapshot want = SatCensusRead(*ss);
		std::vector<uint8_t> blob;
		CHECK_MSG(SatCensusSave(seq, &blob), "round %d: sslm_seq_save", round);
		if (blob.size() < 160) {
			CHECK_MSG(false, "round %d: a %zu-byte blob cannot hold the SSB5 fixed header", round,
			          blob.size());
			continue;
		}
		CHECK_MSG(std::memcmp(blob.data(), "SSB5", 4) == 0, "round %d: sslm_seq_save writes 'SSB5'", round);
		CHECK_MSG(SatCensusReadLE64(blob, 92) == want.total, "round %d: total at 92", round);
		CHECK_MSG(SatCensusReadLE64(blob, 124) == want.kv, "round %d: kv_landing at 124", round);
		CHECK_MSG(SatCensusReadLE64(blob, 132) == want.kch, "round %d: k_channel_landing at 132", round);
		CHECK_MSG(SatCensusReadLE64(blob, 140) == want.rq, "round %d: rope_q at 140", round);
		CHECK_MSG(SatCensusReadLE64(blob, 148) == want.rk, "round %d: rope_k at 148", round);

		sslm_seq r = nullptr;
		CHECK_MSG(sslm_seq_restore(f.model, &f.pool, blob.data(), blob.size(), &r) == SSLM_OK && r,
		          "round %d: restore the SSB5 blob", round);
		if (!r) continue;
		const superslm::SequenceLayerState* rs = SslmSeqLiveStateForTest(r);
		CHECK(rs != nullptr);
		if (rs) {
			const SatCensusSnapshot got = SatCensusRead(*rs);
			CHECK_MSG(got.total == want.total && got.kv == want.kv && got.kch == want.kch &&
			              got.rq == want.rq && got.rk == want.rk,
			          "round %d: restored census {%llu; %llu %llu %llu %llu}, saved {%llu; %llu %llu %llu %llu}",
			          round, static_cast<unsigned long long>(got.total),
			          static_cast<unsigned long long>(got.kv), static_cast<unsigned long long>(got.kch),
			          static_cast<unsigned long long>(got.rq), static_cast<unsigned long long>(got.rk),
			          static_cast<unsigned long long>(want.total), static_cast<unsigned long long>(want.kv),
			          static_cast<unsigned long long>(want.kch), static_cast<unsigned long long>(want.rq),
			          static_cast<unsigned long long>(want.rk));
			CHECK_MSG(SatCensusSiteSum(*rs) == rs->kv_saturation_count,
			          "round %d: the restored per-site counts sum to the restored total", round);
		}
		CHECK(sslm_seq_release(r) == SSLM_OK);
	}
	CHECK(sslm_seq_release(seq) == SSLM_OK);
}

// A legacy 'SSB4' blob -- the real SSB5 blob with its four per-site fields removed -- restores:
// the saved total is kept, the per-site counts read 0, and it decodes the same next token as the
// SSB5 restore. An SSB5 blob cut inside its fixed header rejects SSLM_INVALID_ARGUMENT.
static void TestSlm19x_LegacySsb4RestoreAndTruncatedSsb5() {
	SatCensusAbiFixture f;
	if (!f.Open(/*block_count=*/3)) return;
	sslm_seq seq = SatCensusPrefilled(f);
	if (!seq) return;
	const superslm::SequenceLayerState* ss = SslmSeqLiveStateForTest(seq);
	CHECK(ss != nullptr);
	const uint64_t saved_total = ss ? ss->kv_saturation_count : 0;
	CHECK_MSG(saved_total > 0, "the fixture prompt no longer saturates; the cell needs a prompt that clamps");
	std::vector<uint8_t> ssb5;
	CHECK(SatCensusSave(seq, &ssb5));
	CHECK(sslm_seq_release(seq) == SSLM_OK);
	if (ssb5.size() < 160 || std::memcmp(ssb5.data(), "SSB5", 4) != 0) {
		CHECK_MSG(false, "sslm_seq_save did not write an 'SSB5' blob");
		return;
	}

	std::vector<uint8_t> ssb4 = {'S', 'S', 'B', '4'};
	ssb4.insert(ssb4.end(), ssb5.begin() + 4, ssb5.begin() + 124);
	ssb4.insert(ssb4.end(), ssb5.begin() + 156, ssb5.end());

	sslm_seq r4 = nullptr;
	CHECK_MSG(sslm_seq_restore(f.model, &f.pool, ssb4.data(), ssb4.size(), &r4) == SSLM_OK && r4,
	          "a legacy 'SSB4' blob restores");
	int32_t tok4 = -1;
	if (r4) {
		const superslm::SequenceLayerState* rs = SslmSeqLiveStateForTest(r4);
		CHECK(rs != nullptr);
		if (rs) {
			CHECK_MSG(rs->kv_saturation_count == saved_total, "legacy SSB4 restore keeps the saved total");
			CHECK_MSG(rs->kv_landing_saturation_count == 0 && rs->k_channel_landing_saturation_count == 0 &&
			              rs->rope_q_saturation_count == 0 && rs->rope_k_saturation_count == 0,
			          "legacy SSB4 restore: every per-site count reads 0");
		}
		tok4 = SatCensusNextToken(f.model, r4);
		CHECK(sslm_seq_release(r4) == SSLM_OK);
	}
	sslm_seq r5 = nullptr;
	CHECK(sslm_seq_restore(f.model, &f.pool, ssb5.data(), ssb5.size(), &r5) == SSLM_OK && r5);
	if (r5) {
		const int32_t tok5 = SatCensusNextToken(f.model, r5);
		CHECK_MSG(tok5 >= 0 && tok4 == tok5, "legacy SSB4 restore decodes %d, the SSB5 restore %d",
		          static_cast<int>(tok4), static_cast<int>(tok5));
		CHECK(sslm_seq_release(r5) == SSLM_OK);
	}

	// 150 bytes: enough for the SSB4 fixed header, short of SSB5's -> rejected on the magic's size.
	sslm_seq rt = nullptr;
	CHECK_MSG(sslm_seq_restore(f.model, &f.pool, ssb5.data(), 150, &rt) == SSLM_INVALID_ARGUMENT,
	          "an SSB5 blob cut at 150 bytes rejects SSLM_INVALID_ARGUMENT");
	CHECK(rt == nullptr);
}

void RunSlm18xSaturationCensusCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestSlm18x_AdoptPrefixCarriesPerSiteSaturationCensus();
	TestSlm18x_PrefillFillsPerSiteSaturationCensus();
	TestSlm19x_SaveRestoreCarriesPerSiteSaturationCensus();
	TestSlm19x_LegacySsb4RestoreAndTruncatedSsb5();
	checks += GChecks;
	failures += GFailures;
}
