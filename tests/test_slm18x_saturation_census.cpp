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
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <malloc.h>
#endif

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

// ---------------------------------------------------------------------------------------------
// T-3262: pool teardown order (C7 code review S-1, D-SLM8593) and the pin seam's guarantee.
//
// This unit hosts the cells because it is the translation unit of superslm_tests that can call
// the C ABI and already carries the fixture-model helpers; the cells are about pool teardown, not
// the saturation census.
//
// The order under test uses 1.11.0 calls only: map a model, create a pool on it, unmap the model
// while the pool lives (sslm_model_unmap checks sequences, not pools, and returns SSLM_OK), then
// destroy the pool. The defect this guards against is a pool destroy that writes into the model it
// was made on: after sslm_model_unmap freed the model, that write lands in freed memory. The fix
// (D-SLM8593) leaves the model alone in production pool destroy and keeps unmap accepting, so the
// cell asserts unmap == SSLM_OK as the 1.11.0 behaviour and asserts that the destroy writes nothing
// into the freed model.
//
// Two observers, each failing on the defect:
//  - A sanitizer build (the linux-x64-asan job): AddressSanitizer reports the write as a
//    heap-use-after-free inside sslm_kv_pool_destroy and the process aborts
//    (-fno-sanitize-recover=all). The deterministic observer below stands down there, because the
//    sanitizer's allocator keeps the freed block from being reissued; it stands down under
//    ThreadSanitizer too, whose allocator need not reissue the block (the ASan job catches the defect).
//  - Every other build: the freed model block is reissued to the test. The cell reads the model's
//    block size (the allocator's own usable size, taken while the model is live), allocates blocks
//    of that size after the unmap until one is placed at the freed model's address, zero-fills all
//    of them, destroys the pool, and requires every byte still zero. A write into the freed model
//    lands in the reissued block as a nonzero byte (a uint32 decremented from zero is 0xFFFFFFFF
//    in the counter's own bytes, whatever its offset). The blocks belong to the test when the write
//    happens, so the observation itself performs no invalid access. If no block lands on the freed
//    model's address in an attempt, the observer saw nothing in that attempt; the attempt is repeated
//    (up to 64 times: the Windows heap does not hand a freed block back every time) and the cell
//    fails if no attempt lands, because a pass would be a cell that cannot fail. The observer has been
//    run on Linux (glibc) and Windows (MSVC) allocators. On any other platform (macOS) an attempt
//    that never lands is reported as a NOTE and does not fail the cell.
// ---------------------------------------------------------------------------------------------

#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS)
// The pin seam exists only where the library is built with the seam macro (superslm_test_injection
// and the forced-tier libraries); build.bat's source list does not define it, so the seam cell is
// compiled out there rather than left unresolved at link.
extern "C" sslm_status KvPagesTestPinPagePositions(sslm_model model, int64_t page_positions);
#endif

// A sanitizer allocator (AddressSanitizer or ThreadSanitizer, GCC or Clang) quarantines or relocates
// freed blocks, so the deterministic observer below cannot rely on a reissued block there.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define SSLM_T3262_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define SSLM_T3262_SANITIZER 1
#endif
#endif

namespace {

#if defined(SSLM_T3262_SANITIZER)
constexpr bool kT3262Sanitizer = true;
#else
constexpr bool kT3262Sanitizer = false;
#endif

// Platforms where the deterministic observer has been run and must find its block.
#if defined(_WIN32) || defined(__linux__)
constexpr bool kT3262ObserverVerified = true;
#else
constexpr bool kT3262ObserverVerified = false;
#endif

// The allocator's usable size of a live block obtained from plain operator new, or 0 where it
// cannot be read. sslm_model_map allocates its model with plain `new`.
size_t T3262HeapBlockSize(void* p) {
#if defined(_WIN32)
	return _msize(p);
#elif defined(__APPLE__)
	return malloc_size(p);
#elif defined(__linux__)
	return malloc_usable_size(p);
#else
	(void)p;
	return 0;
#endif
}

enum class T3262Verb { kBlock, kPage, kContiguous };

const char* T3262VerbName(T3262Verb v) {
	switch (v) {
		case T3262Verb::kBlock: return "sslm_kv_pool_create";
		case T3262Verb::kPage: return "sslm_kv_page_pool_create";
		case T3262Verb::kContiguous: return "sslm_kv_pool_create_contiguous";
	}
	return "?";
}

constexpr uint32_t kT3262Blocks = 2;

struct T3262Model {
	std::vector<uint8_t> bytes;
	sslm_model model = nullptr;
	int64_t cap = 0;

	bool Open() {
		const std::string path = ResolveFixturePath("t2572_arm_c_non_qknorm_fixture.sslm");
		CHECK_MSG(!path.empty(), "fixture t2572_arm_c_non_qknorm_fixture.sslm not found");
		if (path.empty()) return false;
		CHECK_MSG(superslm_marshal::ReadFile(path.c_str(), bytes), "failed to read %s", path.c_str());
		if (bytes.empty()) return false;
		superslm::SslmModelView view;
		std::string err;
		if (superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) !=
		    superslm::SslmModelStatus::Ok) {
			CHECK_MSG(false, "SslmModel::Load(fixture) failed: %s", err.c_str());
			return false;
		}
		cap = static_cast<int64_t>(view.config.context_cap);
		const sslm_status ms = sslm_model_map(bytes.data(), bytes.size(), &model);
		CHECK_MSG(ms == SSLM_OK, "sslm_model_map(fixture) == %d, want SSLM_OK", static_cast<int>(ms));
		return ms == SSLM_OK;
	}
};

struct T3262Pool {
	void* buf = nullptr;
	sslm_kv_pool pool = nullptr;

	bool Create(sslm_model model, int64_t cap, T3262Verb verb) {
		size_t size = 0;
		uint32_t page_count = 0;
		switch (verb) {
			case T3262Verb::kBlock:
				size = kT3262Blocks * sslm_kv_block_size(model) +
				       sslm_kv_pool_overhead_size(model, kT3262Blocks);
				break;
			case T3262Verb::kPage: {
				const int64_t b = sslm_kv_page_positions(model);
				page_count = static_cast<uint32_t>(kT3262Blocks * (cap / b));
				size = static_cast<size_t>(page_count) * sslm_kv_page_size(model) +
				       sslm_kv_page_pool_overhead_size(model, page_count);
				break;
			}
			case T3262Verb::kContiguous:
				size = sslm_kv_pool_contiguous_size(model, kT3262Blocks);
				break;
		}
		CHECK_MSG(size != 0 && size != SIZE_MAX, "%s: pool size %zu is unusable", T3262VerbName(verb), size);
		if (size == 0 || size == SIZE_MAX) return false;
		buf = ::operator new(size, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		sslm_status st = SSLM_INVALID_ARGUMENT;
		switch (verb) {
			case T3262Verb::kBlock: st = sslm_kv_pool_create(model, buf, size, kT3262Blocks, &pool); break;
			case T3262Verb::kPage: st = sslm_kv_page_pool_create(model, buf, size, page_count, &pool); break;
			case T3262Verb::kContiguous:
				st = sslm_kv_pool_create_contiguous(model, buf, size, kT3262Blocks, &pool);
				break;
		}
		CHECK_MSG(st == SSLM_OK, "%s == %d, want SSLM_OK", T3262VerbName(verb), static_cast<int>(st));
		return st == SSLM_OK;
	}
	void Free() {
		if (buf) ::operator delete(buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
		buf = nullptr;
	}
};

}  // namespace

// One attempt: map -> create pool -> unmap -> destroy pool, on one pool-create verb. Returns whether a
// reissued block landed on the freed model's address (false where the observer stands down). A write
// into the freed model that reached a reissued block fails the cell here.
static bool T3262TeardownAttempt(T3262Verb verb) {
	T3262Model m;
	if (!m.Open()) return false;
	T3262Pool p;
	if (!p.Create(m.model, m.cap, verb)) {
		if (m.model) sslm_model_unmap(m.model);
		p.Free();
		return false;
	}
	const uintptr_t model_addr = reinterpret_cast<uintptr_t>(m.model);
	const size_t model_block = T3262HeapBlockSize(m.model);  // read while the model is live

	// 1.11.0 behaviour kept: unmap does not refuse while a pool lives.
	const sslm_status us = sslm_model_unmap(m.model);
	CHECK_MSG(us == SSLM_OK, "%s: sslm_model_unmap with a live pool == %d, want SSLM_OK (1.11.0)",
	          T3262VerbName(verb), static_cast<int>(us));
	m.model = nullptr;
	if (us != SSLM_OK) {
		// The model is still live; tear down in the valid order so the process stays sound.
		CHECK(sslm_kv_pool_destroy(p.pool) == SSLM_OK);
		p.Free();
		return false;
	}

	// Reissue the freed model block to the test, so a write into it is observable without any
	// invalid access on the test's side. Skipped under AddressSanitizer: its quarantine keeps the
	// block from being reissued, and the sanitizer is the observer there.
	constexpr size_t kMaxModelBlock = 64 * 1024;
	constexpr int kGroomBlocks = 8192;
	std::vector<unsigned char*> groom;
	bool landed = false;
	const bool observe = !kT3262Sanitizer && model_block != 0 && model_block <= kMaxModelBlock;
	if (observe) {
		groom.reserve(kGroomBlocks);
		for (int i = 0; i < kGroomBlocks; ++i) {
			unsigned char* b = static_cast<unsigned char*>(::operator new(model_block));
			std::memset(b, 0, model_block);
			groom.push_back(b);
			if (reinterpret_cast<uintptr_t>(b) == model_addr) landed = true;
		}
	}

	const sslm_status ds = sslm_kv_pool_destroy(p.pool);  // the write, at 1.12.0
	CHECK_MSG(ds == SSLM_OK, "%s: sslm_kv_pool_destroy == %d, want SSLM_OK", T3262VerbName(verb),
	          static_cast<int>(ds));

	if (observe) {
		size_t dirty_bytes = 0;
		size_t dirty_blocks = 0;
		for (const unsigned char* b : groom) {
			size_t d = 0;
			for (size_t i = 0; i < model_block; ++i) d += (b[i] != 0);
			dirty_bytes += d;
			dirty_blocks += (d != 0);
		}
		CHECK_MSG(dirty_bytes == 0,
		          "%s: sslm_kv_pool_destroy after sslm_model_unmap wrote %zu byte(s) in %zu block(s) of "
		          "the freed model's memory (model block %zu bytes)",
		          T3262VerbName(verb), dirty_bytes, dirty_blocks, model_block);
	} else if (!kT3262Sanitizer) {
		CHECK_MSG(false, "%s: model block size %zu is unreadable or above %zu; the write observer is off",
		          T3262VerbName(verb), model_block, kMaxModelBlock);
	}
	for (unsigned char* b : groom) ::operator delete(b);
	p.Free();
	return landed;
}

// Runs attempts on one verb until the reissued block lands on the freed model, so the observation is
// never vacuous. Windows' low-fragmentation heap does not hand a freed block back deterministically
// (measured: some attempts land, some do not), hence the retry; a landing attempt that sees the write
// fails the cell, and an attempt that does not land cannot pass the cell on its own. Under a sanitizer
// one attempt is made and the sanitizer observes.
static void TestT3262_PoolDestroyAfterModelUnmapWritesNothingIntoTheFreedModel(T3262Verb verb) {
	constexpr int kMaxAttempts = 64;
	if (kT3262Sanitizer) {
		T3262TeardownAttempt(verb);
		return;
	}
	for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
		const int failures_before = GFailures;
		if (T3262TeardownAttempt(verb)) return;
		if (GFailures != failures_before) return;  // a refusal or setup failure: more attempts add noise only
	}
	if (kT3262ObserverVerified) {
		CHECK_MSG(false,
		          "%s: no reissued block landed on the freed model's address in %d attempts; the write "
		          "observer cannot see the destroy, so a pass here would be vacuous",
		          T3262VerbName(verb), kMaxAttempts);
	} else {
		std::printf("NOTE %s: write observer unverified on this allocator and found no landing block in %d "
		            "attempts\n",
		            T3262VerbName(verb), kMaxAttempts);
	}
}

static void TestT3262_PoolDestroyAfterModelUnmapAllVerbs() {
	TestT3262_PoolDestroyAfterModelUnmapWritesNothingIntoTheFreedModel(T3262Verb::kBlock);
	TestT3262_PoolDestroyAfterModelUnmapWritesNothingIntoTheFreedModel(T3262Verb::kPage);
	TestT3262_PoolDestroyAfterModelUnmapWritesNothingIntoTheFreedModel(T3262Verb::kContiguous);
}

#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS)
// The seam guarantee that survives the fix: a test build's KvPagesTestPinPagePositions refuses a pin
// while any pool made on the model lives, leaves B unchanged on the refusal, keeps refusing until the
// LAST pool is destroyed, and then admits the pin. Run once per pool-create verb.
static void TestT3262_PinRefusedWhileAPoolLivesOnEveryCreateVerb(T3262Verb verb) {
	T3262Model m;
	if (!m.Open()) return;
	const char* name = T3262VerbName(verb);
	const int64_t b0 = sslm_kv_page_positions(m.model);
	const int64_t pin = (b0 == m.cap) ? 1 : m.cap;
	CHECK_MSG(b0 >= 1 && pin != b0, "%s: default B %lld, pin target %lld", name, static_cast<long long>(b0),
	          static_cast<long long>(pin));
	T3262Pool first, second;
	if (first.Create(m.model, m.cap, verb) && second.Create(m.model, m.cap, verb)) {
		// Two pools live: refused, B unchanged.
		CHECK_MSG(KvPagesTestPinPagePositions(m.model, pin) == SSLM_INVALID_ARGUMENT,
		          "%s: a pin with two pools live must be refused", name);
		CHECK(sslm_kv_page_positions(m.model) == b0);
		// One destroyed, one live: still refused.
		CHECK(sslm_kv_pool_destroy(first.pool) == SSLM_OK);
		first.pool = nullptr;
		CHECK_MSG(KvPagesTestPinPagePositions(m.model, pin) == SSLM_INVALID_ARGUMENT,
		          "%s: a pin with one pool still live must be refused", name);
		CHECK(sslm_kv_page_positions(m.model) == b0);
		// None live: admitted, and B is the pin.
		CHECK(sslm_kv_pool_destroy(second.pool) == SSLM_OK);
		second.pool = nullptr;
		CHECK_MSG(KvPagesTestPinPagePositions(m.model, pin) == SSLM_OK,
		          "%s: a pin with no pool live must be admitted", name);
		CHECK_MSG(sslm_kv_page_positions(m.model) == pin, "%s: B after the pin == %lld, want %lld", name,
		          static_cast<long long>(sslm_kv_page_positions(m.model)), static_cast<long long>(pin));
	}
	if (first.pool) CHECK(sslm_kv_pool_destroy(first.pool) == SSLM_OK);
	if (second.pool) CHECK(sslm_kv_pool_destroy(second.pool) == SSLM_OK);
	first.Free();
	second.Free();
	if (m.model) CHECK(sslm_model_unmap(m.model) == SSLM_OK);
}

static void TestT3262_PinRefusedWhileAPoolLivesAllVerbs() {
	TestT3262_PinRefusedWhileAPoolLivesOnEveryCreateVerb(T3262Verb::kBlock);
	TestT3262_PinRefusedWhileAPoolLivesOnEveryCreateVerb(T3262Verb::kPage);
	TestT3262_PinRefusedWhileAPoolLivesOnEveryCreateVerb(T3262Verb::kContiguous);
}

#endif  // SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS

// Entry point for the T-3262 cells (called from tests/test_main.cpp beside the census cells).
void RunPoolTeardownOrderCells(int& checks, int& failures) {
	GChecks = 0;
	GFailures = 0;
	TestT3262_PoolDestroyAfterModelUnmapAllVerbs();
#if defined(SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS)
	TestT3262_PinRefusedWhileAPoolLivesAllVerbs();
#endif
	checks += GChecks;
	failures += GFailures;
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
