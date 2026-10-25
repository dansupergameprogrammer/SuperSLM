// Schema fast-decode plan (rev 1.2.x), cells 5.2 and 8.6 (vi), fix round 3 (code review round 1,
// findings 1-3): a REAL allocation failure inside sslm_seq_jump_forward, at every allocation the call
// makes, on a sequence that already holds a damped anti-LM state at another order.
//
// The suite's slots (cell 5.2's pass slot, 8.6 (vi)'s create slot) throw before EnsureAntiLm touches the
// state, so they cannot see a re-key that a later failure leaves behind (finding 1), nor a create that
// fails after the old state is destroyed (finding 2, a dangling pointer: the next save reads freed
// memory, the next re-key or release frees it again). This executable replaces the global operator new
// (the decode_threading_alloc_main.cpp pattern, so it is its own binary) and fails allocation number k,
// counted from the start of one verb call, for every k the call reaches.
//
// Each trial: a POTION-bound sequence, the prompt, the verb at damped order 3 with budget 1 (an order-3
// state whose history is the start chain's first token), then the verb at damped order 2 with budget 1
// and allocation k failing. Plan §5.8 and cell 5.2: the call returns SSLM_ALLOCATION_FAILED with nothing
// admitted, and the anti-LM order, its history and the whole save blob are unchanged; a retry then gives
// the blob of a sequence that never failed; a further save and the release are clean. A trial whose
// verb call returns SSLM_OK (the failed allocation was absorbed) passes if its blob is the never-failed
// one. A trial whose failure comes after tokens were admitted (the history update after the pass; code
// review round 1, observation 9) is outside these cells: its status and blob are not asserted.
//
// Fix round 4 (code review re-check, finding 9): every trial then makes the sequence's next use, two damped
// decodes, which must not crash and must equal the decodes of a sequence restored from the failed
// sequence's own save blob (and the never-failed decodes where the blob is the never-failed one). A second
// sweep fails each allocation of one damped decode step (sslm_decode_step_v2) on the same order-3 state, at
// order 2 (a re-key) and at order 3, and holds its next use to the same checks. On POSIX every trial runs in a forked child, so a crash or a sanitizer abort in one trial fails
// that trial and the sweep goes on; elsewhere the trials run in-process.
//
// Allocation indices are the global operator new's (throwing and nothrow, scalar and array); the
// aligned forms are not replaced and not counted. Reads F-SFD from SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR;
// a missing fixture fails. Exit status 0 iff every trial passed.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "superslm/sslm_abi.h"
#include "schema_fast_decode/sfd_api.h"

// The paged-KV test seams (the library this binary links defines SUPERSLM_ENABLE_KV_PAGES_TEST_SEAMS).
extern "C" sslm_status KvPagesTestPinPagePositions(sslm_model model, int64_t page_positions);
extern "C" sslm_status sslm_pkv_test_only_seq_table_entry(sslm_seq seq, uint32_t i, uint32_t* out_page,
                                                         uint32_t* out_mapped);

namespace {
std::atomic<bool> g_armed{false};
std::atomic<long> g_index{0};
long g_fail_at = -1;
std::atomic<bool> g_fired{false};
// The sizes of the first kSizeCap counted allocations (slice TF aligns two decodes' allocation sequences).
constexpr long kSizeCap = 4096;
std::size_t g_sizes[kSizeCap];

void* CountedNew(std::size_t size) {
	if (g_armed.load(std::memory_order_relaxed)) {
		const long i = g_index.fetch_add(1, std::memory_order_relaxed);
		if (i >= 0 && i < kSizeCap) g_sizes[i] = size;
		if (i == g_fail_at) {
			g_armed.store(false, std::memory_order_relaxed);
			g_fired.store(true, std::memory_order_relaxed);
			throw std::bad_alloc{};
		}
	}
	if (void* p = std::malloc(size ? size : 1)) return p;
	throw std::bad_alloc{};
}
}  // namespace

void* operator new(std::size_t size) { return CountedNew(size); }
void* operator new[](std::size_t size) { return CountedNew(size); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
	try {
		return CountedNew(size);
	} catch (...) {
		return nullptr;
	}
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
	try {
		return CountedNew(size);
	} catch (...) {
		return nullptr;
	}
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

int g_checks = 0, g_failures = 0;
#define CHECK_MSG(cond, ...) \
	do { \
		++g_checks; \
		if (!(cond)) { \
			++g_failures; \
			std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
			std::printf(__VA_ARGS__); \
			std::printf("\n"); \
		} \
	} while (0)

constexpr int32_t kHidden = 64;  // F-SFD's hidden size
const char* const kPotion = "potion_shop_order";
const char* const kPrompt = "Customer: I would like to buy a health potion, please.\nOrder:";

uint32_t Le32(const uint8_t* p) {
	return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t Le64(const uint8_t* p) { return uint64_t(Le32(p)) | (uint64_t(Le32(p + 4)) << 32); }

struct Aligned {
	uint8_t* p = nullptr;
	explicit Aligned(size_t n) {
		p = static_cast<uint8_t*>(::operator new(n ? n : 1, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
	}
	~Aligned() { ::operator delete(p, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)); }
	Aligned(const Aligned&) = delete;
	Aligned& operator=(const Aligned&) = delete;
};

// The save blob's anti-LM order and history (the suite's ParseBlob offsets).
struct AntiLm {
	bool ok = false;
	int32_t order = 0;
	std::vector<int32_t> hist;
};
AntiLm ReadAntiLm(const std::vector<uint8_t>& b) {
	AntiLm a;
	if (b.size() < 156) return a;
	const bool v6 = std::memcmp(b.data(), "SSB6", 4) == 0;
	if (!v6 && std::memcmp(b.data(), "SSB5", 4) != 0) return a;
	a.order = static_cast<int32_t>(Le32(b.data() + 108));
	const int64_t n = static_cast<int64_t>(Le64(b.data() + 112));
	const size_t off = (v6 ? 172 : 156) + size_t{kHidden};
	if (n < 0 || n > 4096 || off + 4 * static_cast<size_t>(n) > b.size()) return a;
	for (int64_t i = 0; i < n; ++i) a.hist.push_back(static_cast<int32_t>(Le32(b.data() + off + 4 * size_t(i))));
	a.ok = true;
	return a;
}

std::string Str(const std::vector<int32_t>& v) {
	std::string s = "{";
	for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
	return s + "}";
}

struct Engine {
	std::vector<uint8_t> file;
	Aligned* bytes = nullptr;
	sslm_model m = nullptr;
	Aligned* pool_buf = nullptr;
	sslm_kv_pool pool = nullptr;
	Aligned* ws_buf = nullptr;
	sslm_workspace ws = nullptr;
	std::vector<int32_t> prompt;
	bool ok = false;

	// `page` > 0 pins the paged-KV page size (KvPagesTestPinPagePositions) before the pool is made.
	explicit Engine(int64_t page = 0) {
		const char* dir = std::getenv("SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR");
		CHECK_MSG(dir != nullptr, "SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR is not set");
		if (!dir) return;
		std::ifstream in(std::string(dir) + "/fsfd.sslm", std::ios::binary);
		file.assign(std::istreambuf_iterator<char>(in), {});
		CHECK_MSG(!file.empty(), "%s/fsfd.sslm was not read", dir);
		if (file.empty()) return;
		bytes = new Aligned(file.size());
		std::memcpy(bytes->p, file.data(), file.size());
		const sslm_status ms = sslm_model_map(bytes->p, file.size(), &m);
		CHECK_MSG(ms == SSLM_OK, "sslm_model_map == %d", static_cast<int>(ms));
		if (ms != SSLM_OK) return;
		if (page > 0) {
			const sslm_status ps = KvPagesTestPinPagePositions(m, page);
			CHECK_MSG(ps == SSLM_OK, "KvPagesTestPinPagePositions(%lld) == %d", static_cast<long long>(page),
			          static_cast<int>(ps));
			if (ps != SSLM_OK) return;
		}
		const uint32_t blocks = 8;
		const size_t size = blocks * sslm_kv_block_size(m) + sslm_kv_pool_overhead_size(m, blocks);
		pool_buf = new Aligned(size);
		CHECK_MSG(sslm_kv_pool_create(m, pool_buf->p, size, blocks, &pool) == SSLM_OK, "sslm_kv_pool_create");
		const sslm_config cfg{3, 64, 2, 0u};
		const size_t wsz = sslm_workspace_size(m, &cfg);
		ws_buf = new Aligned(wsz);
		CHECK_MSG(sslm_workspace_create(m, &cfg, ws_buf->p, wsz, &ws) == SSLM_OK, "sslm_workspace_create");
		prompt.assign(512, 0);
		int32_t n = 512;
		const sslm_status ts = sslm_tokenize(m, kPrompt, prompt.data(), &n);
		CHECK_MSG(ts == SSLM_OK && n > 0, "sslm_tokenize == %d", static_cast<int>(ts));
		prompt.resize(ts == SSLM_OK && n > 0 ? size_t(n) : 0);
		ok = pool && ws && !prompt.empty();
	}
	~Engine() {
		if (ws) sslm_workspace_destroy(ws);
		if (pool) sslm_kv_pool_destroy(pool);
		if (m) sslm_model_unmap(m);
		delete ws_buf;
		delete pool_buf;
		delete bytes;
	}
	Engine(const Engine&) = delete;
	Engine& operator=(const Engine&) = delete;

	sslm_decode_params Damped(int32_t order) const {
		sslm_decode_params p{};
		sslm_decode_params_init(m, SSLM_DECODE_MODE_DAMPED_GREEDY, 2, &p);
		p.anti_lm_max_order = order;
		return p;
	}
	std::vector<uint8_t> Save(sslm_seq s) const {
		std::vector<uint8_t> b(sslm_seq_state_size(m) + 4096);
		size_t n = b.size();
		const sslm_status st = sslm_seq_save(s, b.data(), &n);
		CHECK_MSG(st == SSLM_OK, "sslm_seq_save == %d", static_cast<int>(st));
		b.resize(st == SSLM_OK ? n : 0);
		return b;
	}
	sslm_status Verb(sslm_seq s, int32_t order, int32_t* count) const {
		const sslm_decode_params p = Damped(order);
		int32_t buf[8] = {-7, -7, -7, -7, -7, -7, -7, -7};
		*count = -1;
		return superslm::test::sfd::JumpCanonical(m, s, 1, &p, ws, buf, 8, count);
	}
	// A POTION-bound sequence, the prompt, then the verb at order 3, budget 1. Null on a setup failure.
	sslm_seq Setup() {
		sslm_seq s = nullptr;
		if (sslm_seq_create(m, &pool, &s) != SSLM_OK || !s) {
			CHECK_MSG(false, "sslm_seq_create");
			return nullptr;
		}
		sslm_schema h = nullptr;
		int32_t consumed = 0, count = -1;
		const bool ok_setup =
		    sslm_schema_lookup(m, kPotion, &h) == SSLM_OK && sslm_seq_set_schema(s, h) == SSLM_OK &&
		    sslm_prefill(m, s, prompt.data(), int32_t(prompt.size()), int32_t(prompt.size()), SSLM_SPAN_PROMPT, ws,
		                 &consumed) == SSLM_OK &&
		    Verb(s, 3, &count) == SSLM_OK && count == 1;
		CHECK_MSG(ok_setup, "setup: bind, prompt and verb@3 (count %d)", count);
		if (!ok_setup) {
			sslm_seq_release(s);
			return nullptr;
		}
		return s;
	}
};

// Fix round 4 (code review re-check, finding 9): the sequence's next use after a failure. Two damped decodes
// at `order` must not crash, and they must give what a sequence restored from the failed sequence's own save
// blob gives: the blob is the sequence's whole declared state (sslm_seq_restore rebuilds the anti-LM tables
// from the saved history), so a half-applied update that the blob does not carry -- a context entry with
// total 0, say -- is a divergence here and a division by zero in the decode. Where the failed sequence's
// blob is the never-failed sequence's, the decodes must also equal the never-failed decodes.
struct Decoded {
	std::vector<int> st;
	std::vector<int32_t> tok;
	bool operator==(const Decoded& o) const { return st == o.st && tok == o.tok; }
};
std::string Str(const Decoded& d) {
	std::string s = "{";
	for (size_t i = 0; i < d.st.size(); ++i)
		s += (i ? "," : "") + std::to_string(d.st[i]) + ":" + std::to_string(d.tok[i]);
	return s + "}";
}
constexpr int kNextDecodes = 2;
Decoded DecodeN(Engine& e, sslm_seq s, int32_t order, int n = kNextDecodes) {
	Decoded d;
	const sslm_decode_params p = e.Damped(order);
	for (int i = 0; i < n; ++i) {
		int32_t out = -100;
		d.st.push_back(static_cast<int>(sslm_decode_step_v2(e.m, &s, 1, &p, e.ws, &out)));
		d.tok.push_back(out);
		std::fflush(stdout);
	}
	return d;
}
// Returns the live sequence's decodes; checks them against the restored copy and, when `ref_blob` matches,
// against `ref`.
void NextUse(Engine& e, sslm_seq s, int32_t order, const char* tag, long k, const std::vector<uint8_t>& ref_blob,
             const Decoded& ref) {
	const std::vector<uint8_t> snap = e.Save(s);
	std::printf("[%s k %ld] next use: %d damped decodes at order %d\n", tag, k, kNextDecodes, order);
	std::fflush(stdout);
	const Decoded live = DecodeN(e, s, order);
	sslm_seq r = nullptr;
	const sslm_status rs = snap.empty() ? SSLM_INVALID_ARGUMENT : sslm_seq_restore(e.m, &e.pool, snap.data(), snap.size(), &r);
	CHECK_MSG(rs == SSLM_OK && r, "[%s k %ld] restoring the failed sequence's blob == %d", tag, k, static_cast<int>(rs));
	if (rs == SSLM_OK && r) {
		const Decoded restored = DecodeN(e, r, order);
		CHECK_MSG(live == restored, "[%s k %ld] after the failure the sequence decodes %s, its own restored blob %s", tag,
		          k, Str(live).c_str(), Str(restored).c_str());
		sslm_seq_release(r);
	}
	if (snap == ref_blob) {
		CHECK_MSG(live == ref, "[%s k %ld] the blob is the never-failed one, but the decodes are %s, never-failed %s",
		          tag, k, Str(live).c_str(), Str(ref).c_str());
	}
}

void Arm(long k) {
	g_fail_at = k;
	g_fired = false;
	g_index = 0;
	g_armed = true;
}

// One verb trial at allocation index k. Returns true iff every check in it passed.
bool Trial(Engine& e, long k, const std::vector<uint8_t>& ref_after, const Decoded& ref_next) {
	const int failures_before = g_failures;
	sslm_seq s = e.Setup();
	if (!s) return false;
	const std::vector<uint8_t> before = e.Save(s);
	const AntiLm a0 = ReadAntiLm(before);
	CHECK_MSG(a0.ok && a0.order == 3 && a0.hist.size() == 1, "[k %ld] setup: order %d history %s", k, a0.order,
	          Str(a0.hist).c_str());
	int32_t count = -1;
	Arm(k);
	const sslm_status st = e.Verb(s, 2, &count);
	g_armed = false;
	const long seen = g_index.load();
	const bool fired = g_fired.load();
	std::printf("[k %ld] verb@2 on an order-3 state, allocation #%ld failing (fired %d, %ld allocations): status %d "
	            "count %d\n", k, k, fired, seen, static_cast<int>(st), count);
	std::fflush(stdout);
	if (fired && st != SSLM_OK && count > 0) {
		// The failure came after the admission (the anti-LM history update that follows the pass): tokens
		// were admitted, so this is not the failed call of cells 5.2 and 8.6 (vi), and nothing is asserted of
		// the status or the blob. Fix round 4 (re-check finding 9): the next use below is held to no crash and
		// to determinism.
		std::printf("NOTE [k %ld] status %d after %d tokens were admitted: outside cells 5.2 and 8.6 (vi); the next "
		            "use is checked\n", k, static_cast<int>(st), count);
	} else if (fired && st != SSLM_OK) {
		const std::vector<uint8_t> after = e.Save(s);
		const AntiLm a1 = ReadAntiLm(after);
		CHECK_MSG(st == SSLM_ALLOCATION_FAILED && count == 0, "[k %ld] status %d count %d, want SSLM_ALLOCATION_FAILED "
		          "and nothing admitted", k, static_cast<int>(st), count);
		CHECK_MSG(a1.order == a0.order, "[k %ld] the failed call changed the anti-LM order %d -> %d", k, a0.order,
		          a1.order);
		CHECK_MSG(a1.hist == a0.hist, "[k %ld] the failed call changed the anti-LM history %s -> %s", k,
		          Str(a0.hist).c_str(), Str(a1.hist).c_str());
		CHECK_MSG(after == before, "[k %ld] the failed call changed the save blob", k);
		std::fflush(stdout);
		const sslm_status rs = e.Verb(s, 2, &count);
		CHECK_MSG(rs == SSLM_OK && count == 1 && e.Save(s) == ref_after, "[k %ld] the retry: status %d count %d, blob "
		          "%s the never-failed sequence's", k, static_cast<int>(rs), count,
		          e.Save(s) == ref_after ? "equals" : "differs from");
	} else {
		CHECK_MSG(st == SSLM_OK && count == 1 && e.Save(s) == ref_after, "[k %ld] status %d count %d with the failure "
		          "absorbed; want SSLM_OK and the never-failed blob", k, static_cast<int>(st), count);
	}
	std::fflush(stdout);
	NextUse(e, s, 2, "verb", k, ref_after, ref_next);
	(void)e.Save(s);
	sslm_seq_release(s);
	return g_failures == failures_before;
}

// Fix round 4: one trial of the damped decode step's own allocations. The setup's order-3 state, then one
// damped decode at `order` (2 re-keys, 3 does not) with allocation k failing, then the next use. The failed
// decode's own status and resting state are main's and are not asserted (code review re-check, non-blocking:
// main's decode clears ready_for_logits before the finish).
bool DecodeTrial(Engine& e, int32_t order, long k, const std::vector<uint8_t>& ref_after, const Decoded& ref_next) {
	const int failures_before = g_failures;
	sslm_seq s = e.Setup();
	if (!s) return false;
	const sslm_decode_params p = e.Damped(order);
	int32_t out = -100;
	Arm(k);
	const sslm_status st = sslm_decode_step_v2(e.m, &s, 1, &p, e.ws, &out);
	g_armed = false;
	std::printf("[decode@%d k %ld] damped decode on an order-3 state, allocation #%ld failing (fired %d, %ld "
	            "allocations): status %d token %d\n", order, k, k, g_fired.load() ? 1 : 0, g_index.load(),
	            static_cast<int>(st), out);
	std::fflush(stdout);
	const std::string tag = "decode@" + std::to_string(order);
	NextUse(e, s, order, tag.c_str(), k, ref_after, ref_next);
	(void)e.Save(s);
	sslm_seq_release(s);
	return g_failures == failures_before;
}

template <typename F>
void RunTrial(const char* tag, long k, std::vector<std::string>* failed, F&& body) {
	std::fflush(stdout);
#if !defined(_WIN32)
	const pid_t pid = fork();
	if (pid == 0) {
		const bool pass = body();
		std::fflush(stdout);
		std::_Exit(pass ? 0 : 1);
	}
	int status = 0;
	const bool waited = pid > 0 && waitpid(pid, &status, 0) == pid;
	const bool pass = waited && WIFEXITED(status) && WEXITSTATUS(status) == 0;
	++g_checks;
	if (!pass) {
		++g_failures;
		if (waited && WIFSIGNALED(status))
			std::printf("FAIL [%s k %ld] the trial died with signal %d\n", tag, k, WTERMSIG(status));
		else if (waited && WIFEXITED(status) && WEXITSTATUS(status) != 1)
			std::printf("FAIL [%s k %ld] the trial exited %d (a sanitizer report, above)\n", tag, k, WEXITSTATUS(status));
	}
#else
	const bool pass = body();
#endif
	if (!pass) failed->push_back(std::string(tag) + ":" + std::to_string(k));
}

// ===================================================================================================
// Plan rev 1.3.3, slice TF (§4.5, C16, §11.5): the decode step's finish as a transaction.
//
// TF.1  The allocation sweep: allocation k of the target decode fails, for every k it makes. Status
//       SSLM_ALLOCATION_FAILED; on the ready branch the blob is byte-equal to the blob before the call
//       (the finish's entry); on the layer branch, whose entry is inside the call, a failure in the finish
//       rests ready at the committed row, byte-equal to the ready twin's blob at the same output position
//       (arm (c'), below; rev 1.3.3's "(d) against (a)" with exact mode's j = 0 commit shape), and a
//       failure in the layer pass rests as before the call. With the failure gone, the retry and the next
//       two decodes equal the never-failed run: statuses, tokens, blobs (the history is in the blob).
// TF.3  AntiLmUpdate fails (and every other damped finish allocation): the walk, current_token, and the
//       anti-LM order and history equal their values at the finish's entry, and the retry emits the
//       never-failed token. Read from the TF.1 damped sweeps' rests.
// TF.4  (a) A create or re-key fails: the target decode runs at order 3 on a sequence whose state is at
//       order 2 (a re-key; a create in arm (b), which has no state yet). TF.1's checks, and the old state
//       (order and history) survives intact. Arm (b) of TF.4, the dead end, is in the main suite.
// TF.5  S-CHOICE is armed with a legal token other than the model's at the target decode, and the finish
//       fails: the pin is still armed, and the retry applies it at the same position (the retry equals the
//       never-failed pinned run).
// TF.6  A decode batch of three ready sequences, allocation k of the batched call failing: the sequences
//       before the failing one keep their advance, the failing one rests at its entry, the ones after it
//       are untouched, and the failing one's retry alone equals its never-failed decode.
// TF.7  Vitality: with the guard stubbed (S-GUARDOFF), TF.1 (b) turns red: some trial's rest differs from
//       the blob before the call. TF.7 holds iff that is so and TF.1 (b) itself holds with the guard on.
//
// The arms, all on POTION from the suite's prompt, at greedy and at damped order 2, with the engine's
// workspace and without one:
//   (a)  the layer branch: N decodes, then the target decode embeds t_{N-1}, runs the layers and finishes;
//   (b)  the ready branch, off: the first decode after the prompt;
//   (c)  canonical J1 after a span: the §5.6 drain, a choice, the drain again, then the target decode;
//   (c') canonical at budget 1 after N decodes: the verb admits the pending t_{N-1} alone (exact mode's
//        j = 0 commit shape), and the target decode runs the ready branch at (a)'s output position.
// Exact mode's own arms, (d) after a j = 0 and after a j >= 2 commit, are J1-X cells (TF.1d, TF.2d) and
// land with that slice. N is the first position >= 3 whose state after t_{N-1} is forced with a canonical
// first token other than t_N, so (c') exists there and that token is TF.5's pin.
// ===================================================================================================

enum class TfArm { kLayer, kReady, kCanonSpan, kCanonPending, kExact0, kExact2 };
const char* TfArmName(TfArm a) {
	switch (a) {
	case TfArm::kLayer: return "(a) layer branch";
	case TfArm::kReady: return "(b) first token after the prompt";
	case TfArm::kCanonSpan: return "(c) canonical after a span";
	case TfArm::kCanonPending: return "(c') canonical pending-only at N";
	case TfArm::kExact0: return "(d0) exact, j = 0 at N";
	case TfArm::kExact2: return "(d2) exact, j >= 2 at N";
	}
	return "?";
}
constexpr TfArm kTfArms[] = {TfArm::kLayer, TfArm::kReady, TfArm::kCanonSpan, TfArm::kCanonPending};

struct TfCfg {
	bool damped = false;
	bool ws = false;
};
std::string TfCfgName(const TfCfg& c) {
	return std::string(c.damped ? "damped@2" : "greedy") + (c.ws ? ", workspace" : ", no workspace");
}

struct TfBlob {
	bool ok = false;
	uint32_t walk = 0;
	int64_t ctx = 0;
	int32_t layer = 0, current = 0;
	uint32_t ready = 0;
	AntiLm al;
};
TfBlob ReadTf(const std::vector<uint8_t>& b) {
	TfBlob o;
	if (b.size() < 156) return o;
	o.walk = Le32(b.data() + 48);
	o.ctx = static_cast<int64_t>(Le64(b.data() + 60));
	o.layer = static_cast<int32_t>(Le32(b.data() + 68));
	o.current = static_cast<int32_t>(Le32(b.data() + 72));
	o.ready = Le32(b.data() + 120);
	o.al = ReadAntiLm(b);
	o.ok = true;
	return o;
}
std::string TfFields(const std::vector<uint8_t>& b) {
	const TfBlob t = ReadTf(b);
	if (!t.ok) return "<no blob>";
	char buf[200];
	std::snprintf(buf, sizeof buf, "walk %u ctx %lld layer %d current %d ready %u order %d history %zu", t.walk,
	              static_cast<long long>(t.ctx), t.layer, t.current, t.ready, t.al.order, t.al.hist.size());
	return buf;
}

// Off's state at one cfg: its never-failed stream, the target position N, and TF.5's pins.
struct TfCtx {
	TfCfg cfg;
	bool ok = false;
	std::vector<int32_t> stream;
	int n = -1;
	int32_t pin = -1;    // legal at the state after t_{N-1}, != t_N
	int32_t pin_b = -1;  // legal at the start state, != t_0 (-1: none)
	int32_t pin_c = -1;  // legal at arm (c)'s target state, != the model's token there (-1: none)
};

sslm_decode_params TfParams(const Engine& e, const TfCfg& c, int32_t order) {
	sslm_decode_params p{};
	sslm_decode_params_init(e.m, c.damped ? SSLM_DECODE_MODE_DAMPED_GREEDY : SSLM_DECODE_MODE_GREEDY, 2, &p);
	if (c.damped) p.anti_lm_max_order = order;
	return p;
}
sslm_status TfDecode(const Engine& e, const TfCfg& c, sslm_seq s, int32_t order, int32_t* out) {
	const sslm_decode_params p = TfParams(e, c, order);
	*out = -100;
	return sslm_decode_step_v2(e.m, &s, 1, &p, c.ws ? e.ws : nullptr, out);
}
// The canonical verb with the decode's own params (NULL under greedy, §5.8), and the cfg's workspace.
sslm_status TfVerb(const Engine& e, const TfCfg& c, sslm_seq s, int32_t budget, int32_t order, int32_t* count) {
	const sslm_decode_params p = TfParams(e, c, order);
	std::vector<int32_t> buf(size_t(budget), -7);
	*count = -1;
	return superslm::test::sfd::JumpCanonical(e.m, s, budget, c.damped ? &p : nullptr, c.ws ? e.ws : nullptr, buf.data(), budget,
	                             count);
}
// Exact mode (J1-X, §5.10) with the decode's own params and the cfg's workspace; `draft` is the host draft.
// `out`, when given, is the caller's output buffer, sized before the call: a call under an armed allocation
// window passes one, so the window counts the verb's allocations and not this harness's.
sslm_status TfExact(const Engine& e, const TfCfg& c, sslm_seq s, int32_t budget, const std::vector<int32_t>& draft,
                    int32_t order, int32_t* count, std::vector<int32_t>* out = nullptr) {
	const sslm_decode_params p = TfParams(e, c, order);
	sslm_jump_options o{};
	o.struct_size = static_cast<uint32_t>(sizeof(sslm_jump_options));
	o.mode = SSLM_JUMP_EXACT;
	static const int32_t kNone = -7;
	o.draft = draft.empty() ? &kNone : draft.data();
	o.draft_count = static_cast<int32_t>(draft.size());
	const size_t cap = size_t(budget > 0 ? budget : 1);
	std::vector<int32_t> local;
	if (!out) local.assign(cap, -7);
	std::vector<int32_t>& buf = out ? *out : local;
	if (buf.size() < cap) {
		*count = -1;
		return SSLM_INVALID_ARGUMENT;
	}
	*count = -1;
	return superslm::test::sfd::JumpWithOptions(e.m, s, budget, c.damped ? &p : nullptr, &o, c.ws ? e.ws : nullptr,
	                                            buf.data(), int32_t(cap), count);
}
int32_t TfChainHead(const Engine& e, sslm_seq s, int32_t* count) {
	int32_t id = -1, acc = 0;
	*count = 0;
	if (sslm_seq_forced_tokens(e.m, s, &id, 1, count, &acc) != SSLM_OK) *count = -1;
	return *count > 0 ? id : -1;
}

sslm_seq TfNewBound(const Engine& e, const TfCfg& c, sslm_kv_pool* pool = nullptr) {
	sslm_seq s = nullptr;
	if (sslm_seq_create(e.m, pool ? pool : const_cast<sslm_kv_pool*>(&e.pool), &s) != SSLM_OK || !s) return nullptr;
	sslm_schema h = nullptr;
	int32_t consumed = 0;
	if (sslm_schema_lookup(e.m, kPotion, &h) != SSLM_OK || sslm_seq_set_schema(s, h) != SSLM_OK ||
	    sslm_prefill(e.m, s, e.prompt.data(), int32_t(e.prompt.size()), int32_t(e.prompt.size()), SSLM_SPAN_PROMPT,
	                 c.ws ? e.ws : nullptr, &consumed) != SSLM_OK) {
		sslm_seq_release(s);
		return nullptr;
	}
	return s;
}
bool TfDrain(const Engine& e, const TfCfg& c, sslm_seq s) {
	for (int guard = 0; guard < 16; ++guard) {
		int32_t n = 0;
		TfChainHead(e, s, &n);
		if (n == 0) return true;
		int32_t count = -1;
		if (n < 0 || TfVerb(e, c, s, 64, 2, &count) != SSLM_OK) return false;
	}
	return false;
}
// The arm's sequence, resting at the target decode's entry. Null on a setup failure.
sslm_seq TfBuild(const Engine& e, const TfCtx& x, TfArm arm, sslm_kv_pool* pool = nullptr) {
	sslm_seq s = TfNewBound(e, x.cfg, pool);
	if (!s) return nullptr;
	bool ok = true;
	int32_t t = -100, count = -1;
	switch (arm) {
	case TfArm::kReady: break;
	case TfArm::kLayer:
	case TfArm::kCanonPending:
		for (int i = 0; i < x.n && ok; ++i) ok = TfDecode(e, x.cfg, s, 2, &t) == SSLM_OK && t >= 0;
		if (ok && arm == TfArm::kCanonPending) ok = TfVerb(e, x.cfg, s, 1, 2, &count) == SSLM_OK && count == 0;
		break;
	case TfArm::kCanonSpan:
		ok = TfDrain(e, x.cfg, s) && TfDecode(e, x.cfg, s, 2, &t) == SSLM_OK && t >= 0 && TfDrain(e, x.cfg, s);
		break;
	case TfArm::kExact0:
	case TfArm::kExact2: {
		// N decodes (t_{N-1} pending), then an exact call that commits p alone (j = 0: the draft is x.pin, the
		// canonical head at N, legal there and not the model's t_N) or j >= 2 (off's own next tokens).
		for (int i = 0; i < x.n && ok; ++i) ok = TfDecode(e, x.cfg, s, 2, &t) == SSLM_OK && t >= 0;
		std::vector<int32_t> d;
		if (arm == TfArm::kExact0) d = {x.pin};
		else
			for (size_t q = size_t(x.n); q < x.stream.size() && d.size() < 7; ++q) d.push_back(x.stream[q]);
		ok = ok && TfExact(e, x.cfg, s, int32_t(d.size()) + 1, d, 2, &count) == SSLM_OK &&
		     (arm == TfArm::kExact0 ? count == 0 : count >= 2);
		break;
	}
	}
	if (!ok) {
		sslm_seq_release(s);
		return nullptr;
	}
	return s;
}

struct TfRun {
	std::vector<int> st;
	std::vector<int32_t> tok;
	std::vector<std::vector<uint8_t>> blob;
	bool operator==(const TfRun& o) const { return st == o.st && tok == o.tok && blob == o.blob; }
};
TfRun TfContinue(Engine& e, const TfCfg& c, sslm_seq s, int32_t order, int n) {
	TfRun r;
	for (int i = 0; i < n; ++i) {
		int32_t t = -100;
		r.st.push_back(static_cast<int>(TfDecode(e, c, s, order, &t)));
		r.tok.push_back(t);
		r.blob.push_back(e.Save(s));
	}
	return r;
}
std::string Str(const TfRun& r) {
	std::string s = "{";
	for (size_t i = 0; i < r.st.size(); ++i) s += (i ? "," : "") + std::to_string(r.st[i]) + ":" + std::to_string(r.tok[i]);
	return s + "}";
}

// A token S-CHOICE can pin at an arm's target decode other than the model's own: legal at that state, found
// by the engine itself (the arm's state is saved once; each candidate is pinned on a restored copy and is legal
// iff the decode emits it rather than a dead end). -1 when the state has one legal token only.
int32_t TfLegalOther(Engine& e, const TfCtx& x, TfArm arm) {
	sslm_seq s = TfBuild(e, x, arm);
	if (!s) return -1;
	const std::vector<uint8_t> b = e.Save(s);
	int32_t model = -100;
	const bool ok = TfDecode(e, x.cfg, s, 2, &model) == SSLM_OK && model >= 0;
	sslm_seq_release(s);
	if (!ok) return -1;
	for (int32_t t = 0; t < 8192; ++t) {
		if (t == model) continue;
		sslm_seq q = nullptr;
		if (sslm_seq_restore(e.m, &e.pool, b.data(), b.size(), &q) != SSLM_OK || !q) return -1;
		superslm::test::sfd::ArmChoice(t);
		int32_t out = -100;
		const sslm_status st = TfDecode(e, x.cfg, q, 2, &out);
		superslm::test::sfd::DisarmChoice();
		sslm_seq_release(q);
		if (st == SSLM_OK && out == t) return t;
	}
	return -1;
}

TfCtx TfMakeCtx(Engine& e, const TfCfg& c) {
	TfCtx x;
	x.cfg = c;
	sslm_seq s = TfNewBound(e, c);
	if (!s) return x;
	for (int i = 0; i < 64; ++i) {
		int32_t t = -100;
		if (TfDecode(e, c, s, 2, &t) != SSLM_OK || t < 0) break;
		x.stream.push_back(t);
	}
	sslm_seq_release(s);
	for (int n = 3; n + 1 < static_cast<int>(x.stream.size()) && x.n < 0; ++n) {
		TfCtx probe = x;
		probe.n = n;
		sslm_seq q = TfBuild(e, probe, TfArm::kLayer);
		if (!q) continue;
		int32_t count = 0;
		const int32_t head = TfChainHead(e, q, &count);
		if (count > 0 && head >= 0 && head != x.stream[size_t(n)]) {
			x.n = n;
			x.pin = head;
		}
		sslm_seq_release(q);
	}
	x.ok = x.n >= 3;
	if (x.ok) {
		x.pin_b = TfLegalOther(e, x, TfArm::kReady);
		x.pin_c = TfLegalOther(e, x, TfArm::kCanonSpan);
	}
	CHECK_MSG(x.ok, "[TF %s] no target position: off's stream has %zu tokens and no position n >= 3 whose state is "
	          "forced with a canonical first token other than the model's", TfCfgName(c).c_str(), x.stream.size());
	return x;
}

// The never-failed run of an arm from its target decode (three decodes at `order`, `pin` armed for the
// first), the blob before it, and the number of allocations the target decode makes.
struct TfRef {
	bool ok = false;
	std::vector<uint8_t> before;
	TfRun run;
	long allocs = 0;
	std::vector<std::size_t> sizes;  // the target decode's allocation sizes, in order
};
TfRef TfReference(Engine& e, const TfCtx& x, TfArm arm, int32_t order, int32_t pin) {
	TfRef r;
	sslm_seq s = TfBuild(e, x, arm);
	if (!s) return r;
	r.before = e.Save(s);
	if (pin >= 0) superslm::test::sfd::ArmChoice(pin);
	Arm(-1);
	int32_t t = -100;
	const int st = static_cast<int>(TfDecode(e, x.cfg, s, order, &t));
	g_armed = false;
	r.allocs = g_index.load();
	r.sizes.assign(g_sizes, g_sizes + std::min(r.allocs, kSizeCap));
	superslm::test::sfd::DisarmChoice();
	r.run.st.push_back(st);
	r.run.tok.push_back(t);
	r.run.blob.push_back(e.Save(s));
	const TfRun rest = TfContinue(e, x.cfg, s, order, 2);
	r.run.st.insert(r.run.st.end(), rest.st.begin(), rest.st.end());
	r.run.tok.insert(r.run.tok.end(), rest.tok.begin(), rest.tok.end());
	r.run.blob.insert(r.run.blob.end(), rest.blob.begin(), rest.blob.end());
	sslm_seq_release(s);
	r.ok = st == SSLM_OK && t >= 0;
	return r;
}

// The run with each blob's saturation counters zeroed: kv_saturation_count (offset 92) and the four
// per-site counts (124-155) of the SSB header. forced_token_count (100) is not one and stays compared.
TfRun TfNoSat(TfRun r) {
	for (std::vector<uint8_t>& b : r.blob) {
		if (b.size() < 156) continue;
		std::memset(b.data() + 92, 0, 8);
		std::memset(b.data() + 124, 0, 32);
	}
	return r;
}

// One trial's outcome, as the forked child's exit code.
enum : int { kTfFail = 1, kTf3Fail = 2, kTfInFinish = 4, kTfRestDiffers = 8 };

// One TF.1/TF.3/TF.4/TF.5 trial: the arm's sequence, `pin` armed, allocation k of the target decode at
// `order` failing. `twin_before` is (c')'s blob before its target decode, the layer branch's finish entry.
int TfTrial(Engine& e, const TfCtx& x, TfArm arm, long k, int32_t order, int32_t pin, const TfRef& ref,
            const std::vector<uint8_t>& twin_before, long finish_from, const char* cell) {
	const int failures_before = g_failures;
	int mask = 0;
	sslm_seq s = TfBuild(e, x, arm);
	if (!s) {
		CHECK_MSG(false, "[%s %s, %s, k %ld] setup failed", cell, TfArmName(arm), TfCfgName(x.cfg).c_str(), k);
		return kTfFail;
	}
	const std::vector<uint8_t> before = e.Save(s);
	if (pin >= 0) superslm::test::sfd::ArmChoice(pin);
	Arm(k);
	int32_t t = -100;
	const sslm_status st = TfDecode(e, x.cfg, s, order, &t);
	g_armed = false;
	const bool fired = g_fired.load();
	const bool pin_left = superslm::test::sfd::ChoiceArmed();
	const std::vector<uint8_t> rest = e.Save(s);
	// TF.7 counts a rest that differs only on a trial that fired and failed (an absorbed failure advances the
	// decode, so its blob differs for a reason that is not the guard's).
	if (fired && st != SSLM_OK && rest != before) mask |= kTfRestDiffers;
	const char* an = TfArmName(arm);
	const std::string cn = TfCfgName(x.cfg);
	if (!fired || st == SSLM_OK) {
		superslm::test::sfd::DisarmChoice();
		// Absorbed (or past the last allocation): the call must be the never-failed one.
		const TfRun cont = TfContinue(e, x.cfg, s, order, 2);
		CHECK_MSG(static_cast<int>(st) == ref.run.st[0] && t == ref.run.tok[0] && rest == ref.run.blob[0] &&
		              cont.tok == std::vector<int32_t>(ref.run.tok.begin() + 1, ref.run.tok.end()),
		          "[%s %s, %s, k %ld] the failure was absorbed (status %d token %d), but the run differs from the "
		          "never-failed one %s", cell, an, cn.c_str(), k, static_cast<int>(st), t, Str(ref.run).c_str());
	} else {
		CHECK_MSG(st == SSLM_ALLOCATION_FAILED, "[%s %s, %s, k %ld] status %d; want SSLM_ALLOCATION_FAILED", cell, an,
		          cn.c_str(), k, static_cast<int>(st));
		// On the layer branch the finish's allocations are the target decode's last ones, from `finish_from`
		// (TfFinishFrom: aligned against the ready twin's decode, whose finish is the same code). An
		// allocation failing before that is in the embed or the layer pass, which rests mid-token as the
		// layer budget always has (layer_index L, ready 0) and resumes there: outside the finish, so only the
		// retry decides it. The split is by index, not by the rest, so a finish failure rolled back to before
		// the call (XF6) is still judged against the twin.
		const bool in_finish = arm != TfArm::kLayer || k >= finish_from;
		if (in_finish) mask |= kTfInFinish;
		const std::vector<uint8_t>& entry = arm == TfArm::kLayer ? twin_before : before;
		if (arm != TfArm::kLayer) {
			CHECK_MSG(rest == before, "[%s %s, %s, k %ld] the failed finish does not rest at its entry: %s, before the "
			          "call %s", cell, an, cn.c_str(), k, TfFields(rest).c_str(), TfFields(before).c_str());
		} else if (in_finish) {
			CHECK_MSG(rest == twin_before, "[%s %s, %s, k %ld] a finish failure on the layer branch must rest ready at "
			          "the committed row, equal to the ready twin at the same position: %s, twin %s", cell, an,
			          cn.c_str(), k, TfFields(rest).c_str(), TfFields(twin_before).c_str());
		}
		if (x.cfg.damped && in_finish) {
			const TfBlob a = ReadTf(rest), b = ReadTf(entry);
			const bool same = a.ok && b.ok && a.walk == b.walk && a.current == b.current && a.al.order == b.al.order &&
			                  a.al.hist == b.al.hist;
			if (!same) {
				mask |= kTf3Fail;
				std::printf("FAIL [TF.3 via %s %s, %s, k %ld] the walk, current_token or anti-LM state moved: walk %u -> %u, "
				            "current %d -> %d, order %d -> %d, history %s -> %s\n", cell, an, cn.c_str(), k, b.walk,
				            a.walk, b.current, a.current, b.al.order, a.al.order, Str(b.al.hist).c_str(),
				            Str(a.al.hist).c_str());
			}
		}
		if (order != 2 && in_finish) {  // TF.4 (a): the old state survives the failed create or re-key
			const AntiLm a = ReadAntiLm(rest), b = ReadAntiLm(entry);
			CHECK_MSG(a.order == b.order && a.hist == b.hist, "[%s %s, %s, k %ld] the old anti-LM state did not survive: "
			          "order %d -> %d, history %s -> %s", cell, an, cn.c_str(), k, b.order, a.order, Str(b.hist).c_str(),
			          Str(a.hist).c_str());
		}
		if (pin >= 0) {
			CHECK_MSG(pin_left, "[%s %s, %s, k %ld] the failed finish consumed the armed S-CHOICE pin", cell, an,
			          cn.c_str(), k);
		}
		const TfRun cont = TfContinue(e, x.cfg, s, order, 3);
		superslm::test::sfd::DisarmChoice();
		// A failure inside the layer pass (outside the finish, so outside §4.5) is compared without the SSB
		// saturation counters: the failed pass's sites have already counted, and its resumed retry counts
		// them again. §4.5 makes the finish a transaction and says the finish increments no counter; it is
		// silent on counters a failed layer pass leaves, so this suite does not pin them (the record routes
		// the question to the planner). Every other byte, and every status and token, must match.
		const bool same = in_finish ? cont == ref.run : TfNoSat(cont) == TfNoSat(ref.run);
		CHECK_MSG(same, "[%s %s, %s, k %ld] the retry and the next two decodes %s; never-failed %s (blobs %s)%s",
		          cell, an, cn.c_str(), k, Str(cont).c_str(), Str(ref.run).c_str(),
		          cont.blob == ref.run.blob ? "equal" : "differ", in_finish ? "" : " (a layer-pass failure, counters masked)");
		if (x.cfg.damped && !cont.tok.empty() && cont.tok[0] != ref.run.tok[0]) mask |= kTf3Fail;
	}
	(void)e.Save(s);
	sslm_seq_release(s);
	if (g_failures != failures_before) mask |= kTfFail;
	return mask;
}

// Fork one trial (POSIX); its exit code is the mask. A signal or a sanitizer exit is a failure.
template <typename F>
int TfFork(const char* tag, long k, F&& body) {
	std::fflush(stdout);
#if !defined(_WIN32)
	const pid_t pid = fork();
	if (pid == 0) {
		const int mask = body();
		std::fflush(stdout);
		std::_Exit(mask & 0x3f);
	}
	int status = 0;
	const bool waited = pid > 0 && waitpid(pid, &status, 0) == pid;
	if (waited && WIFEXITED(status) && WEXITSTATUS(status) < 0x40) return WEXITSTATUS(status);
	if (waited && WIFSIGNALED(status)) std::printf("FAIL [%s k %ld] the trial died with signal %d\n", tag, k, WTERMSIG(status));
	else std::printf("FAIL [%s k %ld] the trial exited abnormally (a sanitizer report, above)\n", tag, k);
	return kTfFail;
#else
	// In process (no fork on Windows): the trial's checks are reported in its mask, as a forked child's are, and
	// never in this process's own tally. Left in the tally, a trial expected to fail -- TF.7's guard-off sweep,
	// whose verdict is its rest_differs count -- read as a setup failure and turned TF.1, TF.3, TF.4 (a), TF.5,
	// TF.6 and TF.7 all red "setup failed" (MSVC ctest, 2026-10-21). A real trial failure still sets kTfFail in the
	// mask, which each cell turns red on, so nothing is weakened: it is only counted where the POSIX run counts it.
	const int checks = g_checks, failures = g_failures;
	const int mask = body();
	g_checks = checks;
	g_failures = failures;
	return mask;
#endif
}

struct TfTally {
	long trials = 0, failed = 0, tf3_failed = 0, in_finish = 0, rest_differs = 0;
	long ref_allocs = -1;  // the never-failed target decode's allocations; -1 when it did not run clean
};

// The (d) arms' sweep verdict, or "" when it passes. Greedy with a workspace is allocation-free by design (the
// decode borrows the workspace and has nothing of its own to allocate), so there zero trials is the expected
// outcome and is asserted: the never-failed reference decode ran clean and made no allocation. Every other cfg
// must have allocations to fail, and every trial must pass.
std::string TfDArmVerdict(const TfCfg& c, const TfTally& t) {
	const bool alloc_free = !c.damped && c.ws;
	if (alloc_free) {
		if (t.ref_allocs == 0 && t.trials == 0 && t.failed == 0) return "";
		if (t.ref_allocs < 0) return "the never-failed reference decode failed, or the arm did not build";
		return "greedy with a workspace must be allocation-free, but the target decode made " +
		       std::to_string(t.ref_allocs) + " allocations (" + std::to_string(t.failed) + " of " +
		       std::to_string(t.trials) + " trials failed)";
	}
	if (t.failed == 0 && t.trials > 0) return "";
	return t.trials ? std::to_string(t.failed) + " of " + std::to_string(t.trials) + " trials"
	                : std::string("no allocation to fail, or the arm did not build");
}

// The first allocation of the layer branch's finish. The ready twin's target decode is the same decode
// minus the embed and the layer pass: its allocations are the shared preamble (a common prefix of the
// two size sequences) and then the finish (a common suffix). The split must be exact -- prefix plus
// suffix covering the twin's allocations -- or the cell cannot place the finish and says so.
long TfFinishFrom(const TfRef& a, const TfRef& twin, const char* cell, const std::string& cn) {
	const size_t na = a.sizes.size(), nt = twin.sizes.size();
	size_t pre = 0, suf = 0;
	while (pre < nt && pre < na && a.sizes[pre] == twin.sizes[pre]) ++pre;
	while (suf < nt && suf < na && a.sizes[na - 1 - suf] == twin.sizes[nt - 1 - suf]) ++suf;
	// Prefer the finish: where the two runs overlap, the twin's trailing allocations are the finish's.
	const size_t finish = std::min(suf, nt);
	const bool exact = na == size_t(a.allocs) && nt == size_t(twin.allocs) && pre + finish >= nt;
	CHECK_MSG(exact, "[%s (a), %s] cannot place the layer branch's finish: the twin's %zu allocations are not a common "
	          "prefix (%zu) plus a common suffix (%zu) of the layer branch's %zu", cell, cn.c_str(), nt, pre, suf, na);
	std::printf("[%s (a), %s] the layer branch's %ld allocations: the finish's are the last %zu (twin %ld: preamble %zu, "
	            "finish %zu)\n", cell, cn.c_str(), a.allocs, finish, twin.allocs, nt - finish, finish);
	return exact ? a.allocs - long(finish) : a.allocs;
}

TfTally TfSweep(Engine& e, const TfCtx& x, TfArm arm, int32_t order, int32_t pin, const std::vector<uint8_t>& twin_before,
                const char* cell) {
	TfTally t;
	const TfRef ref = TfReference(e, x, arm, order, pin);
	CHECK_MSG(ref.ok, "[%s %s, %s] the never-failed reference decode failed", cell, TfArmName(arm),
	          TfCfgName(x.cfg).c_str());
	if (!ref.ok) {
		t.failed = 1;
		return t;
	}
	t.ref_allocs = ref.allocs;
	long finish_from = 0;
	if (arm == TfArm::kLayer) {
		const int fails = g_failures;
		finish_from = TfFinishFrom(ref, TfReference(e, x, TfArm::kCanonPending, order, pin), cell, TfCfgName(x.cfg));
		if (g_failures != fails) t.failed = 1;
	}
	for (long k = 0; k < ref.allocs; ++k) {
		++t.trials;
		const int m =
		    TfFork(cell, k, [&] { return TfTrial(e, x, arm, k, order, pin, ref, twin_before, finish_from, cell); });
		if (m & kTfFail) ++t.failed;
		if (m & kTf3Fail) ++t.tf3_failed;
		if (m & kTfInFinish) ++t.in_finish;
		if (m & kTfRestDiffers) ++t.rest_differs;
	}
	return t;
}

// TF.6: where each sequence's allocations sit in the batched call. The batch's sizes are a shared preamble
// (the solo decodes' common first P allocations) followed by each sequence's own allocations in order, which are
// its solo decode's allocations after P. `from[i]` is sequence i's first allocation index; `from[3]` is the end.
// False (and the cell fails) when no P gives that split exactly.
bool TfBatchSplit(const std::vector<std::size_t>& batch, const std::vector<TfRef>& solo, long (&from)[4]) {
	size_t lo = SIZE_MAX;
	for (const TfRef& r : solo) lo = std::min(lo, r.sizes.size());
	for (size_t pre = 0; pre <= lo; ++pre) {
		bool common = true;
		for (size_t q = 0; q < pre && common; ++q)
			for (const TfRef& r : solo) common = common && r.sizes[q] == solo[0].sizes[q];
		if (!common) break;
		std::vector<std::size_t> want(solo[0].sizes.begin(), solo[0].sizes.begin() + long(pre));
		long at = long(pre);
		for (int i = 0; i < 3; ++i) {
			from[i] = at;
			want.insert(want.end(), solo[size_t(i)].sizes.begin() + long(pre), solo[size_t(i)].sizes.end());
			at = long(want.size());
		}
		from[3] = at;
		if (want == batch) return true;
	}
	return false;
}

// TF.6: one trial of the batch. Returns kTfFail, plus kTfInFinish when allocation k belonged to sequence 1 (the
// second sequence, the one the plan names) and the call failed there.
int TfBatchTrial(Engine& e, const TfCtx& x, const TfArm (&arms)[3], long k, const std::vector<std::vector<uint8_t>>& after,
                 const std::vector<int32_t>& after_tok, const std::vector<TfRef>& solo, const long (&from)[4]) {
	const int failures_before = g_failures;
	int mask = 0;
	sslm_seq seqs[3] = {};
	std::vector<std::vector<uint8_t>> before;
	for (int i = 0; i < 3; ++i) {
		seqs[i] = TfBuild(e, x, arms[i]);
		if (!seqs[i]) {
			CHECK_MSG(false, "[TF.6 %s, k %ld] setup failed", TfCfgName(x.cfg).c_str(), k);
			for (sslm_seq s : seqs)
				if (s) sslm_seq_release(s);
			return kTfFail;
		}
		before.push_back(e.Save(seqs[i]));
	}
	const sslm_decode_params p = TfParams(e, x.cfg, 2);
	int32_t out[3] = {-100, -100, -100};
	Arm(k);
	const sslm_status st = sslm_decode_step_v2(e.m, seqs, 3, &p, x.cfg.ws ? e.ws : nullptr, out);
	g_armed = false;
	std::vector<std::vector<uint8_t>> now;
	for (sslm_seq s : seqs) now.push_back(e.Save(s));
	const std::string cn = TfCfgName(x.cfg);
	if (st == SSLM_OK || !g_fired.load()) {
		CHECK_MSG(now == after, "[TF.6 %s, k %ld] the failure was absorbed, but the blobs differ from the never-failed "
		          "batch's", cn.c_str(), k);
	} else {
		// The owner of allocation k, by index (-1: the shared preamble, before any sequence's own work).
		int o = -1;
		for (int i = 0; i < 3; ++i)
			if (k >= from[i] && k < from[i + 1]) o = i;
		if (o == 1) mask |= kTfInFinish;
		CHECK_MSG(st == SSLM_ALLOCATION_FAILED, "[TF.6 %s, k %ld] status %d; want SSLM_ALLOCATION_FAILED", cn.c_str(), k,
		          static_cast<int>(st));
		for (int j = 0; j < 3; ++j) {
			if (j < o) {
				// §4.5: the sequences before the failing one keep their advance.
				CHECK_MSG(now[size_t(j)] == after[size_t(j)] && out[j] == after_tok[size_t(j)],
				          "[TF.6 %s, k %ld] sequence %d, before the failing one (%d), lost its advance: token %d (never-failed "
				          "%d), %s, never-failed %s", cn.c_str(), k, j, o, out[j], after_tok[size_t(j)],
				          TfFields(now[size_t(j)]).c_str(), TfFields(after[size_t(j)]).c_str());
			} else {
				CHECK_MSG(now[size_t(j)] == before[size_t(j)], "[TF.6 %s, k %ld] sequence %d (%s) %s: %s, before %s",
				          cn.c_str(), k, j, TfArmName(arms[j]),
				          j == o ? "is the failing one and does not rest at its entry"
				                 : "comes after the failing one (or the preamble failed) and was touched",
				          TfFields(now[size_t(j)]).c_str(), TfFields(before[size_t(j)]).c_str());
			}
		}
		if (o >= 0) {
			int32_t t = -100;
			const int rs = static_cast<int>(TfDecode(e, x.cfg, seqs[o], 2, &t));
			CHECK_MSG(rs == solo[size_t(o)].run.st[0] && t == solo[size_t(o)].run.tok[0] &&
			              e.Save(seqs[o]) == solo[size_t(o)].run.blob[0],
			          "[TF.6 %s, k %ld] the failing sequence's retry alone: status %d token %d; never-failed %d:%d", cn.c_str(),
			          k, rs, t, solo[size_t(o)].run.st[0], solo[size_t(o)].run.tok[0]);
		}
	}
	for (sslm_seq s : seqs) sslm_seq_release(s);
	if (g_failures != failures_before) mask |= kTfFail;
	return mask;
}

struct TfVerdicts {
	std::vector<std::string> red;  // "cell: reason"
	void Red(const std::string& cell, const std::string& why) { red.push_back(cell + ": " + why); }
};

void RunTfCells(Engine& e, TfVerdicts* v, long* trials) {
	const TfCfg cfgs[] = {{false, false}, {false, true}, {true, false}, {true, true}};
	bool tf1b_damped_nows_green = false, tf1_any_finish = false;
	long tf6_second = 0;  // TF.6 trials that failed inside the second sequence's own allocations
	long tf3_trials = 0;
	for (const TfCfg& c : cfgs) {
		const TfCtx x = TfMakeCtx(e, c);
		const std::string cn = TfCfgName(c);
		if (!x.ok) {
			v->Red("TF.1", "no target position at " + cn);
			continue;
		}
		std::vector<uint8_t> twin_before;
		{
			sslm_seq s = TfBuild(e, x, TfArm::kCanonPending);
			CHECK_MSG(s != nullptr, "[TF %s] arm (c') did not build", cn.c_str());
			if (s) {
				twin_before = e.Save(s);
				sslm_seq_release(s);
			}
		}
		std::printf("[TF %s] off's stream %s; target position N %d, pin %d, start pin %d, (c) pin %d\n", cn.c_str(),
		            Str(x.stream).c_str(), x.n, x.pin, x.pin_b, x.pin_c);
		// TF.1, with TF.3 read from its damped trials.
		for (TfArm arm : kTfArms) {
			const TfTally t = TfSweep(e, x, arm, 2, -1, twin_before, "TF.1");
			*trials += t.trials;
			std::printf("[TF.1 %s, %s] %ld trials, %ld in the finish, %ld failed\n", TfArmName(arm), cn.c_str(), t.trials,
			            t.in_finish, t.failed);
			if (t.failed) v->Red("TF.1", std::string(TfArmName(arm)) + ", " + cn + ": " + std::to_string(t.failed) + " of " +
			                                 std::to_string(t.trials) + " trials");
			if (t.in_finish) tf1_any_finish = true;
			if (arm == TfArm::kReady && c.damped && !c.ws) {
				tf1b_damped_nows_green = t.failed == 0 && t.trials > 0;
				CHECK_MSG(t.in_finish > 0, "[TF.1 (b), %s] the sweep reached no allocation in the finish", cn.c_str());
			}
			if (c.damped) {
				tf3_trials += t.in_finish;
				if (t.tf3_failed)
					v->Red("TF.3", std::string(TfArmName(arm)) + ", " + cn + ": " + std::to_string(t.tf3_failed) + " trials");
			}
		}
		// TF.4 (a): the target decode at order 3, a re-key (a create in arm (b)).
		if (c.damped) {
			for (TfArm arm : kTfArms) {
				const TfTally t = TfSweep(e, x, arm, 3, -1, twin_before, "TF.4");
				*trials += t.trials;
				std::printf("[TF.4 %s, %s, order 2 -> 3] %ld trials, %ld in the finish, %ld failed\n", TfArmName(arm),
				            cn.c_str(), t.trials, t.in_finish, t.failed);
				if (t.failed) v->Red("TF.4 (a)", std::string(TfArmName(arm)) + ", " + cn + ": " + std::to_string(t.failed) +
				                                     " of " + std::to_string(t.trials) + " trials");
			}
		}
		// TF.5: S-CHOICE armed at the target decode.
		struct PinArm {
			TfArm arm;
			int32_t pin;
		};
		const PinArm pins[] = {{TfArm::kLayer, x.pin},
		                       {TfArm::kCanonPending, x.pin},
		                       {TfArm::kReady, x.pin_b},
		                       {TfArm::kCanonSpan, x.pin_c}};
		for (const PinArm& pa : pins) {
			if (pa.pin < 0) {  // a missing pin is a failure, not a skip
				v->Red("TF.5", std::string(TfArmName(pa.arm)) + ", " + cn + ": no legal token other than the model's at "
				                   "the target decode to pin");
				continue;
			}
			const TfTally t = TfSweep(e, x, pa.arm, 2, pa.pin, twin_before, "TF.5");
			*trials += t.trials;
			std::printf("[TF.5 %s, %s, pin %d] %ld trials, %ld in the finish, %ld failed\n", TfArmName(pa.arm), cn.c_str(),
			            pa.pin, t.trials, t.in_finish, t.failed);
			if (t.failed) v->Red("TF.5", std::string(TfArmName(pa.arm)) + ", " + cn + ": " + std::to_string(t.failed) +
			                                 " of " + std::to_string(t.trials) + " trials");
		}
		// TF.6: a batch of three ready sequences.
		{
			const TfArm arms[3] = {TfArm::kReady, TfArm::kCanonPending, TfArm::kCanonSpan};
			std::vector<TfRef> solo;
			for (TfArm a : arms) solo.push_back(TfReference(e, x, a, 2, -1));
			std::vector<std::vector<uint8_t>> after;
			std::vector<int32_t> after_tok;
			std::vector<std::size_t> batch_sizes;
			long allocs = 0;
			sslm_seq seqs[3] = {};
			bool built = true;
			for (int i = 0; i < 3; ++i) built = (seqs[i] = TfBuild(e, x, arms[i])) != nullptr && built;
			if (built) {
				const sslm_decode_params p = TfParams(e, c, 2);
				int32_t out[3] = {-100, -100, -100};
				Arm(-1);
				const sslm_status st = sslm_decode_step_v2(e.m, seqs, 3, &p, c.ws ? e.ws : nullptr, out);
				g_armed = false;
				allocs = g_index.load();
				batch_sizes.assign(g_sizes, g_sizes + std::min(allocs, kSizeCap));
				for (sslm_seq s : seqs) after.push_back(e.Save(s));
				after_tok.assign(out, out + 3);
				CHECK_MSG(st == SSLM_OK, "[TF.6 %s] the never-failed batch: status %d", cn.c_str(), static_cast<int>(st));
				for (int i = 0; i < 3; ++i)
					CHECK_MSG(after[size_t(i)] == solo[size_t(i)].run.blob[0], "[TF.6 %s] sequence %d's batched decode "
					          "differs from its solo decode", cn.c_str(), i);
			}
			for (sslm_seq s : seqs)
				if (s) sslm_seq_release(s);
			long from[4] = {0, 0, 0, 0};
			const bool placed = built && TfBatchSplit(batch_sizes, solo, from);
			CHECK_MSG(!built || placed, "[TF.6 %s] cannot place each sequence's allocations in the batched call (%ld): "
			          "they are not a common preamble plus each solo decode's own allocations in order", cn.c_str(), allocs);
			long failed = 0, in_second = 0;
			for (long k = 0; placed && k < allocs; ++k) {
				++*trials;
				const int m = TfFork("TF.6", k, [&] { return TfBatchTrial(e, x, arms, k, after, after_tok, solo, from); });
				if (m & kTfFail) ++failed;
				if (m & kTfInFinish) ++in_second;
			}
			std::printf("[TF.6 %s] %ld trials (sequences from %ld, %ld, %ld), %ld failed in sequence 1, %ld failed\n",
			            cn.c_str(), allocs, from[0], from[1], from[2], in_second, failed);
			if (!built || !placed || failed)
				v->Red("TF.6", cn + ": " + (!built    ? std::string("setup failed")
				                            : !placed ? std::string("the batch's allocations could not be placed")
				                                      : std::to_string(failed) + " of " + std::to_string(allocs) + " trials"));
			if (placed && allocs > 0) tf6_second += in_second;
		}
		// TF.7: TF.1 (b) with the guard stubbed, and TF.2 (b)'s status exits (final norm, damped refused) with the
		// guard stubbed: S-FINISHEXIT must sit behind the guard, so each exit then rests away from its entry.
		if (c.damped && !c.ws) {
			namespace sfd = superslm::test::sfd;
			for (const sfd::FinishExit ex : {sfd::FinishExit::kFinalNorm, sfd::FinishExit::kDampedRefused}) {
				const char* en = ex == sfd::FinishExit::kFinalNorm ? "final norm" : "damped refused";
				sslm_seq s = TfBuild(e, x, TfArm::kReady);
				if (!s) {
					v->Red("TF.7", std::string("status exit ") + en + ": arm (b) did not build");
					continue;
				}
				const std::vector<uint8_t> before = e.Save(s);
				sfd::SetFinishGuardDisabled(true);
				sfd::ArmFinishExit(ex, 0);
				int32_t t = -100;
				const sslm_status st = TfDecode(e, c, s, 2, &t);
				const uint64_t fired = sfd::FinishExitFired();
				sfd::DisarmFinishExit();
				sfd::SetFinishGuardDisabled(false);
				const bool differs = e.Save(s) != before;
				sslm_seq_release(s);
				std::printf("[TF.7 %s] guard off, status exit %s: status %d, fired %llu, rest %s the entry\n", cn.c_str(), en,
				            static_cast<int>(st), static_cast<unsigned long long>(fired), differs ? "differs from" : "equals");
				if (fired == 0 || st == SSLM_OK)
					v->Red("TF.7", std::string("status exit ") + en + ": the exit was not taken with the guard off (status " +
					                   std::to_string(static_cast<int>(st)) + ", fired " + std::to_string(fired) + ")");
				else if (!differs)
					v->Red("TF.7", std::string("status exit ") + en + ": with the guard off the rest still equals the entry, "
					                   "so the guard is not what restores it (S-FINISHEXIT is not behind the guard)");
			}
			superslm::test::sfd::SetFinishGuardDisabled(true);
			const TfTally t = TfSweep(e, x, TfArm::kReady, 2, -1, twin_before, "TF.7 (guard off)");
			superslm::test::sfd::SetFinishGuardDisabled(false);
			*trials += t.trials;
			std::printf("[TF.7 %s] guard off: %ld trials, %ld rest differently from before the call; TF.1 (b) with the guard "
			            "on: %s\n", cn.c_str(), t.trials, t.rest_differs, tf1b_damped_nows_green ? "green" : "RED");
			if (t.rest_differs == 0) v->Red("TF.7", "with the guard off, no trial's rest differs: the oracle cannot fail");
			if (!tf1b_damped_nows_green) v->Red("TF.7", "TF.1 (b) is not green with the guard on");
		}
	}
	CHECK_MSG(tf1_any_finish, "[TF.1] no sweep reached an allocation in the finish");
	CHECK_MSG(tf3_trials > 0, "[TF.3] no damped trial failed in the finish");
	if (!tf1_any_finish) v->Red("TF.1", "no allocation in any finish");
	if (tf3_trials == 0) v->Red("TF.3", "no damped finish trial");
	// TF.6 is red-if-zero: some trial must fail in the second sequence, the one the plan names.
	if (tf6_second == 0) v->Red("TF.6", "no trial failed inside the second sequence's own allocations");
}

// ===================================================================================================
// Plan rev 1.3.3, slice J1-X (§5.10, §11.4/§11.5): the allocation cells that need a real operator new.
//
// TF.1d  TF.1 in exact mode's arms: (d0) after an exact call that committed p alone (j = 0), (d2) after one
//        that committed j >= 2 drafted tokens. Both leave the sequence ready at the committed row, so the
//        target decode runs the ready branch: allocation k failing rests at its entry, byte-equal to the blob
//        before the call, and the retry and the next two decodes equal the never-failed run. Every cfg.
// 5.X3 (ii)  An allocation failure inside the exact call (§5.10.3 "Failures inside the call"), at every
//        allocation the call makes, damped at order 2 with no workspace (the check's FinishRow allocates on
//        that path) and with one. Either the call returns a non-OK status with the sequence unchanged (a pass
//        or prepare failure before any row is kept), or it returns OK having confirmed j <= the never-failed
//        j drafted tokens, which must be off's own; either way the next three decodes equal off's at the same
//        output positions (status, token, blob). Red if no trial returned OK with j below the never-failed j
//        (the sweep never failed a check row's finish), or if some trial broke the rule.
//
// Both are gated by an exact probe: until J1-X lands the verb is the red stub (SSLM_STATUS_NEXT_FREE), and
// each cell is red for that reason alone.
// ===================================================================================================

bool XExactLive(Engine& e, const TfCtx& x) {
	sslm_seq s = TfBuild(e, x, TfArm::kLayer);
	if (!s) return false;
	int32_t count = -1;
	const std::vector<int32_t> d = {x.pin};
	const sslm_status st = TfExact(e, x.cfg, s, 2, d, 2, &count);
	sslm_seq_release(s);
	return st != SSLM_STATUS_NEXT_FREE;
}

// A fresh pool of its own, so two runs built the same way draw the same physical pages (the table entries are
// then comparable across them).
struct XPool {
	Aligned* buf = nullptr;
	sslm_kv_pool pool = nullptr;
	explicit XPool(const Engine& e, uint32_t blocks = 2) {
		const size_t size = blocks * sslm_kv_block_size(e.m) + sslm_kv_pool_overhead_size(e.m, blocks);
		buf = new Aligned(size);
		if (sslm_kv_pool_create(e.m, buf->p, size, blocks, &pool) != SSLM_OK) pool = nullptr;
	}
	~XPool() {
		if (pool) sslm_kv_pool_destroy(pool);
		delete buf;
	}
	XPool(const XPool&) = delete;
	XPool& operator=(const XPool&) = delete;
};

// One observation: the decode's status and token, the blob, sslm_seq_kv_stats and the table entries.
struct XObsA {
	int st = 0;
	int32_t tok = -100;
	std::vector<uint8_t> blob;
	sslm_seq_kv_stats_out kv{};
	std::vector<uint32_t> table;
};
XObsA XObserveA(const Engine& e, sslm_seq s, int st, int32_t tok) {
	XObsA o;
	o.st = st;
	o.tok = tok;
	o.blob = e.Save(s);
	o.kv.struct_size = sizeof(o.kv);
	sslm_seq_kv_stats(s, &o.kv);
	uint32_t mapped = 0;
	for (uint32_t i = 0;; ++i) {
		uint32_t page = 0;
		if (sslm_pkv_test_only_seq_table_entry(s, i, &page, &mapped) != SSLM_OK || i >= mapped) break;
		o.table.push_back(page);
	}
	return o;
}
std::string XDiffA(const XObsA& a, const XObsA& b) {
	char buf[300];
	if (a.st != b.st || a.tok != b.tok) {
		std::snprintf(buf, sizeof buf, "status/token %d:%d, want %d:%d", a.st, a.tok, b.st, b.tok);
		return buf;
	}
	if (a.blob != b.blob) return "blob differs: " + TfFields(a.blob) + "; want " + TfFields(b.blob);
	const sslm_seq_kv_stats_out& p = a.kv;
	const sslm_seq_kv_stats_out& q = b.kv;
	if (p.context_length != q.context_length || p.mapped_private_pages != q.mapped_private_pages ||
	    p.shared_pages != q.shared_pages || p.reserve_pages != q.reserve_pages ||
	    p.materialized_pages != q.materialized_pages) {
		std::snprintf(buf, sizeof buf, "kv stats: ctx %lld/%lld mapped %u/%u reserve %u/%u materialized %u/%u",
		              static_cast<long long>(p.context_length), static_cast<long long>(q.context_length),
		              p.mapped_private_pages, q.mapped_private_pages, p.reserve_pages, q.reserve_pages,
		              p.materialized_pages, q.materialized_pages);
		return buf;
	}
	if (a.table != b.table) return "table entries differ (" + std::to_string(a.table.size()) + " / " +
	                               std::to_string(b.table.size()) + " mapped)";
	return "";
}

// 5.X3 (ii): one trial, at page size 2 with S-SATROW armed on the last drafted row. Every run here is built in a
// fresh pool of its own, so the table entries compare:
//   off  -- N decodes, then d.size() + 4 more, observed after each;
//   ref  -- the never-failed exact call (S-SATROW armed), then 3 decodes;
//   the trial -- the exact call with allocation k failing.
// Returns 1 on a failed check, 2 when the call returned OK with j below the never-failed j.
int XCallTrial(Engine& e, const TfCtx& x, long k, const std::vector<int32_t>& d) {
	const int failures_before = g_failures;
	int mask = 0;
	const std::string cn = TfCfgName(x.cfg);
	const int32_t sat_row = int32_t(d.size());  // rows 0 (p) .. k: the last drafted row
	std::vector<XObsA> off;
	{
		XPool pa(e);
		sslm_seq o = pa.pool ? TfBuild(e, x, TfArm::kLayer, &pa.pool) : nullptr;
		for (size_t i = 0; o && i < d.size() + 4; ++i) {
			int32_t t = -100;
			const sslm_status st = TfDecode(e, x.cfg, o, 2, &t);
			off.push_back(XObserveA(e, o, static_cast<int>(st), t));
			if (st != SSLM_OK || t < 0) break;
		}
		if (o) sslm_seq_release(o);
	}
	int32_t ref_count = -1;
	std::vector<XObsA> ref;
	{
		XPool pc(e);
		sslm_seq r = pc.pool ? TfBuild(e, x, TfArm::kLayer, &pc.pool) : nullptr;
		if (r) {
			superslm::test::sfd::ArmSatRow(sat_row);
			TfExact(e, x.cfg, r, int32_t(d.size()) + 1, d, 2, &ref_count);
			superslm::test::sfd::DisarmSatRow();
			for (int i = 0; i < 3; ++i) {
				int32_t t = -100;
				const sslm_status st = TfDecode(e, x.cfg, r, 2, &t);
				ref.push_back(XObserveA(e, r, static_cast<int>(st), t));
			}
			sslm_seq_release(r);
		}
	}
	if (off.size() < d.size() + 1 || ref.size() != 3 || ref_count < 0) {
		CHECK_MSG(false, "[5.X3 (ii) %s, k %ld] setup: off %zu observations, never-failed count %d", cn.c_str(), k,
		          off.size(), ref_count);
		return 1;
	}
	XPool pb(e);
	sslm_seq s = pb.pool ? TfBuild(e, x, TfArm::kLayer, &pb.pool) : nullptr;
	if (!s) {
		CHECK_MSG(false, "[5.X3 (ii) %s, k %ld] setup failed", cn.c_str(), k);
		return 1;
	}
	const XObsA before = XObserveA(e, s, 0, 0);
	int32_t count = -1;
	std::vector<int32_t> out(d.size() + 1, -7);  // allocated before the window opens
	superslm::test::sfd::ArmSatRow(sat_row);
	Arm(k);
	const sslm_status st = TfExact(e, x.cfg, s, int32_t(d.size()) + 1, d, 2, &count, &out);
	g_armed = false;
	superslm::test::sfd::DisarmSatRow();
	int32_t j = count;
	if (st != SSLM_OK) {
		// A pass or prepare failure before any row is kept: nothing changed, pages and stats included.
		const XObsA now = XObserveA(e, s, 0, 0);
		CHECK_MSG(st == SSLM_ALLOCATION_FAILED && XDiffA(now, before).empty(),
		          "[5.X3 (ii) %s, k %ld] status %d: a failed call must return SSLM_ALLOCATION_FAILED with nothing changed "
		          "(%s)", cn.c_str(), k, static_cast<int>(st), XDiffA(now, before).c_str());
		superslm::test::sfd::ArmSatRow(sat_row);
		const sslm_status rs = TfExact(e, x.cfg, s, int32_t(d.size()) + 1, d, 2, &count);
		superslm::test::sfd::DisarmSatRow();
		CHECK_MSG(rs == SSLM_OK && count == ref_count, "[5.X3 (ii) %s, k %ld] the retry: status %d count %d; never-failed %d",
		          cn.c_str(), k, static_cast<int>(rs), count, ref_count);
		j = count;
	} else {
		CHECK_MSG(count >= 0 && count <= ref_count, "[5.X3 (ii) %s, k %ld] OK with count %d; the never-failed call "
		          "confirmed %d, and a failure can only stop the check earlier", cn.c_str(), k, count, ref_count);
		if (count >= 0 && count < ref_count) mask |= 2;
	}
	// The next three decodes: against the never-failed exact run when the whole draft was kept (its last row's
	// injected count is then committed), else against off at the same output positions (the injected row was
	// rejected, so its count must not survive, and its pages went back).
	for (int i = 0; i < 3 && j >= 0; ++i) {
		int32_t t = -100;
		const sslm_status ds = TfDecode(e, x.cfg, s, 2, &t);
		const XObsA now = XObserveA(e, s, static_cast<int>(ds), t);
		const size_t at = size_t(j) + size_t(i);
		const XObsA* want = j == ref_count ? &ref[size_t(i)] : at < off.size() ? &off[at] : nullptr;
		if (!want) break;
		const std::string diff = XDiffA(now, *want);
		CHECK_MSG(diff.empty(), "[5.X3 (ii) %s, k %ld] decode %d after the call (j %d, status %d): %s", cn.c_str(), k, i, j,
		          static_cast<int>(st), diff.c_str());
		if (ds != SSLM_OK) break;
	}
	sslm_seq_release(s);
	if (g_failures != failures_before) mask |= 1;
	return mask;
}

void RunXAllocCells(Engine& e, TfVerdicts* v, long* trials) {
	const TfCfg cfgs[] = {{false, false}, {false, true}, {true, false}, {true, true}};
	bool live_checked = false;
	for (const TfCfg& c : cfgs) {
		const TfCtx x = TfMakeCtx(e, c);
		const std::string cn = TfCfgName(c);
		if (!x.ok) {
			for (const char* cell : {"TF.1d", "TF.4 (a) (d)", "TF.5 (d)"}) v->Red(cell, "no target position at " + cn);
			continue;
		}
		if (!live_checked) {
			live_checked = true;
			if (!XExactLive(e, x)) {
				for (const char* cell : {"TF.1d", "TF.4 (a) (d)", "TF.5 (d)"})
					v->Red(cell, "exact mode is unimplemented (the J1-X verb returned SSLM_STATUS_NEXT_FREE, the red stub)");
				return;
			}
		}
		// TF.1d, first: exact mode's resting blob before the target decode equals canonical mode's pending-only
		// commit at the same output position ((c'), §11.5 "(d)'s blob equals (a)'s"). (d0) at N; (d2) at N + j.
		std::vector<int32_t> d2;
		for (size_t q = size_t(x.n); q < x.stream.size() && d2.size() < 7; ++q) d2.push_back(x.stream[q]);
		int32_t j2 = -1;
		if (sslm_seq s = TfBuild(e, x, TfArm::kLayer)) {
			TfExact(e, c, s, int32_t(d2.size()) + 1, d2, 2, &j2);
			sslm_seq_release(s);
		}
		for (TfArm arm : {TfArm::kExact0, TfArm::kExact2}) {
			TfCtx at = x;
			if (arm == TfArm::kExact2) at.n = x.n + std::max(j2, 0);
			std::vector<uint8_t> exact_blob, twin_blob;
			if (sslm_seq s = TfBuild(e, x, arm)) {
				exact_blob = e.Save(s);
				sslm_seq_release(s);
			}
			if (sslm_seq s = TfBuild(e, at, TfArm::kCanonPending)) {
				twin_blob = e.Save(s);
				sslm_seq_release(s);
			}
			if (exact_blob.empty() || twin_blob.empty() || exact_blob != twin_blob)
				v->Red("TF.1d", std::string(TfArmName(arm)) + ", " + cn + ": the exact commit's resting blob " +
				                    (exact_blob.empty() || twin_blob.empty() ? std::string("did not build")
				                                                             : TfFields(exact_blob) + " is not canonical "
				                                                               "(c')'s at the same position " +
				                                                               TfFields(twin_blob)));
		}
		// TF.1d: the ready branch after an exact commit; TfTrial's (b)/(c) rule applies (rest == before).
		for (TfArm arm : {TfArm::kExact0, TfArm::kExact2}) {
			const TfTally t = TfSweep(e, x, arm, 2, -1, {}, "TF.1d");
			*trials += t.trials;
			std::printf("[TF.1d %s, %s] %ld trials, %ld failed (reference %ld allocations)\n", TfArmName(arm), cn.c_str(),
			            t.trials, t.failed, t.ref_allocs);
			const std::string why = TfDArmVerdict(c, t);
			if (!why.empty()) v->Red("TF.1d", std::string(TfArmName(arm)) + ", " + cn + ": " + why);
		}
		// TF.4 (a) in the (d) arms: the target decode at order 3, a re-key of the order-2 state.
		if (c.damped)
			for (TfArm arm : {TfArm::kExact0, TfArm::kExact2}) {
				const TfTally t = TfSweep(e, x, arm, 3, -1, {}, "TF.4");
				*trials += t.trials;
				std::printf("[TF.4 (a) %s, %s] %ld trials, %ld failed\n", TfArmName(arm), cn.c_str(), t.trials, t.failed);
				if (t.failed || t.trials == 0)
					v->Red("TF.4 (a) (d)", std::string(TfArmName(arm)) + ", " + cn + ": " +
					                           (t.trials ? std::to_string(t.failed) + " of " + std::to_string(t.trials) +
					                                           " trials"
					                                     : std::string("no allocation to fail, or the arm did not build")));
			}
		// TF.5 in the (d) arms: S-CHOICE armed with a legal token other than the model's.
		for (TfArm arm : {TfArm::kExact0, TfArm::kExact2}) {
			const int32_t pin = TfLegalOther(e, x, arm);
			if (pin < 0) {
				v->Red("TF.5 (d)", std::string(TfArmName(arm)) + ", " + cn + ": no legal token other than the model's to pin");
				continue;
			}
			const TfTally t = TfSweep(e, x, arm, 2, pin, {}, "TF.5");
			*trials += t.trials;
			std::printf("[TF.5 %s, %s, pin %d] %ld trials, %ld failed (reference %ld allocations)\n", TfArmName(arm),
			            cn.c_str(), pin, t.trials, t.failed, t.ref_allocs);
			const std::string why = TfDArmVerdict(c, t);
			if (!why.empty()) v->Red("TF.5 (d)", std::string(TfArmName(arm)) + ", " + cn + ": " + why);
		}
	}
}

// 5.X3 (ii) runs on its own engine, the page size pinned to 2.
void RunX53ii(TfVerdicts* v, long* trials) {
	Engine e(2);
	if (!e.ok) {
		v->Red("5.X3 (ii)", "the page-size-2 engine did not load");
		return;
	}
	long short_commits = 0;
	bool any = false;
	for (const TfCfg& c : {TfCfg{true, false}, TfCfg{true, true}}) {
		const TfCtx x = TfMakeCtx(e, c);
		const std::string cn = TfCfgName(c);
		if (!x.ok) {
			v->Red("5.X3 (ii)", "no target position at " + cn);
			continue;
		}
		if (!XExactLive(e, x)) {
			v->Red("5.X3 (ii)", "exact mode is unimplemented (the J1-X verb returned SSLM_STATUS_NEXT_FREE, the red stub)");
			return;
		}
		any = true;
		std::vector<int32_t> d;
		for (size_t q = size_t(x.n); q < x.stream.size() && d.size() < 7; ++q) d.push_back(x.stream[q]);
		long allocs = 0;
		{
			XPool pc(e);
			sslm_seq s = pc.pool ? TfBuild(e, x, TfArm::kLayer, &pc.pool) : nullptr;
			if (s) {
				int32_t ref_count = -1;
				std::vector<int32_t> out(d.size() + 1, -7);  // allocated before the window opens
				superslm::test::sfd::ArmSatRow(int32_t(d.size()));
				Arm(-1);
				const sslm_status st = TfExact(e, c, s, int32_t(d.size()) + 1, d, 2, &ref_count, &out);
				g_armed = false;
				superslm::test::sfd::DisarmSatRow();
				allocs = g_index.load();
				CHECK_MSG(st == SSLM_OK && ref_count >= 2, "[5.X3 (ii) %s] the never-failed call: status %d count %d",
				          cn.c_str(), static_cast<int>(st), ref_count);
				sslm_seq_release(s);
			}
		}
		long failed = 0;
		for (long k = 0; k < allocs; ++k) {
			++*trials;
			const int m = TfFork("5.X3 (ii)", k, [&] { return XCallTrial(e, x, k, d); });
			if (m & 1) ++failed;
			if (m & 2) ++short_commits;
		}
		std::printf("[5.X3 (ii) %s, page 2] %ld trials, %ld failed\n", cn.c_str(), allocs, failed);
		if (failed || allocs == 0)
			v->Red("5.X3 (ii)", cn + ": " + (allocs ? std::to_string(failed) + " of " + std::to_string(allocs) + " trials"
			                                         : std::string("the call made no allocation to fail")));
	}
	if (any && short_commits == 0)
		v->Red("5.X3 (ii)", "no trial failed inside a check row's finish (OK with j below the never-failed j): the sweep "
		                    "never reached FinishRow");
}

}  // namespace

int main() {
	std::vector<std::string> failed;
	long trials = 0;
	{
		Engine e;
		if (!e.ok) {
			std::printf("schema fast-decode alloc sweep: setup failed (%d failures)\n", g_failures);
			return 1;
		}
		// The verb sweep's never-failed reference, and the number of allocations one verb@2 call makes.
		std::vector<uint8_t> ref_after;
		Decoded ref_next;
		long total = 0;
		{
			sslm_seq s = e.Setup();
			if (!s) return 1;
			int32_t count = -1;
			Arm(-1);
			const sslm_status st = e.Verb(s, 2, &count);
			g_armed = false;
			total = g_index.load();
			CHECK_MSG(st == SSLM_OK && count == 1, "reference verb@2: status %d count %d", static_cast<int>(st), count);
			ref_after = e.Save(s);
			const AntiLm a = ReadAntiLm(ref_after);
			ref_next = DecodeN(e, s, 2);
			std::printf("reference: verb@2 on an order-3 state makes %ld allocations; after it, order %d history %s; "
			            "the next decodes %s\n", total, a.order, Str(a.hist).c_str(), Str(ref_next).c_str());
			sslm_seq_release(s);
		}
		CHECK_MSG(total > 0, "the verb call made no allocation to fail");
		for (long k = 0; k < total; ++k) {
			++trials;
			RunTrial("verb", k, &failed, [&] { return Trial(e, k, ref_after, ref_next); });
		}
		// Fix round 4: the damped decode step's own allocations, at a re-keying order and at the live one.
		for (int32_t order : {2, 3}) {
			std::vector<uint8_t> dref_after;
			Decoded dref_next;
			long dtotal = 0;
			{
				sslm_seq s = e.Setup();
				if (!s) return 1;
				const sslm_decode_params p = e.Damped(order);
				int32_t out = -100;
				Arm(-1);
				const sslm_status st = sslm_decode_step_v2(e.m, &s, 1, &p, e.ws, &out);
				g_armed = false;
				dtotal = g_index.load();
				CHECK_MSG(st == SSLM_OK && out >= 0, "reference decode@%d: status %d token %d", order,
				          static_cast<int>(st), out);
				dref_after = e.Save(s);
				dref_next = DecodeN(e, s, order);
				std::printf("reference: decode@%d on an order-3 state makes %ld allocations (token %d); the next "
				            "decodes %s\n", order, dtotal, out, Str(dref_next).c_str());
				sslm_seq_release(s);
			}
			CHECK_MSG(dtotal > 0, "decode@%d made no allocation to fail", order);
			const std::string tag = "decode@" + std::to_string(order);
			for (long k = 0; k < dtotal; ++k) {
				++trials;
				RunTrial(tag.c_str(), k, &failed, [&] { return DecodeTrial(e, order, k, dref_after, dref_next); });
			}
		}
	}
	std::string list;
	for (size_t i = 0; i < failed.size(); ++i) list += (i ? "," : "") + failed[i];
	std::printf("schema fast-decode alloc sweep: %ld trials, %zu failed {%s}; %d checks, %d failures\n", trials,
	            failed.size(), list.c_str(), g_checks, g_failures);
	// Plan rev 1.3.3, slice TF: the decode step's finish.
	TfVerdicts tf;
	long tf_trials = 0;
	const int checks_before_tf = g_checks, failures_before_tf = g_failures;
	{
		Engine e;
		if (e.ok) RunTfCells(e, &tf, &tf_trials);
		else ++g_failures;
	}
	static const char* const kTfCells[] = {"TF.1", "TF.3", "TF.4 (a)", "TF.5", "TF.6", "TF.7"};
	if (g_failures != failures_before_tf)  // the TF setup itself failed (in this process, not a trial)
		for (const char* cell : kTfCells) tf.Red(cell, "setup failed");
	for (const char* cell : kTfCells) {
		std::string why;
		for (const std::string& r : tf.red)
			if (r.rfind(std::string(cell) + ":", 0) == 0) why += (why.empty() ? "" : "; ") + r.substr(std::strlen(cell) + 2);
		std::printf("[schema fast-decode alloc] cell %-8s %s%s%s\n", cell, why.empty() ? "green" : "RED", why.empty() ? "" : " -- ",
		            why.c_str());
	}
	// Each trial runs in a forked child, so its checks are reported in its verdict, not counted here.
	std::printf("schema fast-decode TF sweep: %ld trials; %d setup checks, %d setup failures\n", tf_trials,
	            g_checks - checks_before_tf, g_failures - failures_before_tf);
	// Plan rev 1.3.3, slice J1-X: TF.1d and 5.X3 (ii).
	TfVerdicts xv;
	long x_trials = 0;
	const int checks_before_x = g_checks, failures_before_x = g_failures;
	{
		Engine e;
		if (e.ok) RunXAllocCells(e, &xv, &x_trials);
		else ++g_failures;
	}
	RunX53ii(&xv, &x_trials);
	static const char* const kXCells[] = {"TF.1d", "TF.4 (a) (d)", "TF.5 (d)", "5.X3 (ii)"};
	if (g_failures != failures_before_x)
		for (const char* cell : kXCells) xv.Red(cell, "setup failed");
	for (const char* cell : kXCells) {
		std::string why;
		for (const std::string& r : xv.red)
			if (r.rfind(std::string(cell) + ":", 0) == 0) why += (why.empty() ? "" : "; ") + r.substr(std::strlen(cell) + 2);
		std::printf("[schema fast-decode alloc] cell %-12s %s%s%s\n", cell, why.empty() ? "green" : "RED", why.empty() ? "" : " -- ",
		            why.c_str());
	}
	std::printf("schema fast-decode J1-X sweep: %ld trials; %d setup checks, %d setup failures\n", x_trials,
	            g_checks - checks_before_x, g_failures - failures_before_x);
	return failed.empty() && g_failures == 0 && tf.red.empty() && xv.red.empty() ? 0 : 1;
}
