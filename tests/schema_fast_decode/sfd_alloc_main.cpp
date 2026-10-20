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
// review round 1, observation 9) is outside these cells: it is reported, and only its save and release
// are held to the sanitizers. On POSIX every trial runs in a forked child, so a crash or a sanitizer abort in one trial fails
// that trial and the sweep goes on; elsewhere the trials run in-process.
//
// Allocation indices are the global operator new's (throwing and nothrow, scalar and array); the
// aligned forms are not replaced and not counted. Reads F-SFD from SUPERSLM_SCHEMA_FAST_DECODE_FIXTURE_DIR;
// a missing fixture fails. Exit status 0 iff every trial passed.
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

namespace {
std::atomic<bool> g_armed{false};
std::atomic<long> g_index{0};
long g_fail_at = -1;
std::atomic<bool> g_fired{false};

void* CountedNew(std::size_t size) {
	if (g_armed.load(std::memory_order_relaxed)) {
		const long i = g_index.fetch_add(1, std::memory_order_relaxed);
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

	Engine() {
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
		return sslm_seq_jump_forward(m, s, 1, &p, ws, buf, 8, count);
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

// One trial at allocation index k. Returns true iff every check in it passed.
bool Trial(Engine& e, long k, const std::vector<uint8_t>& ref_after, long* seen) {
	const int failures_before = g_failures;
	sslm_seq s = e.Setup();
	if (!s) return false;
	const std::vector<uint8_t> before = e.Save(s);
	const AntiLm a0 = ReadAntiLm(before);
	CHECK_MSG(a0.ok && a0.order == 3 && a0.hist.size() == 1, "[k %ld] setup: order %d history %s", k, a0.order,
	          Str(a0.hist).c_str());
	int32_t count = -1;
	g_fail_at = k;
	g_fired = false;
	g_index = 0;
	g_armed = true;
	const sslm_status st = e.Verb(s, 2, &count);
	g_armed = false;
	*seen = g_index.load();
	const bool fired = g_fired.load();
	std::printf("[k %ld] verb@2 on an order-3 state, allocation #%ld failing (fired %d, %ld allocations): status %d "
	            "count %d\n", k, k, fired, *seen, static_cast<int>(st), count);
	std::fflush(stdout);
	if (fired && st != SSLM_OK && count > 0) {
		// The failure came after the admission (the anti-LM history update that follows the pass): tokens
		// were admitted, so this is not the failed call of cells 5.2 and 8.6 (vi). Code review round 1
		// lists it as observation 9, outside this round; the trial still saves and releases below.
		std::printf("NOTE [k %ld] status %d after %d tokens were admitted: outside cells 5.2 and 8.6 (vi)\n", k,
		            static_cast<int>(st), count);
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
	(void)e.Save(s);
	sslm_seq_release(s);
	return g_failures == failures_before;
}

}  // namespace

int main() {
	std::vector<long> failed;
	long trials = 0;
	{
		Engine e;
		if (!e.ok) {
			std::printf("schema fast-decode alloc sweep: setup failed (%d failures)\n", g_failures);
			return 1;
		}
		// The never-failed reference, and the number of allocations one verb@2 call makes.
		std::vector<uint8_t> ref_after;
		long total = 0;
		{
			sslm_seq s = e.Setup();
			if (!s) return 1;
			int32_t count = -1;
			g_fail_at = -1;
			g_index = 0;
			g_armed = true;
			const sslm_status st = e.Verb(s, 2, &count);
			g_armed = false;
			total = g_index.load();
			CHECK_MSG(st == SSLM_OK && count == 1, "reference verb@2: status %d count %d", static_cast<int>(st), count);
			ref_after = e.Save(s);
			const AntiLm a = ReadAntiLm(ref_after);
			std::printf("reference: verb@2 on an order-3 state makes %ld allocations; after it, order %d history %s\n",
			            total, a.order, Str(a.hist).c_str());
			sslm_seq_release(s);
		}
		CHECK_MSG(total > 0, "the verb call made no allocation to fail");
		for (long k = 0; k < total; ++k) {
			++trials;
			std::fflush(stdout);
#if !defined(_WIN32)
			const pid_t pid = fork();
			if (pid == 0) {
				long seen = 0;
				const bool pass = Trial(e, k, ref_after, &seen);
				std::fflush(stdout);
				std::_Exit(pass ? 0 : 1);
			}
			int status = 0;
			const bool waited = pid > 0 && waitpid(pid, &status, 0) == pid;
			const bool pass = waited && WIFEXITED(status) && WEXITSTATUS(status) == 0;
			++g_checks;
			if (!pass) {
				++g_failures;
				failed.push_back(k);
				if (waited && WIFSIGNALED(status))
					std::printf("FAIL [k %ld] the trial died with signal %d\n", k, WTERMSIG(status));
				else if (waited && WIFEXITED(status) && WEXITSTATUS(status) != 1)
					std::printf("FAIL [k %ld] the trial exited %d (a sanitizer report, above)\n", k, WEXITSTATUS(status));
			}
#else
			long seen = 0;
			if (!Trial(e, k, ref_after, &seen)) failed.push_back(k);
#endif
		}
	}
	std::string list;
	for (size_t i = 0; i < failed.size(); ++i) list += (i ? "," : "") + std::to_string(failed[i]);
	std::printf("schema fast-decode alloc sweep: %ld trials, %zu failed {%s}; %d checks, %d failures\n", trials,
	            failed.size(), list.c_str(), g_checks, g_failures);
	return failed.empty() && g_failures == 0 ? 0 : 1;
}
