// Decode-threading plan (rev 1.2) §8 7.4 (F12): the one-row split adds no heap allocation. The
// counting global operator new of tests/t2138-abi-red-suite/dim7_contract_red.cpp:64, reused in its
// own executable (it replaces operator new for the whole process), with a hook that allocates
// nothing: per decode token and per one-token prefill call, count(bit 1 on) <= count(bit 1 off).
//
// Reads the F-DEF fixture from SUPERSLM_DECODE_THREADING_FIXTURE_DIR; a missing fixture fails.
// Exit status 0 iff every check passed.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "superslm/layer_marshal.h"
#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#include "../src/forward/parallel_split.h"

static long long g_new_calls = 0;

void* operator new(std::size_t size) {
	++g_new_calls;
	if (void* p = std::malloc(size ? size : 1)) return p;
	throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

static int g_checks = 0, g_failures = 0;
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

namespace {

// Runs every task inline on the calling thread and allocates nothing.
struct CountingHook {
	int calls = 0;
	static void Run(void* host_ctx, int32_t task_count, sslm_task_fn task, void* task_ctx) {
		++static_cast<CountingHook*>(host_ctx)->calls;
		for (int32_t i = 0; i < task_count; ++i) task(task_ctx, i);
	}
};

struct Counts {
	std::vector<long long> decode, prefill;
	int run_calls = 0;
};

Counts Measure(sslm_model model, uint32_t bits) {
	Counts out;
	const uint32_t blocks = 2;
	const size_t psize = blocks * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, blocks);
	void* pool_buf = ::operator new(psize, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	sslm_kv_pool pool = nullptr;
	CHECK_MSG(sslm_kv_pool_create(model, pool_buf, psize, blocks, &pool) == SSLM_OK, "pool");
	const sslm_config cfg{1, 16, 2, 0u};
	const size_t wsz = sslm_workspace_size(model, &cfg);
	void* ws_buf = ::operator new(wsz, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	sslm_workspace ws = nullptr;
	CHECK_MSG(sslm_workspace_create(model, &cfg, ws_buf, wsz, &ws) == SSLM_OK, "workspace");
	CountingHook hook;
	const sslm_parallel_for pf{&CountingHook::Run, &hook, 4, bits};
	CHECK_MSG(sslm_workspace_set_parallel_for(ws, &pf) == SSLM_OK, "install, bits %u", bits);
	sslm_seq seq = nullptr;
	CHECK_MSG(sslm_seq_create(model, &pool, &seq) == SSLM_OK, "seq");
	sslm_decode_params params{};
	sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, 2, &params);
	for (int32_t i = 0; i < 12; ++i) {
		const int32_t tok = (i * 37 + 11) % 256;
		int32_t consumed = 0;
		const long long before = g_new_calls;
		const sslm_status st = sslm_prefill(model, seq, &tok, 1, 1, SSLM_SPAN_PROMPT, ws, &consumed);
		out.prefill.push_back(g_new_calls - before);
		CHECK_MSG(st == SSLM_OK && consumed == 1, "prefill %d: status %d", i, static_cast<int>(st));
	}
	for (int i = 0; i < 12; ++i) {
		int32_t tok = -2;
		const long long before = g_new_calls;
		const sslm_status st = sslm_decode_step_v2(model, &seq, 1, &params, ws, &tok);
		out.decode.push_back(g_new_calls - before);
		CHECK_MSG(st == SSLM_OK && tok >= 0, "decode %d: status %d", i, static_cast<int>(st));
	}
	out.run_calls = hook.calls;
	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	::operator delete(ws_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	::operator delete(pool_buf, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	return out;
}

}  // namespace

int main() {
	const char* dir = std::getenv("SUPERSLM_DECODE_THREADING_FIXTURE_DIR");
	CHECK_MSG(dir != nullptr && *dir != '\0', "SUPERSLM_DECODE_THREADING_FIXTURE_DIR is not set");
	std::vector<uint8_t> raw;
	if (dir != nullptr && *dir != '\0') {
		const std::string path = std::string(dir) + "/fdef.sslm";
		CHECK_MSG(superslm_marshal::ReadFile(path.c_str(), raw) && !raw.empty(), "fixture %s missing", path.c_str());
	}
	if (!raw.empty()) {
		uint8_t* bytes =
		    static_cast<uint8_t*>(::operator new(raw.size(), std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
		std::memcpy(bytes, raw.data(), raw.size());
		sslm_model model = nullptr;
		CHECK_MSG(sslm_model_map(bytes, raw.size(), &model) == SSLM_OK, "sslm_model_map");
		if (model) {
			superslm::test::SetMinRowBytesPerTask(8 * 1024);  // every M = 1 site threads (§3.9)
			const Counts off = Measure(model, 0u);
			const Counts on = Measure(model, SSLM_PARALLEL_FOR_MATVEC);
			superslm::test::ResetMinRowBytesPerTask();
			CHECK_MSG(on.run_calls > off.run_calls && on.run_calls >= 12 * 14,
			          "preamble: bit 1 on made %d run calls, off %d", on.run_calls, off.run_calls);
			for (size_t i = 0; i < on.prefill.size(); ++i) {
				CHECK_MSG(on.prefill[i] <= off.prefill[i], "7.4 one-token prefill %zu: %lld allocations on, %lld off",
				          i, on.prefill[i], off.prefill[i]);
			}
			for (size_t i = 0; i < on.decode.size(); ++i) {
				CHECK_MSG(on.decode[i] <= off.decode[i], "7.4 decode token %zu: %lld allocations on, %lld off", i,
				          on.decode[i], off.decode[i]);
			}
			std::printf("7.4: per decode token %lld allocations off, %lld on; per one-token prefill %lld off, %lld on\n",
			            off.decode.back(), on.decode.back(), off.prefill.back(), on.prefill.back());
			sslm_model_unmap(model);
		}
		::operator delete(bytes, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	}
	std::printf("superslm_decode_threading_alloc: %d checks, %d failures\n", g_checks, g_failures);
	return g_failures == 0 && g_checks > 0 ? 0 : 1;
}
