// Local evidence for cells 6.3 and 9.1 against the v1.9.0 library (not a suite cell).
// Built twice: against v1.9.0's libsuperslm.a (no hook bits exist) and against the D1 seam library
// (bit 1 on, seam 8 KiB so every M = 1 site threads).
//   x190 save <fixture> <out.blob>              one-token prefill of a 64-token prompt, then save
//   x190 cont <fixture> <in.blob> <rt.blob> <out.blob>
//        restore, save again (round-trip bytes), decode 8 greedy tokens, save; prints the tokens
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "superslm/parallel_for.h"
#include "superslm/sslm_abi.h"
#ifdef SSLM_PARALLEL_FOR_MATVEC
#include "../src/forward/parallel_split.h"
#endif

static std::vector<uint8_t> ReadAll(const char* p) {
	std::vector<uint8_t> v;
	FILE* f = std::fopen(p, "rb");
	if (!f) return v;
	std::fseek(f, 0, SEEK_END);
	v.resize(static_cast<size_t>(std::ftell(f)));
	std::fseek(f, 0, SEEK_SET);
	if (std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
	std::fclose(f);
	return v;
}
static void WriteAll(const char* p, const std::vector<uint8_t>& v) {
	FILE* f = std::fopen(p, "wb");
	std::fwrite(v.data(), 1, v.size(), f);
	std::fclose(f);
}

struct Hook {
	long calls = 0;
	static void Run(void* ctx, int32_t n, sslm_task_fn task, void* tctx) {
		++static_cast<Hook*>(ctx)->calls;
		for (int32_t i = n - 1; i >= 0; --i) task(tctx, i);  // reverse order: any order must do
	}
};

#define DIE(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(2); } while (0)

int main(int argc, char** argv) {
	if (argc < 4) DIE("usage");
	const std::string mode = argv[1];
	std::vector<uint8_t> raw = ReadAll(argv[2]);
	if (raw.empty()) DIE("fixture");
	uint8_t* bytes = static_cast<uint8_t*>(::operator new(raw.size(), std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES)));
	std::memcpy(bytes, raw.data(), raw.size());
	sslm_model model = nullptr;
	if (sslm_model_map(bytes, raw.size(), &model) != SSLM_OK) DIE("map");
	const uint32_t blocks = 16;
	const size_t psize = blocks * sslm_kv_block_size(model) + sslm_kv_pool_overhead_size(model, blocks);
	void* pool_buf = ::operator new(psize, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	sslm_kv_pool pool = nullptr;
	if (sslm_kv_pool_create(model, pool_buf, psize, blocks, &pool) != SSLM_OK) DIE("pool");
	const sslm_config cfg{1, 16, 2, 0u};
	const size_t wsz = sslm_workspace_size(model, &cfg);
	void* ws_buf = ::operator new(wsz, std::align_val_t(SSLM_ABI_ALIGNMENT_BYTES));
	sslm_workspace ws = nullptr;
	if (sslm_workspace_create(model, &cfg, ws_buf, wsz, &ws) != SSLM_OK) DIE("ws");
	Hook hook;
#ifdef SSLM_PARALLEL_FOR_MATVEC
	superslm::test::SetMinRowBytesPerTask(8 * 1024);
	const sslm_parallel_for pf{&Hook::Run, &hook, 4, SSLM_PARALLEL_FOR_MATVEC};
	if (sslm_workspace_set_parallel_for(ws, &pf) != SSLM_OK) DIE("install");
	const char* lib = "d1-threaded";
#else
	const char* lib = "v1.9.0-nohook";
#endif
	auto save = [&](sslm_seq s) {
		std::vector<uint8_t> b(sslm_seq_state_size(model));
		size_t n = b.size();
		if (sslm_seq_save(s, b.data(), &n) != SSLM_OK) DIE("save");
		b.resize(n);
		return b;
	};
	sslm_seq seq = nullptr;
	if (mode == "save") {
		if (sslm_seq_create(model, &pool, &seq) != SSLM_OK) DIE("seq");
		for (int i = 0; i < 64; ++i) {
			const int32_t tok = static_cast<int32_t>((i * 29 + 3) % 256);
			int32_t consumed = 0;
			const sslm_status st = sslm_prefill(model, seq, &tok, 1, 1, SSLM_SPAN_PROMPT, ws, &consumed);
			if (st != SSLM_OK || consumed != 1) DIE("prefill %d: %d", i, static_cast<int>(st));
		}
		WriteAll(argv[3], save(seq));
		std::printf("%s save: 64 one-token prefills, %ld run calls\n", lib, hook.calls);
	} else if (mode == "cont" && argc >= 6) {
		std::vector<uint8_t> in = ReadAll(argv[3]);
		const sslm_status rs = sslm_seq_restore(model, &pool, in.data(), in.size(), &seq);
		if (rs != SSLM_OK) DIE("restore: %d", static_cast<int>(rs));
		WriteAll(argv[4], save(seq));
		sslm_decode_params params{};
		sslm_decode_params_init(model, SSLM_DECODE_MODE_GREEDY, 2, &params);
		std::printf("%s cont: tokens", lib);
		for (int i = 0; i < 8; ++i) {
			int32_t tok = -2;
			const sslm_status st = sslm_decode_step_v2(model, &seq, 1, &params, ws, &tok);
			if (st != SSLM_OK) DIE("decode %d: %d", i, static_cast<int>(st));
			std::printf(" %d", tok);
		}
		WriteAll(argv[5], save(seq));
		std::printf("; %ld run calls\n", hook.calls);
	} else {
		DIE("mode");
	}
	sslm_seq_release(seq);
	sslm_workspace_destroy(ws);
	sslm_kv_pool_destroy(pool);
	sslm_model_unmap(model);
	return 0;
}
