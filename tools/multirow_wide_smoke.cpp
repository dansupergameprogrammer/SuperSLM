// Multi-row GPU plan (rev 1.3.2), slice MR1 gate: "Wine one-layer smoke equal on pkv_def and pkv_qk". Drives
// the test-only entry sslm_gpu_test_only_mr1_wide_one_layer_smoke (src/gpu/multirow_wide.h) over a grid of
// shapes and reports each. Build with -DSUPERSLM_MULTIROW_WIDE over the engine sources (the entry exists only
// there); tools/multirow_wide_smoke.sh builds it with mingw-w64 and runs it under Wine's software GPU.
//
// Each case: two sequences of one model fed the same p0 tokens to full depth, then M row tokens. The
// reference runs each row one-row (its first layer alone, then the rest); the other runs the first layer for
// all M rows as one widened submission at variant H. A case passes when every real row's SeqState equals
// one-row's byte for byte, every padded row (M < H) leaves as it entered, every real row's first-layer K/V
// rows equal one-row's, and no other K/V byte changed.
//
// Grid: H/M in {2/2, 4/4, 8/8, 4/3, 8/5}, p0 in {0, 3, 13} (13 + 8 rows crosses a 16-position page), each on
// dedicated sequences and on two sequences of one K/V pool (16-position pages, so the two sequences'
// pages interleave and every K/V address goes through the page table).
//
// Usage: multirow_wide_smoke.exe --model=PATH [--adapter=PATH] [--shaders=DIR] [--quick]
//   --adapter  decode both sequences with this runtime adapter (tools/gen_multirow_adapter_fixture.py)
//   --quick    p0 = 3 only
// Exit 0 when every case passes.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "multirow_wide.h"
#include "superslm/gpu_1p0.h"
#include "superslm/model.h"

#if !defined(SUPERSLM_MULTIROW_WIDE)
#error "multirow_wide_smoke needs the engine built with -DSUPERSLM_MULTIROW_WIDE"
#endif

using enum SslmGpuStatus;

namespace {

bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(path.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long sz = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(sz > 0 ? static_cast<size_t>(sz) : 0);
	const size_t n = sz > 0 ? std::fread(out->data(), 1, out->size(), f) : 0;
	std::fclose(f);
	return n == out->size();
}

// A fixed token stream inside the vocabulary (the twin capture's, tests/multirow-gpu/mr1_twin_capture.cpp).
int32_t TokenAt(int i, int32_t vocab) {
	return static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u) % static_cast<uint32_t>(vocab));
}

constexpr uint32_t kFullDepth = 1u << 20;  // any budget at or above a whole token's dispatches

SslmGpuStatus FeedFullDepth(SslmGpuContext* ctx, SslmGpuSequenceHandle* seq, const SslmGpuAdapterHandle* adapter,
                            int32_t token) {
	SslmGpuStatus st = sslm_gpu_seq_embed_token(ctx, seq, token);
	if (st != SSLM_OK) return st;
	if ((st = sslm_decode_step_gpu(ctx, seq, adapter, kFullDepth)) != SSLM_OK) return st;
	int32_t ready = 0;
	SslmGpuStatus slice = SSLM_OK;
	if ((st = sslm_gpu_ready(ctx, seq, 1, &ready, &slice)) != SSLM_OK) return st;
	return ready ? slice : SSLM_DEVICE_LOST;
}

struct Case {
	uint32_t h, m;
	int p0;
	bool pooled;
};

}  // namespace

int main(int argc, char** argv) {
	std::string model_path, adapter_path, shaders;
	bool quick = false;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
		else if (a.rfind("--adapter=", 0) == 0) adapter_path = a.substr(10);
		else if (a.rfind("--shaders=", 0) == 0) shaders = a.substr(10);
		else if (a == "--quick") quick = true;
		else {
			std::printf("usage: multirow_wide_smoke --model=PATH [--adapter=PATH] [--shaders=DIR] [--quick]\n");
			return 2;
		}
	}
	std::vector<uint8_t> model_bytes, adapter_bytes;
	superslm::SslmModelView view{}, adapter_view{};
	std::string err;
	if (model_path.empty() || !ReadFileBytes(model_path, &model_bytes) ||
	    superslm::SslmModel::Load(model_bytes.data(), model_bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
		std::printf("FAIL cannot load model '%s' %s\n", model_path.c_str(), err.c_str());
		return 1;
	}
	if (!adapter_path.empty() &&
	    (!ReadFileBytes(adapter_path, &adapter_bytes) ||
	     superslm::SslmModel::Load(adapter_bytes.data(), adapter_bytes.size(), adapter_view, &err) !=
	         superslm::SslmModelStatus::Ok)) {
		std::printf("FAIL cannot load adapter '%s' %s\n", adapter_path.c_str(), err.c_str());
		return 1;
	}
	const int32_t vocab = static_cast<int32_t>(view.config.vocab_size);

	GpuContextConfig cfg{};
	cfg.shader_dir = shaders.empty() ? nullptr : shaders.c_str();
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	SslmGpuAdapterHandle* adapter = nullptr;
	SslmGpuKvPoolHandle* pool = nullptr;
	if (sslm_gpu_context_create(cfg, &ctx) != SSLM_OK || sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != SSLM_OK) {
		std::printf("FAIL context or model map\n");
		return 1;
	}
	if (!adapter_path.empty() && sslm_gpu_adapter_map(ctx, model, &adapter_view, &adapter) != SSLM_OK) {
		std::printf("FAIL adapter map\n");
		return 1;
	}

	std::vector<Case> cases;
	const int p0s_full[] = {0, 3, 13};
	const int p0s_quick[] = {3};
	const uint32_t shapes[][2] = {{2, 2}, {4, 4}, {8, 8}, {4, 3}, {8, 5}};
	for (int pooled = 0; pooled < 2; ++pooled) {
		for (const auto& s : shapes) {
			if (quick) {
				for (int p0 : p0s_quick) cases.push_back({s[0], s[1], p0, pooled != 0});
			} else {
				for (int p0 : p0s_full) cases.push_back({s[0], s[1], p0, pooled != 0});
			}
		}
	}
	int failed = 0;
	for (const Case& c : cases) {
		SslmGpuSequenceHandle* seqs[2] = {nullptr, nullptr};
		bool ok = true;
		// A fresh pool per case: a widened run writes device K/V its released sequence never read back, so
		// a reused pool's mirror would not describe the device bytes the stray check compares against.
		if (c.pooled && sslm_gpu_kv_pool_create(ctx, model, 16, &pool) != SSLM_OK) {
			std::printf("FAIL kv pool create\n");
			return 1;
		}
		for (SslmGpuSequenceHandle*& s : seqs) {
			const SslmGpuStatus st =
			    c.pooled ? sslm_gpu_seq_create_budgeted(ctx, model, pool, 32, &s) : sslm_gpu_seq_create(ctx, model, 32, &s);
			if (st != SSLM_OK) {
				std::printf("FAIL sequence create -> %d\n", static_cast<int>(st));
				ok = false;
			} else if (adapter && sslm_gpu_seq_bind_adapter(ctx, s, adapter) != SSLM_OK) {
				std::printf("FAIL adapter bind\n");
				ok = false;
			}
		}
		// The same prefix on both, interleaved token by token so pooled pages alternate between them.
		for (int i = 0; ok && i < c.p0; ++i) {
			for (SslmGpuSequenceHandle* s : seqs) {
				const SslmGpuStatus st = FeedFullDepth(ctx, s, adapter, TokenAt(i, vocab));
				if (st != SSLM_OK) {
					std::printf("FAIL prefix token %d -> %d\n", i, static_cast<int>(st));
					ok = false;
					break;
				}
			}
		}
		std::vector<int32_t> rows(c.m);
		for (uint32_t r = 0; r < c.m; ++r) rows[r] = TokenAt(c.p0 + static_cast<int>(r), vocab);
		SslmGpuMr1WideSmokeReport rep{};
		if (ok) {
			const SslmGpuStatus st =
			    sslm_gpu_test_only_mr1_wide_one_layer_smoke(ctx, seqs[0], seqs[1], adapter, rows.data(), c.m, c.h, &rep);
			if (st != SSLM_OK) {
				std::printf("FAIL smoke call -> %d (%s)\n", static_cast<int>(st), rep.detail);
				ok = false;
			}
		}
		const bool equal = ok && rep.rows_equal == c.m && rep.padded_rows_inert == c.h - c.m &&
		                   rep.kv_row_bytes_compared > 0 && rep.kv_row_bytes_differing == 0 && rep.kv_stray_bytes == 0;
		std::printf("%s H=%u M=%u p0=%d %s%s: rows equal %u/%u, padded inert %u/%u, K/V row bytes %llu (%llu differ), "
		            "stray %llu %s\n",
		            equal ? "PASS" : "FAIL", c.h, c.m, c.p0, c.pooled ? "pooled" : "dedicated",
		            adapter ? " +adapter" : "", rep.rows_equal, c.m, rep.padded_rows_inert, c.h - c.m,
		            static_cast<unsigned long long>(rep.kv_row_bytes_compared),
		            static_cast<unsigned long long>(rep.kv_row_bytes_differing),
		            static_cast<unsigned long long>(rep.kv_stray_bytes), rep.detail);
		if (!equal) ++failed;
		for (SslmGpuSequenceHandle* s : seqs) {
			if (s) sslm_gpu_seq_release(ctx, s);
		}
		if (pool) sslm_gpu_kv_pool_destroy(ctx, pool);
		pool = nullptr;
	}
	if (adapter) sslm_gpu_adapter_unmap(ctx, adapter);
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
	std::printf("MR1 one-layer smoke: %s (%zu cases, %d failed)\n", failed ? "FAIL" : "PASS", cases.size(), failed);
	return failed ? 1 : 0;
}
