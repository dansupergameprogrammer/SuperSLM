// Multi-row GPU plan (rev 1.3.2), cell 8.3 at MR4: Canonical over the fixture's order spans gives the tokens and the
// bytes of 1.17's one-row backend. Built twice by mr4_wine.sh, from 1.17's tree (origin/claude/sfd-gpu-1p17, the R-17
// base) and from the branch, from this one source, which uses only 1.17's public verbs: the canonical loop (plan SFD
// §5.6: sslm_gpu_seq_jump_forward until sslm_gpu_seq_forced_tokens reports no chain, then one decode-bridge token)
// on F-SFD's schemas, at chunk budgets 1, 2, 3, 4, 5, 8 and 64, dedicated and pooled. On the branch the verb's rows
// run on the M-row recorder; on 1.17 they run on the schema-content chunk path. The runner requires the two binaries'
// lines to be equal, and every stream to equal the CPU's pinned stream where 1.17's suite pins one (G7.1's digests).
//
// Each line: "canon <schema> <dedicated|pooled> budget <b> tokens <n> <sha256 of the LE32 ids> end <status> blob <fnv>
// walk <state> ready <flag> ctx <len>", after a 41-token-or-less prompt of the 1.17 suite's own shape. A failed call
// prints "FAIL ...".
// Usage: mr4_83_canon.exe --fixtures=DIR
// MRX (E-11, ruling 20): RE-POINT to an Exact loop. The canonical loop here passes opts NULL, which MRX makes Exact;
// the GPU refuses Exact until MR5, so after MRX this file's branch binary stops at its first jump. It stays as 1.17's
// reference (mrx_gpu.cpp's recorder is its successor for saves); an Exact loop against the same 1.17 lines replaces it
// at MR5. Marked, not deleted (red-suite-mrx.md).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#include "superslm/sha256.h"

namespace {
using S = SslmGpuStatus;
bool ReadAll(const std::string& p, std::vector<uint8_t>* out) {
	std::FILE* f = std::fopen(p.c_str(), "rb");
	if (!f) return false;
	std::fseek(f, 0, SEEK_END);
	const long n = std::ftell(f);
	std::fseek(f, 0, SEEK_SET);
	out->resize(n > 0 ? static_cast<size_t>(n) : 0);
	const bool ok = n > 0 && std::fread(out->data(), 1, out->size(), f) == out->size();
	std::fclose(f);
	return ok;
}
uint64_t Fnv(const std::vector<uint8_t>& b) {
	uint64_t h = 1469598103934665603ull;
	for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
	return h;
}
std::string Digest(const std::vector<int32_t>& ids) {
	superslm::Sha256 h;
	for (int32_t x : ids) {
		const uint32_t u = static_cast<uint32_t>(x);
		const uint8_t b[4] = {static_cast<uint8_t>(u), static_cast<uint8_t>(u >> 8), static_cast<uint8_t>(u >> 16), static_cast<uint8_t>(u >> 24)};
		h.Update(b, 4);
	}
	uint8_t out[32];
	h.Final(out);
	std::string s;
	char x[3];
	for (uint8_t c : out) { std::snprintf(x, sizeof x, "%02x", c); s += x; }
	return s;
}
S Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* s) {
	int32_t ready = 0;
	S out = S::SSLM_OK;
	const S st = sslm_gpu_ready(ctx, s, 1, &ready, &out);
	return st != S::SSLM_OK ? st : out;
}
bool Save(SslmGpuContext* ctx, SslmGpuSequenceHandle* s, std::vector<uint8_t>* b) {
	size_t need = 0;
	sslm_gpu_seq_save(ctx, s, nullptr, &need);
	b->assign(need ? need : 1, 0);
	size_t sz = b->size();
	if (sslm_gpu_seq_save(ctx, s, b->data(), &sz) != S::SSLM_OK) return false;
	b->resize(sz);
	return true;
}
}  // namespace

int main(int argc, char** argv) {
	std::string dir;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--fixtures=", 0) == 0) dir = a.substr(11);
	}
	std::vector<uint8_t> bytes;
	superslm::SslmModelView view{};
	std::string err;
	if (!ReadAll(dir + "/fsfd.sslm", &bytes) || superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
		std::printf("FAIL load fsfd.sslm %s\n", err.c_str());
		return 1;
	}
	const uint32_t L = view.config.num_hidden_layers;
	const uint32_t full = superslm_gpu::DispatchesPerLayer(false) * L;  // F-SFD has no QK norm
	SslmGpuContext* ctx = nullptr;
	SslmGpuModelHandle* model = nullptr;
	if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != S::SSLM_OK || sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != S::SSLM_OK) {
		std::printf("FAIL context\n");
		return 1;
	}
	// The prompts: ordinary text ids (the stream shape the multi-row suites use), one per schema.
	auto Prompt = [](int n, uint32_t stream) {
		std::vector<int32_t> v;
		for (int i = 0; i < n; ++i) v.push_back(static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u + 7919u * stream) % 20000u + 200u));
		return v;
	};
	int fails = 0;
	for (const char* schema : {"potion_shop_order", "prompt_result", "longkey", "utf8enum"})
		for (int pooled = 0; pooled <= 1; ++pooled)
			for (int budget : {1, 2, 3, 4, 5, 8, 64}) {
				SslmGpuKvPoolHandle* pool = nullptr;
				SslmGpuSequenceHandle* s = nullptr;
				S c = pooled ? sslm_gpu_kv_pool_create(ctx, model, 64, &pool) : S::SSLM_OK;
				if (c == S::SSLM_OK) c = pooled ? sslm_gpu_seq_create_budgeted(ctx, model, pool, 240, &s) : sslm_gpu_seq_create(ctx, model, 240, &s);
				const int32_t idx = SslmGpuSchemaLookupForG5Bridge(model, schema);
				if (c == S::SSLM_OK) c = SslmGpuSeqSetSchemaForG5Bridge(ctx, s, idx);
				const std::vector<int32_t> p = Prompt(12, 83u + static_cast<uint32_t>(idx));
				if (c == S::SSLM_OK) c = SslmGpuSeqPrefillPromptForG5Bridge(ctx, s, p.data(), static_cast<int32_t>(p.size()), full);
				if (c != S::SSLM_OK) {
					std::printf("FAIL setup %s %d %d: %u\n", schema, pooled, budget, (unsigned)c);
					++fails;
					if (s) sslm_gpu_seq_release(ctx, s);
					if (pool) sslm_gpu_kv_pool_destroy(ctx, pool);
					continue;
				}
				std::vector<int32_t> stream;
				int32_t pending = -1, last = p.back();
				S end = S::SSLM_OK;
				for (int d = 0; d < 200 && end == S::SSLM_OK; ++d) {
					for (int g = 0; g < 400; ++g) {
						std::vector<int32_t> out(static_cast<size_t>(budget > 64 ? budget : 64), -7);
						int32_t n = 0;
						end = sslm_gpu_seq_jump_forward(ctx, s, pending, budget, nullptr, out.data(), static_cast<int32_t>(out.size()), &n);
						if (end != S::SSLM_OK) break;
						stream.insert(stream.end(), out.begin(), out.begin() + n);
						int32_t ready = 0;
						sslm_gpu_seq_ready_for_logits(ctx, s, &ready);
						if (ready) pending = -1;
						int32_t count = 0, acc = 0;
						end = sslm_gpu_seq_forced_tokens(ctx, s, nullptr, 0, &count, &acc);
						if (end != S::SSLM_OK || count == 0) break;
					}
					if (end != S::SSLM_OK) break;
					int32_t t = -1;
					end = SslmGpuSeqDecodeStepForG5Bridge(ctx, s, pending >= 0 ? pending : last, full, &t);
					if (end != S::SSLM_OK || t < 0) break;  // -2: a dead end, the stream's end
					stream.push_back(t);
					pending = last = t;
				}
				std::vector<uint8_t> blob;
				if (!Save(ctx, s, &blob)) { std::printf("FAIL save %s\n", schema); ++fails; }
				int32_t ready = -1;
				sslm_gpu_seq_ready_for_logits(ctx, s, &ready);
				std::printf("canon %s %s budget %d tokens %zu %s end %u blob %016llx walk %u ready %d ctx %lld\n", schema, pooled ? "pooled" : "dedicated",
				            budget, stream.size(), Digest(stream).c_str(), (unsigned)end, (unsigned long long)Fnv(blob), SslmGpuSeqWalkStateForG5Bridge(s), ready,
				            (long long)*SslmGpuSeqHandleContextLengthForBench(s));
				(void)Drain;
				sslm_gpu_seq_release(ctx, s);
				if (pool) sslm_gpu_kv_pool_destroy(ctx, pool);
			}
	sslm_gpu_model_unmap(ctx, model);
	sslm_gpu_context_destroy(ctx);
	return fails ? 1 : 0;
}
