// Multi-row GPU plan (rev 1.3.2), cell 9.2 at MR3 (against the base tag): blobs written by the tag's binary restore
// on the branch, and the reverse. Built twice by mr3_wine.sh, from the base tag's tree and from the branch (with
// -DMR3_BRANCH, where the branch's blobs are written through embed_rows chunks).
//
//   --write=DIR   writes <model>_slm5.blob (dedicated) and <model>_slm6.blob (pooled) after a 21-token prompt: one row
//                 at a time on the tag; chunks of 8, 8 and 5 (one layer per call) on the branch
//   --read=DIR    restores each blob in DIR, re-saves it (FNV printed), finishes and greedy-decodes 12 tokens, and
//                 prints one line per blob; the runner compares the lines of every (writer, reader) pair
// Every printed line is "blob <model> <kind> resave <fnv> tokens <t0 .. t12>"; a failed call prints "FAIL ...".
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "superslm/gpu_1p0.h"
#include "superslm/gpu_1p0_bench_bridge.h"
#include "superslm/gpu_port.h"
#include "superslm/model.h"
#if defined(MR3_BRANCH)
#include "mr2_entry.h"
#endif

namespace {
using S = SslmGpuStatus;
bool ReadFile(const std::string& p, std::vector<uint8_t>* out) {
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
bool WriteFile(const std::string& p, const std::vector<uint8_t>& b) {
	std::FILE* f = std::fopen(p.c_str(), "wb");
	if (!f) return false;
	const bool ok = std::fwrite(b.data(), 1, b.size(), f) == b.size();
	std::fclose(f);
	return ok;
}
uint64_t Fnv(const std::vector<uint8_t>& b) {
	uint64_t h = 1469598103934665603ull;
	for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
	return h;
}
int32_t Tok(int i, int32_t V) { return static_cast<int32_t>((1103515245u * static_cast<uint32_t>(i + 7) + 12345u + 7919u * 92u) % static_cast<uint32_t>(V)); }
bool Save(SslmGpuContext* ctx, SslmGpuSequenceHandle* s, std::vector<uint8_t>* b) {
	size_t need = 0;
	sslm_gpu_seq_save(ctx, s, nullptr, &need);
	b->assign(need ? need : 1, 0);
	size_t sz = b->size();
	if (sslm_gpu_seq_save(ctx, s, b->data(), &sz) != S::SSLM_OK) return false;
	b->resize(sz);
	return true;
}
S Drain(SslmGpuContext* ctx, SslmGpuSequenceHandle* s) {
	int32_t ready = 0;
	S out = S::SSLM_OK;
	const S st = sslm_gpu_ready(ctx, s, 1, &ready, &out);
	return st != S::SSLM_OK ? st : out;
}
// The 21-token prompt: one row at a time (tag), or chunks of 8, 8, 5 (branch).
bool Prompt(SslmGpuContext* ctx, SslmGpuSequenceHandle* s, int32_t V, uint32_t dpl, uint32_t L) {
#if defined(MR3_BRANCH)
	int at = 0;
	for (int M : {8, 8, 5}) {
		std::vector<int32_t> rows;
		for (int i = 0; i < M; ++i) rows.push_back(Tok(at + i, V));
		SslmGpuRowsSpec spec{sizeof(SslmGpuRowsSpec), SSLM_GPU_ROWS_PROMPT, rows.data(), M, 0u};
		int32_t adm = 0;
		if (sslm_gpu_seq_embed_rows(ctx, s, &spec, &adm) != S::SSLM_OK || adm != M) return false;
		for (int g = 0; g < 64; ++g) {
			if (sslm_decode_step_gpu(ctx, s, nullptr, dpl) != S::SSLM_OK || Drain(ctx, s) != S::SSLM_OK) return false;
			int32_t open = -1;
			sslm_gpu_seq_rows_open(s, &open);
			if (open == 0) break;
		}
		at += M;
	}
	return true;
#else
	for (int i = 0; i < 21; ++i) {
		if (sslm_gpu_seq_embed_token(ctx, s, Tok(i, V)) != S::SSLM_OK) return false;
		while (*SslmGpuSeqHandleLayerIndexForBench(s) < L)
			if (sslm_decode_step_gpu(ctx, s, nullptr, dpl) != S::SSLM_OK || Drain(ctx, s) != S::SSLM_OK) return false;
	}
	return true;
#endif
}
}  // namespace

int main(int argc, char** argv) {
	std::vector<std::string> models;
	std::string wdir, rdir;
	for (int i = 1; i < argc; ++i) {
		const std::string a = argv[i];
		if (a.rfind("--model=", 0) == 0) models.push_back(a.substr(8));
		else if (a.rfind("--write=", 0) == 0) wdir = a.substr(8);
		else if (a.rfind("--read=", 0) == 0) rdir = a.substr(7);
	}
	int fails = 0;
	for (const std::string& path : models) {
		std::vector<uint8_t> bytes;
		superslm::SslmModelView view{};
		std::string err;
		if (!ReadFile(path, &bytes) || superslm::SslmModel::Load(bytes.data(), bytes.size(), view, &err) != superslm::SslmModelStatus::Ok) {
			std::printf("FAIL load %s\n", path.c_str());
			return 1;
		}
		bool qk = false;
		for (uint32_t l = 0; l < view.config.num_hidden_layers; ++l)
			if (view.weights.Tensor("layer" + std::to_string(l) + ".q_norm.gain")) qk = true;
		const uint32_t L = view.config.num_hidden_layers, dpl = superslm_gpu::DispatchesPerLayer(qk);
		const int32_t V = static_cast<int32_t>(view.config.vocab_size);
		std::string base = path.substr(path.find_last_of("/\\") + 1);
		base = base.substr(0, base.find('.'));
		SslmGpuContext* ctx = nullptr;
		SslmGpuModelHandle* model = nullptr;
		SslmGpuKvPoolHandle* pool = nullptr;
		if (sslm_gpu_context_create(GpuContextConfig{}, &ctx) != S::SSLM_OK || sslm_gpu_model_map(ctx, &view, GpuResidencyConfig{}, &model) != S::SSLM_OK ||
		    sslm_gpu_kv_pool_create(ctx, model, 24, &pool) != S::SSLM_OK) {
			std::printf("FAIL context %s\n", base.c_str());
			return 1;
		}
		std::vector<SslmGpuSequenceHandle*> live;
		for (int pooled = 0; pooled <= 1; ++pooled) {
			const char* kind = pooled ? "slm6" : "slm5";
			const std::string file = base + "_" + kind + ".blob";
			if (!wdir.empty()) {
				SslmGpuSequenceHandle* s = nullptr;
				const S c = pooled ? sslm_gpu_seq_create_budgeted(ctx, model, pool, 64, &s) : sslm_gpu_seq_create(ctx, model, 64, &s);
				std::vector<uint8_t> b;
				if (c != S::SSLM_OK || !Prompt(ctx, s, V, dpl, L) || !Save(ctx, s, &b) || !WriteFile(wdir + "/" + file, b)) {
					std::printf("FAIL write %s %s\n", base.c_str(), kind);
					++fails;
				} else {
					std::printf("wrote %s %s %016llx\n", base.c_str(), kind, (unsigned long long)Fnv(b));
				}
				if (s) live.push_back(s);
			}
			if (!rdir.empty()) {
				std::vector<uint8_t> b, re;
				SslmGpuSequenceHandle* s = nullptr;
				uint32_t shared = 0;
				if (!ReadFile(rdir + "/" + file, &b)) { std::printf("FAIL read %s\n", file.c_str()); ++fails; continue; }
				const S st = pooled ? sslm_gpu_seq_restore_into_pool(ctx, model, pool, b.data(), b.size(), nullptr, &s, &shared)
				                    : sslm_gpu_seq_restore(ctx, model, b.data(), b.size(), &s);
				if (st != S::SSLM_OK || !s || !Save(ctx, s, &re)) { std::printf("FAIL restore %s %s: %u\n", base.c_str(), kind, (unsigned)st); ++fails; continue; }
				live.push_back(s);
				std::string toks;
				int32_t t = -1;
				if (SslmGpuSeqFinishTokenForG5Bridge(ctx, s, &t) != S::SSLM_OK) { std::printf("FAIL finish %s %s\n", base.c_str(), kind); ++fails; continue; }
				toks += std::to_string(t);
				for (int i = 0; i < 12; ++i) {
					int32_t o = -1;
					if (SslmGpuSeqDecodeStepForG5Bridge(ctx, s, t, dpl * L, &o) != S::SSLM_OK) { ++fails; break; }
					toks += " " + std::to_string(o);
					t = o >= 0 ? o : Tok(i, V);
				}
				std::printf("blob %s %s resave %016llx tokens %s\n", base.c_str(), kind, (unsigned long long)Fnv(re), toks.c_str());
			}
		}
		for (auto* s : live) sslm_gpu_seq_release(ctx, s);
		sslm_gpu_kv_pool_destroy(ctx, pool);
		sslm_gpu_model_unmap(ctx, model);
		sslm_gpu_context_destroy(ctx);
	}
	return fails ? 1 : 0;
}
